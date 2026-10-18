"""Frozen development and confirmation benchmark for structural auto policy v3."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Iterable

from spectra.exact import (
    ExactProblem,
    ExactRuntime,
    decode_certified_output,
    pack_session_packet,
    verify_session_output,
)
from .binary_list import BinaryListProblem
from .cases import Case
from .transfer_panel import FAMILIES, make_case, topology

COUNTS = (1, 8, 32, 63, 64, 65, 128, 1024)
CONFIRMATION_COUNTS = (1, 8, 64, 128, 1024)
MODES = ("raw", "scc", "route", "auto")
REPEATS = 17
WARMUPS = 2
SCHEMA = "spectra.auto_selector_benchmark.v1"


def seed_for(label: str) -> int:
    payload = ("SPECTRA-auto-selector-v3-confirmation/" + label).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def gmean(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values or any(value <= 0 for value in values):
        raise ValueError("positive nonempty values required")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def confirmation_case(family: str, replica: int) -> Case:
    """Generate one query-blind case after the v3 policy has been frozen."""
    if family not in FAMILIES or replica not in range(4):
        raise ValueError("fixed confirmation family/replica required")
    n = 256 if replica < 2 else 1024
    rng = random.Random(seed_for(f"{family}/{replica}"))
    candidate_edges = topology(family, n, rng)

    side = [rng.randrange(2) for _ in range(n)]
    side[0], side[1] = 0, 1
    edges = {edge for edge in candidate_edges if side[edge[0]] != side[edge[1]]}
    left = [vertex for vertex, value in enumerate(side) if value == 0]
    right = [vertex for vertex, value in enumerate(side) if value == 1]
    # Preserve connectivity pressure without sacrificing the known reference
    # colouring: every vertex receives one deterministic opposite-side bridge.
    for vertex in range(n):
        pool = right if side[vertex] == 0 else left
        other = pool[(vertex * 17 + replica * 13) % len(pool)]
        if vertex != other:
            edges.add((min(vertex, other), max(vertex, other)))

    palette = 8
    if replica == 0:
        masks = [3] * n  # strong two-colour contraction
    else:
        masks = []
        for vertex in range(n):
            reference = side[vertex]
            if vertex % 11 == 0:
                masks.append(1 << reference)
            else:
                alternate = rng.randrange(2, palette)
                masks.append((1 << reference) | (1 << alternate))

    if replica == 3:
        # One prospectively specified base-infeasible regime.
        masks[:3] = [3, 3, 3]
        edges.update(((0, 1), (0, 2), (1, 2)))

    problem = BinaryListProblem.create(n, palette, tuple(sorted(edges)), masks)
    widths = {
        0: (0, 1, 2, 4),
        1: (2, 4, 6, 7),
        2: (8, 12, 16, 24),
        3: (0, 1, 2, 4),
    }[replica]
    queries = []
    for index in range(1024):
        width = widths[index % len(widths)]
        row = []
        if width:
            for vertex in rng.sample(range(n), min(width, n)):
                if index % 5 == 0:
                    allowed = masks[vertex]
                else:
                    allowed = 1 << side[vertex]
                row.append((vertex, allowed))
        # Deterministic mixed outcomes on otherwise feasible replicas.
        if replica != 3 and (index + 1) % 31 == 0:
            vertex = (index * 29) % n
            row.append((vertex, 0))
        if row and (index + 1) % 17 == 0:
            row.append(row[0])  # duplicate-pin normalization path
        queries.append(tuple(row))

    name = f"confirm-{family}-r{replica}-n{n}"
    return Case(
        name,
        name,
        "prospective auto-selector-v3 confirmation; generated without solver calls",
        problem,
        tuple(queries),
    )


def cases(panel: str) -> tuple[Case, ...]:
    if panel == "development":
        return tuple(make_case(family, replica) for family in FAMILIES for replica in range(6))
    if panel == "confirmation":
        return tuple(confirmation_case(family, replica) for family in FAMILIES for replica in range(4))
    raise ValueError("panel must be development or confirmation")


def decisions(payload: bytes, n: int, count: int) -> tuple[int, ...]:
    return tuple(row[0] for row in decode_certified_output(payload, n, count))


def run(library: Path, panel: str, destination: Path, repeats: int = REPEATS) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    if repeats < 9 or repeats % 2 == 0:
        raise ValueError("odd repeats >= 9 required")
    destination.mkdir(parents=True)
    runtime = ExactRuntime(library)
    selected_counts = COUNTS if panel == "development" else CONFIRMATION_COUNTS
    records = []

    gc.disable()
    try:
        for case in cases(panel):
            problem = ExactProblem.create(
                case.problem.n,
                case.problem.palette,
                case.problem.edges,
                case.problem.masks,
            )
            family = case.name.split("-r", 1)[0].replace("confirm-", "")
            for count in selected_counts:
                rows = case.queries[:count]
                packet = pack_session_packet(problem, rows)
                first = {
                    mode: runtime.solve_session_packet(packet, mode=mode)
                    for mode in MODES
                }
                reference_decisions = decisions(first["raw"].output, problem.n, count)
                for mode, answer in first.items():
                    if not verify_session_output(packet, answer.output):
                        raise AssertionError(f"independent replay failed: {case.name}/{count}/{mode}")
                    if decisions(answer.output, problem.n, count) != reference_decisions:
                        raise AssertionError(f"decision disagreement: {case.name}/{count}/{mode}")
                selected = first["auto"].info["selected_mode"]
                reason = first["auto"].info["auto_policy_reason"]
                if selected in ("raw", "scc") and first["auto"].output != first[selected].output:
                    raise AssertionError("auto output differs from selected explicit engine")

                for _ in range(WARMUPS):
                    for mode in MODES:
                        warm_packet = pack_session_packet(problem, rows)
                        runtime.solve_session_packet(warm_packet, mode=mode)

                samples = {mode: [] for mode in MODES}
                output_hashes = {mode: set() for mode in MODES}
                schedule = random.Random(seed_for(f"schedule/{panel}/{case.name}/{count}"))
                for _ in range(repeats):
                    order = list(MODES)
                    schedule.shuffle(order)
                    for mode in order:
                        started = time.perf_counter_ns()
                        current_packet = pack_session_packet(problem, rows)
                        answer = runtime.solve_session_packet(current_packet, mode=mode)
                        elapsed = time.perf_counter_ns() - started
                        samples[mode].append(elapsed)
                        output_hashes[mode].add(hashlib.sha256(answer.output).hexdigest())
                        if not verify_session_output(current_packet, answer.output):
                            raise AssertionError("timed output replay failed")

                if any(len(values) != 1 for values in output_hashes.values()):
                    raise AssertionError("output bytes changed across repetitions")
                medians = {mode: statistics.median(values) for mode, values in samples.items()}
                oracle = min(("raw", "scc", "route"), key=medians.get)
                pins = sum(map(len, rows))
                info = first["auto"].info
                record = {
                    "panel": panel,
                    "case": case.name,
                    "family": family,
                    "count": count,
                    "vertices": problem.n,
                    "edges": len(problem.edges),
                    "pins": pins,
                    "pins_per_request": pins / count,
                    "base_satisfiable": bool(info.get("base_satisfiable", True)),
                    "components": info.get("scc_components"),
                    "selected_mode": selected,
                    "selection_reason": reason,
                    "oracle_mode": oracle,
                    "medians_ns": medians,
                    "selection_regret": medians[selected] / medians[oracle],
                    "realized_regret": medians["auto"] / medians[oracle],
                    "auto_raw_ratio": medians["auto"] / medians["raw"],
                    "samples_ns": samples,
                    "output_sha256": {mode: next(iter(values)) for mode, values in output_hashes.items()},
                }
                records.append(record)
                print(
                    panel,
                    case.name,
                    count,
                    f"selected={selected}",
                    f"oracle={oracle}",
                    f"regret={record['realized_regret']:.4f}",
                    flush=True,
                )
    finally:
        gc.enable()

    realized = [row["realized_regret"] for row in records]
    selection = [row["selection_regret"] for row in records]
    grouped = defaultdict(list)
    family_raw = defaultdict(list)
    for row in records:
        grouped[(row["family"], row["count"])].append(row["realized_regret"])
        family_raw[row["family"]].append(row["auto_raw_ratio"])

    limits = (
        {"global": 1.03, "group": 1.05, "fraction": .95, "worst": 1.10, "selection": 1.02}
        if panel == "development"
        else {"global": 1.05, "group": 1.07, "fraction": .90, "worst": 1.12, "selection": 1.03}
    )
    rule_failures = []
    for row in records:
        selected, reason, count = row["selected_mode"], row["selection_reason"], row["count"]
        if reason in ("base_infeasible", "high_pin_density", "short_weak_compression") and selected != "raw":
            rule_failures.append(f"{row['case']}/{count}: {reason} selected {selected}")
        if reason in ("scc_tile64", "scc_strong_compression") and selected != "scc":
            rule_failures.append(f"{row['case']}/{count}: {reason} selected {selected}")
        if count < 8 and selected != "raw":
            rule_failures.append(f"{row['case']}/{count}: sub-eight bank selected {selected}")

    summary = {
        "schema": SCHEMA,
        "classification": f"{panel} selector panel; confirmation is project-owned, not external replication",
        "panel": panel,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cases": len({row["case"] for row in records}),
        "cells": len(records),
        "counts": list(selected_counts),
        "repeats": repeats,
        "global_realized_regret_geomean": gmean(realized),
        "global_selection_regret_geomean": gmean(selection),
        "worst_realized_regret": max(realized),
        "fraction_within_1_05": sum(value <= 1.05 for value in realized) / len(realized),
        "worst_group_regret": max(gmean(values) for values in grouped.values()),
        "worst_family_auto_raw_ratio": max(gmean(values) for values in family_raw.values()),
        "selected_modes": {
            mode: sum(row["selected_mode"] == mode for row in records)
            for mode in ("raw", "scc", "route")
        },
        "oracle_modes": {
            mode: sum(row["oracle_mode"] == mode for row in records)
            for mode in ("raw", "scc", "route")
        },
        "limits": limits,
        "rule_failures": rule_failures,
    }
    checks = {
        "global_realized_regret": summary["global_realized_regret_geomean"] <= limits["global"],
        "group_regret": summary["worst_group_regret"] <= limits["group"],
        "fraction_within_1_05": summary["fraction_within_1_05"] >= limits["fraction"],
        "worst_cell": summary["worst_realized_regret"] <= limits["worst"],
        "selection_regret": summary["global_selection_regret_geomean"] <= limits["selection"],
        "family_not_slower_than_raw": (
            True if panel == "development" else summary["worst_family_auto_raw_ratio"] <= 1.05
        ),
        "structural_rules": not rule_failures,
    }
    summary["checks"] = checks
    summary["pass"] = all(checks.values())
    payload = {"summary": summary, "records": records}
    destination.joinpath("results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise AssertionError("automatic selector gate failed")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--panel", choices=("development", "confirmation"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args(argv)
    run(args.library.resolve(), args.panel, args.out.resolve(), repeats=args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
