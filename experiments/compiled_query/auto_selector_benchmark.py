"""Position-balanced, calibrated benchmark for structural auto policy v11."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
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

MODES = ("raw", "scc", "route", "auto")
DEVELOPMENT_COUNTS = (1, 8, 32, 63, 64, 65, 128, 1024)
CONFIRMATION_COUNTS = (1, 8, 32, 64, 128, 1024)
WARMUPS = 2
MIN_SAMPLE_NS = 1_000_000
MAX_BATCH_SIZE = 256
SCHEMA = "spectra.auto_selector_benchmark.v8"
POLICY = "structural-v11"


@dataclass(frozen=True)
class PanelCase:
    case: Case
    family: str
    regime: str
    counts: tuple[int, ...]


def seed_for(namespace: str, label: str) -> int:
    payload = (namespace + "/" + label).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def gmean(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values or any(value <= 0 for value in values):
        raise ValueError("positive nonempty values required")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _topology_at_size(family: str, n: int, rng: random.Random):
    """Preserve the family generator while admitting protocol sizes like 320/384.

    The established topology helper is defined only for powers of two. Generate
    the next power-of-two supergraph, then take its first ``n`` vertices. The
    later opposite-side bridge pass restores deterministic connectivity pressure,
    so this does not inspect requests, answers, selector decisions, or timings.
    """
    if n < 8:
        raise ValueError("selector panel requires n >= 8")
    super_n = 1 << (n - 1).bit_length()
    return tuple(
        (left, right)
        for left, right in topology(family, super_n, rng)
        if left < n and right < n
    )


def _known_bipartite_problem(
    family: str,
    label: str,
    *,
    n: int,
    strong_contraction: bool,
    base_infeasible: bool,
    namespace: str,
    minimum_edge_factor: float = 0.0,
) -> tuple[BinaryListProblem, list[int], random.Random]:
    rng = random.Random(seed_for(namespace, label))
    candidate_edges = _topology_at_size(family, n, rng)
    side = [rng.randrange(2) for _ in range(n)]
    side[0], side[1] = 0, 1
    edges = {edge for edge in candidate_edges if side[edge[0]] != side[edge[1]]}
    left = [vertex for vertex, value in enumerate(side) if value == 0]
    right = [vertex for vertex, value in enumerate(side) if value == 1]
    for vertex in range(n):
        pool = right if side[vertex] == 0 else left
        other = pool[(vertex * 17 + len(label) * 13) % len(pool)]
        if vertex != other:
            edges.add((min(vertex, other), max(vertex, other)))

    if minimum_edge_factor:
        target = math.ceil(minimum_edge_factor * n)
        while len(edges) < target:
            a = left[rng.randrange(len(left))]
            b = right[rng.randrange(len(right))]
            edges.add((min(a, b), max(a, b)))

    palette = 8
    if strong_contraction:
        masks = [3] * n
    else:
        masks = []
        for vertex in range(n):
            reference = side[vertex]
            if vertex % 13 == 0:
                masks.append(1 << reference)
            else:
                alternate = rng.randrange(2, palette)
                masks.append((1 << reference) | (1 << alternate))

    if base_infeasible:
        masks[:3] = [3, 3, 3]
        edges.update(((0, 1), (0, 2), (1, 2)))

    return (
        BinaryListProblem.create(n, palette, tuple(sorted(edges)), masks),
        side,
        rng,
    )


def _queries(
    problem: BinaryListProblem,
    side: list[int],
    rng: random.Random,
    *,
    width: int,
    base_infeasible: bool,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    rows = []
    for index in range(1024):
        row: list[tuple[int, int]] = []
        if width:
            for vertex in rng.sample(range(problem.n), min(width, problem.n)):
                allowed = problem.masks[vertex] if index % 5 == 0 else 1 << side[vertex]
                row.append((vertex, allowed))
        if not base_infeasible and (index + 1) % 31 == 0:
            row.append(((index * 29) % problem.n, 0))
        if row and (index + 1) % 17 == 0:
            row.append(row[0])
        rows.append(tuple(row))
    return tuple(rows)


def generated_case(
    family: str,
    regime: str,
    *,
    n: int,
    width: int,
    strong_contraction: bool = False,
    base_infeasible: bool = False,
    namespace: str,
    minimum_edge_factor: float = 0.0,
) -> Case:
    label = f"{family}/{regime}/n{n}/w{width}"
    problem, side, rng = _known_bipartite_problem(
        family,
        label,
        n=n,
        strong_contraction=strong_contraction,
        base_infeasible=base_infeasible,
        namespace=namespace,
        minimum_edge_factor=minimum_edge_factor,
    )
    rows = _queries(
        problem,
        side,
        rng,
        width=width,
        base_infeasible=base_infeasible,
    )
    name = f"auto-v10-{family}-{regime}-n{n}"
    return Case(
        name,
        name,
        f"{namespace}; generated query-blind before timing",
        problem,
        rows,
    )


def v3_confirmation_case(family: str, replica: int) -> Case:
    """Recreate the panel that rejected v3; it is development evidence for v4."""
    if family not in FAMILIES or replica not in range(4):
        raise ValueError("fixed v3 confirmation family/replica required")
    n = 256 if replica < 2 else 1024
    rng = random.Random(seed_for("SPECTRA-auto-selector-v3-confirmation", f"{family}/{replica}"))
    candidate_edges = topology(family, n, rng)
    side = [rng.randrange(2) for _ in range(n)]
    side[0], side[1] = 0, 1
    edges = {edge for edge in candidate_edges if side[edge[0]] != side[edge[1]]}
    left = [vertex for vertex, value in enumerate(side) if value == 0]
    right = [vertex for vertex, value in enumerate(side) if value == 1]
    for vertex in range(n):
        pool = right if side[vertex] == 0 else left
        other = pool[(vertex * 17 + replica * 13) % len(pool)]
        if vertex != other:
            edges.add((min(vertex, other), max(vertex, other)))

    palette = 8
    if replica == 0:
        masks = [3] * n
    else:
        masks = []
        for vertex in range(n):
            reference = side[vertex]
            if vertex % 11 == 0:
                masks.append(1 << reference)
            else:
                masks.append((1 << reference) | (1 << rng.randrange(2, palette)))
    if replica == 3:
        masks[:3] = [3, 3, 3]
        edges.update(((0, 1), (0, 2), (1, 2)))

    problem = BinaryListProblem.create(n, palette, tuple(sorted(edges)), masks)
    widths = {0: (0, 1, 2, 4), 1: (2, 4, 6, 7), 2: (8, 12, 16, 24), 3: (0, 1, 2, 4)}[replica]
    rows = []
    for index in range(1024):
        width = widths[index % len(widths)]
        row = []
        if width:
            for vertex in rng.sample(range(n), min(width, n)):
                allowed = masks[vertex] if index % 5 == 0 else 1 << side[vertex]
                row.append((vertex, allowed))
        if replica != 3 and (index + 1) % 31 == 0:
            row.append(((index * 29) % n, 0))
        if row and (index + 1) % 17 == 0:
            row.append(row[0])
        rows.append(tuple(row))
    name = f"v3-failure-{family}-r{replica}-n{n}"
    return Case(name, name, "retained failed-v3 confirmation; v4 development only", problem, tuple(rows))


def development_cases() -> tuple[PanelCase, ...]:
    result: list[PanelCase] = []
    for family in FAMILIES:
        for replica in range(6):
            case = make_case(family, replica)
            result.append(PanelCase(case, family, f"legacy-r{replica}", DEVELOPMENT_COUNTS))
        for replica in range(4):
            case = v3_confirmation_case(family, replica)
            result.append(PanelCase(case, family, f"v3-failure-r{replica}", CONFIRMATION_COUNTS))
        for regime, n, width in (
            ("density-below", 512, 14),
            ("density-above", 384, 14),
            ("density-low-large", 1536, 28),
        ):
            case = generated_case(
                family,
                regime,
                n=n,
                width=width,
                namespace="SPECTRA-auto-selector-v4-density-development",
            )
            result.append(PanelCase(case, family, regime, CONFIRMATION_COUNTS))
    return tuple(result)


def confirmation_cases() -> tuple[PanelCase, ...]:
    result: list[PanelCase] = []
    namespace = "SPECTRA-auto-selector-v10-confirmation"
    regimes = (
        ("strong-sparse", 352, 3, True, False, 0.0),
        ("dense-small-uncompressed", 84, 12, False, False, 0.0),
        ("dense-boundary-uncompressed", 92, 12, False, False, 0.0),
        ("dense-at96", 96, 12, False, False, 0.0),
        ("dense-above96", 108, 12, False, False, 0.0),
        ("dense-medium", 184, 12, False, False, 0.0),
        ("edge-heavy-warmup", 752, 12, False, False, 5.0),
        ("base-infeasible", 440, 4, False, True, 0.0),
    )
    for family in FAMILIES:
        for regime, n, width, strong, infeasible, edge_factor in regimes:
            case = generated_case(
                family,
                regime,
                n=n,
                width=width,
                strong_contraction=strong,
                base_infeasible=infeasible,
                namespace=namespace,
                minimum_edge_factor=edge_factor,
            )
            result.append(PanelCase(case, family, regime, CONFIRMATION_COUNTS))
    return tuple(result)

def panel_cases(panel: str) -> tuple[PanelCase, ...]:
    if panel == "development":
        return development_cases()
    if panel == "confirmation":
        return confirmation_cases()
    raise ValueError("panel must be development or confirmation")


def decisions(payload: bytes, n: int, count: int) -> tuple[int, ...]:
    return tuple(row[0] for row in decode_certified_output(payload, n, count))


def expected_policy(*, base: bool, pins: int, count: int, vertices: int,
                    edges: int, components: int) -> tuple[str, str]:
    high_pin = bool(count and pins >= 8 * count
                    and 24 * pins >= vertices * count)
    if count <= 8 and vertices > 512:
        return "raw", "large_tiny_bank"
    if not base:
        return "raw", "base_infeasible"
    if high_pin and count < 64:
        return "raw", "dense_pin_short"
    if high_pin and vertices < 96 and components < 2 * vertices:
        return "raw", "dense_small_compressed"
    if count >= 64 and components == 2 * vertices and edges >= 4 * vertices:
        return "raw", "dense_uncompressed_bank"
    if count >= 64:
        return "scc", "scc_long_bank"
    if count >= 8 and 2 * components <= vertices:
        return "scc", "scc_strong_compression"
    return "raw", "short_weak_compression"

def _time_batch(
    runtime: ExactRuntime,
    problem: ExactProblem,
    rows,
    mode: str,
    batch_size: int,
):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    current = b""
    answer = None
    started = time.perf_counter_ns()
    for _ in range(batch_size):
        current = pack_session_packet(problem, rows)
        answer = runtime.solve_session_packet(current, mode=mode)
    elapsed = time.perf_counter_ns() - started
    if answer is None:
        raise AssertionError("timed batch produced no answer")
    return elapsed, current, answer


def _calibrate_batch(
    runtime: ExactRuntime,
    problem: ExactProblem,
    rows,
    mode: str,
) -> int:
    batch_size = 1
    for _ in range(8):
        elapsed, current, answer = _time_batch(
            runtime, problem, rows, mode, batch_size
        )
        if not verify_session_output(current, answer.output):
            raise AssertionError("calibration output replay failed")
        if elapsed >= MIN_SAMPLE_NS or batch_size >= MAX_BATCH_SIZE:
            return batch_size
        target = math.ceil(
            batch_size * MIN_SAMPLE_NS / max(elapsed, 1)
        )
        batch_size = min(
            MAX_BATCH_SIZE, max(batch_size + 1, target)
        )
    return batch_size


def run(library: Path, panel: str, destination: Path, repeats: int) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    if repeats < 16 or repeats % len(MODES) != 0:
        raise ValueError("repeats must be a multiple of four and at least 16")
    destination.mkdir(parents=True)
    runtime = ExactRuntime(library)
    records = []
    rule_failures: list[str] = []

    gc.disable()
    try:
        for spec in panel_cases(panel):
            case = spec.case
            problem = ExactProblem.create(
                case.problem.n,
                case.problem.palette,
                case.problem.edges,
                case.problem.masks,
            )
            for count in spec.counts:
                rows = case.queries[:count]
                packet = pack_session_packet(problem, rows)
                first = {mode: runtime.solve_session_packet(packet, mode=mode) for mode in MODES}
                reference = decisions(first["raw"].output, problem.n, count)
                for mode, answer in first.items():
                    if not verify_session_output(packet, answer.output):
                        raise AssertionError(f"independent replay failed: {case.name}/{count}/{mode}")
                    if decisions(answer.output, problem.n, count) != reference:
                        raise AssertionError(f"decision disagreement: {case.name}/{count}/{mode}")
                info = first["auto"].info
                selected = info["selected_mode"]
                reason = info["auto_policy_reason"]
                if info["auto_policy"] != POLICY:
                    rule_failures.append(f"{case.name}/{count}: policy={info['auto_policy']}")
                pins = sum(map(len, rows))
                expected_mode, expected_reason = expected_policy(
                    base=bool(info["base_satisfiable"]),
                    pins=pins,
                    count=count,
                    vertices=problem.n,
                    edges=len(problem.edges),
                    components=int(info["components"]),
                )
                if (selected, reason) != (expected_mode, expected_reason):
                    rule_failures.append(
                        f"{case.name}/{count}: got {selected}/{reason}, "
                        f"expected {expected_mode}/{expected_reason}"
                    )
                if selected not in ("raw", "scc"):
                    raise AssertionError("automatic policy selected an unsupported mode")
                if first["auto"].output != first[selected].output:
                    raise AssertionError("auto output differs from selected explicit engine")

                for _ in range(WARMUPS):
                    for mode in MODES:
                        current = pack_session_packet(problem, rows)
                        runtime.solve_session_packet(current, mode=mode)

                batch_sizes = {
                    mode: _calibrate_batch(runtime, problem, rows, mode)
                    for mode in MODES
                }
                samples = {mode: [] for mode in MODES}
                hashes = {mode: set() for mode in MODES}
                execution_order = []
                schedule = random.Random(seed_for(
                    "SPECTRA-auto-selector-v10-schedule",
                    f"{panel}/{case.name}/{count}",
                ))
                base_order = list(MODES)
                for repetition in range(repeats):
                    offset = repetition % len(MODES)
                    if offset == 0:
                        schedule.shuffle(base_order)
                    order = (*base_order[offset:], *base_order[:offset])
                    execution_order.append(list(order))
                    for mode in order:
                        elapsed, current, answer = _time_batch(
                            runtime, problem, rows, mode, batch_sizes[mode]
                        )
                        samples[mode].append(elapsed / batch_sizes[mode])
                        hashes[mode].add(hashlib.sha256(answer.output).hexdigest())
                        if answer.output != first[mode].output:
                            raise AssertionError(
                                "timed output differs from the verified warm result"
                            )
                        if not verify_session_output(current, answer.output):
                            raise AssertionError("timed output replay failed")
                if any(len(values) != 1 for values in hashes.values()):
                    raise AssertionError("output bytes changed across repetitions")
                medians = {
                    mode: statistics.median(values)
                    for mode, values in samples.items()
                }
                oracle = min(("raw", "scc", "route"), key=medians.get)
                selection_ratios = [
                    selected_ns / oracle_ns
                    for selected_ns, oracle_ns in zip(
                        samples[selected], samples[oracle]
                    )
                ]
                realized_ratios = [
                    auto_ns / oracle_ns
                    for auto_ns, oracle_ns in zip(
                        samples["auto"], samples[oracle]
                    )
                ]
                auto_raw_ratios = [
                    auto_ns / raw_ns
                    for auto_ns, raw_ns in zip(
                        samples["auto"], samples["raw"]
                    )
                ]
                record = {
                    "panel": panel,
                    "case": case.name,
                    "family": spec.family,
                    "regime": spec.regime,
                    "count": count,
                    "vertices": problem.n,
                    "edges": len(problem.edges),
                    "pins": pins,
                    "pins_per_request": pins / count,
                    "pin_density": pins / (count * problem.n),
                    "base_satisfiable": bool(info["base_satisfiable"]),
                    "components": int(info["components"]),
                    "selected_mode": selected,
                    "selection_reason": reason,
                    "expected_mode": expected_mode,
                    "expected_reason": expected_reason,
                    "oracle_mode": oracle,
                    "medians_ns": medians,
                    "batch_sizes": batch_sizes,
                    "execution_order": execution_order,
                    "selection_regret": statistics.median(selection_ratios),
                    "realized_regret": statistics.median(realized_ratios),
                    "auto_raw_ratio": statistics.median(auto_raw_ratios),
                    "independent_median_realized_regret": (
                        medians["auto"] / medians[oracle]
                    ),
                    "samples_ns_per_call": samples,
                    "output_sha256": {
                        mode: next(iter(values)) for mode, values in hashes.items()
                    },
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
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    family_raw: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[(row["family"], row["count"])].append(row["realized_regret"])
        family_raw[row["family"]].append(row["auto_raw_ratio"])

    if panel == "development":
        limits = {
            "global": 1.025,
            "selection": 1.015,
            "group": 1.05,
            "fraction": 0.97,
            "worst": 1.10,
            "family_raw": None,
        }
    else:
        limits = {
            "global": 1.04,
            "selection": 1.025,
            "group": 1.06,
            "fraction": 0.95,
            "worst": 1.10,
            "family_raw": 1.03,
        }

    summary = {
        "schema": SCHEMA,
        "policy": POLICY,
        "classification": (
            "exposed development evidence" if panel == "development"
            else "fresh project-owned prospective confirmation; not external replication"
        ),
        "panel": panel,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cases": len({row["case"] for row in records}),
        "cells": len(records),
        "repeats": repeats,
        "sampling": {
            "method": (
                "position-balanced median of paired calibrated-batch "
                "per-call samples"
            ),
            "minimum_sample_ns": MIN_SAMPLE_NS,
            "maximum_batch_size": MAX_BATCH_SIZE,
        },
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
        "selection_regret": summary["global_selection_regret_geomean"] <= limits["selection"],
        "group_regret": summary["worst_group_regret"] <= limits["group"],
        "fraction_within_1_05": summary["fraction_within_1_05"] >= limits["fraction"],
        "worst_cell": summary["worst_realized_regret"] <= limits["worst"],
        "family_not_slower_than_raw": (
            True if limits["family_raw"] is None
            else summary["worst_family_auto_raw_ratio"] <= limits["family_raw"]
        ),
        "structural_rules": not rule_failures,
    }
    summary["checks"] = checks
    summary["pass"] = all(checks.values())
    payload = {"summary": summary, "records": records}
    destination.joinpath("results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args(argv)
    run(args.library.resolve(), args.panel, args.out.resolve(), repeats=args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
