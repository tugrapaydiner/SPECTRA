"""Three-way development benchmark for the exposed transfer suite.

The first prospective opening failed during workload admission. This module now
measures the corrected generator on exposed graphs; its thresholds are development
targets and cannot be promoted into a prospective or state-of-the-art claim.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

from spectra.exact import (
    ExactProblem,
    ExactRuntime,
    decode_certified_output,
    pack_session_packet,
    verify_session_output,
)

from .cadical_exact import CADICAL_COMMIT, CadicalExactRuntime
from .cases import Case, load_case
from .direct_twosat import Direct2SATRuntime
from .fresh_transfer import EXPECTED, GRAPH_INVENTORY

SCHEMA = "spectra.fresh_transfer_benchmark.v2"
COUNTS = (1, 8, 64, 1024)
ARMS = (
    "spectra",
    "cadical",
    "direct_auto",
    "direct_dag",
    "direct_tile",
)
TARGET_SAMPLE_NS = 5_000_000
MAX_BATCH = 32
CADICAL_GATES = {
    1: {"geometric_mean": 0.30, "every_case": 0.50},
    8: {"geometric_mean": 0.25, "every_case": 0.40},
    64: {"geometric_mean": 0.15, "every_case": 0.25},
    1024: {"geometric_mean": 0.10, "every_case": 0.20},
}
DIRECT_GATES = {
    1: {"geometric_mean": 0.85, "every_case": 1.10},
    8: {"geometric_mean": 0.75, "every_case": 1.00},
    64: {"geometric_mean": 0.50, "every_case": 0.75},
    1024: {"geometric_mean": 0.35, "every_case": 0.60},
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gmean(values) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _statuses(output: bytes, n: int, count: int) -> tuple[int, ...]:
    return tuple(
        code
        for code, _labels, _proof
        in decode_certified_output(output, n, count)
    )


def _gate(ratios: list[float], limits: dict[str, float]) -> dict:
    geometric_mean = _gmean(ratios)
    maximum = max(ratios)
    return {
        "geometric_mean": geometric_mean,
        "maximum": maximum,
        "geometric_mean_limit": limits["geometric_mean"],
        "every_case_limit": limits["every_case"],
        "geometric_mean_pass": geometric_mean <= limits["geometric_mean"],
        "every_case_pass": maximum <= limits["every_case"],
    }


def run(
    spectra_library: Path,
    cadical_library: Path,
    direct_library: Path,
    destination: Path,
    *,
    cases: tuple[Case, ...],
    repeats: int = 30,
    enforce: bool = False,
) -> dict:
    if type(repeats) is not int or repeats < len(ARMS) or repeats % len(ARMS):
        raise ValueError("repeats must be a positive multiple of five")
    if len(cases) != len(GRAPH_INVENTORY):
        raise ValueError("fresh transfer requires the complete six-graph inventory")
    if len({case.cluster for case in cases}) != len(cases):
        raise ValueError("fresh transfer graph clusters must be distinct")
    if any(len(case.queries) < max(COUNTS) for case in cases):
        raise ValueError("every transfer case needs at least 1,024 rows")
    destination.mkdir(parents=True, exist_ok=False)

    spectra = ExactRuntime(spectra_library)
    cadical = CadicalExactRuntime(cadical_library)
    direct = Direct2SATRuntime(direct_library)

    def execute(arm: str, packet: bytes):
        if arm == "spectra":
            return spectra.solve_session_packet(packet, mode="auto")
        if arm == "cadical":
            return cadical.solve_session_packet(packet)
        engine = {
            "direct_auto": "auto",
            "direct_dag": "dag",
            "direct_tile": "tile",
        }[arm]
        return direct.solve_session_packet(packet, engine=engine)

    def timed(arm: str, packet: bytes, batch: int):
        answer = None
        started = time.perf_counter_ns()
        for _ in range(batch):
            answer = execute(arm, packet)
        elapsed = time.perf_counter_ns() - started
        if answer is None:
            raise AssertionError("empty timed batch")
        return elapsed / batch, answer

    def calibrate(arm: str, packet: bytes) -> int:
        batch = 1
        for _ in range(8):
            elapsed, _answer = timed(arm, packet, batch)
            total = elapsed * batch
            if total >= TARGET_SAMPLE_NS or batch >= MAX_BATCH:
                return batch
            batch = min(
                MAX_BATCH,
                max(
                    batch + 1,
                    math.ceil(batch * TARGET_SAMPLE_NS / max(total, 1)),
                ),
            )
        return batch

    records = []
    gc.disable()
    try:
        for case in cases:
            problem = ExactProblem.create(
                case.problem.n,
                case.problem.palette,
                case.problem.edges,
                case.problem.masks,
            )
            for count in COUNTS:
                rows = case.queries[:count]
                packet = pack_session_packet(problem, rows)
                frozen_expected = EXPECTED[:count]
                first = {arm: execute(arm, packet) for arm in ARMS}
                output_hashes = {}
                infos = {}
                for arm, answer in first.items():
                    if not verify_session_output(packet, answer.output):
                        raise AssertionError(
                            f"{case.name}/{count}/{arm}: replay failed"
                        )
                    if _statuses(answer.output, problem.n, count) != frozen_expected:
                        raise AssertionError(
                            f"{case.name}/{count}/{arm}: frozen status sequence differs"
                        )
                    output_hashes[arm] = _digest(answer.output)
                    infos[arm] = answer.info

                for _ in range(3):
                    for arm in ARMS:
                        answer = execute(arm, packet)
                        if _statuses(answer.output, problem.n, count) != frozen_expected:
                            raise AssertionError("warm status sequence differs")
                        if not verify_session_output(packet, answer.output):
                            raise AssertionError("warm output replay failed")

                batch_sizes = {arm: calibrate(arm, packet) for arm in ARMS}
                samples = {arm: [] for arm in ARMS}
                execution_orders = []
                seed = int.from_bytes(
                    hashlib.sha256(
                        f"fresh-transfer/{case.name}/{count}".encode()
                    ).digest()[:8],
                    "big",
                )
                rng = random.Random(seed)
                blocks = repeats // len(ARMS)
                for _block in range(blocks):
                    base = list(ARMS)
                    rng.shuffle(base)
                    for shift in range(len(ARMS)):
                        order = base[shift:] + base[:shift]
                        execution_orders.append(order)
                        for arm in order:
                            elapsed, answer = timed(
                                arm, packet, batch_sizes[arm]
                            )
                            if (
                                _statuses(answer.output, problem.n, count)
                                != frozen_expected
                            ):
                                raise AssertionError("timed status sequence differs")
                            if not verify_session_output(packet, answer.output):
                                raise AssertionError("timed output replay failed")
                            samples[arm].append(elapsed)

                medians = {
                    arm: statistics.median(values)
                    for arm, values in samples.items()
                }
                direct_arms = ("direct_auto", "direct_dag", "direct_tile")
                best_direct_arm = min(direct_arms, key=medians.get)
                spectra_over_cadical = medians["spectra"] / medians["cadical"]
                spectra_over_direct = (
                    medians["spectra"] / medians[best_direct_arm]
                )
                record = {
                    "case": case.name,
                    "cluster": case.cluster,
                    "provenance": case.provenance,
                    "count": count,
                    "vertices": problem.n,
                    "edges": len(problem.edges),
                    "pins": sum(map(len, rows)),
                    "sat": sum(frozen_expected),
                    "unsat": count - sum(frozen_expected),
                    "packet_bytes": len(packet),
                    "packet_sha256": _digest(packet),
                    "output_sha256": output_hashes,
                    "info": infos,
                    "batch_sizes": batch_sizes,
                    "execution_orders": execution_orders,
                    "samples_ns_per_call": samples,
                    "medians_ns": medians,
                    "best_direct_arm": best_direct_arm,
                    "spectra_over_cadical": spectra_over_cadical,
                    "cadical_over_spectra": 1.0 / spectra_over_cadical,
                    "spectra_over_best_direct": spectra_over_direct,
                    "best_direct_over_spectra": 1.0 / spectra_over_direct,
                }
                records.append(record)
                print(
                    case.name,
                    count,
                    f"S/C={spectra_over_cadical:.6f}",
                    f"S/D={spectra_over_direct:.6f}",
                    "direct=" + best_direct_arm,
                    flush=True,
                )
    finally:
        gc.enable()

    summary = {}
    gates = {}
    for count in COUNTS:
        selected = [row for row in records if row["count"] == count]
        cadical_ratios = [row["spectra_over_cadical"] for row in selected]
        direct_ratios = [row["spectra_over_best_direct"] for row in selected]
        summary[str(count)] = {
            "cells": len(selected),
            "spectra_over_cadical_geometric_mean": _gmean(cadical_ratios),
            "spectra_over_cadical_maximum": max(cadical_ratios),
            "cadical_over_spectra_geometric_mean": 1.0 / _gmean(cadical_ratios),
            "spectra_over_best_direct_geometric_mean": _gmean(direct_ratios),
            "spectra_over_best_direct_maximum": max(direct_ratios),
            "best_direct_over_spectra_geometric_mean": 1.0 / _gmean(direct_ratios),
            "spectra_cadical_wins": sum(ratio < 1 for ratio in cadical_ratios),
            "spectra_direct_wins": sum(ratio < 1 for ratio in direct_ratios),
        }
        gates[str(count)] = {
            "cadical": _gate(cadical_ratios, CADICAL_GATES[count]),
            "direct": _gate(direct_ratios, DIRECT_GATES[count]),
        }

    correctness = {
        "complete_inventory": len(records) == len(GRAPH_INVENTORY) * len(COUNTS),
        "all_rows_completed": all(
            row["sat"] + row["unsat"] == row["count"] for row in records
        ),
        "frozen_status_schedule": all(
            row["sat"] == sum(EXPECTED[: row["count"]]) for row in records
        ),
        "official_cadical_commit": all(
            row["info"]["cadical"]["cadical_commit"] == CADICAL_COMMIT
            for row in records
        ),
        "portable_cadical_proofs": all(
            row["info"]["cadical"]["portable_unsat_certificate"] is True
            for row in records
        ),
    }
    gate_pass = all(correctness.values()) and all(
        competitor["geometric_mean_pass"] and competitor["every_case_pass"]
        for by_count in gates.values()
        for competitor in by_count.values()
    )
    payload = {
        "schema": SCHEMA,
        "classification": (
            "exposed six-family transfer development under matched "
            "proof-carrying first-use accounting"
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "repeats": repeats,
        "schedule": "position-balanced five-arm blocks",
        "target_sample_ns": TARGET_SAMPLE_NS,
        "max_batch": MAX_BATCH,
        "correctness": correctness,
        "gates": gates,
        "gate_pass": gate_pass,
        "summary": summary,
        "records": records,
    }
    destination.joinpath("results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destination.joinpath("summary.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA + ".summary",
                "classification": payload["classification"],
                "platform": payload["platform"],
                "machine": payload["machine"],
                "correctness": correctness,
                "gates": gates,
                "gate_pass": gate_pass,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if enforce and not gate_pass:
        raise AssertionError("transfer development target failed")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra-library", type=Path, required=True)
    parser.add_argument("--cadical-library", type=Path, required=True)
    parser.add_argument("--direct-library", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)
    cases = tuple(
        load_case(args.cases / name / "case.json")[0]
        for name in GRAPH_INVENTORY
    )
    result = run(
        args.spectra_library.resolve(),
        args.cadical_library.resolve(),
        args.direct_library.resolve(),
        args.out,
        cases=cases,
        repeats=args.repeats,
        enforce=args.enforce,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print("gate_pass=", result["gate_pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
