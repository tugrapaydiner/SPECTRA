"""Frozen matched benchmark against the official CaDiCaL 3.0.1 core.

The four input banks are already opened development cases. This script implements
the precommitted measurement and gate calculation; it is not prospective evidence.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

from spectra.exact import (
    ExactProblem,
    ExactRuntime,
    decode_certified_output,
    pack_session_packet,
    verify_session_output,
)

from .cadical_exact import CadicalExactRuntime
from .cases import Case, load_case

SCHEMA = "spectra.cadical_301_challenge.v1"
COUNTS = (1, 8, 64, 1024)
ARMS = ("spectra", "cadical")
TARGET_SAMPLE_NS = 5_000_000
MAX_BATCH = 32
GATES = {
    8: {"geometric_mean": 0.50, "every_case": 0.75},
    64: {"geometric_mean": 0.30, "every_case": 0.45},
    1024: {"geometric_mean": 0.20, "every_case": 0.30},
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


def run(
    spectra_library: Path,
    cadical_library: Path,
    destination: Path,
    *,
    cases: tuple[Case, ...],
    repeats: int = 30,
    enforce: bool = False,
) -> dict:
    if type(repeats) is not int or repeats < 10 or repeats % 2:
        raise ValueError("repeats must be an even integer of at least ten")
    if len(cases) != 4 or len({case.cluster for case in cases}) != 4:
        raise ValueError("challenge requires exactly four distinct graph clusters")
    if any(len(case.queries) < max(COUNTS) for case in cases):
        raise ValueError("every challenge case needs at least 1,024 rows")
    destination.mkdir(parents=True, exist_ok=False)

    spectra = ExactRuntime(spectra_library)
    cadical = CadicalExactRuntime(cadical_library)

    def execute(arm: str, packet: bytes):
        if arm == "spectra":
            return spectra.solve_session_packet(packet, mode="auto")
        return cadical.solve_session_packet(packet)

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
                first = {arm: execute(arm, packet) for arm in ARMS}
                expected = None
                output_hashes = {}
                infos = {}
                for arm, answer in first.items():
                    if not verify_session_output(packet, answer.output):
                        raise AssertionError(
                            f"{case.name}/{count}/{arm}: replay failed"
                        )
                    statuses = _statuses(answer.output, problem.n, count)
                    if expected is None:
                        expected = statuses
                    elif statuses != expected:
                        raise AssertionError(
                            f"{case.name}/{count}: status sequence differs"
                        )
                    output_hashes[arm] = _digest(answer.output)
                    infos[arm] = answer.info

                for _ in range(3):
                    for arm in ARMS:
                        answer = execute(arm, packet)
                        if _statuses(answer.output, problem.n, count) != expected:
                            raise AssertionError("warm answer sequence differs")
                        if not verify_session_output(packet, answer.output):
                            raise AssertionError("warm output replay failed")

                batch_sizes = {
                    arm: calibrate(arm, packet) for arm in ARMS
                }
                samples = {arm: [] for arm in ARMS}
                execution_orders = []
                seed = int.from_bytes(
                    hashlib.sha256(
                        f"official-cadical-301/{case.name}/{count}".encode()
                    ).digest()[:8],
                    "big",
                )
                first_arm = seed & 1
                for repeat in range(repeats):
                    order = list(ARMS)
                    if (repeat + first_arm) & 1:
                        order.reverse()
                    execution_orders.append(order)
                    for arm in order:
                        elapsed, answer = timed(
                            arm, packet, batch_sizes[arm]
                        )
                        if _statuses(answer.output, problem.n, count) != expected:
                            raise AssertionError("timed answer sequence differs")
                        if not verify_session_output(packet, answer.output):
                            raise AssertionError("timed output replay failed")
                        samples[arm].append(elapsed)

                medians = {
                    arm: statistics.median(values)
                    for arm, values in samples.items()
                }
                ratio = medians["spectra"] / medians["cadical"]
                record = {
                    "case": case.name,
                    "cluster": case.cluster,
                    "provenance": case.provenance,
                    "count": count,
                    "vertices": problem.n,
                    "edges": len(problem.edges),
                    "pins": sum(map(len, rows)),
                    "sat": sum(expected),
                    "unsat": count - sum(expected),
                    "packet_bytes": len(packet),
                    "packet_sha256": _digest(packet),
                    "output_sha256": output_hashes,
                    "spectra_info": infos["spectra"],
                    "cadical_info": infos["cadical"],
                    "batch_sizes": batch_sizes,
                    "execution_orders": execution_orders,
                    "samples_ns_per_call": samples,
                    "medians_ns": medians,
                    "spectra_over_cadical": ratio,
                    "cadical_over_spectra": 1.0 / ratio,
                }
                records.append(record)
                print(
                    case.name,
                    count,
                    f"spectra/cadical={ratio:.6f}",
                    f"speedup={1.0 / ratio:.3f}x",
                    flush=True,
                )
    finally:
        gc.enable()

    summary = {}
    gate_checks = {}
    for count in COUNTS:
        selected = [row for row in records if row["count"] == count]
        ratios = [row["spectra_over_cadical"] for row in selected]
        current = {
            "cells": len(selected),
            "spectra_over_cadical_geometric_mean": _gmean(ratios),
            "spectra_over_cadical_maximum": max(ratios),
            "cadical_over_spectra_geometric_mean": 1.0 / _gmean(ratios),
            "spectra_wins": sum(ratio < 1 for ratio in ratios),
            "cadical_wins_or_ties": sum(ratio >= 1 for ratio in ratios),
        }
        summary[str(count)] = current
        if count in GATES:
            limits = GATES[count]
            gate_checks[str(count)] = {
                "geometric_mean_limit": limits["geometric_mean"],
                "every_case_limit": limits["every_case"],
                "geometric_mean_pass": (
                    current["spectra_over_cadical_geometric_mean"]
                    <= limits["geometric_mean"]
                ),
                "every_case_pass": (
                    current["spectra_over_cadical_maximum"]
                    <= limits["every_case"]
                ),
            }

    correctness = {
        "four_distinct_graph_clusters": len(records) == 4 * len(COUNTS),
        "all_rows_completed": all(
            row["sat"] + row["unsat"] == row["count"]
            for row in records
        ),
        "official_cadical_commit": all(
            row["cadical_info"]["cadical_commit"]
            == "c60730422e758ef1cebe7aeddf2dda31c996bf04"
            for row in records
        ),
        "portable_proofs": all(
            row["cadical_info"]["portable_unsat_certificate"] is True
            for row in records
        ),
    }
    gate_pass = all(correctness.values()) and all(
        check["geometric_mean_pass"] and check["every_case_pass"]
        for check in gate_checks.values()
    )
    payload = {
        "schema": SCHEMA,
        "classification": (
            "official CaDiCaL 3.0.1 solver-core development challenge on "
            "already opened mixed banks; not prospective confirmation"
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "repeats": repeats,
        "schedule": "alternating position-balanced two-arm rounds",
        "target_sample_ns": TARGET_SAMPLE_NS,
        "max_batch": MAX_BATCH,
        "correctness": correctness,
        "gates": gate_checks,
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
                "classification": payload["classification"],
                "platform": payload["platform"],
                "machine": payload["machine"],
                "correctness": correctness,
                "gates": gate_checks,
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
        raise AssertionError("official CaDiCaL large-gap gate failed")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra-library", type=Path, required=True)
    parser.add_argument("--cadical-library", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)
    cases = tuple(
        load_case(args.cases / name / "case.json")[0]
        for name in (
            "DSJC250.5.col",
            "flat300_28_0.col",
            "le450_15c.col",
            "miles1000.col",
        )
    )
    result = run(
        args.spectra_library.resolve(),
        args.cadical_library.resolve(),
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
