"""Matched full-cost challenge against an independent direct 2-SAT service.

The WAP05/06 inputs are exposed development cases. This file cannot create a new
holdout or an independent competitor result. It measures one ready ``SBC1`` packet
through four complete ``SCQ2`` implementations under a position-balanced schedule.
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
import sys
import time

from experiments.compiled_query.cases import Case
from experiments.compiled_query.twosat_baseline import TwoSatBaseline
from spectra.exact import (
    ExactRuntime,
    decode_certified_output,
    pack_session_packet,
    verify_session_output,
)

SCHEMA = "spectra.direct_twosat_challenge.v1"
CASES = ("wap05a.col", "wap06a.col")
COUNTS = (1, 8, 64, 1024)
ARMS = ("auto", "raw", "scc", "twosat")
REPEATS = 32
MIN_SAMPLE_NS = 2_000_000
MAX_BATCH = 128
LONG_COUNTS = (64, 1024)
LONG_GEOMEAN_LIMIT = 0.90
LONG_CELL_LIMIT = 1.10
SHORT_CELL_LIMIT = 1.25


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _gmean(values) -> float:
    values = tuple(values)
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean needs positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _answer(arm, exact, twosat, packet):
    if arm == "twosat":
        return twosat.solve_session_packet(packet)
    return exact.solve_session_packet(packet, mode=arm)


def _time_batch(arm, exact, twosat, packet, batch_size):
    output = None
    answer = None
    started = time.perf_counter_ns()
    for _ in range(batch_size):
        answer = _answer(arm, exact, twosat, packet)
        output = answer.output
    elapsed = time.perf_counter_ns() - started
    if output is None or answer is None:
        raise AssertionError("timed batch produced no answer")
    return elapsed, output, answer


def _calibrate(arm, exact, twosat, packet):
    batch_size = 1
    for _ in range(8):
        elapsed, output, _answer_value = _time_batch(
            arm, exact, twosat, packet, batch_size
        )
        if not verify_session_output(packet, output):
            raise AssertionError(f"{arm}: calibration output failed replay")
        if elapsed >= MIN_SAMPLE_NS or batch_size >= MAX_BATCH:
            return batch_size
        target = math.ceil(batch_size * MIN_SAMPLE_NS / max(elapsed, 1))
        batch_size = min(MAX_BATCH, max(batch_size + 1, target))
    return batch_size


def run(exact_library, twosat_library, cases, destination, *, repeats=REPEATS):
    if type(repeats) is not int or repeats < 16 or repeats % len(ARMS):
        raise ValueError("repeats must be a multiple of four and at least sixteen")
    destination.mkdir(parents=True, exist_ok=False)
    exact = ExactRuntime(exact_library)
    twosat = TwoSatBaseline(twosat_library)
    records = []

    gc.disable()
    try:
        for case_name in CASES:
            case = Case.decode((cases / f"{case_name}.json").read_bytes())
            for count in COUNTS:
                if count > len(case.queries):
                    raise ValueError(f"{case_name} has fewer than {count} requests")
                rows = case.queries[:count]
                packet = pack_session_packet(case.problem, rows)
                first = {
                    arm: _answer(arm, exact, twosat, packet)
                    for arm in ARMS
                }
                for arm, answer in first.items():
                    if answer.query_count != count:
                        raise AssertionError(f"{arm}: query count differs")
                    if not verify_session_output(packet, answer.output):
                        raise AssertionError(f"{arm}: SCQ2 replay failed")
                decision_hashes = {
                    arm: _digest(bytes(
                        code for code, _labels, _proof in decode_certified_output(
                            answer.output, case.problem.n, count
                        )
                    ))
                    for arm, answer in first.items()
                }
                if len(set(decision_hashes.values())) != 1:
                    raise AssertionError("competitors returned different decisions")

                batch_sizes = {
                    arm: _calibrate(arm, exact, twosat, packet)
                    for arm in ARMS
                }
                samples = {arm: [] for arm in ARMS}
                output_hashes = {
                    arm: {_digest(answer.output)} for arm, answer in first.items()
                }
                schedule = random.Random(int.from_bytes(
                    hashlib.sha256(
                        f"SPECTRA/direct-twosat-v1/{case_name}/{count}".encode()
                    ).digest()[:8],
                    "big",
                ))
                base_order = list(ARMS)
                execution_order = []
                for repetition in range(repeats):
                    offset = repetition % len(ARMS)
                    if offset == 0:
                        schedule.shuffle(base_order)
                    order = (*base_order[offset:], *base_order[:offset])
                    execution_order.append(list(order))
                    for arm in order:
                        elapsed, output, answer = _time_batch(
                            arm, exact, twosat, packet, batch_sizes[arm]
                        )
                        samples[arm].append(elapsed / batch_sizes[arm])
                        output_hashes[arm].add(_digest(output))
                        if answer.query_count != count:
                            raise AssertionError(f"{arm}: timed query count differs")
                if any(len(values) != 1 for values in output_hashes.values()):
                    raise AssertionError("one competitor changed its output across repeats")

                medians = {
                    arm: statistics.median(values)
                    for arm, values in samples.items()
                }
                paired_ratios = {
                    arm: [
                        value / baseline
                        for value, baseline in zip(samples[arm], samples["twosat"])
                    ]
                    for arm in ("auto", "raw", "scc")
                }
                record = {
                    "case": case_name,
                    "vertices": case.problem.n,
                    "edges": len(case.problem.edges),
                    "queries": count,
                    "pins": sum(map(len, rows)),
                    "packet_bytes": len(packet),
                    "packet_sha256": _digest(packet),
                    "decision_sha256": next(iter(decision_hashes.values())),
                    "batch_sizes": batch_sizes,
                    "execution_order": execution_order,
                    "samples_ns_per_call": samples,
                    "medians_ns": medians,
                    "paired_median_ratios_to_twosat": {
                        arm: statistics.median(values)
                        for arm, values in paired_ratios.items()
                    },
                    "output_sha256": {
                        arm: next(iter(values))
                        for arm, values in output_hashes.items()
                    },
                    "auto_selected_mode": first["auto"].info["selected_mode"],
                    "auto_policy_reason": first["auto"].info["auto_policy_reason"],
                    "twosat_components": first["twosat"].components,
                    "twosat_implication_arcs": first["twosat"].implication_arcs,
                    "twosat_dense_closure": first["twosat"].dense_closure,
                    "twosat_closure_bytes": first["twosat"].closure_bytes,
                    "twosat_assignment_fallbacks": first["twosat"].assignment_fallbacks,
                }
                records.append(record)
                print(
                    case_name,
                    count,
                    "auto/twosat=",
                    f"{record['paired_median_ratios_to_twosat']['auto']:.6f}",
                    flush=True,
                )
    finally:
        gc.enable()

    long_ratios = [
        record["paired_median_ratios_to_twosat"]["auto"]
        for record in records
        if record["queries"] in LONG_COUNTS
    ]
    short_ratios = [
        record["paired_median_ratios_to_twosat"]["auto"]
        for record in records
        if record["queries"] not in LONG_COUNTS
    ]
    summary = {
        "schema": SCHEMA,
        "classification": (
            "project-owned direct 2-SAT challenger on exposed WAP development cases; "
            "not a holdout or external replication"
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "repeats": repeats,
        "sampling": {
            "method": "position-balanced paired calibrated-batch per-call samples",
            "minimum_sample_ns": MIN_SAMPLE_NS,
            "maximum_batch_size": MAX_BATCH,
        },
        "cells": len(records),
        "long_auto_twosat_geometric_mean": _gmean(long_ratios),
        "long_auto_twosat_max": max(long_ratios),
        "short_auto_twosat_max": max(short_ratios),
        "limits": {
            "long_geometric_mean": LONG_GEOMEAN_LIMIT,
            "long_every_cell": LONG_CELL_LIMIT,
            "short_every_cell": SHORT_CELL_LIMIT,
        },
        "gate": {
            "all_eight_cells_completed": len(records) == len(CASES) * len(COUNTS),
            "long_geometric_mean_below_limit": (
                _gmean(long_ratios) <= LONG_GEOMEAN_LIMIT
            ),
            "long_every_cell_below_limit": max(long_ratios) <= LONG_CELL_LIMIT,
            "short_every_cell_below_limit": max(short_ratios) <= SHORT_CELL_LIMIT,
            "no_assignment_fallbacks": all(
                record["twosat_assignment_fallbacks"] == 0 for record in records
            ),
        },
    }
    summary["gate_pass"] = all(summary["gate"].values())
    payload = {"summary": summary, "records": records}
    (destination / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-library", type=Path, required=True)
    parser.add_argument("--twosat-library", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        args.exact_library.resolve(),
        args.twosat_library.resolve(),
        args.cases.resolve(),
        args.out,
        repeats=args.repeats,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if args.enforce and not payload["summary"]["gate_pass"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
