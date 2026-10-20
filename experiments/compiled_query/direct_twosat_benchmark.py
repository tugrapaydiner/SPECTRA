"""Matched exploratory benchmark against an independent native 2-SAT service.

All inputs are already exposed synthetic development cases. The result is a
challenge diagnostic, not a holdout or a superiority claim.
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
from .cases import synthetic_cases
from .direct_twosat import Direct2SATRuntime

SCHEMA = "spectra.direct_twosat_challenge.v1"
ARMS = ("raw", "scc", "auto", "direct")
COUNTS = (1, 8, 64, 1024)
TARGET_SAMPLE_NS = 2_000_000
MAX_BATCH = 64


def _gmean(values):
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _statuses(output: bytes, n: int, count: int) -> tuple[int, ...]:
    return tuple(
        code for code, _labels, _proof
        in decode_certified_output(output, n, count)
    )


def run(
    spectra_library: Path,
    direct_library: Path,
    destination: Path,
    *,
    repeats: int = 24,
) -> dict:
    if type(repeats) is not int or repeats < len(ARMS) or repeats % len(ARMS):
        raise ValueError("repeats must be a positive multiple of four")
    destination.mkdir(parents=True, exist_ok=False)
    spectra = ExactRuntime(spectra_library)
    direct = Direct2SATRuntime(direct_library)
    cases = synthetic_cases(count=max(COUNTS), n=96)

    def execute(arm: str, packet: bytes):
        if arm == "direct":
            return direct.solve_session_packet(packet)
        return spectra.solve_session_packet(packet, mode=arm)

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
                max(batch + 1, math.ceil(batch * TARGET_SAMPLE_NS / max(total, 1))),
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
                outputs = {}
                infos = {}
                for arm, answer in first.items():
                    if not verify_session_output(packet, answer.output):
                        raise AssertionError(f"{case.name}/{count}/{arm}: replay failed")
                    current = _statuses(answer.output, problem.n, count)
                    if expected is None:
                        expected = current
                    elif current != expected:
                        raise AssertionError(
                            f"{case.name}/{count}: decision sequence differs"
                        )
                    outputs[arm] = _digest(answer.output)
                    infos[arm] = getattr(answer, "info", {})

                for _ in range(3):
                    for arm in ARMS:
                        answer = execute(arm, packet)
                        if _statuses(answer.output, problem.n, count) != expected:
                            raise AssertionError("warm result differs")

                batch_sizes = {
                    arm: calibrate(arm, packet) for arm in ARMS
                }
                samples = {arm: [] for arm in ARMS}
                orders = []
                rng = random.Random(int.from_bytes(hashlib.sha256(
                    f"direct-2sat/{case.name}/{count}".encode()
                ).digest()[:8], "big"))
                blocks = repeats // len(ARMS)
                for _block in range(blocks):
                    base = list(ARMS)
                    rng.shuffle(base)
                    for shift in range(len(ARMS)):
                        order = base[shift:] + base[:shift]
                        orders.append(order)
                        for arm in order:
                            elapsed, answer = timed(arm, packet, batch_sizes[arm])
                            if _statuses(answer.output, problem.n, count) != expected:
                                raise AssertionError("timed result differs")
                            samples[arm].append(elapsed)

                medians = {
                    arm: statistics.median(values)
                    for arm, values in samples.items()
                }
                best_spectra_arm = min(("raw", "scc", "auto"), key=medians.get)
                record = {
                    "case": case.name,
                    "cluster": case.cluster,
                    "count": count,
                    "vertices": problem.n,
                    "edges": len(problem.edges),
                    "pins": sum(map(len, rows)),
                    "sat": sum(expected),
                    "unsat": count - sum(expected),
                    "packet_bytes": len(packet),
                    "packet_sha256": _digest(packet),
                    "output_sha256": outputs,
                    "direct_info": infos["direct"],
                    "batch_sizes": batch_sizes,
                    "execution_orders": orders,
                    "samples_ns_per_call": samples,
                    "medians_ns": medians,
                    "best_spectra_arm": best_spectra_arm,
                    "direct_over_best_spectra": (
                        medians["direct"] / medians[best_spectra_arm]
                    ),
                    "auto_over_direct": medians["auto"] / medians["direct"],
                    "scc_over_direct": medians["scc"] / medians["direct"],
                }
                records.append(record)
                print(
                    case.name,
                    count,
                    "direct/best=%.4f" % record["direct_over_best_spectra"],
                    "best=" + best_spectra_arm,
                    flush=True,
                )
    finally:
        gc.enable()

    by_count = {}
    for count in COUNTS:
        rows = [record for record in records if record["count"] == count]
        by_count[str(count)] = {
            "cells": len(rows),
            "direct_over_best_spectra_geomean": _gmean(
                [row["direct_over_best_spectra"] for row in rows]
            ),
            "auto_over_direct_geomean": _gmean(
                [row["auto_over_direct"] for row in rows]
            ),
            "scc_over_direct_geomean": _gmean(
                [row["scc_over_direct"] for row in rows]
            ),
            "direct_wins": sum(
                row["direct_over_best_spectra"] < 1 for row in rows
            ),
            "spectra_wins_or_ties": sum(
                row["direct_over_best_spectra"] >= 1 for row in rows
            ),
        }
    payload = {
        "schema": SCHEMA,
        "classification": (
            "exposed synthetic development diagnostic; no holdout or SOTA claim"
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "repeats": repeats,
        "schedule": "position-balanced four-arm blocks",
        "target_sample_ns": TARGET_SAMPLE_NS,
        "max_batch": MAX_BATCH,
        "summary": by_count,
        "records": records,
    }
    destination.joinpath("results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra-library", type=Path, required=True)
    parser.add_argument("--direct-library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=24)
    args = parser.parse_args(argv)
    result = run(
        args.spectra_library.resolve(),
        args.direct_library.resolve(),
        args.out,
        repeats=args.repeats,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())