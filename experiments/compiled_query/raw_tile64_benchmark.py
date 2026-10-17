"""Fixed complete-call benchmark for raw-tile64 versus the serial raw control.

This is an engineering gate on disclosed development structures, not a new
scientific holdout. All modes return and independently replay the same SCQ2 bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from .binary_list import BinaryListProblem
from .exact import ExactRuntime, pack_session_packet, verify_session_output

SCHEMA = "spectra.raw_tile64.benchmark.v1"
COUNTS = (1, 8, 64, 256, 1024)
REPETITIONS = 9
MODES = ("raw", "raw-tile", "auto")


def problems() -> tuple[tuple[str, BinaryListProblem], ...]:
    n = 384
    rng = random.Random(2026101701)
    sparse = tuple(
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if rng.random() < 0.012
    )
    dense_n = 256
    return (
        ("pairs", BinaryListProblem.create(
            n, 2, tuple((v, v + 1) for v in range(0, n - 1, 2)), (3,) * n)),
        ("path", BinaryListProblem.create(
            n, 2, tuple((v, v + 1) for v in range(n - 1)), (3,) * n)),
        ("mixed_path", BinaryListProblem.create(
            n, 3, tuple((v, v + 1) for v in range(n - 1)),
            tuple((3, 6, 5)[v % 3] for v in range(n)))),
        ("sparse_random", BinaryListProblem.create(n, 2, sparse, (3,) * n)),
        ("dense_bipartite", BinaryListProblem.create(
            dense_n, 2,
            tuple((u, v) for u in range(dense_n // 2)
                  for v in range(dense_n // 2, dense_n)),
            (3,) * dense_n)),
    )


def query_rows(problem: BinaryListProblem, count: int, seed: int) -> tuple:
    rng = random.Random(seed)
    choices = problem.colors()
    rows = []
    for index in range(count):
        width = min(12, problem.n)
        vertices = rng.sample(range(problem.n), width)
        pins = [(vertex, 1 << rng.choice(choices[vertex])) for vertex in vertices]
        # Retain immediate and propagated contradictions without solver filtering.
        if index % 11 == 0:
            vertex, allowed = pins[0]
            pins.append((vertex, problem.masks[vertex] ^ allowed))
        if index % 17 == 0:
            pins.reverse()
        rows.append(tuple(pins))
    return tuple(rows)


def geometric(values) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bootstrap_upper(case_repetition_ratios: list[list[float]], seed: int,
                    samples: int = 5000) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        case_values = []
        for _case in range(len(case_repetition_ratios)):
            ratios = case_repetition_ratios[rng.randrange(len(case_repetition_ratios))]
            sample = [ratios[rng.randrange(len(ratios))] for _ in range(len(ratios))]
            case_values.append(statistics.median(sample))
        draws.append(geometric(case_values))
    draws.sort()
    return draws[int(0.025 * samples)], draws[int(0.975 * samples)]


def run(library: Path) -> dict:
    runtime = ExactRuntime(library)
    records = []
    by_count: dict[int, list[dict]] = {count: [] for count in COUNTS}

    for case_index, (name, problem) in enumerate(problems()):
        full_rows = query_rows(problem, max(COUNTS), 2026101717 + 1009 * case_index)
        for count in COUNTS:
            rows = full_rows[:count]
            packet = pack_session_packet(problem, rows)
            baseline = runtime.solve_session_packet(packet, mode="raw")
            tiled = runtime.solve_session_packet(packet, mode="raw-tile")
            auto = runtime.solve_session_packet(packet, mode="auto")
            if baseline.output != tiled.output or baseline.output != auto.output:
                raise AssertionError(f"{name}/{count}: complete output bytes differ")
            if not verify_session_output(packet, baseline.output):
                raise AssertionError(f"{name}/{count}: source-only output replay failed")
            expected_auto = "raw" if count < 64 else "raw-tile"
            if auto.info["selected_mode"] != expected_auto:
                raise AssertionError(
                    f"{name}/{count}: auto selected {auto.info['selected_mode']}, "
                    f"expected {expected_auto}"
                )
            if count >= 64 and tiled.info["bank_kernel"] != "tile64":
                raise AssertionError(f"{name}/{count}: raw-tile did not use tile64")
            expected_hash = hashlib.sha256(baseline.output).hexdigest()

            # Warm both code paths before timing. Packet preparation is excluded;
            # native parsing, construction, solving, proof/witness creation,
            # checking, serialization, disposal and Python bytes ownership remain.
            for mode in MODES:
                for _ in range(2):
                    warmed = runtime.solve_session_packet(packet, mode=mode)
                    if hashlib.sha256(warmed.output).hexdigest() != expected_hash:
                        raise AssertionError("warm output changed")

            timings = {mode: [] for mode in MODES}
            for repetition in range(REPETITIONS):
                order = MODES[repetition % len(MODES):] + MODES[:repetition % len(MODES)]
                for mode in order:
                    start = time.perf_counter_ns()
                    result = runtime.solve_session_packet(packet, mode=mode)
                    elapsed = time.perf_counter_ns() - start
                    if hashlib.sha256(result.output).hexdigest() != expected_hash:
                        raise AssertionError("timed output changed")
                    timings[mode].append(elapsed)

            medians = {mode: statistics.median(values) for mode, values in timings.items()}
            paired = [
                timings["auto"][index] / timings["raw"][index]
                for index in range(REPETITIONS)
            ]
            item = {
                "case": name,
                "n": problem.n,
                "edges": len(problem.edges),
                "queries": count,
                "pins": sum(map(len, rows)),
                "sat": baseline.sat_count,
                "unsat": baseline.unsat_count,
                "output_sha256": expected_hash,
                "selected_mode": auto.info["selected_mode"],
                "auto_reason": auto.info["auto_policy_reason"],
                "bank_kernel": auto.info["bank_kernel"],
                "median_ns": medians,
                "auto_over_raw": medians["auto"] / medians["raw"],
                "raw_tile_over_raw": medians["raw-tile"] / medians["raw"],
                "paired_auto_over_raw": paired,
                "raw_samples_ns": timings["raw"],
                "raw_tile_samples_ns": timings["raw-tile"],
                "auto_samples_ns": timings["auto"],
            }
            records.append(item)
            by_count[count].append(item)

    summaries = {}
    for count, items in by_count.items():
        case_ratios = [item["auto_over_raw"] for item in items]
        repetition_ratios = [item["paired_auto_over_raw"] for item in items]
        lower, upper = bootstrap_upper(repetition_ratios, 701 + count)
        summaries[str(count)] = {
            "geometric_auto_over_raw": geometric(case_ratios),
            "minimum_case_ratio": min(case_ratios),
            "maximum_case_ratio": max(case_ratios),
            "bootstrap_95": [lower, upper],
            "cases": len(items),
        }

    failures = []
    for count in (1, 8):
        if summaries[str(count)]["maximum_case_ratio"] > 1.10:
            failures.append(f"{count}-query auto/raw regression exceeds 10%")
    if summaries["64"]["bootstrap_95"][1] > 0.95:
        failures.append("64-query upper bootstrap bound exceeds 0.95")
    if summaries["64"]["maximum_case_ratio"] > 1.02:
        failures.append("64-query case regression exceeds 2%")
    if summaries["1024"]["bootstrap_95"][1] > 0.80:
        failures.append("1024-query upper bootstrap bound exceeds 0.80")
    if summaries["1024"]["maximum_case_ratio"] > 0.95:
        failures.append("1024-query case ratio is not below 0.95")

    return {
        "schema": SCHEMA,
        "source_scope": "disclosed deterministic development structures",
        "timing_boundary": (
            "ready SBC1 call: native parse/construction, decisions, complete outputs, "
            "independent native checking, serialization, disposal, Python bytes return"
        ),
        "repetitions": REPETITIONS,
        "records": records,
        "summaries": summaries,
        "gate": {
            "pass": not failures,
            "failures": failures,
            "requirements": {
                "q1_q8_max_ratio": 1.10,
                "q64_upper_bootstrap": 0.95,
                "q64_max_case_ratio": 1.02,
                "q1024_upper_bootstrap": 0.80,
                "q1024_max_case_ratio": 0.95,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.library)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for count, summary in result["summaries"].items():
        print(count, json.dumps(summary, sort_keys=True))
    print("gate", json.dumps(result["gate"], sort_keys=True))
    if not result["gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
