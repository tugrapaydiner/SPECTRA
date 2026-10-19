"""Position-balanced development comparison against the native 2-SAT challenger.

This is exposed development evidence, not a holdout.  It compares identical
SBC1 packets and keeps the service-only and service-plus-replay endpoints separate.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import sys
import time
from typing import Callable

from spectra.exact import (
    ExactProblem,
    ExactRuntime,
    decode_certified_output,
    pack_session_packet,
    verify_session_output,
)
from .twosat_baseline import TwoSatRuntime

SCHEMA = "spectra.native_twosat_development.v1"
ARMS = (
    "spectra/raw",
    "spectra/scc",
    "spectra/auto",
    "twosat/raw",
    "twosat/scc",
    "twosat/auto",
)
PREFIXES = (1, 8, 64, 1024)
TARGET_BATCH_NS = 500_000
MAX_BATCH_SIZE = 16


@dataclass(frozen=True)
class DevelopmentCase:
    name: str
    family: str
    tags: tuple[str, ...]
    problem: ExactProblem
    rows: tuple[tuple[tuple[int, int], ...], ...]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gmean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _colours(mask: int) -> tuple[int, ...]:
    return tuple(index for index in range(64) if mask & (1 << index))


def _edge_conflict(problem: ExactProblem) -> tuple[tuple[int, int], ...] | None:
    for left, right in problem.edges:
        shared = problem.masks[left] & problem.masks[right]
        if shared:
            colour = (shared & -shared).bit_length() - 1
            return ((left, 1 << colour), (right, 1 << colour))
    return None


def _rows(
    problem: ExactProblem,
    *,
    count: int,
    seed: int,
    width: int,
    first: str = "sat",
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if count < 1:
        raise ValueError("count must be positive")
    conflict = _edge_conflict(problem)
    if first == "sat":
        result: list[tuple[tuple[int, int], ...]] = [()]
    elif first == "query-unsat":
        if conflict is None:
            raise ValueError("query-unsat case needs a shared-colour edge")
        result = [conflict]
    elif first == "empty-domain":
        result = [((0, 0),)]
    else:
        raise ValueError("unknown first-row role")

    rng = random.Random(seed)
    vertices = tuple(range(problem.n))
    while len(result) < count:
        row_index = len(result)
        if row_index % 29 == 0:
            result.append(((row_index % problem.n, 0),))
            continue
        if row_index % 23 == 0 and conflict is not None:
            result.append(conflict)
            continue
        chosen = rng.sample(vertices, min(width, problem.n))
        pins = []
        for vertex in chosen:
            options = _colours(problem.masks[vertex])
            pins.append((vertex, 1 << rng.choice(options)))
        result.append(tuple(sorted(pins)))
    return tuple(result)


def _make_cases() -> tuple[DevelopmentCase, ...]:
    count = max(PREFIXES)
    isolated_masks = tuple(
        ((1 << 0) | (1 << 63), (1 << 7) | (1 << 31))[index % 2]
        for index in range(96)
    )
    isolated = ExactProblem.create(96, 64, (), isolated_masks)

    path_edges = tuple((index, index + 1) for index in range(95))
    path = ExactProblem.create(96, 2, path_edges, (0b11,) * 96)

    dense_edges = tuple(
        (left, right)
        for left in range(48)
        for right in range(48, 96)
    )
    dense = ExactProblem.create(96, 2, dense_edges, (0b11,) * 96)

    mixed = ExactProblem.create(
        96,
        3,
        path_edges,
        tuple((0b011, 0b110, 0b101)[index % 3] for index in range(96)),
    )

    odd_cycle = ExactProblem.create(
        97,
        2,
        tuple((index, index + 1) for index in range(96)) + ((0, 96),),
        (0b11,) * 97,
    )

    high_path = ExactProblem.create(
        256,
        2,
        tuple((index, index + 1) for index in range(255)),
        (0b11,) * 256,
    )

    return (
        DevelopmentCase(
            "isolated-high-colours",
            "synthetic",
            ("weak-compression", "sparse", "low-pin", "palette64"),
            isolated,
            _rows(isolated, count=count, seed=101, width=1),
        ),
        DevelopmentCase(
            "path",
            "synthetic",
            ("strong-compression", "sparse", "low-pin", "mixed-result"),
            path,
            _rows(path, count=count, seed=202, width=8),
        ),
        DevelopmentCase(
            "path-query-unsat-first",
            "synthetic",
            ("strong-compression", "sparse", "single-query-unsat"),
            path,
            _rows(path, count=count, seed=303, width=8, first="query-unsat"),
        ),
        DevelopmentCase(
            "dense-bipartite",
            "synthetic",
            ("strong-compression", "dense", "low-pin", "mixed-result"),
            dense,
            _rows(dense, count=count, seed=404, width=8),
        ),
        DevelopmentCase(
            "mixed-palette-path",
            "synthetic",
            ("moderate-compression", "sparse", "palette3", "mixed-result"),
            mixed,
            _rows(mixed, count=count, seed=505, width=8),
        ),
        DevelopmentCase(
            "odd-cycle-base-unsat",
            "synthetic",
            ("base-unsat", "sparse", "proof-kind3"),
            odd_cycle,
            _rows(odd_cycle, count=count, seed=606, width=8),
        ),
        DevelopmentCase(
            "high-pin-path",
            "synthetic",
            ("strong-compression", "sparse", "high-pin", "mixed-result"),
            high_path,
            _rows(high_path, count=count, seed=707, width=128),
        ),
    )


def _decision_codes(output: bytes, vertices: int, rows: int) -> tuple[int, ...]:
    return tuple(
        code
        for code, _labels, _proof in decode_certified_output(
            output, vertices, rows
        )
    )


def _set_affinity() -> list[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        available = sorted(os.sched_getaffinity(0))
        if available:
            os.sched_setaffinity(0, {available[0]})
            return [available[0]]
    except OSError:
        return None
    return None


def _position_balanced_orders(
    *,
    repeats: int,
    seed: int,
) -> list[list[str]]:
    if repeats < len(ARMS) or repeats % len(ARMS):
        raise ValueError("repeats must be a positive multiple of six")
    rng = random.Random(seed)
    orders: list[list[str]] = []
    for _block in range(repeats // len(ARMS)):
        base = list(ARMS)
        rng.shuffle(base)
        for offset in range(len(ARMS)):
            orders.append(base[offset:] + base[:offset])
    return orders


def _time_batch(
    call: Callable[[], object],
    *,
    packet: bytes,
    batch_size: int,
    verify: bool,
):
    answer = None
    started = time.perf_counter_ns()
    for _ in range(batch_size):
        answer = call()
        if verify and not verify_session_output(packet, answer.output):
            raise AssertionError("timed output failed standard-library replay")
    elapsed = time.perf_counter_ns() - started
    if answer is None:
        raise AssertionError("empty timed batch")
    return elapsed, answer


def _common_batch_size(calls: dict[str, Callable[[], object]], packet: bytes) -> int:
    elapsed = []
    for call in calls.values():
        duration, answer = _time_batch(
            call, packet=packet, batch_size=1, verify=False
        )
        if not verify_session_output(packet, answer.output):
            raise AssertionError("calibration output failed replay")
        elapsed.append(duration)
    fastest = max(1, min(elapsed))
    return min(
        MAX_BATCH_SIZE,
        max(1, math.ceil(TARGET_BATCH_NS / fastest)),
    )


def _measure_endpoint(
    calls: dict[str, Callable[[], object]],
    *,
    packet: bytes,
    batch_size: int,
    repeats: int,
    seed: int,
    verify: bool,
) -> tuple[dict[str, list[float]], list[list[str]], dict[str, set[str]]]:
    samples = {arm: [] for arm in ARMS}
    hashes = {arm: set() for arm in ARMS}
    orders = _position_balanced_orders(repeats=repeats, seed=seed)
    for order in orders:
        for arm in order:
            elapsed, answer = _time_batch(
                calls[arm],
                packet=packet,
                batch_size=batch_size,
                verify=verify,
            )
            if not verify_session_output(packet, answer.output):
                raise AssertionError("measured output failed replay")
            samples[arm].append(elapsed / batch_size)
            hashes[arm].add(_sha256(answer.output))
    return samples, orders, hashes


def _median_ratios(
    samples: dict[str, list[float]],
) -> dict[str, float]:
    count = len(next(iter(samples.values())))
    if any(len(values) != count for values in samples.values()):
        raise AssertionError("sample counts differ")
    return {
        "spectra_auto_over_twosat_auto": statistics.median(
            samples["spectra/auto"][index] / samples["twosat/auto"][index]
            for index in range(count)
        ),
        "spectra_scc_over_twosat_scc": statistics.median(
            samples["spectra/scc"][index] / samples["twosat/scc"][index]
            for index in range(count)
        ),
        "spectra_auto_over_twosat_best_explicit": statistics.median(
            samples["spectra/auto"][index]
            / min(
                samples["twosat/raw"][index],
                samples["twosat/scc"][index],
            )
            for index in range(count)
        ),
    }


def _event(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _manifest(root: Path) -> dict:
    artifacts = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256.json":
            data = path.read_bytes()
            artifacts[path.relative_to(root).as_posix()] = {
                "bytes": len(data),
                "sha256": _sha256(data),
            }
    return {
        "schema": "spectra.native_twosat_development_manifest.v1",
        "artifacts": artifacts,
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "spectra/exact.py",
        root / "spectra/exact_proof.py",
        root / "spectra/_native/exact_binary.cpp",
        root / "experiments/compiled_query/twosat_baseline.py",
        root / "experiments/compiled_query/native_twosat.cpp",
        root / "experiments/compiled_query/TWOSAT_CHALLENGE_PROTOCOL.md",
    )
    return {
        path.relative_to(root).as_posix(): _sha256(path.read_bytes())
        for path in paths
    }


def run(
    spectra_library: Path,
    twosat_library: Path,
    destination: Path,
    *,
    repeats: int = 24,
) -> dict:
    if repeats < len(ARMS) or repeats % len(ARMS):
        raise ValueError("repeats must be a positive multiple of six")
    destination.mkdir(parents=True, exist_ok=False)
    journal = destination / "events.jsonl"
    spectra_receipt = spectra_library.with_name("build.json")
    twosat_receipt = twosat_library.with_name("twosat-build.json")
    if not spectra_receipt.is_file() or not twosat_receipt.is_file():
        raise ValueError("both native builds need adjacent receipts")
    shutil.copy2(spectra_receipt, destination / "spectra-build.json")
    shutil.copy2(twosat_receipt, destination / "twosat-build.json")
    spectra = ExactRuntime(spectra_library)
    twosat = TwoSatRuntime(twosat_library)
    affinity = _set_affinity()
    records = []
    _event(
        journal,
        {
            "event": "run-start",
            "schema": SCHEMA,
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "repeats": repeats,
            "source_commit": os.environ.get("SOURCE_COMMIT", ""),
            "source_tree": os.environ.get("SOURCE_TREE", ""),
        },
    )
    gc.disable()
    try:
        for case_index, case in enumerate(_make_cases()):
            for prefix in PREFIXES:
                rows = case.rows[:prefix]
                cell = f"{case.name}-q{prefix}"
                _event(
                    journal,
                    {
                        "event": "cell-start",
                        "case": case.name,
                        "prefix": prefix,
                        "tags": list(case.tags),
                    },
                )
                packet = pack_session_packet(case.problem, rows)
                calls: dict[str, Callable[[], object]] = {
                    "spectra/raw": lambda p=packet: spectra.solve_session_packet(
                        p, mode="raw"
                    ),
                    "spectra/scc": lambda p=packet: spectra.solve_session_packet(
                        p, mode="scc"
                    ),
                    "spectra/auto": lambda p=packet: spectra.solve_session_packet(
                        p, mode="auto"
                    ),
                    "twosat/raw": lambda p=packet: twosat.solve_session_packet(
                        p, mode="raw"
                    ),
                    "twosat/scc": lambda p=packet: twosat.solve_session_packet(
                        p, mode="scc"
                    ),
                    "twosat/auto": lambda p=packet: twosat.solve_session_packet(
                        p, mode="auto"
                    ),
                }
                first = {arm: call() for arm, call in calls.items()}
                decisions = {}
                for arm, answer in first.items():
                    if not verify_session_output(packet, answer.output):
                        raise AssertionError(f"{case.name}/{prefix}/{arm}: replay failed")
                    decisions[arm] = _decision_codes(
                        answer.output, case.problem.n, len(rows)
                    )
                reference = decisions[ARMS[0]]
                if any(value != reference for value in decisions.values()):
                    raise AssertionError(
                        f"{case.name}/{prefix}: arm decisions differ"
                    )

                packet_name = f"packets/{cell}.sbc1"
                _write_new(destination / packet_name, packet)
                output_paths = {}
                for arm, answer in first.items():
                    safe_arm = arm.replace("/", "-")
                    output_name = f"outputs/{cell}-{safe_arm}.scq2"
                    _write_new(destination / output_name, answer.output)
                    output_paths[arm] = output_name

                for _ in range(3):
                    for arm in ARMS:
                        warmed = calls[arm]()
                        if not verify_session_output(packet, warmed.output):
                            raise AssertionError("warm output failed replay")

                batch_size = _common_batch_size(calls, packet)
                endpoints = {}
                for endpoint_index, (endpoint, verify) in enumerate(
                    (("service", False), ("verified_service", True))
                ):
                    samples, orders, hashes = _measure_endpoint(
                        calls,
                        packet=packet,
                        batch_size=batch_size,
                        repeats=repeats,
                        seed=0x2A5A7000
                        + case_index * 100_000
                        + prefix * 10
                        + endpoint_index,
                        verify=verify,
                    )
                    if any(len(values) != 1 for values in hashes.values()):
                        raise AssertionError(
                            f"{case.name}/{prefix}/{endpoint}: output changed"
                        )
                    medians = {
                        arm: statistics.median(values)
                        for arm, values in samples.items()
                    }
                    endpoints[endpoint] = {
                        "batch_size": batch_size,
                        "execution_orders": orders,
                        "samples_ns_per_call": samples,
                        "medians_ns": medians,
                        "paired_median_ratios": _median_ratios(samples),
                    }

                answers = {
                    arm: {
                        "sat": answer.sat_count,
                        "unsat": answer.unsat_count,
                        "proof_bytes": answer.proof_bytes,
                        "output_bytes": len(answer.output),
                        "output_sha256": _sha256(answer.output),
                        "output": output_paths[arm],
                        "info": answer.info,
                    }
                    for arm, answer in first.items()
                }
                record = {
                    "case": case.name,
                    "family": case.family,
                    "tags": list(case.tags),
                    "prefix": prefix,
                    "vertices": case.problem.n,
                    "edges": len(case.problem.edges),
                    "palette": case.problem.palette,
                    "pins": sum(len(row) for row in rows),
                    "packet": packet_name,
                    "packet_bytes": len(packet),
                    "packet_sha256": _sha256(packet),
                    "decisions_sha256": _sha256(bytes(reference)),
                    "answers": answers,
                    "endpoints": endpoints,
                }
                records.append(record)
                _event(
                    journal,
                    {
                        "event": "cell-complete",
                        "case": case.name,
                        "prefix": prefix,
                        "service_ratios": endpoints["service"]["paired_median_ratios"],
                        "verified_service_ratios": endpoints[
                            "verified_service"
                        ]["paired_median_ratios"],
                    },
                )
                print(
                    cell,
                    "service_auto/best="
                    f"{endpoints['service']['paired_median_ratios']['spectra_auto_over_twosat_best_explicit']:.6f}",
                    "verified_auto/best="
                    f"{endpoints['verified_service']['paired_median_ratios']['spectra_auto_over_twosat_best_explicit']:.6f}",
                    flush=True,
                )
    except BaseException as error:
        _event(
            journal,
            {
                "event": "run-failure",
                "error_type": type(error).__name__,
                "message": str(error),
                "completed_cells": len(records),
            },
        )
        raise
    finally:
        gc.enable()

    summaries = {}
    for endpoint in ("service", "verified_service"):
        ratios = [
            record["endpoints"][endpoint]["paired_median_ratios"][
                "spectra_auto_over_twosat_best_explicit"
            ]
            for record in records
        ]
        auto_ratios = [
            record["endpoints"][endpoint]["paired_median_ratios"][
                "spectra_auto_over_twosat_auto"
            ]
            for record in records
        ]
        summaries[endpoint] = {
            "cells": len(records),
            "spectra_auto_over_twosat_best_explicit_geomean": _gmean(ratios),
            "spectra_auto_over_twosat_best_explicit_wins": sum(
                ratio < 1 for ratio in ratios
            ),
            "spectra_auto_over_twosat_auto_geomean": _gmean(auto_ratios),
            "spectra_auto_over_twosat_auto_wins": sum(
                ratio < 1 for ratio in auto_ratios
            ),
            "worst_spectra_auto_over_twosat_best_explicit": max(ratios),
        }

    payload = {
        "schema": SCHEMA,
        "classification": (
            "exposed deterministic development panel; descriptive only; "
            "not a holdout or final competition result"
        ),
        "source_commit": os.environ.get("SOURCE_COMMIT", ""),
        "source_tree": os.environ.get("SOURCE_TREE", ""),
        "source_hashes": _source_hashes(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "github_runner": {
            "name": os.environ.get("RUNNER_NAME", ""),
            "os": os.environ.get("RUNNER_OS", ""),
            "architecture": os.environ.get("RUNNER_ARCH", ""),
            "image": os.environ.get("ImageOS", ""),
        },
        "cpu_affinity": affinity,
        "repeats": repeats,
        "builds": {
            "spectra": json.loads(
                (destination / "spectra-build.json").read_text(encoding="utf-8")
            ),
            "twosat": json.loads(
                (destination / "twosat-build.json").read_text(encoding="utf-8")
            ),
        },
        "prefixes": list(PREFIXES),
        "arms": list(ARMS),
        "timing": {
            "clock": "time.perf_counter_ns",
            "target_batch_ns": TARGET_BATCH_NS,
            "maximum_batch_size": MAX_BATCH_SIZE,
            "schedule": "complete position-balanced six-arm blocks",
            "service_boundary": (
                "ready SBC1 through native parse, construction, query bank, "
                "complete output ownership, and disposal"
            ),
            "verified_service_boundary": (
                "service boundary plus standard-library replay on every call"
            ),
        },
        "primary_endpoint": "verified_service",
        "summary": summaries,
        "records": records,
    }
    results_path = destination / "results.json"
    results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _event(
        journal,
        {
            "event": "run-complete",
            "cells": len(records),
            "primary_endpoint": "verified_service",
        "summary": summaries,
        },
    )
    manifest = _manifest(destination)
    (destination / "SHA256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-library", type=Path, required=True)
    parser.add_argument("--twosat-library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=24)
    args = parser.parse_args(argv)
    payload = run(
        args.spectra_library.resolve(),
        args.twosat_library.resolve(),
        args.out.resolve(),
        repeats=args.repeats,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
