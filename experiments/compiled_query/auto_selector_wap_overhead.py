"""Paired auto/SCC overhead guard for exposed WAP long banks."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from spectra.exact import ExactProblem, ExactRuntime, pack_session_packet, verify_session_output
from .cases import Case

CASES = ("wap05a.col", "wap06a.col")
COUNT = 1024
REPEATS = 31
POLICY = "structural-v8"


def run(library: Path, cases: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    runtime = ExactRuntime(library)
    records = []
    gc.disable()
    try:
        for name in CASES:
            case = Case.decode((cases / (name + ".json")).read_bytes())
            problem = ExactProblem.create(
                case.problem.n, case.problem.palette,
                case.problem.edges, case.problem.masks,
            )
            rows = case.queries[:COUNT]
            packet = pack_session_packet(problem, rows)
            first = {
                mode: runtime.solve_session_packet(packet, mode=mode)
                for mode in ("scc", "auto")
            }
            if first["auto"].info["selected_mode"] != "scc":
                raise AssertionError(f"{name}: auto did not select SCC")
            if first["auto"].info["auto_policy"] != POLICY:
                raise AssertionError(f"{name}: policy identity differs")
            if first["auto"].output != first["scc"].output:
                raise AssertionError(f"{name}: auto output differs from SCC")
            if not verify_session_output(packet, first["auto"].output):
                raise AssertionError(f"{name}: independent replay failed")
            for _ in range(5):
                for mode in ("scc", "auto"):
                    runtime.solve_session_packet(
                        pack_session_packet(problem, rows), mode=mode
                    )
            samples = {"scc": [], "auto": []}
            rng = random.Random(int.from_bytes(hashlib.sha256(
                f"SPECTRA-auto-v7-WAP-overhead/{name}".encode()
            ).digest()[:8], "big"))
            for _ in range(REPEATS):
                order = ["scc", "auto"]
                rng.shuffle(order)
                for mode in order:
                    started = time.perf_counter_ns()
                    current = pack_session_packet(problem, rows)
                    answer = runtime.solve_session_packet(current, mode=mode)
                    elapsed = time.perf_counter_ns() - started
                    if answer.output != first["scc"].output:
                        raise AssertionError("timed output changed")
                    if not verify_session_output(current, answer.output):
                        raise AssertionError("timed output replay failed")
                    samples[mode].append(elapsed)
            medians = {mode: statistics.median(values) for mode, values in samples.items()}
            ratio = medians["auto"] / medians["scc"]
            records.append({
                "case": name,
                "count": COUNT,
                "auto_scc_ratio": ratio,
                "auto_median_ns": medians["auto"],
                "scc_median_ns": medians["scc"],
                "samples_ns": samples,
                "output_sha256": hashlib.sha256(first["scc"].output).hexdigest(),
            })
    finally:
        gc.enable()
    ratios = [record["auto_scc_ratio"] for record in records]
    summary = {
        "schema": "spectra.auto_selector_wap_overhead.v1",
        "classification": "exposed WAP development guard; not a holdout",
        "policy": POLICY,
        "count": COUNT,
        "repeats": REPEATS,
        "geometric_mean_ratio": math.exp(statistics.fmean(math.log(v) for v in ratios)),
        "maximum_case_ratio": max(ratios),
        "limit": 1.03,
    }
    summary["pass"] = summary["maximum_case_ratio"] <= summary["limit"]
    payload = {"summary": summary, "records": records}
    (destination / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise AssertionError("automatic dispatch overhead exceeded the frozen limit")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.library.resolve(), args.cases.resolve(), args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
