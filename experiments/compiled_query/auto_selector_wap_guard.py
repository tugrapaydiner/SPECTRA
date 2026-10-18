"""Real WAP regression guard for structural automatic engine selection v4."""
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
COUNTS = (64, 128, 1024)
REPEATS = 31
SCHEMA = "spectra.auto_selector_wap_guard.v3"
POLICY = "structural-v5"


def gmean(values):
    return math.exp(statistics.fmean(math.log(value) for value in values))


def run(library: Path, cases: Path, destination: Path, repeats: int = REPEATS) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    if repeats < 9 or repeats % 2 == 0:
        raise ValueError("odd repeats >= 9 required")
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
            for count in COUNTS:
                rows = case.queries[:count]
                packet = pack_session_packet(problem, rows)
                first = {
                    mode: runtime.solve_session_packet(packet, mode=mode)
                    for mode in ("raw", "scc", "auto")
                }
                if first["auto"].info["selected_mode"] != "scc":
                    raise AssertionError(f"{name}/{count}: auto did not select SCC")
                if first["auto"].info["auto_policy"] != POLICY:
                    raise AssertionError(f"{name}/{count}: policy identity differs")
                if first["auto"].output != first["scc"].output:
                    raise AssertionError(f"{name}/{count}: auto output differs from SCC")
                if first["raw"].output != first["scc"].output:
                    raise AssertionError(f"{name}/{count}: raw and SCC outputs differ")
                if not verify_session_output(packet, first["auto"].output):
                    raise AssertionError(f"{name}/{count}: replay failed")

                for _ in range(5):
                    for mode in ("raw", "auto"):
                        current = pack_session_packet(problem, rows)
                        runtime.solve_session_packet(current, mode=mode)

                samples = {"raw": [], "auto": []}
                hashes = {"raw": set(), "auto": set()}
                rng = random.Random(int.from_bytes(hashlib.sha256(
                    f"SPECTRA-auto-wap-v5/{name}/{count}".encode()
                ).digest()[:8], "big"))
                for _ in range(repeats):
                    order = ["raw", "auto"]
                    rng.shuffle(order)
                    for mode in order:
                        started = time.perf_counter_ns()
                        current = pack_session_packet(problem, rows)
                        answer = runtime.solve_session_packet(current, mode=mode)
                        elapsed = time.perf_counter_ns() - started
                        samples[mode].append(elapsed)
                        hashes[mode].add(hashlib.sha256(answer.output).hexdigest())
                        if not verify_session_output(current, answer.output):
                            raise AssertionError("timed output replay failed")
                if any(len(values) != 1 for values in hashes.values()):
                    raise AssertionError("output changed across repetitions")
                if hashes["raw"] != hashes["auto"]:
                    raise AssertionError("raw and automatic output identities differ")
                medians = {mode: statistics.median(values) for mode, values in samples.items()}
                records.append({
                    "case": name,
                    "count": count,
                    "selected_mode": "scc",
                    "selection_reason": first["auto"].info["auto_policy_reason"],
                    "raw_median_ns": medians["raw"],
                    "auto_median_ns": medians["auto"],
                    "auto_raw_ratio": medians["auto"] / medians["raw"],
                    "samples_ns": samples,
                    "output_sha256": next(iter(hashes["auto"])),
                })
    finally:
        gc.enable()

    long_rows = [row for row in records if row["count"] == 1024]
    ratios = [row["auto_raw_ratio"] for row in long_rows]
    summary = {
        "schema": SCHEMA,
        "policy": POLICY,
        "classification": "exposed WAP development guard; not a holdout",
        "cases": len(CASES),
        "counts": list(COUNTS),
        "repeats": repeats,
        "all_selected_scc": all(row["selected_mode"] == "scc" for row in records),
        "long_bank_geometric_mean_ratio": gmean(ratios),
        "long_bank_max_case_ratio": max(ratios),
        "limits": {"geometric_mean": 0.75, "per_case": 0.80},
    }
    summary["pass"] = (
        summary["all_selected_scc"]
        and summary["long_bank_geometric_mean_ratio"] <= 0.75
        and summary["long_bank_max_case_ratio"] <= 0.80
    )
    payload = {"summary": summary, "records": records}
    destination.joinpath("results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise AssertionError("WAP automatic selector guard failed")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args(argv)
    run(args.library.resolve(), args.cases.resolve(), args.out.resolve(), args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
