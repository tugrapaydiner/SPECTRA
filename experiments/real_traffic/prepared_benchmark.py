"""Repeat the fair comparison with one prepared independent proof checker.

The solver arms, workload, answer materialisation, and native SAT witness checker
are identical to ``fair_benchmark``. The only change is that immutable implications
from the original graph and lists are reconstructed once per session. Every UNSAT
query still verifies both retained contradiction paths against original data plus
that query's unit restrictions.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from experiments.real_traffic import PreparedContradictionChecker, TrafficCase
from experiments.real_traffic import fair_benchmark as fair


def run_child(case_path: Path, arm: str, repeat: int,
              library: Path, output: Path) -> None:
    cpu = fair.pin_current_process()
    process_started = time.perf_counter_ns()
    load_started = time.perf_counter_ns()
    case = TrafficCase.read(case_path)
    load_ns = time.perf_counter_ns() - load_started

    proof_setup_started = time.perf_counter_ns()
    proof_checker = PreparedContradictionChecker(case.edges, case.masks)
    proof_setup_ns = time.perf_counter_ns() - proof_setup_started
    original_prove_unsat = fair.prove_unsat

    def prepared_prove_unsat(
            observed_case: TrafficCase,
            query: tuple[tuple[int, int], ...],
            index: int) -> tuple[dict, int]:
        if observed_case is not case:
            raise AssertionError("proof checker received a different case")
        started = time.perf_counter_ns()
        certificate = case.contradictions[index]
        if certificate is None:
            raise AssertionError("solver reported UNSAT without a retained contradiction")
        receipt = proof_checker.verify(query, certificate)
        return receipt, time.perf_counter_ns() - started

    fair.prove_unsat = prepared_prove_unsat
    try:
        if arm == "spectra":
            totals, rows = fair.run_spectra(case, library)
        elif arm == "minicard":
            totals, rows = fair.run_minicard(case, library)
        elif arm == "cudd_dynamic":
            totals, rows = fair.run_cudd(case, library, dynamic=True)
        elif arm == "cudd_minfill":
            totals, rows = fair.run_cudd(case, library, dynamic=False)
        else:
            raise ValueError("unknown arm")
    finally:
        fair.prove_unsat = original_prove_unsat
        proof_dispose_started = time.perf_counter_ns()
        del prepared_prove_unsat
        del proof_checker
        gc.collect()
        proof_dispose_ns = time.perf_counter_ns() - proof_dispose_started

    totals["proof_checker_setup_ns"] = proof_setup_ns
    totals["proof_checker_dispose_ns"] = proof_dispose_ns
    totals["setup_ns"] += proof_setup_ns
    totals["dispose_ns"] += proof_dispose_ns
    totals["session_ns"] += proof_setup_ns + proof_dispose_ns
    process_ns = time.perf_counter_ns() - process_started
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "schema": "spectra.real_traffic.prepared_benchmark_cell.v1",
        "arm": arm,
        "repeat": repeat,
        "cpu": cpu,
        "case_sha256": case.case_sha256,
        "queries": len(rows),
        "sat_queries": sum(row["status"] == "SAT" for row in rows),
        "unsat_queries": sum(row["status"] == "UNSAT" for row in rows),
        "load_and_replay_ns": load_ns,
        "process_ns": process_ns,
        "peak_rss_kib": peak_rss_kib,
        **totals,
        "query_rows": rows,
    }
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(
        {key: value for key, value in payload.items() if key != "query_rows"},
        sort_keys=True,
    ))


def run_coordinator(case_path: Path, output: Path, repeats: int,
                    library: Path) -> None:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output.mkdir(parents=True, exist_ok=False)
    cells_dir = output / "cells"
    logs_dir = output / "logs"
    cells_dir.mkdir()
    logs_dir.mkdir()
    cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    cells = []
    failures = []
    for repeat, arm in fair.arm_schedule(repeats):
        cell_path = cells_dir / f"{repeat:02d}-{arm}.json"
        command = [
            sys.executable, "-m", "experiments.real_traffic.prepared_benchmark",
            "child", "--case", str(case_path.resolve()), "--arm", arm,
            "--repeat", str(repeat), "--library", str(library.resolve()),
            "--out", str(cell_path.resolve()),
        ]
        env = dict(os.environ)
        if cpu is not None:
            env["SPECTRA_CPU"] = str(cpu)
        started = time.perf_counter_ns()
        result = subprocess.run(command, text=True, capture_output=True, env=env)
        wall_ns = time.perf_counter_ns() - started
        stem = f"{repeat:02d}-{arm}"
        (logs_dir / f"{stem}.stdout").write_text(result.stdout)
        (logs_dir / f"{stem}.stderr").write_text(result.stderr)
        if result.returncode != 0 or not cell_path.is_file():
            failures.append({
                "arm": arm,
                "repeat": repeat,
                "returncode": result.returncode,
                "process_wall_ns": wall_ns,
                "stderr_tail": result.stderr[-4000:],
            })
            continue
        cell = json.loads(cell_path.read_text())
        cell["process_wall_ns"] = wall_ns
        cell_path.write_text(
            json.dumps(cell, sort_keys=True, separators=(",", ":")) + "\n"
        )
        cells.append(cell)
    summary = fair.summarize(cells, failures)
    summary["schema"] = "spectra.real_traffic.prepared_benchmark_summary.v1"
    summary["proof_checker"] = (
        "Original graph/list implications prepared once per complete session; "
        "query unit implications and both contradiction paths checked per UNSAT request."
    )
    summary["boundary"] = (
        "X01 remains exposed development data. The solver arms, query bank, answer "
        "materialisation, and native SAT checker are unchanged from the preceding fair "
        "comparison. This run can measure audit overhead but cannot establish confirmation, "
        "deployment, or flagship status."
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (output / "raw.jsonl").open("w") as stream:
        for cell in cells:
            stream.write(json.dumps(cell, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    child = subparsers.add_parser("child")
    child.add_argument("--case", type=Path, required=True)
    child.add_argument("--arm", choices=fair.ARMS, required=True)
    child.add_argument("--repeat", type=int, required=True)
    child.add_argument("--library", type=Path, required=True)
    child.add_argument("--out", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--case", type=Path, required=True)
    run.add_argument("--library", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    library = fair.resolve_library(args.library)
    if args.command == "child":
        run_child(args.case, args.arm, args.repeat, library, args.out)
    else:
        run_coordinator(args.case, args.out, args.repeats, library)


if __name__ == "__main__":
    main()
