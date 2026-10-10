"""Strong development comparison with native MiniCard decoding and causal ablations."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
from typing import Sequence

from experiments.real_traffic import PreparedContradictionChecker, TrafficCase
from experiments.real_traffic import fair_benchmark as fair
from experiments.real_traffic.model_decoder import ModelDecoderRuntime

ARMS = (
    "spectra_scc",
    "spectra_hybrid",
    "spectra_parity",
    "spectra_none",
    "minicard_native",
    "minicard_python",
    "cudd_minfill",
)
MODES = {
    "spectra_scc": "scc",
    "spectra_hybrid": "hybrid",
    "spectra_parity": "parity",
    "spectra_none": "none",
}


def percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def run_spectra_mode(case: TrafficCase, library: Path, selected_mode: str):
    from spectra.cnf.quotient_query import QuotientRuntime

    original_prepare = QuotientRuntime.prepare

    def forced_prepare(self, n, k, edges, *, masks=(), mode="hybrid",
                       max_build_bytes=fair.MAX_BYTES):
        del mode
        return original_prepare(
            self, n, k, edges, masks=masks, mode=selected_mode,
            max_build_bytes=max_build_bytes,
        )

    QuotientRuntime.prepare = forced_prepare
    try:
        return fair.run_spectra(case, library)
    finally:
        QuotientRuntime.prepare = original_prepare


def add_owned_costs(totals: dict, *, setup_ns: int, dispose_ns: int,
                    prefix: str) -> None:
    totals[f"{prefix}_setup_ns"] = setup_ns
    totals[f"{prefix}_dispose_ns"] = dispose_ns
    totals["setup_ns"] += setup_ns
    totals["dispose_ns"] += dispose_ns
    totals["session_ns"] += setup_ns + dispose_ns


def run_child(case_path: Path, arm: str, repeat: int, library: Path,
              decoder_library: Path, output: Path) -> None:
    cpu = fair.pin_current_process()
    process_started = time.perf_counter_ns()
    load_started = time.perf_counter_ns()
    case = TrafficCase.read(case_path)
    load_ns = time.perf_counter_ns() - load_started

    proof_setup_started = time.perf_counter_ns()
    proof_checker = PreparedContradictionChecker(case.edges, case.masks)
    proof_setup_ns = time.perf_counter_ns() - proof_setup_started
    original_prove_unsat = fair.prove_unsat

    def prepared_prove_unsat(observed_case, query, index):
        if observed_case is not case:
            raise AssertionError("proof checker received a different case")
        started = time.perf_counter_ns()
        certificate = case.contradictions[index]
        if certificate is None:
            raise AssertionError("solver reported UNSAT without a retained contradiction")
        receipt = proof_checker.verify(query, certificate)
        return receipt, time.perf_counter_ns() - started

    fair.prove_unsat = prepared_prove_unsat
    decoder = None
    decoder_setup_ns = decoder_dispose_ns = 0
    original_model_to_labels = fair.model_to_labels
    try:
        if arm in MODES:
            totals, rows = run_spectra_mode(case, library, MODES[arm])
        elif arm == "minicard_python":
            totals, rows = fair.run_minicard(case, library)
        elif arm == "minicard_native":
            decoder_setup_started = time.perf_counter_ns()
            decoder = ModelDecoderRuntime(decoder_library).prepare(case.masks)
            decoder_setup_ns = time.perf_counter_ns() - decoder_setup_started
            fair.model_to_labels = lambda model, masks: decoder.decode(model)
            totals, rows = fair.run_minicard(case, library)
            totals["decoder_info"] = decoder.info
        elif arm == "cudd_minfill":
            totals, rows = fair.run_cudd(case, library, dynamic=False)
        else:
            raise ValueError("unknown arm")
    finally:
        fair.model_to_labels = original_model_to_labels
        if decoder is not None:
            decoder_dispose_started = time.perf_counter_ns()
            decoder.close()
            decoder = None
            decoder_dispose_ns = time.perf_counter_ns() - decoder_dispose_started
        fair.prove_unsat = original_prove_unsat
        proof_dispose_started = time.perf_counter_ns()
        del prepared_prove_unsat
        del proof_checker
        gc.collect()
        proof_dispose_ns = time.perf_counter_ns() - proof_dispose_started

    add_owned_costs(
        totals, setup_ns=proof_setup_ns, dispose_ns=proof_dispose_ns,
        prefix="proof_checker",
    )
    if arm == "minicard_native":
        add_owned_costs(
            totals, setup_ns=decoder_setup_ns, dispose_ns=decoder_dispose_ns,
            prefix="model_decoder",
        )
    process_ns = time.perf_counter_ns() - process_started
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "schema": "spectra.real_traffic.strong_benchmark_cell.v1",
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


def schedule(repeats: int) -> list[tuple[int, str]]:
    jobs = []
    for repeat in range(repeats):
        rotation = repeat % len(ARMS)
        order = list(ARMS[rotation:] + ARMS[:rotation])
        if repeat % 2:
            order.reverse()
        jobs.extend((repeat, arm) for arm in order)
    return jobs


def summarize(cells: Sequence[dict], failures: Sequence[dict]) -> dict:
    arms: dict[str, dict] = {}
    for arm in ARMS:
        selected = [cell for cell in cells if cell["arm"] == arm]
        if not selected:
            continue
        sessions = [cell["session_ns"] for cell in selected]
        queries = [row["complete_ns"] for cell in selected for row in cell["query_rows"]]
        arms[arm] = {
            "completed_repeats": len(selected),
            "mean_session_ns": statistics.fmean(sessions),
            "median_session_ns": statistics.median(sessions),
            "mean_complete_query_ns": statistics.fmean(queries),
            "p95_complete_query_ns": percentile(queries, 0.95),
            "mean_process_wall_ns": statistics.fmean(
                cell["process_wall_ns"] for cell in selected),
            "peak_rss_kib": max(cell["peak_rss_kib"] for cell in selected),
            "sat_queries_per_repeat": selected[0]["sat_queries"],
            "unsat_queries_per_repeat": selected[0]["unsat_queries"],
        }
        for key in ("quotient_vertices", "quotient_arcs", "index_payload_bytes",
                    "clauses", "manager_nodes"):
            values = [cell[key] for cell in selected if key in cell]
            if values:
                arms[arm][key] = values[-1]
        decoder = [cell.get("decoder_info") for cell in selected if cell.get("decoder_info")]
        if decoder:
            arms[arm]["decoder_info"] = decoder[-1]

    external = [name for name in ("minicard_native", "minicard_python", "cudd_minfill")
                if name in arms]
    best_external = min(external, key=lambda name: arms[name]["mean_session_ns"]) \
        if external else None
    candidate = arms.get("spectra_scc")
    ratios = {}
    if candidate is not None:
        for name, result in arms.items():
            if name != "spectra_scc":
                ratios[name] = candidate["mean_session_ns"] / result["mean_session_ns"]
    return {
        "schema": "spectra.real_traffic.strong_benchmark_summary.v1",
        "decision": "DEVELOPMENT_ONLY",
        "arms": arms,
        "best_external_baseline": best_external,
        "spectra_to_best_external_mean_ratio": (
            candidate["mean_session_ns"] / arms[best_external]["mean_session_ns"]
            if candidate is not None and best_external is not None else None),
        "candidate_mean_ratios": ratios,
        "failures": list(failures),
        "boundary": (
            "X01 is exposed development data. Native MiniCard decoding strengthens the "
            "control but still uses the pinned PySAT MiniCard core. The no/parity/hybrid "
            "arms use the same SPECTRA executor and isolate representation effects. No "
            "confirmation, deployment, or flagship conclusion follows."
        ),
    }


def run_coordinator(case_path: Path, output: Path, repeats: int,
                    library: Path, decoder_library: Path) -> None:
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
    for repeat, arm in schedule(repeats):
        cell_path = cells_dir / f"{repeat:02d}-{arm}.json"
        command = [
            sys.executable, "-m", "experiments.real_traffic.strong_benchmark",
            "child", "--case", str(case_path.resolve()), "--arm", arm,
            "--repeat", str(repeat), "--library", str(library.resolve()),
            "--decoder", str(decoder_library.resolve()),
            "--out", str(cell_path.resolve()),
        ]
        env = dict(os.environ)
        if cpu is not None:
            env["SPECTRA_CPU"] = str(cpu)
        started = time.perf_counter_ns()
        completed = subprocess.run(command, text=True, capture_output=True, env=env)
        wall_ns = time.perf_counter_ns() - started
        stem = f"{repeat:02d}-{arm}"
        (logs_dir / f"{stem}.stdout").write_text(completed.stdout)
        (logs_dir / f"{stem}.stderr").write_text(completed.stderr)
        if completed.returncode != 0 or not cell_path.is_file():
            failures.append({
                "arm": arm,
                "repeat": repeat,
                "returncode": completed.returncode,
                "process_wall_ns": wall_ns,
                "stderr_tail": completed.stderr[-4000:],
            })
            continue
        cell = json.loads(cell_path.read_text())
        cell["process_wall_ns"] = wall_ns
        cell_path.write_text(
            json.dumps(cell, sort_keys=True, separators=(",", ":")) + "\n")
        cells.append(cell)
    summary = summarize(cells, failures)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (output / "raw.jsonl").open("w") as stream:
        for cell in cells:
            stream.write(json.dumps(cell, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def resolve_library(path: Path, pattern: str) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise ValueError("native library path is missing or ambiguous")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    child = subparsers.add_parser("child")
    child.add_argument("--case", type=Path, required=True)
    child.add_argument("--arm", choices=ARMS, required=True)
    child.add_argument("--repeat", type=int, required=True)
    child.add_argument("--library", type=Path, required=True)
    child.add_argument("--decoder", type=Path, required=True)
    child.add_argument("--out", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--case", type=Path, required=True)
    run.add_argument("--library", type=Path, required=True)
    run.add_argument("--decoder", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    library = resolve_library(args.library, "_spectra_quotient_query*.so")
    decoder = resolve_library(args.decoder, "_spectra_model_decoder*.so")
    if args.command == "child":
        run_child(args.case, args.arm, args.repeat, library, decoder, args.out)
    else:
        run_coordinator(args.case, args.out, args.repeats, library, decoder)


if __name__ == "__main__":
    main()
