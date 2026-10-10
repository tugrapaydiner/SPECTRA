"""Complete-cost development comparison on the fixed Abilene trace case."""
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
from typing import Any, Iterable, Sequence

from experiments.real_traffic import (
    TrafficCase,
    binary_clauses,
    solve_with_proof,
    verify_contradiction,
    verify_labels,
)
from experiments.real_traffic.observer import restrictions_to_units

ARMS = ("spectra", "minicard", "cudd_dynamic", "cudd_minfill")
MAX_WORK = 100_000_000
MAX_BYTES = 512 * 1024 * 1024


def _percentile(values: Sequence[int], fraction: float) -> float:
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


def _choices(mask: int) -> tuple[int, ...]:
    return tuple(color for color in range(mask.bit_length()) if mask >> color & 1)


def model_to_labels(model: Iterable[int], masks: Sequence[int]) -> tuple[int, ...]:
    positive = {literal for literal in model if literal > 0}
    labels = []
    for vertex, mask in enumerate(masks):
        choices = _choices(mask)
        if not choices or len(choices) > 2:
            raise ValueError("model conversion requires binary lists")
        labels.append(choices[0] if len(choices) == 1
                      else choices[int(vertex + 1 in positive)])
    return tuple(labels)


def assignment_to_labels(model: dict[str, bool], masks: Sequence[int],
                         names: Sequence[str]) -> tuple[int, ...]:
    if set(model) != set(names):
        raise AssertionError("CUDD did not materialize every original variable")
    labels = []
    for vertex, mask in enumerate(masks):
        choices = _choices(mask)
        if not choices or len(choices) > 2:
            raise ValueError("BDD conversion requires binary lists")
        labels.append(choices[0] if len(choices) == 1
                      else choices[int(bool(model[names[vertex]]))])
    return tuple(labels)


def minfill_order(vertices: int, edges: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    adjacency = [set() for _ in range(vertices)]
    for left, right in edges:
        if not (0 <= left < vertices and 0 <= right < vertices) or left == right:
            raise ValueError("invalid min-fill edge")
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(range(vertices))
    order: list[int] = []
    while remaining:
        def score(vertex: int) -> tuple[int, int, int]:
            neighbors = sorted(adjacency[vertex] & remaining)
            fill = sum(
                right not in adjacency[left]
                for index, left in enumerate(neighbors)
                for right in neighbors[index + 1:]
            )
            return fill, len(neighbors), vertex

        vertex = min(remaining, key=score)
        neighbors = list(adjacency[vertex] & remaining)
        for index, left in enumerate(neighbors):
            for right in neighbors[index + 1:]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        for neighbor in neighbors:
            adjacency[neighbor].discard(vertex)
        remaining.remove(vertex)
        order.append(vertex)
    if sorted(order) != list(range(vertices)):
        raise AssertionError("min-fill order is not a permutation")
    return tuple(order)


def _pin_current_process() -> int | None:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    requested = os.environ.get("SPECTRA_CPU")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        return None
    cpu = int(requested) if requested is not None else allowed[0]
    if cpu not in allowed:
        raise ValueError("requested CPU is outside process affinity")
    os.sched_setaffinity(0, {cpu})
    return cpu


def _prove_unsat(case: TrafficCase, query: tuple[tuple[int, int], ...]) -> tuple[dict, int]:
    started = time.perf_counter_ns()
    outcome = solve_with_proof(case.edges, case.masks, query)
    if outcome.status != "UNSAT" or outcome.contradiction is None:
        raise AssertionError("solver reported UNSAT for a satisfiable request")
    receipt = verify_contradiction(case.edges, case.masks, query, outcome.contradiction)
    return receipt, time.perf_counter_ns() - started


def _record_sat(case: TrafficCase, query: tuple[tuple[int, int], ...],
                labels: Sequence[int]) -> tuple[str, int]:
    started = time.perf_counter_ns()
    if not verify_labels(case.edges, case.masks, labels, query):
        raise AssertionError("returned SAT witness fails the original trace case")
    digest = __import__("hashlib").sha256(bytes(labels)).hexdigest()
    return digest, time.perf_counter_ns() - started


def _run_spectra(case: TrafficCase, library: Path) -> tuple[dict, list[dict]]:
    from spectra.cnf.quotient_query import QuotientRuntime

    setup_started = time.perf_counter_ns()
    runtime = QuotientRuntime(library)
    prepared = runtime.prepare(
        case.vertices, case.palette, case.edges,
        masks=case.masks, mode="scc", max_build_bytes=MAX_BYTES,
    )
    info = prepared.info
    setup_ns = time.perf_counter_ns() - setup_started
    rows = []
    proof_ns = check_ns = engine_ns = 0
    work = branches = backtracks = reductions = 0
    try:
        for index, (query, expected) in enumerate(zip(
                case.queries, case.statuses)):
            query_started = time.perf_counter_ns()
            engine_started = time.perf_counter_ns()
            result = prepared.solve(
                query, max_work=MAX_WORK, max_state_bytes=MAX_BYTES,
            )
            engine_elapsed = time.perf_counter_ns() - engine_started
            engine_ns += engine_elapsed
            work += result.work
            branches += result.branches
            backtracks += result.backtracks
            reductions += result.reductions
            if result.status == "SAT_VERIFIED":
                observed = "SAT"
                digest, observer_elapsed = _record_sat(case, query, result.labels)
                check_ns += observer_elapsed
                proof_edges = 0
            elif result.reason in {"exhausted", "base_contradiction", "restriction_conflict"}:
                observed = "UNSAT"
                receipt, observer_elapsed = _prove_unsat(case, query)
                proof_ns += observer_elapsed
                proof_edges = receipt["path_edges"]
            else:
                raise AssertionError(f"SPECTRA stopped without an exact answer: {result.reason}")
            if observed != expected:
                raise AssertionError("SPECTRA status differs from the retained trace case")
            rows.append({
                "index": index,
                "timestamp": case.timestamps[index],
                "status": observed,
                "engine_ns": engine_elapsed,
                "complete_ns": time.perf_counter_ns() - query_started,
                "proof_edges": proof_edges,
                "witness_sha256": digest if observed == "SAT" else None,
                "work": result.work,
                "branches": result.branches,
                "backtracks": result.backtracks,
                "reductions": result.reductions,
            })
    finally:
        dispose_started = time.perf_counter_ns()
        prepared.close()
        dispose_ns = time.perf_counter_ns() - dispose_started
    session_ns = setup_ns + sum(row["complete_ns"] for row in rows) + dispose_ns
    return ({
        "setup_ns": setup_ns,
        "dispose_ns": dispose_ns,
        "session_ns": session_ns,
        "engine_ns": engine_ns,
        "proof_ns": proof_ns,
        "check_ns": check_ns,
        "work": work,
        "branches": branches,
        "backtracks": backtracks,
        "reductions": reductions,
        "quotient_vertices": info["quotient_vertices"],
        "quotient_arcs": info["quotient_arcs"],
        "index_payload_bytes": info["total_owned_index_payload_bytes"],
    }, rows)


def _run_minicard(case: TrafficCase) -> tuple[dict, list[dict]]:
    from pysat.solvers import Solver

    setup_started = time.perf_counter_ns()
    clauses = binary_clauses(case.edges, case.masks)
    solver = Solver(name="minicard", bootstrap_with=clauses)
    setup_ns = time.perf_counter_ns() - setup_started
    rows = []
    proof_ns = check_ns = engine_ns = 0
    try:
        for index, (query, expected) in enumerate(zip(
                case.queries, case.statuses)):
            query_started = time.perf_counter_ns()
            assumptions = restrictions_to_units(query, case.masks)
            engine_started = time.perf_counter_ns()
            sat = bool(solver.solve(assumptions=list(assumptions)))
            engine_elapsed = time.perf_counter_ns() - engine_started
            engine_ns += engine_elapsed
            if sat:
                model = solver.get_model()
                if model is None:
                    raise AssertionError("MiniCard returned SAT without a model")
                labels = model_to_labels(model, case.masks)
                observed = "SAT"
                digest, observer_elapsed = _record_sat(case, query, labels)
                check_ns += observer_elapsed
                proof_edges = 0
            else:
                observed = "UNSAT"
                receipt, observer_elapsed = _prove_unsat(case, query)
                proof_ns += observer_elapsed
                proof_edges = receipt["path_edges"]
            if observed != expected:
                raise AssertionError("MiniCard status differs from the retained trace case")
            rows.append({
                "index": index,
                "timestamp": case.timestamps[index],
                "status": observed,
                "engine_ns": engine_elapsed,
                "complete_ns": time.perf_counter_ns() - query_started,
                "proof_edges": proof_edges,
                "witness_sha256": digest if observed == "SAT" else None,
            })
    finally:
        dispose_started = time.perf_counter_ns()
        solver.delete()
        dispose_ns = time.perf_counter_ns() - dispose_started
    session_ns = setup_ns + sum(row["complete_ns"] for row in rows) + dispose_ns
    return ({
        "setup_ns": setup_ns,
        "dispose_ns": dispose_ns,
        "session_ns": session_ns,
        "engine_ns": engine_ns,
        "proof_ns": proof_ns,
        "check_ns": check_ns,
        "clauses": len(clauses),
    }, rows)


def _bdd_literal(variable: Any, literal: int) -> Any:
    return variable if literal > 0 else ~variable


def _run_cudd(case: TrafficCase, *, dynamic: bool) -> tuple[dict, list[dict]]:
    from dd.cudd import BDD

    setup_started = time.perf_counter_ns()
    bdd = BDD()
    names = tuple(f"v{vertex:04d}" for vertex in range(case.vertices))
    order = tuple(range(case.vertices)) if dynamic else minfill_order(case.vertices, case.edges)
    bdd.declare(*(names[vertex] for vertex in order))
    bdd.configure(reordering=dynamic)
    variables = tuple(bdd.var(name) for name in names)
    clauses = binary_clauses(case.edges, case.masks)
    root = bdd.true
    for clause in clauses:
        if len(clause) == 1:
            term = _bdd_literal(variables[abs(clause[0]) - 1], clause[0])
        else:
            term = (
                _bdd_literal(variables[abs(clause[0]) - 1], clause[0])
                | _bdd_literal(variables[abs(clause[1]) - 1], clause[1])
            )
        root = root & term
    manager_nodes = len(bdd)
    levels = tuple(bdd.level_of_var(name) for name in names)
    setup_ns = time.perf_counter_ns() - setup_started
    rows = []
    proof_ns = check_ns = engine_ns = 0
    try:
        for index, (query, expected) in enumerate(zip(
                case.queries, case.statuses)):
            query_started = time.perf_counter_ns()
            units = restrictions_to_units(query, case.masks)
            assignments = {
                names[abs(literal) - 1]: literal > 0 for literal in units
            }
            engine_started = time.perf_counter_ns()
            restricted = bdd.let(assignments, root)
            if restricted == bdd.false:
                sat = False
                model = None
            else:
                sat = True
                model = bdd.pick(restricted, care_vars=names)
            engine_elapsed = time.perf_counter_ns() - engine_started
            engine_ns += engine_elapsed
            if sat:
                if model is None:
                    raise AssertionError("CUDD returned a nonempty relation without a model")
                labels = assignment_to_labels(model, case.masks, names)
                observed = "SAT"
                digest, observer_elapsed = _record_sat(case, query, labels)
                check_ns += observer_elapsed
                proof_edges = 0
            else:
                observed = "UNSAT"
                receipt, observer_elapsed = _prove_unsat(case, query)
                proof_ns += observer_elapsed
                proof_edges = receipt["path_edges"]
            if observed != expected:
                raise AssertionError("CUDD status differs from the retained trace case")
            rows.append({
                "index": index,
                "timestamp": case.timestamps[index],
                "status": observed,
                "engine_ns": engine_elapsed,
                "complete_ns": time.perf_counter_ns() - query_started,
                "proof_edges": proof_edges,
                "witness_sha256": digest if observed == "SAT" else None,
            })
            del restricted
    finally:
        dispose_started = time.perf_counter_ns()
        del root
        del variables
        try:
            del term
        except UnboundLocalError:
            pass
        try:
            bdd.collect_garbage()
        except AttributeError:
            pass
        del bdd
        gc.collect()
        dispose_ns = time.perf_counter_ns() - dispose_started
    session_ns = setup_ns + sum(row["complete_ns"] for row in rows) + dispose_ns
    return ({
        "setup_ns": setup_ns,
        "dispose_ns": dispose_ns,
        "session_ns": session_ns,
        "engine_ns": engine_ns,
        "proof_ns": proof_ns,
        "check_ns": check_ns,
        "clauses": len(clauses),
        "manager_nodes": manager_nodes,
        "dynamic_reordering": dynamic,
        "final_levels": levels,
    }, rows)


def run_child(case_path: Path, arm: str, repeat: int,
              library: Path | None, output: Path) -> None:
    cpu = _pin_current_process()
    process_started = time.perf_counter_ns()
    load_started = time.perf_counter_ns()
    case = TrafficCase.read(case_path)
    load_ns = time.perf_counter_ns() - load_started
    if arm == "spectra":
        if library is None:
            raise ValueError("SPECTRA arm requires a native library")
        totals, rows = _run_spectra(case, library)
    elif arm == "minicard":
        totals, rows = _run_minicard(case)
    elif arm == "cudd_dynamic":
        totals, rows = _run_cudd(case, dynamic=True)
    elif arm == "cudd_minfill":
        totals, rows = _run_cudd(case, dynamic=False)
    else:
        raise ValueError("unknown arm")
    process_ns = time.perf_counter_ns() - process_started
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "schema": "spectra.real_traffic.benchmark_cell.v1",
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
    print(json.dumps({key: value for key, value in payload.items() if key != "query_rows"},
                     sort_keys=True))


def _arm_schedule(repeats: int) -> list[tuple[int, str]]:
    schedule = []
    for repeat in range(repeats):
        rotation = repeat % len(ARMS)
        order = list(ARMS[rotation:] + ARMS[:rotation])
        if repeat % 2:
            order.reverse()
        schedule.extend((repeat, arm) for arm in order)
    return schedule


def _summary(cells: Sequence[dict], failures: Sequence[dict]) -> dict:
    arms: dict[str, dict] = {}
    for arm in ARMS:
        selected = [cell for cell in cells if cell["arm"] == arm]
        if not selected:
            continue
        session = [cell["session_ns"] for cell in selected]
        process = [cell["process_wall_ns"] for cell in selected]
        query = [row["complete_ns"] for cell in selected for row in cell["query_rows"]]
        arms[arm] = {
            "completed_repeats": len(selected),
            "mean_session_ns": statistics.fmean(session),
            "median_session_ns": statistics.median(session),
            "mean_process_wall_ns": statistics.fmean(process),
            "median_process_wall_ns": statistics.median(process),
            "mean_complete_query_ns": statistics.fmean(query),
            "p95_complete_query_ns": _percentile(query, 0.95),
            "peak_rss_kib": max(cell["peak_rss_kib"] for cell in selected),
            "sat_queries_per_repeat": selected[0]["sat_queries"],
            "unsat_queries_per_repeat": selected[0]["unsat_queries"],
        }
        for key in ("manager_nodes", "quotient_vertices", "quotient_arcs",
                    "index_payload_bytes", "clauses"):
            values = [cell[key] for cell in selected if key in cell]
            if values:
                arms[arm][key] = values[-1]
    baseline_names = [name for name in ("minicard", "cudd_dynamic", "cudd_minfill")
                      if name in arms]
    best = min(baseline_names, key=lambda name: arms[name]["mean_session_ns"]) \
        if baseline_names else None
    ratio = (arms["spectra"]["mean_session_ns"] / arms[best]["mean_session_ns"]
             if best is not None and "spectra" in arms else None)
    return {
        "schema": "spectra.real_traffic.benchmark_summary.v1",
        "decision": "DEVELOPMENT_ONLY",
        "arms": arms,
        "best_external_baseline": best,
        "spectra_to_best_external_mean_ratio": ratio,
        "failures": list(failures),
        "boundary": (
            "X01 is exposed development data. Ratios from this run can choose or reject "
            "a direction, but cannot establish confirmation, deployment, or flagship status."
        ),
    }


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
    for repeat, arm in _arm_schedule(repeats):
        cell_path = cells_dir / f"{repeat:02d}-{arm}.json"
        command = [
            sys.executable, "-m", "experiments.real_traffic.benchmark",
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
                "arm": arm, "repeat": repeat, "returncode": result.returncode,
                "process_wall_ns": wall_ns,
                "stderr_tail": result.stderr[-4000:],
            })
            continue
        cell = json.loads(cell_path.read_text())
        cell["process_wall_ns"] = wall_ns
        cell_path.write_text(json.dumps(cell, sort_keys=True, separators=(",", ":")) + "\n")
        cells.append(cell)
    summary = _summary(cells, failures)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (output / "raw.jsonl").open("w") as stream:
        for cell in cells:
            stream.write(json.dumps(cell, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def resolve_library(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.glob("_spectra_quotient_query*.so"))
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
    child.add_argument("--out", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--case", type=Path, required=True)
    run.add_argument("--library", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.command == "child":
        run_child(args.case, args.arm, args.repeat,
                  resolve_library(args.library), args.out)
    else:
        run_coordinator(args.case, args.out, args.repeats,
                        resolve_library(args.library))


if __name__ == "__main__":
    main()
