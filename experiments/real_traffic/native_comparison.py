"""Direct native development comparison for the compiled support relation.

Both arms execute the complete query bank in one native call, publish one status
byte and one original-address witness slot per query, pass the same independently
owned native auditor, and charge setup, execution, audit, and disposal.  JSON and
receipt hashing are retained separately as evidence costs.
"""
from __future__ import annotations

import argparse
import hashlib
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

from experiments.real_traffic import TrafficCase, hash_json
from experiments.real_traffic.quotient_session import QuotientSessionRuntime
from experiments.real_traffic.minicard_session import MiniCardSessionRuntime
from experiments.real_traffic.session_audit import SessionAuditRuntime
from spectra.cnf.quotient_query import QuotientRuntime

MAX_BYTES = 512 * 1024 * 1024
ARMS = ("quotient_session", "minicard_direct")


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
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def pin_current_process() -> int | None:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        return None
    requested = os.environ.get("SPECTRA_CPU")
    cpu = int(requested) if requested is not None else allowed[0]
    if cpu not in allowed:
        raise ValueError("requested CPU is outside process affinity")
    os.sched_setaffinity(0, {cpu})
    return cpu


def decode_case(path: Path, expected_sha256: str) -> TrafficCase:
    raw = json.loads(path.read_text())
    for field in ("edges", "route_links"):
        raw[field] = tuple(tuple(row) for row in raw[field])
    for field in (
        "demand_indices", "demand_names", "plan_a", "plan_b", "masks",
        "timestamps", "statuses", "witness_sha256", "contradictions",
    ):
        raw[field] = tuple(raw[field])
    raw["queries"] = tuple(
        tuple(tuple(item) for item in query) for query in raw["queries"])
    case = TrafficCase(**raw)
    if (case.case_sha256 != expected_sha256
            or case.case_sha256 != hash_json(case.canonical_without_hash())):
        raise ValueError("benchmark case identity differs")
    return case


def distribute(total: int, count: int) -> tuple[int, ...]:
    if total < 0 or count < 1:
        raise ValueError("invalid timing distribution")
    base, extra = divmod(total, count)
    return tuple(base + (index < extra) for index in range(count))


def prepare_auditor(case: TrafficCase, library: Path):
    return SessionAuditRuntime(library).prepare(
        case.edges, case.masks, case.queries, case.statuses,
        case.contradictions, max_bytes=MAX_BYTES)


def rows_from_audit(case: TrafficCase, statuses: bytes, engine_ns,
                    batch_ns: int, audit) -> list[dict]:
    count = len(case.queries)
    if len(statuses) != count or len(engine_ns) != count or len(audit.query_ns) != count:
        raise AssertionError("query accounting geometry differs")
    batch_residual = batch_ns - sum(engine_ns)
    audit_residual = audit.elapsed_ns - sum(audit.query_ns)
    if batch_residual < 0 or audit_residual < 0:
        raise AssertionError("component timing exceeds its enclosing call")
    batch_shared = distribute(batch_residual, count)
    audit_shared = distribute(audit_residual, count)
    rows = []
    for index in range(count):
        status = "UNSAT" if statuses[index] else "SAT"
        if status != case.statuses[index]:
            raise AssertionError("native status differs from retained outcome")
        complete = (int(engine_ns[index]) + batch_shared[index]
                    + int(audit.query_ns[index]) + audit_shared[index])
        rows.append({
            "index": index,
            "timestamp": case.timestamps[index],
            "status": status,
            "engine_ns": int(engine_ns[index]),
            "batch_publication_ns": batch_shared[index],
            "observer_ns": int(audit.query_ns[index]),
            "observer_call_ns": audit_shared[index],
            "complete_ns": complete,
        })
    if sum(row["complete_ns"] for row in rows) != batch_ns + audit.elapsed_ns:
        raise AssertionError("query timings no longer reconstruct the measured calls")
    return rows


def run_quotient_session(case: TrafficCase, quotient_library: Path,
                         session_library: Path, audit_library: Path):
    setup_started = time.perf_counter_ns()
    quotient = QuotientRuntime(quotient_library).prepare(
        case.vertices, case.palette, case.edges, masks=case.masks,
        mode="hybrid", max_build_bytes=MAX_BYTES)
    certificate = quotient.certificate()
    prepared = QuotientSessionRuntime(session_library).prepare(
        certificate, max_bytes=MAX_BYTES)
    auditor = prepare_auditor(case, audit_library)
    quotient_info = quotient.info
    session_info = prepared.info
    audit_info = auditor.info
    setup_ns = time.perf_counter_ns() - setup_started

    statuses, labels, engine_ns, operations, batch_ns = prepared.solve_raw(
        case.queries, compiled=True, max_bytes=MAX_BYTES)
    audit = auditor.verify(statuses, labels)
    if audit.sat != case.sat_queries or audit.unsat != case.unsat_queries:
        raise AssertionError("audited quotient-session totals differ")
    rows = rows_from_audit(case, statuses, engine_ns, batch_ns, audit)

    dispose_started = time.perf_counter_ns()
    auditor.close(); prepared.close(); quotient.close()
    dispose_ns = time.perf_counter_ns() - dispose_started
    return ({
        "setup_ns": setup_ns,
        "batch_ns": batch_ns,
        "engine_ns": sum(engine_ns),
        "audit_ns": audit.elapsed_ns,
        "sat_check_ns": audit.sat_check_ns,
        "proof_check_ns": audit.proof_check_ns,
        "proof_edges": audit.proof_edges,
        "dispose_ns": dispose_ns,
        "session_ns": setup_ns + batch_ns + audit.elapsed_ns + dispose_ns,
        "quotient_vertices": quotient_info["quotient_vertices"],
        "quotient_arcs": quotient_info["quotient_arcs"],
        "quotient_payload_bytes": quotient_info["total_owned_index_payload_bytes"],
        "session_payload_bytes": session_info["payload_bytes"],
        "audit_payload_bytes": audit_info["payload_bytes"],
        "word_operations": sum(operations),
    }, rows, statuses, labels)


def run_minicard(case: TrafficCase, minicard_library: Path, audit_library: Path):
    setup_started = time.perf_counter_ns()
    prepared = MiniCardSessionRuntime(minicard_library).prepare(
        case.edges, case.masks, max_bytes=MAX_BYTES)
    auditor = prepare_auditor(case, audit_library)
    solver_info = prepared.info
    audit_info = auditor.info
    setup_ns = time.perf_counter_ns() - setup_started

    statuses, labels, engine_ns, decisions, conflicts, batch_ns = prepared.solve_raw(
        case.queries, max_bytes=MAX_BYTES)
    audit = auditor.verify(statuses, labels)
    if audit.sat != case.sat_queries or audit.unsat != case.unsat_queries:
        raise AssertionError("audited MiniCard totals differ")
    rows = rows_from_audit(case, statuses, engine_ns, batch_ns, audit)

    dispose_started = time.perf_counter_ns()
    auditor.close(); prepared.close()
    dispose_ns = time.perf_counter_ns() - dispose_started
    return ({
        "setup_ns": setup_ns,
        "batch_ns": batch_ns,
        "engine_ns": sum(engine_ns),
        "audit_ns": audit.elapsed_ns,
        "sat_check_ns": audit.sat_check_ns,
        "proof_check_ns": audit.proof_check_ns,
        "proof_edges": audit.proof_edges,
        "dispose_ns": dispose_ns,
        "session_ns": setup_ns + batch_ns + audit.elapsed_ns + dispose_ns,
        "variables": solver_info["variables"],
        "clauses": solver_info["clauses"],
        "adapter_payload_bytes": solver_info["adapter_payload_bytes"],
        "audit_payload_bytes": audit_info["payload_bytes"],
        "decisions": sum(decisions),
        "conflicts": sum(conflicts),
    }, rows, statuses, labels)


def run_child(case_path: Path, case_sha256: str, arm: str, repeat: int,
              quotient_library: Path, session_library: Path,
              minicard_library: Path, audit_library: Path, output: Path) -> None:
    cpu = pin_current_process()
    process_started = time.perf_counter_ns()
    load_started = time.perf_counter_ns()
    case = decode_case(case_path, case_sha256)
    load_ns = time.perf_counter_ns() - load_started
    if arm == "quotient_session":
        totals, rows, statuses, labels = run_quotient_session(
            case, quotient_library, session_library, audit_library)
    else:
        totals, rows, statuses, labels = run_minicard(
            case, minicard_library, audit_library)

    evidence_started = time.perf_counter_ns()
    receipts = {
        "statuses_sha256": hashlib.sha256(statuses).hexdigest(),
        "labels_sha256": hashlib.sha256(labels).hexdigest(),
        "rows_sha256": hashlib.sha256(json.dumps(
            rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    evidence_ns = time.perf_counter_ns() - evidence_started
    payload = {
        "schema": "spectra.real_traffic.native_comparison_cell.v1",
        "arm": arm,
        "repeat": repeat,
        "cpu": cpu,
        "case_sha256": case.case_sha256,
        "queries": len(rows),
        "sat_queries": sum(row["status"] == "SAT" for row in rows),
        "unsat_queries": sum(row["status"] == "UNSAT" for row in rows),
        "load_and_hash_ns": load_ns,
        "evidence_ns": evidence_ns,
        "process_ns": time.perf_counter_ns() - process_started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        **totals,
        **receipts,
        "query_rows": rows,
    }
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "query_rows"}, sort_keys=True))


def schedule(repeats: int):
    return tuple(
        (repeat, arm)
        for repeat in range(repeats)
        for arm in (ARMS if repeat % 2 == 0 else tuple(reversed(ARMS)))
    )


def summarize(cells, failures, case_sha256: str):
    arms = {}
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
            "p95_session_ns": percentile(sessions, .95),
            "mean_complete_query_ns": statistics.fmean(queries),
            "p95_complete_query_ns": percentile(queries, .95),
            "mean_setup_ns": statistics.fmean(cell["setup_ns"] for cell in selected),
            "mean_batch_ns": statistics.fmean(cell["batch_ns"] for cell in selected),
            "mean_engine_ns": statistics.fmean(cell["engine_ns"] for cell in selected),
            "mean_audit_ns": statistics.fmean(cell["audit_ns"] for cell in selected),
            "mean_dispose_ns": statistics.fmean(cell["dispose_ns"] for cell in selected),
            "peak_rss_kib": max(cell["peak_rss_kib"] for cell in selected),
            "sat_queries_per_repeat": selected[0]["sat_queries"],
            "unsat_queries_per_repeat": selected[0]["unsat_queries"],
            "status_hashes": sorted({cell["statuses_sha256"] for cell in selected}),
            "label_hashes": sorted({cell["labels_sha256"] for cell in selected}),
        }
    candidate = arms.get("quotient_session")
    baseline = arms.get("minicard_direct")
    ratio = (candidate["mean_session_ns"] / baseline["mean_session_ns"]
             if candidate and baseline else None)
    return {
        "schema": "spectra.real_traffic.native_comparison_summary.v1",
        "decision": "DEVELOPMENT_ONLY",
        "case_sha256": case_sha256,
        "arms": arms,
        "quotient_session_to_direct_minicard_mean_ratio": ratio,
        "failures": list(failures),
        "boundary": (
            "X01 is exposed development data. Both arms build from the original "
            "binary-list graph, execute the full query bank in one native call, "
            "materialize every original-address witness slot, and pass the same "
            "independently owned native audit of lists, edges, requests, and retained "
            "contradiction paths. Setup, execution, audit, and disposal are charged. "
            "Receipt hashing and JSON transport are reported separately."
        ),
    }


def run_coordinator(case_path: Path, output: Path, repeats: int,
                    quotient: Path, session: Path,
                    minicard: Path, audit: Path) -> None:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    validated = TrafficCase.read(case_path)
    output.mkdir(parents=True, exist_ok=False)
    cells_dir = output / "cells"; cells_dir.mkdir()
    logs_dir = output / "logs"; logs_dir.mkdir()
    cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    cells = []
    failures = []
    for repeat, arm in schedule(repeats):
        cell = cells_dir / f"{repeat:02d}-{arm}.json"
        command = [
            sys.executable, "-m", "experiments.real_traffic.native_comparison",
            "child", "--case", str(case_path.resolve()),
            "--case-sha256", validated.case_sha256,
            "--arm", arm, "--repeat", str(repeat),
            "--quotient", str(quotient.resolve()),
            "--session", str(session.resolve()),
            "--minicard", str(minicard.resolve()),
            "--audit", str(audit.resolve()),
            "--out", str(cell.resolve()),
        ]
        environment = dict(os.environ)
        if cpu is not None:
            environment["SPECTRA_CPU"] = str(cpu)
        started = time.perf_counter_ns()
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=180, env=environment)
        wall_ns = time.perf_counter_ns() - started
        stem = f"{repeat:02d}-{arm}"
        (logs_dir / f"{stem}.stdout").write_text(completed.stdout)
        (logs_dir / f"{stem}.stderr").write_text(completed.stderr)
        if completed.returncode or not cell.exists():
            failures.append({
                "repeat": repeat, "arm": arm,
                "returncode": completed.returncode,
                "process_wall_ns": wall_ns,
                "stderr_tail": completed.stderr[-4000:],
            })
            continue
        payload = json.loads(cell.read_text())
        payload["process_wall_ns"] = wall_ns
        cell.write_text(json.dumps(payload, sort_keys=True,
                                   separators=(",", ":")) + "\n")
        cells.append(payload)
    summary = summarize(cells, failures, validated.case_sha256)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "raw.jsonl").write_text("".join(
        json.dumps(cell, sort_keys=True, separators=(",", ":")) + "\n"
        for cell in cells))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    child = sub.add_parser("child")
    child.add_argument("--case", type=Path, required=True)
    child.add_argument("--case-sha256", required=True)
    child.add_argument("--arm", choices=ARMS, required=True)
    child.add_argument("--repeat", type=int, required=True)
    child.add_argument("--quotient", type=Path, required=True)
    child.add_argument("--session", type=Path, required=True)
    child.add_argument("--minicard", type=Path, required=True)
    child.add_argument("--audit", type=Path, required=True)
    child.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--case", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--repeats", type=int, default=10)
    run.add_argument("--quotient", type=Path, required=True)
    run.add_argument("--session", type=Path, required=True)
    run.add_argument("--minicard", type=Path, required=True)
    run.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "child":
        run_child(args.case, args.case_sha256, args.arm, args.repeat,
                  args.quotient, args.session, args.minicard, args.audit, args.out)
    else:
        run_coordinator(args.case, args.out, args.repeats,
                        args.quotient, args.session, args.minicard, args.audit)


if __name__ == "__main__":
    main()
