#!/usr/bin/env python3
"""Measure or verify a complete CNF workload-admission pilot, never a model win.

The run command refuses existing destinations and retains partial rows on failure.
The verify command needs only the standard library, not PySAT or neural inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.cnf import CNF
from eval.sat_workload import (execution_schedule, generate_cases, independent_check,
                               model_to_witness, run_case, summarize, validate_protocol)

SOURCE_FILES = ("data/cnf.py", "eval/cnf_repair.py", "eval/sat_workload.py",
                "scripts/sat_workload_gate.py", "requirements-sat-research.txt")
EVIDENCE_FILES = ("protocol.json", "cases.json", "rows.jsonl", "summary.json",
                  "environment.json", "source.json")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(path.read_text(), object_pairs_hook=object_pairs,
                      parse_constant=invalid_constant)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+"\n")


def current_sources() -> dict[str, str]:
    return {name: sha((ROOT/name).read_bytes()) for name in SOURCE_FILES}


def smoke(protocol: dict) -> dict:
    """Actual native solver API/semantics check; not part of the research sample."""
    validate_protocol(protocol)
    version = importlib.metadata.version("python-sat")
    if version != protocol["python_sat_version"]:
        raise ValueError("PySAT version differs from the frozen protocol")
    from pysat.solvers import Solver
    rows = []
    for name in protocol["solvers"]:
        for label, formula, expected in (
            ("sat", CNF(3, ((1,), (-1, 2), (-2, -3))), True),
            ("unsat", CNF(1, ((1,), (-1,))), False),
        ):
            with Solver(name=name, bootstrap_with=[list(c) for c in formula.clauses]) as solver:
                solver.conf_budget(protocol["conflict_budget"])
                status = solver.solve_limited()
                if status is not expected:
                    raise AssertionError(f"{name}: wrong result on known {label} formula")
                if expected and not independent_check(formula, model_to_witness(formula, solver.get_model())):
                    raise AssertionError("invalid smoke-test witness")
                stats = solver.accum_stats()
            rows.append({"solver": name, "fixture": label, "expected_satisfiable": expected,
                         "solver_stats": stats})
    return {"python_sat_version": version, "fixtures": rows,
            "scope": "actual solver API smoke fixtures, excluded from experiment outcomes"}


def run(protocol_path: Path, out: Path) -> dict:
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    if out.exists():
        raise FileExistsError("refusing to overwrite existing evidence")
    out.mkdir(parents=True)
    try:
        write_json(out/"protocol.json", protocol)
        source = {"files": current_sources(), "protocol_sha256": sha((out/"protocol.json").read_bytes())}
        try:
            source["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            source["git_tree"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
            source["working_tree_status"] = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            source["git_commit"] = source["git_tree"] = None
            source["working_tree_status"] = "unavailable; exact file hashes still recorded"
        write_json(out/"source.json", source)
        env = {"python": platform.python_version(), "platform": platform.platform(),
               "cpu_count": os.cpu_count(), "cpu_model": next((line.split(":", 1)[1].strip()
                   for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name")), None) if Path("/proc/cpuinfo").is_file() else platform.processor(),
               "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
               "python_sat_version": importlib.metadata.version("python-sat"),
               "physical_energy_joules": None, "process_start_method": "spawn",
               "solve_scope": protocol["cost_scope"]}
        if env["python_sat_version"] != protocol["python_sat_version"]:
            raise ValueError("installed PySAT version differs from protocol")
        write_json(out/"environment.json", env)
        start = time.perf_counter_ns()
        cases = generate_cases(protocol)
        env["generation_ns"] = time.perf_counter_ns()-start
        write_json(out/"environment.json", env)
        write_json(out/"cases.json", cases)
        rows = []
        with (out/"rows.jsonl").open("x") as stream:
            for case, solver, round_id in execution_schedule(protocol, cases):
                row = run_case(case, solver, round_id, protocol)
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                stream.flush()
                rows.append(row)
                print(json.dumps({"row": len(rows), "case": case["case_id"], "solver": solver,
                                  "round": round_id, "status": row["status"]}), flush=True)
        if current_sources() != source["files"] or sha((out/"protocol.json").read_bytes()) != source["protocol_sha256"]:
            raise ValueError("source/protocol changed during execution")
        report = summarize(protocol, cases, rows)
        write_json(out/"summary.json", report)
        hashes = {name: sha((out/name).read_bytes()) for name in EVIDENCE_FILES}
        write_json(out/"sha256.json", hashes)
        return report
    except Exception:
        (out/"failure.txt").write_text(traceback.format_exc())
        raise


def verify(out: Path) -> dict:
    """Read a directory or a bounded archive containing exactly seven files."""
    if out.is_file():
        names = set(EVIDENCE_FILES) | {"sha256.json"}
        with zipfile.ZipFile(out) as archive:
            members = archive.infolist()
            if (len(members) != len(names) or {m.filename for m in members} != names
                    or any(m.is_dir() or m.file_size > 20_000_000 for m in members)
                    or sum(m.file_size for m in members) > 50_000_000):
                raise ValueError("unexpected or oversized archived evidence inventory")
            with tempfile.TemporaryDirectory(prefix="spectra-sat-verify-") as temp:
                target = Path(temp)
                # Fixed allowlisted basenames; no archive paths are extracted.
                for name in sorted(names):
                    (target/name).write_bytes(archive.read(name))
                return verify(target)
    expected_names = set(EVIDENCE_FILES) | {"sha256.json"}
    if {p.name for p in out.iterdir()} != expected_names or any(not p.is_file() or p.is_symlink() for p in out.iterdir()):
        raise ValueError("incomplete, unexpected or nonregular evidence inventory")
    hashes = read_json(out/"sha256.json")
    if type(hashes) is not dict or set(hashes) != set(EVIDENCE_FILES):
        raise ValueError("incomplete evidence hash inventory")
    for name in EVIDENCE_FILES:
        if hashes[name] != sha((out/name).read_bytes()):
            raise ValueError(f"evidence digest mismatch: {name}")
    source = read_json(out/"source.json")
    if source["files"] != current_sources() or source["protocol_sha256"] != hashes["protocol.json"]:
        raise ValueError("runtime source or protocol identity mismatch")
    protocol = read_json(out/"protocol.json")
    environment = read_json(out/"environment.json")
    if environment.get("python_sat_version") != protocol["python_sat_version"] or environment.get("physical_energy_joules") is not None:
        raise ValueError("incompatible recorded environment or unsupported energy claim")
    # Parse JSONL with the same strict duplicate-key/nonfinite rules as JSON.
    raw = (out/"rows.jsonl").read_text()
    if any(not line.strip() for line in raw.splitlines()):
        raise ValueError("blank row in the declared observation inventory")
    rows = [json.loads(line) for line in raw.splitlines()]
    # Canonical round-trip rejects hidden duplicate keys, nonfinite values and
    # noncanonical record serialization; the runner emits this exact format.
    if raw != "".join(json.dumps(r, sort_keys=True, allow_nan=False)+"\n" for r in rows):
        raise ValueError("noncanonical or ambiguous observation JSONL")
    report = summarize(protocol, read_json(out/"cases.json"), rows)
    if report != read_json(out/"summary.json"):
        raise ValueError("stored summary does not exactly reproduce")
    return {"integrity": "PASS", "source_commit": source.get("git_commit"),
            "observations": report["observations"], "instances": report["instances"],
            "sat_witnesses_independently_checked": report["sat_witnesses_independently_checked"],
            "tier_gates": [tier["gate"] for tier in report["tiers"]],
            "scope": "source-bound stored evidence replay, not a new timing or learned result"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify", "smoke"])
    parser.add_argument("--protocol", type=Path, default=ROOT/"config/sat_workload_gate_v1.json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "smoke":
        report = smoke(read_json(args.protocol))
    elif args.out is None:
        parser.error("run/verify requires --out")
    elif args.command == "run":
        report = run(args.protocol, args.out)
    else:
        report = verify(args.out)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
