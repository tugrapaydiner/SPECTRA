"""Retained, exposed one-formula-per-process CNF campaigns.

Run ``python -m experiments.general_competition.campaign --help``.  The runner
never certifies a hidden win or a complete modern field.  Every declared cell is
retained; only independently checked answers earn runtime instead of PAR-2.
Linux/POSIX RLIMIT_AS is a per-process address-space bound, not a cgroup aggregate
memory bound.  This development harness does not claim official isolation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import io
import json
import lzma
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from .cnf_admission import admit_cnf
from .official_score import score_bytes

ROOT = Path(__file__).resolve().parents[2]
MODULE = "experiments.general_competition.campaign"
SOLVED = {"SAT_VERIFIED": "sat-verified", "UNSAT_VERIFIED": "unsat-verified"}
TRACKS = {"standalone-native", "assisted-portfolio", "upstream-baseline"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


STORAGE_SCHEMA = "spectra.general_evidence_storage.v1"
COMPRESSION_MIN_BYTES = 64 * 1024


def _decoded_identity(path: Path) -> tuple[int, str]:
    value = hashlib.sha256()
    size = 0
    with lzma.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
            size += len(chunk)
    return size, value.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def compress_evidence_file(path: Path, root: Path, journal: Path) -> dict:
    """Publish verified XZ bytes and durable metadata before removing raw bytes.

    An interrupted operation leaves either the original, or a verified archive
    with its original identity in the append-only journal (sometimes both).
    No operation overwrites an existing archive or follows an evidence symlink.
    """
    root = root.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("compression source must be a regular non-symlink file")
    path = path.resolve(strict=True)
    logical = str(path.relative_to(root))
    journal.resolve().relative_to(root)
    if journal.is_symlink():
        raise ValueError("compression journal must not be a symlink")
    stored = path.with_name(path.name + ".xz")
    if stored.exists() or stored.is_symlink():
        raise FileExistsError(stored)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".compress-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    os.close(descriptor)
    expected_hash = hashlib.sha256()
    expected_bytes = 0
    try:
        with path.open("rb") as incoming, lzma.open(temporary, "wb", preset=1) as outgoing:
            for chunk in iter(lambda: incoming.read(1 << 20), b""):
                expected_hash.update(chunk)
                expected_bytes += len(chunk)
                outgoing.write(chunk)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if _decoded_identity(temporary) != (expected_bytes, expected_hash.hexdigest()):
            raise ValueError("compressed evidence failed its round-trip identity check")
        if path.stat().st_size != expected_bytes or digest(path) != expected_hash.hexdigest():
            raise ValueError("evidence source changed during compression")
        entry = {"logical_path": logical, "uncompressed_sha256": expected_hash.hexdigest(),
                 "uncompressed_bytes": expected_bytes, "stored_path": str(stored.relative_to(root)),
                 "stored_sha256": digest(temporary), "stored_bytes": temporary.stat().st_size,
                 "compression": "xz", "preset": 1, "roundtrip_verified": True}
        # Atomic create-if-absent: unlike replace(), an existing archive cannot
        # be overwritten by a race between the existence check and publication.
        os.link(temporary, stored)
        temporary.unlink()
        _fsync_directory(stored.parent)
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(journal.parent)
        path.unlink()
        _fsync_directory(path.parent)
        return entry
    finally:
        temporary.unlink(missing_ok=True)


def compress_evidence_scope(scope: Path, root: Path, *, recursive: bool = True) -> dict:
    """Compress large payloads only; JSON receipts and small logs stay readable."""
    journal = scope / "storage.jsonl"
    receipt_path = scope / "storage.json"
    if journal.exists() or receipt_path.exists():
        raise FileExistsError("storage receipts already exist")
    candidates = sorted(scope.rglob("*") if recursive else scope.iterdir())
    entries = []
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix.lower() in {".json", ".jsonl", ".xz", ".gz", ".zip", ".zst", ".bz2"}:
            continue
        if path.stat().st_size < COMPRESSION_MIN_BYTES:
            continue
        entries.append(compress_evidence_file(path, root, journal))
    receipt = {"schema": STORAGE_SCHEMA, "minimum_bytes": COMPRESSION_MIN_BYTES,
               "compression": "xz", "preset": 1, "timing_scope": "after checking; outside solve and check timers",
               "entries": entries}
    write_json(receipt_path, receipt)
    return receipt


def _storage_abort(root: Path, cells: list[dict], scope: Path, error: Exception) -> dict:
    """Keep completed cells but refuse a ranking after a storage failure."""
    summary = {"status": "ABORTED_STORAGE_FAILURE", "cells_retained": len(cells),
               "declared_field_complete": False, "claim": "No ranking accepted",
               "scope": str(scope.relative_to(root)), "error": f"{type(error).__name__}: {error}"}
    write_json(root / "summary.json", summary)
    files = {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*")) if path.is_file()}
    write_json(root / "evidence-sha256.json", files)
    return summary


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _sha(value: object, length: int, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{%d}" % length, value) is None:
        raise ValueError(f"{name} must be a lowercase {length}-digit hash")


def _load_manifest(path: Path, schema: str, key: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != schema:
        raise ValueError(f"expected {schema}")
    entries = data.get(key)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{key} must be a non-empty list")
    ids = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("manifest entries must be objects")
        ident = entry.get("id")
        if not isinstance(ident, str) or not ident or any(c in ident for c in "\n\r\0"):
            raise ValueError("manifest entry needs a non-empty single-line id")
        ids.append(ident)
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {key} id")
    return data


def load_inventory(path: Path) -> dict:
    data = _load_manifest(path, "spectra.general_inventory.v1", "instances")
    for item in data["instances"]:
        if not isinstance(item.get("path"), str) or not item["path"]:
            raise ValueError("instance path is required")
        _sha(item.get("sha256"), 64, "instance sha256")
        if not isinstance(item.get("family"), str) or not item["family"]:
            raise ValueError("every instance needs a declared family")
        if item.get("expected") not in ("SAT", "UNSAT", "UNKNOWN"):
            raise ValueError("expected must be SAT, UNSAT or UNKNOWN")
        if not isinstance(item.get("source_url"), str) or not item["source_url"]:
            raise ValueError("every instance needs a source_url")
    return data


def load_solvers(path: Path) -> dict:
    data = _load_manifest(path, "spectra.general_solver_field.v1", "solvers")
    for solver in data["solvers"]:
        if solver.get("track") not in TRACKS:
            raise ValueError("solver track must explicitly identify native, assisted or upstream")
        if "source_commit" not in solver and "source_archive_sha256" not in solver:
            raise ValueError("solver source_commit or source_archive_sha256 is required")
        if "source_commit" in solver:
            _sha(solver["source_commit"], 40, "solver source_commit")
        if "source_archive_sha256" in solver:
            _sha(solver["source_archive_sha256"], 64, "solver source_archive_sha256")
        if not isinstance(solver.get("proof_format", "DRAT"), str):
            raise ValueError("proof_format must be text")
        exit_codes = solver.get("accepted_exit_codes", {})
        if not isinstance(exit_codes, dict) or not set(exit_codes) <= {"SAT", "UNSAT", "UNKNOWN"}:
            raise ValueError("accepted_exit_codes must map SAT/UNSAT/UNKNOWN to exit codes")
        for codes in exit_codes.values():
            if not isinstance(codes, list) or not codes or any(type(code) is not int or not 0 <= code <= 255 for code in codes):
                raise ValueError("accepted_exit_codes must contain non-empty exit-code lists")
        custom_checker = solver.get("proof_checker")
        if custom_checker is not None:
            if not isinstance(custom_checker, dict):
                raise ValueError("proof_checker must be an object")
            checker_command = custom_checker.get("command")
            if (not isinstance(checker_command, list) or not checker_command
                    or any(not isinstance(arg, str) or not arg for arg in checker_command)
                    or not any("{cnf}" in arg for arg in checker_command)
                    or not any("{proof}" in arg for arg in checker_command)):
                raise ValueError("proof_checker needs an argv with {cnf} and {proof}")
            line = custom_checker.get("success_line")
            if not isinstance(line, str) or not line or "\n" in line or "\r" in line:
                raise ValueError("proof_checker needs an exact success_line")
            codes = custom_checker.get("accepted_exit_codes", [0])
            if not isinstance(codes, list) or not codes or any(type(x) is not int or x < 0 for x in codes):
                raise ValueError("proof_checker accepted_exit_codes must be nonnegative integers")
        expected_artifacts = solver.get("expected_artifact_sha256", {})
        if not isinstance(expected_artifacts, dict):
            raise ValueError("expected_artifact_sha256 must map paths to hashes")
        for artifact, expected_hash in expected_artifacts.items():
            if not isinstance(artifact, str) or not artifact:
                raise ValueError("expected artifact path must be non-empty")
            _sha(expected_hash, 64, "expected artifact SHA-256")
        artifacts = solver.get("artifact_paths", [])
        if not isinstance(artifacts, list) or any(not isinstance(x, str) or not x for x in artifacts):
            raise ValueError("artifact_paths must be a list of paths")
        command = solver.get("command")
        if (not isinstance(command, list) or not command
                or any(not isinstance(arg, str) or not arg for arg in command)):
            raise ValueError("solver command must be a non-empty argv array")
        if not any("{cnf}" in arg for arg in command):
            raise ValueError("solver command must use {cnf}")
        if not any("{proof}" in arg for arg in command):
            raise ValueError("solver command must retain a {proof} destination")
        upstream = solver.get("upstream_engines")
        if not isinstance(upstream, list) or any(not isinstance(x, str) or not x for x in upstream):
            raise ValueError("upstream_engines must be an explicit list")
        if solver["track"] == "standalone-native" and upstream:
            raise ValueError("standalone-native must not include upstream engines")
        if solver["track"] == "assisted-portfolio" and not upstream:
            raise ValueError("assisted-portfolio must disclose upstream engines")
    return data


def run_process(command: list[str], destination: Path, *, timeout: float,
                memory_mib: int, output_mib: int, workdir: Path | None = None) -> dict:
    """Run one process group, retaining streams even on timeout or spawn failure."""
    if os.name != "posix" or not hasattr(os, "wait4"):
        raise RuntimeError("campaign execution requires POSIX wait4 and resource limits")
    import resource
    _positive(timeout, "timeout")
    if type(memory_mib) is not int or memory_mib <= 0 or type(output_mib) is not int or output_mib <= 0:
        raise ValueError("memory_mib and output_mib must be positive integers")
    destination.mkdir(parents=True, exist_ok=False)
    workdir = ROOT if workdir is None else workdir.resolve()
    temporary = destination / "tmp"
    temporary.mkdir()
    environment = os.environ.copy()
    environment["TMPDIR"] = str(temporary.resolve())
    environment["TMP"] = str(temporary.resolve())
    environment["TEMP"] = str(temporary.resolve())
    environment["PYTHONPATH"] = str(ROOT) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    record = {"command": command, "working_directory": str(workdir), "returncode": None, "timed_out": False,
              "elapsed_seconds": 0.0, "user_seconds": None, "system_seconds": None,
              "peak_rss_kib": None, "spawn_error": None}

    def limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_mib << 20, memory_mib << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (output_mib << 20, output_mib << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    started = time.perf_counter()
    with (destination / "stdout.txt").open("xb") as stdout, (destination / "stderr.txt").open("xb") as stderr:
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout,
                                       stderr=stderr, cwd=workdir, env=environment,
                                       start_new_session=True, preexec_fn=limits)
        except (OSError, subprocess.SubprocessError) as error:
            record["spawn_error"] = f"{type(error).__name__}: {error}"
        else:
            try:
                while True:
                    pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                    if pid:
                        process.returncode = os.waitstatus_to_exitcode(status)
                        record.update(returncode=process.returncode,
                                      user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                                      peak_rss_kib=usage.ru_maxrss if sys.platform != "darwin" else usage.ru_maxrss / 1024)
                        break
                    if time.perf_counter() - started >= timeout:
                        record["timed_out"] = True
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        _, status, usage = os.wait4(process.pid, 0)
                        process.returncode = os.waitstatus_to_exitcode(status)
                        record.update(returncode=process.returncode,
                                      user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                                      peak_rss_kib=usage.ru_maxrss if sys.platform != "darwin" else usage.ru_maxrss / 1024)
                        break
                    time.sleep(min(0.01, timeout / 10))
            finally:
                # A child must not keep running after its parent exits or times out.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    record["elapsed_seconds"] = time.perf_counter() - started
    record["stdout_sha256"] = digest(destination / "stdout.txt")
    record["stderr_sha256"] = digest(destination / "stderr.txt")
    write_json(destination / "process.json", record)
    return record


def _worker_command(*args: str) -> list[str]:
    return [sys.executable, "-m", MODULE, *args]


def stage_input(source: Path, expected_hash: str, output: Path, maximum_bytes: int) -> None:
    """Retain byte-identical uncompressed DIMACS and a strict admission receipt."""
    if digest(source) != expected_hash:
        raise ValueError("source SHA-256 differs from the predeclared inventory")
    opener = lzma.open if source.suffix.lower() == ".xz" else gzip.open if source.suffix.lower() == ".gz" else open
    with opener(source, "rb") as incoming, output.open("xb") as outgoing:
        size = 0
        while chunk := incoming.read(1 << 20):
            size += len(chunk)
            if size > maximum_bytes:
                raise MemoryError("decompressed DIMACS exceeds the declared byte limit")
            outgoing.write(chunk)
    receipt = admit_cnf(output, max_source_bytes=maximum_bytes, max_dimacs_bytes=maximum_bytes)
    record = asdict(receipt)
    record.update(original_source=str(source), original_source_sha256=expected_hash,
                  original_source_bytes=source.stat().st_size)
    write_json(output.with_name("admission.json"), record)


def check_model(cnf: Path, stdout: Path, output: Path) -> None:
    """Check every original clause using a separate streaming assignment checker.

    Omitted variables receive false.  This only accepts the resulting complete
    assignment if every original clause is satisfied; it never solves for missing
    values. Both the verbatim witness and this explicit completion are retained.
    """
    receipt = admit_cnf(cnf)
    values = bytearray(receipt.declared_variables + 1)
    assigned = 0
    saw_model = False
    terminated = False
    with stdout.open(encoding="ascii", errors="strict") as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0] != "v":
                continue
            saw_model = True
            for token in fields[1:]:
                if terminated:
                    raise ValueError("model has data after its final zero")
                if re.fullmatch(r"-?[0-9]+", token) is None:
                    raise ValueError("malformed model literal")
                literal = int(token)
                if not literal:
                    terminated = True
                    continue
                variable = abs(literal)
                if variable > receipt.declared_variables:
                    raise ValueError("model variable exceeds DIMACS header")
                sign = 1 if literal > 0 else 2
                if values[variable] and values[variable] != sign:
                    raise ValueError("contradictory model assignments")
                if not values[variable]:
                    assigned += 1
                values[variable] = sign
    if not saw_model or not terminated:
        raise ValueError("SAT answer lacks a terminated DIMACS witness")
    satisfied = False
    clauses = 0
    with cnf.open(encoding="ascii") as stream:
        for line in stream:
            fields = line.split()
            if not fields or line.strip().startswith(("c", "p")):
                continue
            for token in fields:
                literal = int(token)
                if literal == 0:
                    clauses += 1
                    if not satisfied:
                        raise ValueError(f"model fails original clause {clauses}")
                    satisfied = False
                else:
                    value = values[abs(literal)] == 1
                    satisfied |= value if literal > 0 else not value
    if clauses != receipt.declared_clauses:
        raise ValueError("model checker clause count differs")
    with output.open("x", encoding="ascii") as stream:
        stream.write("s SATISFIABLE\n")
        for begin in range(1, receipt.declared_variables + 1, 100):
            stream.write("v " + " ".join(str(v if values[v] == 1 else -v)
                         for v in range(begin, min(begin + 100, receipt.declared_variables + 1))) + "\n")
        stream.write("v 0\n")
    print(json.dumps({"status": "SAT_VERIFIED", "checked_clauses": clauses,
                      "supplied_assignments": assigned,
                      "completed_false_variables": receipt.declared_variables - assigned}))


def native_worker(args: argparse.Namespace) -> int:
    from .competition_input import admit_and_load_cnf
    from .native_cdcl import NativeCDCLRuntime, write_drup
    started = time.perf_counter()
    admitted = admit_and_load_cnf(args.cnf)
    runtime = NativeCDCLRuntime(args.library)
    solver_started = time.perf_counter()
    status, model, proof, stats, _ = runtime.solve_residual(
        admitted.nvars, admitted.clauses, max_conflicts=args.max_conflicts,
        max_learned_literals=args.max_learned_literals)
    native_elapsed = time.perf_counter() - solver_started
    serialization_started = time.perf_counter()
    if status == 10:
        print("s SATISFIABLE")
        for begin in range(0, len(model), 100):
            print("v", *[i + 1 if model[i] else -(i + 1)
                         for i in range(begin, min(begin + 100, len(model)))])
        print("v 0")
    elif status == 20:
        write_drup(args.proof, proof)
        print("s UNSATISFIABLE")
    else:
        print("s UNKNOWN")
    write_json(args.stats, {"schema": "spectra.general_campaign_native_stats.v1",
                           "status_code": status, "stats": asdict(stats),
                           "max_conflicts": args.max_conflicts,
                           "max_learned_literals": args.max_learned_literals,
                           "native_solve_seconds": native_elapsed,
                           "output_serialization_seconds": time.perf_counter() - serialization_started,
                           "worker_seconds": time.perf_counter() - started,
                           "proof_generation_included_in_solve": True,
                           "proof_generation_separately_measurable": False,
                           "independent_checking_included": False})
    return status if status in (10, 20) else 0


def _lines_bounded(path: Path):
    """Read log lines with bounded buffers; oversized lines cannot be status lines."""
    with path.open(encoding="utf-8", errors="replace") as stream:
        while True:
            line = stream.readline(4096)
            if not line:
                return
            if not line.endswith("\n") and len(line) == 4096:
                while line and not line.endswith("\n"):
                    line = stream.readline(4096)
                yield "<oversized log line>"
            else:
                yield line


def _answer(stdout: Path) -> str:
    statuses = []
    for line in _lines_bounded(stdout):
        if line.startswith("s "):
            statuses.append(line.strip())
            if len(statuses) > 1:
                return "INVALID_OUTPUT"
    if not statuses:
        return "UNKNOWN"
    if statuses[0] not in ("s SATISFIABLE", "s UNSATISFIABLE", "s UNKNOWN"):
        return "INVALID_OUTPUT"
    return {"s SATISFIABLE": "SAT", "s UNSATISFIABLE": "UNSAT", "s UNKNOWN": "UNKNOWN"}[statuses[0]]


def _status(process: dict, answer: str, timeout: float, accepted_exit_codes: dict | None = None) -> str:
    codes = {"SAT": [10], "UNSAT": [20], "UNKNOWN": [0]}
    codes.update(accepted_exit_codes or {})
    if process["timed_out"] or process["elapsed_seconds"] > timeout:
        return "TIMEOUT"
    if process["spawn_error"] or process["returncode"] not in {code for values in codes.values() for code in values}:
        return "RESOURCE_FAILURE"
    if answer == "INVALID_OUTPUT":
        return answer
    if process["returncode"] not in codes.get(answer, []):
        return "INVALID_OUTPUT"
    return answer


def _score_csv(cells: list[dict], timeout: float) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("solverid", "instanceid", "runtime", "status", "score", "vresult"))
    for cell in cells:
        runtime = Decimal(str(cell["elapsed_seconds"]))
        score = runtime if cell["status"] in SOLVED else Decimal(str(timeout)) * 2
        writer.writerow((cell["solver_id"], cell["instance_id"], runtime,
                         SOLVED.get(cell["status"], cell["status"].lower()), score,
                         "" if cell["expected"] == "UNKNOWN" else cell["expected"].lower()))
    return stream.getvalue().encode("utf-8")


def score_campaign(cells: list[dict], inventory: dict, field: dict, timeout: float) -> dict:
    expected = {(solver["id"], instance["id"]) for solver in field["solvers"] for instance in inventory["instances"]}
    actual = [(cell["solver_id"], cell["instance_id"]) for cell in cells]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("campaign is missing declared cells, has duplicate cells, or has extra cells")
    declared = {instance["id"]: instance for instance in inventory["instances"]}
    declared_solvers = {solver["id"]: solver for solver in field["solvers"]}
    for entry in cells:
        source = declared[entry["instance_id"]]
        declared_track = declared_solvers[entry["solver_id"]].get("track")
        if declared_track is not None and entry.get("track") != declared_track:
            raise ValueError("cell track differs from declared solver entry")
        if entry["family"] != source["family"] or ("expected" in source and entry["expected"] != source["expected"]):
            raise ValueError("cell metadata differs from declared inventory")
        elapsed = entry["elapsed_seconds"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("cell runtime must be finite and nonnegative")
        if entry["status"] in SOLVED and elapsed > timeout:
            raise ValueError("solved cell exceeds the declared timeout")
    solvers = [solver["id"] for solver in field["solvers"]]
    result = score_bytes(_score_csv(cells, timeout), timeout_seconds=Decimal(str(timeout)), solver_ids=solvers).to_record()
    families = {}
    for family in sorted({instance["family"] for instance in inventory["instances"]}):
        subset = [cell for cell in cells if cell["family"] == family]
        families[family] = score_bytes(_score_csv(subset, timeout), timeout_seconds=Decimal(str(timeout)), solver_ids=solvers).to_record()
    # A wrong result disqualifies the solver across the campaign, including
    # family and subtrack ranks where the particular wrong cell is absent.
    disqualified = {row["solver_id"] for row in result["standings"] if row["disqualified"]}
    for table in [result, *families.values()]:
        for row in table["standings"]:
            row["disqualified"] = row["solver_id"] in disqualified
        table["standings"].sort(key=lambda row: (row["disqualified"], Decimal(row["overall_par2"]), -row["solved"], row["solver_id"]))
        for rank, row in enumerate(table["standings"], 1):
            row["rank"] = rank
        for subtrack in ("sat", "unsat"):
            ordered = sorted((row for row in table["standings"] if row[subtrack + "_par2"] is not None),
                             key=lambda row: (row["disqualified"], Decimal(row[subtrack + "_par2"]), -row[subtrack + "_solved"], row["solver_id"]))
            ranks = {row["solver_id"]: rank for rank, row in enumerate(ordered, 1)}
            for row in table["standings"]:
                row[subtrack + "_rank"] = ranks.get(row["solver_id"])
    result["families"] = families
    result["resource_summary"] = {
        solver: {"statuses": dict(Counter(c["status"] for c in cells if c["solver_id"] == solver)),
                 "peak_rss_kib": max((c.get("peak_rss_kib") or 0 for c in cells if c["solver_id"] == solver), default=0),
                 "checking_seconds": sum(c.get("checking_seconds", 0) for c in cells if c["solver_id"] == solver)}
        for solver in solvers}
    return result


def run_campaign(args: argparse.Namespace) -> dict:
    _positive(args.timeout, "timeout")
    _positive(args.checker_timeout, "checker_timeout")
    compression_enabled = getattr(args, "compress_evidence", False)
    if type(compression_enabled) is not bool:
        raise ValueError("compress_evidence must be Boolean")
    for name in ("memory_mib", "output_mib", "dimacs_mib"):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    inventory_path, field_path = args.inventory.resolve(), args.solvers.resolve()
    inventory, field = load_inventory(inventory_path), load_solvers(field_path)
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(inventory_path, destination / "inventory.json")
    shutil.copyfile(field_path, destination / "solvers.json")
    checker = args.checker.resolve()
    configuration = {"schema": "spectra.general_campaign.v1", "evidence_class": "EXPOSED_DEVELOPMENT",
                     "started_at_utc": datetime.now(timezone.utc).isoformat(),
                     "inventory_sha256": digest(inventory_path), "solver_field_sha256": digest(field_path),
                     "timeout_seconds": args.timeout, "checker_timeout_seconds": args.checker_timeout,
                     "memory_mib": args.memory_mib, "maximum_output_file_mib": args.output_mib,
                     "maximum_dimacs_mib": args.dimacs_mib,
                     "memory_limit_scope": "RLIMIT_AS per process; not aggregate cgroup memory",
                     "execution": "sequential fresh process groups with private solver working/temp directories; shared host/filesystem, not a security sandbox",
                     "timing_boundary": "process spawn through exit, includes parse and proof output; independent checking separately bounded",
                     "input_preparation": "common byte-preserving decompression and admission excluded for every solver",
                     "evidence_compression": {"enabled": compression_enabled, "schema": STORAGE_SCHEMA,
                                              "compression": "xz", "preset": 1,
                                              "minimum_bytes": COMPRESSION_MIN_BYTES,
                                              "outside_solve_and_check_timers": True},
                     "checker": str(checker), "checker_sha256": digest(checker) if checker.is_file() else None,
                     "runner_sha256": digest(Path(__file__)), "python": sys.version,
                     "platform": platform.platform(), "machine": platform.machine(),
                     "cpu_count": os.cpu_count(), "solver_metadata": field["solvers"],
                     "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                     "exclusive_host": False,
                     "complete_roadmap_field_verified": False, "near_duplicate_control": False,
                     "source_to_binary_rebuild_link_verified": False,
                     "source_pin_verification": "Source labels are declarations; artifact hashes bind observed files. Rebuild linkage requires external build evidence.",
                     "claim": "Exposed declared-field development measurement only; no leadership or hidden-set claim."}
    try:
        configuration["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        configuration["working_tree_status"] = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except (OSError, subprocess.CalledProcessError):
        configuration["source_commit"] = None
    cpuinfo = Path("/proc/cpuinfo")
    meminfo = Path("/proc/meminfo")
    configuration["cpu_model"] = next((line.partition(":")[2].strip() for line in cpuinfo.read_text().splitlines() if line.startswith("model name")), None) if cpuinfo.is_file() else None
    configuration["host_memory"] = next((line for line in meminfo.read_text().splitlines() if line.startswith("MemTotal:")), None) if meminfo.is_file() else None
    artifacts = {}
    artifact_identity_failures = {}
    for solver in field["solvers"]:
        paths = list(solver.get("artifact_paths", [])) + list(solver.get("expected_artifact_sha256", {}))
        # Commands execute from ROOT, independent of the manifest location.
        executable_arg = solver["command"][0]
        executable = str((ROOT / executable_arg).resolve()) if "/" in executable_arg and not Path(executable_arg).is_absolute() else shutil.which(executable_arg)
        if executable:
            paths.append(executable)
        custom = solver.get("proof_checker")
        if custom:
            checker_arg = custom["command"][0]
            executable = str((ROOT / checker_arg).resolve()) if "/" in checker_arg and not Path(checker_arg).is_absolute() else shutil.which(checker_arg)
            if executable:
                paths.append(executable)
        artifacts[solver["id"]] = {}
        for filename in paths:
            artifact = Path(filename)
            if not artifact.is_absolute():
                artifact = field_path.parent / artifact
            artifacts[solver["id"]][str(artifact.resolve())] = digest(artifact) if artifact.is_file() else None
        failures = []
        for filename, expected_hash in solver.get("expected_artifact_sha256", {}).items():
            artifact = Path(filename)
            if not artifact.is_absolute():
                artifact = field_path.parent / artifact
            if artifacts[solver["id"]].get(str(artifact.resolve())) != expected_hash:
                failures.append(str(artifact.resolve()))
        if failures:
            artifact_identity_failures[solver["id"]] = failures
    configuration["artifact_sha256"] = artifacts
    configuration["artifact_identity_failures"] = artifact_identity_failures
    write_json(destination / "campaign.json", configuration)
    limits = {"memory_mib": args.memory_mib, "output_mib": args.output_mib}
    cells = []
    admissions = []
    seen_semantics = {}
    duplicates = []
    original_input_mutations = []
    for index, instance in enumerate(inventory["instances"]):
        instance_dir = destination / "instances" / f"{index:05d}"
        instance_dir.mkdir(parents=True)
        source = Path(instance["path"])
        if not source.is_absolute():
            source = inventory_path.parent / source
        cnf = instance_dir / "input.cnf"
        admission = run_process(_worker_command("stage", "--source", str(source.resolve()),
                                "--sha256", instance["sha256"], "--out", str(cnf),
                                "--max-bytes", str(args.dimacs_mib << 20)),
                                instance_dir / "admission-process", timeout=args.checker_timeout, **limits)
        admitted = admission["returncode"] == 0 and not admission["timed_out"] and admission["elapsed_seconds"] <= args.checker_timeout and (instance_dir / "admission.json").is_file()
        admissions.append({"instance_id": instance["id"], "admitted": admitted, "process": admission})
        if admitted:
            receipt = json.loads((instance_dir / "admission.json").read_text())
            semantic = receipt["semantic_sha256"]
            if semantic in seen_semantics:
                duplicates.append([seen_semantics[semantic], instance["id"]])
            else:
                seen_semantics[semantic] = instance["id"]
        # Rotate the entire declared field; metadata is never passed to a solver.
        ordered = field["solvers"][index % len(field["solvers"]):] + field["solvers"][:index % len(field["solvers"])]
        for solver in ordered:
            solver_index = next(i for i, entry in enumerate(field["solvers"]) if entry["id"] == solver["id"])
            cell_dir = instance_dir / f"solver-{solver_index:03d}"
            cell_dir.mkdir()
            if admitted and (not cnf.is_file() or digest(cnf) != receipt["dimacs_sha256"] or
                             not source.is_file() or digest(source) != instance["sha256"]):
                admitted = False
            cell = {"instance_id": instance["id"], "solver_id": solver["id"],
                    "track": solver["track"], "family": instance["family"], "expected": instance["expected"],
                    "status": "ADMISSION_FAILURE", "elapsed_seconds": 0.0, "peak_rss_kib": None,
                    "checking_seconds": 0.0, "proof_bytes": 0, "directory": str(cell_dir.relative_to(destination))}
            if solver["id"] in artifact_identity_failures:
                cell["status"] = "SOURCE_IDENTITY_FAILURE"
                cell["identity_failures"] = artifact_identity_failures[solver["id"]]
            elif admitted:
                # The solver only sees a private copy. Independent checking uses
                # the separately retained original, with hashes on both sides.
                solver_cnf = cell_dir / "input.cnf"
                shutil.copyfile(cnf, solver_cnf)
                proof, stats = cell_dir / "proof.drat", cell_dir / "stats.json"
                replacements = {"{cnf}": str(solver_cnf), "{proof}": str(proof), "{stats}": str(stats)}
                command = []
                for arg in solver["command"]:
                    for token, value in replacements.items():
                        arg = arg.replace(token, value)
                    command.append(arg)
                if "/" in command[0] and not Path(command[0]).is_absolute():
                    command[0] = str((ROOT / command[0]).resolve())
                process = run_process(command, cell_dir / "solve", timeout=args.timeout, workdir=cell_dir, **limits)
                answer = _answer(cell_dir / "solve" / "stdout.txt")
                status = _status(process, answer, args.timeout, solver.get("accepted_exit_codes"))
                expected_cnf_hash = receipt["dimacs_sha256"]
                integrity = {"solver_input_sha256": digest(solver_cnf) if solver_cnf.is_file() else None,
                             "original_cnf_sha256": digest(cnf) if cnf.is_file() else None,
                             "source_sha256": digest(source) if source.is_file() else None}
                input_unchanged = (integrity["solver_input_sha256"] == expected_cnf_hash and
                                   integrity["original_cnf_sha256"] == expected_cnf_hash and
                                   integrity["source_sha256"] == instance["sha256"])
                cell["input_integrity_after_solve"] = integrity
                if not input_unchanged:
                    status = "INVALID_INPUT_MUTATION"
                cell.update(status=status, elapsed_seconds=process["elapsed_seconds"], peak_rss_kib=process["peak_rss_kib"],
                            returncode=process["returncode"], reported_answer=answer)
                custom_checker = solver.get("proof_checker")
                unsupported_proof = solver.get("proof_format", "DRAT").upper() not in ("DRAT", "DRUP") and custom_checker is None
                if status == "UNSAT" and unsupported_proof:
                    cell["status"] = "UNAVAILABLE_CHECKER"
                    cell["proof_format"] = solver["proof_format"]
                elif status in ("SAT", "UNSAT"):
                    success_line = "s VERIFIED"
                    accepted_codes = [0]
                    if status == "SAT":
                        check_command = _worker_command("check-model", "--cnf", str(cnf),
                                        "--stdout", str(cell_dir / "solve" / "stdout.txt"),
                                        "--out", str(cell_dir / "complete.model"))
                    elif custom_checker:
                        check_command = []
                        checker_replacements = {**replacements, "{cnf}": str(cnf)}
                        for arg in custom_checker["command"]:
                            for token, value in checker_replacements.items():
                                arg = arg.replace(token, value)
                            check_command.append(arg)
                        success_line = custom_checker["success_line"]
                        accepted_codes = custom_checker.get("accepted_exit_codes", [0])
                    else:
                        check_command = [str(checker), str(cnf), str(proof)]
                    checked = run_process(check_command, cell_dir / "check", timeout=args.checker_timeout,
                                          workdir=cell_dir / "check", **limits)
                    cell["checking_seconds"] = checked["elapsed_seconds"]
                    if checked["timed_out"] or checked["elapsed_seconds"] > args.checker_timeout:
                        cell["status"] = "CHECKER_TIMEOUT"
                    elif checked["spawn_error"] or checked["returncode"] is None or checked["returncode"] < 0:
                        cell["status"] = "CHECKER_RESOURCE_FAILURE"
                    elif status == "SAT":
                        cell["status"] = ("SAT_VERIFIED" if checked["returncode"] == 0 else
                                          "INVALID_MODEL" if checked["returncode"] == 65 else
                                          "CHECKER_RESOURCE_FAILURE")
                    else:
                        accepted = any(line.strip() == success_line for line in _lines_bounded(cell_dir / "check" / "stdout.txt"))
                        cell["status"] = "UNSAT_VERIFIED" if checked["returncode"] in accepted_codes and accepted and proof.is_file() else "INVALID_PROOF"
                    if cell["status"] in SOLVED and instance["expected"] not in ("UNKNOWN", answer):
                        cell["status"] = "INVALID_VERDICT_CONFLICT"
                final_integrity = {"solver_input_sha256": digest(solver_cnf) if solver_cnf.is_file() else None,
                                   "original_cnf_sha256": digest(cnf) if cnf.is_file() else None,
                                   "source_sha256": digest(source) if source.is_file() else None}
                cell["input_integrity_after_check"] = final_integrity
                if (integrity["original_cnf_sha256"] != expected_cnf_hash or integrity["source_sha256"] != instance["sha256"] or
                        final_integrity["original_cnf_sha256"] != expected_cnf_hash or final_integrity["source_sha256"] != instance["sha256"]):
                    original_input_mutations.append({"instance_id": instance["id"], "solver_id": solver["id"],
                                                     "after_solve": integrity, "after_check": final_integrity})
                if (final_integrity["solver_input_sha256"] != expected_cnf_hash or
                        final_integrity["original_cnf_sha256"] != expected_cnf_hash or
                        final_integrity["source_sha256"] != instance["sha256"]):
                    cell["status"] = "INVALID_INPUT_MUTATION"
                elif input_unchanged:
                    # Identical bytes are already retained once as input.cnf.
                    # Keep altered copies, but avoid N-fold duplicate artifacts.
                    solver_cnf.unlink()
                    cell["identical_input_retained_at"] = str(cnf.relative_to(destination))
                if proof.is_file():
                    cell.update(proof_bytes=proof.stat().st_size, proof_sha256=digest(proof))
                if stats.is_file():
                    cell["stats_sha256"] = digest(stats)
            write_json(cell_dir / "cell.json", cell)
            with (destination / "cells.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(cell, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            cells.append(cell)
            if compression_enabled:
                try:
                    compress_evidence_scope(cell_dir, destination)
                except (OSError, ValueError, lzma.LZMAError) as error:
                    return _storage_abort(destination, cells, cell_dir, error)
            print(json.dumps({"completed": len(cells), "total": len(inventory["instances"]) * len(field["solvers"]),
                              "instance": instance["id"], "solver": solver["id"], "status": cell["status"]}), flush=True)
        if compression_enabled:
            try:
                # No later cell needs this formula after its entire field ran.
                compress_evidence_scope(instance_dir, destination, recursive=False)
            except (OSError, ValueError, lzma.LZMAError) as error:
                return _storage_abort(destination, cells, instance_dir, error)
    write_json(destination / "admissions.json", {"admissions": admissions, "exact_semantic_duplicate_pairs": duplicates})
    csv_bytes = _score_csv(cells, args.timeout)
    (destination / "scores.csv").write_bytes(csv_bytes)
    changed_artifacts = []
    for solver_id, observed in artifacts.items():
        for filename, expected_hash in observed.items():
            artifact = Path(filename)
            actual_hash = digest(artifact) if artifact.is_file() else None
            if actual_hash != expected_hash:
                changed_artifacts.append({"solver_id": solver_id, "path": filename,
                                          "before": expected_hash, "after": actual_hash})
    for filename, expected_hash in ((str(checker), configuration["checker_sha256"]),
                                    (str(Path(__file__)), configuration["runner_sha256"]),
                                    (str(inventory_path), configuration["inventory_sha256"]),
                                    (str(field_path), configuration["solver_field_sha256"])):
        artifact = Path(filename)
        actual_hash = digest(artifact) if artifact.is_file() else None
        if actual_hash != expected_hash:
            changed_artifacts.append({"path": filename, "before": expected_hash, "after": actual_hash})
    write_json(destination / "artifact-integrity.json", {"changed_artifacts": changed_artifacts,
                                                       "initial_identity_failures": artifact_identity_failures,
                                                       "original_input_mutations": original_input_mutations})
    try:
        if changed_artifacts:
            raise ValueError("campaign source, executable, checker or manifest changed during execution")
        if original_input_mutations:
            raise ValueError("original admitted CNF or source changed during execution")
        summary = score_campaign(cells, inventory, field, args.timeout)
        summary.update(evidence_class="EXPOSED_DEVELOPMENT", declared_field_complete=True,
                       complete_roadmap_field_verified=False, exact_semantic_duplicate_pairs=duplicates,
                       claim=configuration["claim"])
    except ValueError as error:
        summary = {"status": "INVALID_CAMPAIGN", "error": str(error), "cells_retained": len(cells), "claim": "No ranking accepted"}
    write_json(destination / "summary.json", summary)
    files = {str(path.relative_to(destination)): digest(path) for path in sorted(destination.rglob("*")) if path.is_file()}
    write_json(destination / "evidence-sha256.json", files)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="run every cell of an exposed declared field")
    for flag in ("inventory", "solvers", "out", "checker"):
        run.add_argument("--" + flag, type=Path, required=True)
    run.add_argument("--timeout", type=float, default=60)
    run.add_argument("--checker-timeout", type=float, default=120)
    run.add_argument("--memory-mib", type=int, default=4096)
    run.add_argument("--output-mib", type=int, default=1024)
    run.add_argument("--dimacs-mib", type=int, default=2048)
    run.add_argument("--compress-evidence", action="store_true",
                     help="losslessly XZ-compress large checked artifacts outside all timing endpoints")
    native = sub.add_parser("native", help="standalone native worker; checking is external")
    for flag in ("library", "cnf", "proof", "stats"):
        native.add_argument("--" + flag, type=Path, required=True)
    native.add_argument("--max-conflicts", type=int, default=(1 << 64) - 1)
    native.add_argument("--max-learned-literals", type=int, default=100_000_000)
    stage = sub.add_parser("stage", help=argparse.SUPPRESS)
    for flag in ("source", "out"):
        stage.add_argument("--" + flag, type=Path, required=True)
    stage.add_argument("--sha256", required=True)
    stage.add_argument("--max-bytes", type=int, required=True)
    check = sub.add_parser("check-model", help=argparse.SUPPRESS)
    for flag in ("cnf", "stdout", "out"):
        check.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "run":
        result = run_campaign(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1 if result.get("status") in ("INVALID_CAMPAIGN", "ABORTED_STORAGE_FAILURE") else 0
    if args.action == "native":
        return native_worker(args)
    if args.action == "stage":
        stage_input(args.source, args.sha256, args.out, args.max_bytes)
    elif args.action == "check-model":
        try:
            check_model(args.cnf, args.stdout, args.out)
        except (ValueError, UnicodeError) as error:
            print(f"INVALID_MODEL: {error}", file=sys.stderr)
            return 65
        except (MemoryError, OSError) as error:
            print(f"CHECKER_RESOURCE_FAILURE: {error}", file=sys.stderr)
            return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
