"""Read-only verification of retained exposed campaign archives.

The default command executes no archived solver or checker commands. It checks
the exact file inventory, retained input and process records, and recomputes the
declared rectangle and scores. An externally supplied manifest hash can anchor
the archive; an unpinned archive only establishes internal consistency.

Optional fresh answer checks run trusted local code on temporary copies outside
the archive. Proof checker argv, exit codes and success lines must be explicitly
supplied by the caller, never adopted from an archived solver manifest. This is
neither a new timed campaign nor independent replication or hidden-set evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal
import csv
import gzip
import hashlib
import io
import json
import lzma
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Mapping

from .campaign import checker_resource_diagnostic, digest, load_inventory, load_solvers, run_process, score_campaign
from .cnf_admission import admit_cnf

MANIFEST = "evidence-sha256.json"
CLAIM = "Exposed declared-field development measurement only; no leadership or hidden-set claim."
MAX_METADATA_BYTES = 32 << 20
MAX_CELLS = 100_000
SOLVED = {"SAT_VERIFIED": "sat-verified", "UNSAT_VERIFIED": "unsat-verified"}
FINAL_STATUSES = {*SOLVED, "ADMISSION_FAILURE", "SOURCE_IDENTITY_FAILURE", "TIMEOUT", "RESOURCE_FAILURE",
                  "INVALID_OUTPUT", "UNKNOWN", "INVALID_INPUT_MUTATION", "UNAVAILABLE_CHECKER", "CHECKER_TIMEOUT",
                  "CHECKER_RESOURCE_FAILURE", "INVALID_MODEL", "INVALID_PROOF", "INVALID_VERDICT_CONFLICT"}


class VerificationError(ValueError):
    """The archive cannot be accepted as a consistent exposed measurement."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _same(actual: object, expected: object, label: str) -> None:
    # Ordinary equality silently equates True, 1 and 1.0 in JSON metadata.
    _require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
             json.dumps(expected, sort_keys=True, allow_nan=False), f"{label} differs")


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _number(token: str) -> float:
    value = float(token)
    _require(math.isfinite(value), "non-finite JSON number")
    return value


def _invalid_constant(token: str):
    raise VerificationError(f"non-finite JSON constant: {token}")


def _parse_json(text: str, label: str) -> dict:
    try:
        value = json.loads(text, object_pairs_hook=_object, parse_float=_number,
                           parse_constant=_invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise VerificationError(f"invalid JSON in {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_json(path: Path) -> dict:
    _require(path.stat().st_size <= MAX_METADATA_BYTES, f"metadata exceeds byte limit: {path.name}")
    return _parse_json(path.read_text(encoding="utf-8"), path.name)


def _hash(value: object, label: str, *, nullable: bool = False) -> None:
    _require((nullable and value is None) or
             (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None),
             f"{label} must be a lowercase SHA-256")


def _finite(value: object, label: str, *, positive: bool = False) -> None:
    try:
        accepted = type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        accepted = False
    _require(accepted, f"{label} must be finite and {'positive' if positive else 'nonnegative'}")


def _relative(name: object) -> str:
    _require(isinstance(name, str) and bool(name), "empty or non-string evidence path")
    _require(not any(ord(c) < 32 for c in name) and "\\" not in name and ":" not in name,
             f"unsafe evidence path: {name!r}")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and all(part not in (".", "..") for part in name.split("/"))
             and path.as_posix() == name, f"non-canonical evidence path: {name!r}")
    return name


def _root(path: Path) -> Path:
    result = Path(os.path.abspath(path))
    for component in (*reversed(result.parents), result):
        _require(not component.is_symlink(), f"symlink in evidence root: {component}")
    _require(result.is_dir(), "evidence root must be a directory")
    return result


def _files(root: Path) -> tuple[set[str], set[str]]:
    files, directories = set(), set()
    def failed(error: OSError) -> None:
        raise VerificationError(f"cannot enumerate evidence: {error}") from error
    for directory, subdirs, filenames in os.walk(root, followlinks=False, onerror=failed):
        for name in [*subdirs, *filenames]:
            path = Path(directory) / name
            relative = _relative(path.relative_to(root).as_posix())
            mode = path.lstat().st_mode
            _require(not stat.S_ISLNK(mode), f"symlink in evidence: {relative}")
            _require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), f"special file in evidence: {relative}")
            (directories if stat.S_ISDIR(mode) else files).add(relative)
    return files, directories


def _verify_inventory(root: Path, expected_manifest_sha256: str | None) -> tuple[dict, set[str]]:
    files, directories = _files(root)
    _require(MANIFEST in files, "missing evidence hash manifest")
    if expected_manifest_sha256 is not None:
        _hash(expected_manifest_sha256, "expected manifest hash")
        _require(digest(root / MANIFEST) == expected_manifest_sha256, "external evidence manifest hash differs")
    hashes = _read_json(root / MANIFEST)
    for name, expected in hashes.items():
        _relative(name)
        _hash(expected, f"hash of {name}")
        _require(name != MANIFEST, "hash manifest must not list itself")
    _require(files == set(hashes) | {MANIFEST},
             f"file inventory differs: missing={sorted(set(hashes) - files)[:8]}, extra={sorted(files - set(hashes) - {MANIFEST})[:8]}")
    for name, expected in hashes.items():
        _require(digest(root / name) == expected, f"file hash differs: {name}")
    return hashes, directories


def _decoded_chunks(path: Path, expected_bytes: int):
    """Bound both decoded size and XZ dictionary memory for untrusted archives."""
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64 << 20)
    size = 0
    with path.open("rb") as stream:
        while not decoder.eof:
            data = stream.read(65536) if decoder.needs_input else b""
            _require(data or not decoder.needs_input, "truncated compressed evidence")
            chunk = decoder.decompress(data, max_length=min(1 << 20, expected_bytes - size + 1))
            size += len(chunk)
            _require(size <= expected_bytes, "compressed evidence exceeds declared decoded size")
            yield chunk
        _require(not decoder.unused_data and not stream.read(1), "extra stream or trailing bytes in compressed evidence")
    _require(size == expected_bytes, "compressed evidence decoded size differs")


class _Archive:
    """Logical files over a checked raw/XZ inventory; never writes evidence."""
    def __init__(self, root: Path, hashes: dict, config: dict, inventory: dict, field: dict):
        self.root, self.hashes, self.storage = root, hashes, {}
        compression = config.get("evidence_compression")
        _require(compression is None or isinstance(compression, dict), "invalid compression configuration")
        enabled = compression is not None and compression.get("enabled") is True
        if compression is not None:
            _same(compression, {"enabled": enabled, "schema": "spectra.general_evidence_storage.v1",
                                "compression": "xz", "preset": 1, "minimum_bytes": 65536,
                                "outside_solve_and_check_timers": True}, "compression configuration")
        scopes = {f"instances/{index:05d}" for index in range(len(inventory["instances"]))}
        scopes |= {f"instances/{index:05d}/solver-{solver_index:03d}"
                   for index in range(len(inventory["instances"])) for solver_index in range(len(field["solvers"]))}
        receipts = {name for name in hashes if name.endswith("/storage.json")}
        _same(sorted(receipts), sorted(scope + "/storage.json" for scope in scopes) if enabled else [], "storage receipt scopes")
        limit = max(config["maximum_output_file_mib"], config["maximum_dimacs_mib"]) << 20
        for name in sorted(receipts):
            receipt = _read_json(root / name)
            scope = name.removesuffix("/storage.json")
            entries = receipt.get("entries")
            _require(isinstance(entries, list), "storage entries must be a list")
            _same({key: value for key, value in receipt.items() if key != "entries"},
                  {"schema": "spectra.general_evidence_storage.v1", "minimum_bytes": 65536,
                   "compression": "xz", "preset": 1, "timing_scope": "after checking; outside solve and check timers"}, "storage receipt policy")
            journal = root / scope / "storage.jsonl"
            journal_entries = []
            if journal.is_file():
                with journal.open(encoding="utf-8") as stream:
                    while line := stream.readline(MAX_METADATA_BYTES + 1):
                        _require(len(line) <= MAX_METADATA_BYTES and len(journal_entries) < len(entries), "oversized or extra storage journal entry")
                        journal_entries.append(_parse_json(line, "storage journal"))
            _same(journal_entries, entries, "storage journal and receipt")
            _require(bool(entries) == journal.is_file(), "storage journal presence differs")
            for entry in entries:
                _require(isinstance(entry, dict) and set(entry) ==
                         {"logical_path", "uncompressed_sha256", "uncompressed_bytes", "stored_path", "stored_sha256",
                          "stored_bytes", "compression", "preset", "roundtrip_verified"}, "invalid storage entry fields")
                logical, stored = _relative(entry["logical_path"]), _relative(entry["stored_path"])
                _require(logical.startswith(scope + "/") and
                         (scope.count("/") == 2 or "/" not in logical[len(scope) + 1:]), "storage entry escapes its scope")
                _require(logical not in self.storage and logical not in hashes and stored == logical + ".xz" and stored in hashes,
                         "ambiguous or missing stored evidence")
                _require(Path(logical).suffix.lower() not in {".json", ".jsonl", ".xz", ".gz", ".zip", ".zst", ".bz2"},
                         "unsupported compressed metadata or archive")
                _same(entry["compression"], "xz", "storage codec")
                _same(entry["preset"], 1, "storage preset")
                _same(entry["roundtrip_verified"], True, "storage roundtrip claim")
                _hash(entry["uncompressed_sha256"], "uncompressed evidence hash")
                _same(entry["stored_sha256"], hashes[stored], "stored evidence hash")
                _require(type(entry["uncompressed_bytes"]) is int and 65536 <= entry["uncompressed_bytes"] <= limit,
                         "invalid or excessive uncompressed evidence size")
                _same(entry["stored_bytes"], (root / stored).stat().st_size, "stored evidence size")
                decoded = hashlib.sha256()
                try:
                    for chunk in _decoded_chunks(root / stored, entry["uncompressed_bytes"]):
                        decoded.update(chunk)
                except lzma.LZMAError as error:
                    raise VerificationError(f"invalid compressed evidence: {error}") from error
                _same(decoded.hexdigest(), entry["uncompressed_sha256"], "decoded evidence hash")
                self.storage[logical] = entry

    def name(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def physical(self, path: Path) -> Path:
        entry = self.storage.get(self.name(path))
        return self.root / entry["stored_path"] if entry else path

    def exists(self, path: Path) -> bool:
        return self.name(path) in self.storage or path.is_file()

    def sha256(self, path: Path) -> str:
        name = self.name(path)
        return self.storage[name]["uncompressed_sha256"] if name in self.storage else self.hashes[name]

    def size(self, path: Path) -> int:
        entry = self.storage.get(self.name(path))
        return entry["uncompressed_bytes"] if entry else path.stat().st_size

    def open(self, path: Path, *, text: bool = False):
        physical = self.physical(path)
        opener = lzma.open if self.name(path) in self.storage else open
        return opener(physical, "rt", encoding="utf-8", errors="replace") if text else opener(physical, "rb")

    def copy(self, path: Path, destination: Path) -> None:
        with self.open(path) as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1 << 20)

    def admission(self, path: Path, limit: int) -> dict:
        if self.name(path) in self.storage:
            # The strict admission reader accepts plain/gzip files. Materialize
            # one original CNF at a time, never the whole archive or its proofs.
            with tempfile.TemporaryDirectory(prefix="spectra-admission-", dir="/tmp") as temporary:
                target = Path(temporary) / path.name
                _require(not target.is_relative_to(self.root), "admission workspace is inside evidence")
                self.copy(path, target)
                return asdict(admit_cnf(target, max_source_bytes=limit, max_dimacs_bytes=limit))
        return asdict(admit_cnf(path, max_source_bytes=limit, max_dimacs_bytes=limit))


def _lines(path: Path, archive: _Archive | None = None):
    with (archive.open(path, text=True) if archive else path.open(encoding="utf-8", errors="replace")) as stream:
        while line := stream.readline(4096):
            if len(line) == 4096 and not line.endswith("\n"):
                while line and not line.endswith("\n"):
                    line = stream.readline(4096)
                yield "<oversized log line>"
            else:
                yield line


def _answer(path: Path, archive: _Archive) -> str:
    statuses = []
    for line in _lines(path, archive):
        if line.startswith("s "):
            statuses.append(line.strip())
            if len(statuses) > 1:
                return "INVALID_OUTPUT"
    if not statuses:
        return "UNKNOWN"
    return {"s SATISFIABLE": "SAT", "s UNSATISFIABLE": "UNSAT", "s UNKNOWN": "UNKNOWN"}.get(statuses[0], "INVALID_OUTPUT")


def _process(path: Path, archive: _Archive) -> dict:
    record = _read_json(path / "process.json")
    _require(type(record.get("timed_out")) is bool, "process timed_out must be Boolean")
    _require(record.get("returncode") is None or type(record["returncode"]) is int, "invalid process returncode")
    _require(record.get("spawn_error") is None or isinstance(record["spawn_error"], str), "invalid process spawn_error")
    _require(isinstance(record.get("command"), list) and bool(record["command"]) and
             all(isinstance(arg, str) and bool(arg) for arg in record["command"]), "invalid recorded argv")
    _require(isinstance(record.get("working_directory"), str), "missing process working directory")
    _finite(record.get("elapsed_seconds"), "process runtime")
    for key in ("user_seconds", "system_seconds", "peak_rss_kib"):
        if record.get(key) is not None:
            _finite(record[key], f"process {key}")
    for stream in ("stdout", "stderr"):
        expected = record.get(stream + "_sha256")
        _hash(expected, f"process {stream} hash")
        _require(archive.sha256(path / (stream + ".txt")) == expected, f"process {stream} receipt differs: {path.name}")
    return record


def _solver_status(process: dict, answer: str, timeout: float, solver: dict) -> str:
    codes = {"SAT": [10], "UNSAT": [20], "UNKNOWN": [0]}
    codes.update(solver.get("accepted_exit_codes", {}))
    if process["timed_out"] or process["elapsed_seconds"] > timeout:
        return "TIMEOUT"
    if process["spawn_error"] or process["returncode"] not in {c for group in codes.values() for c in group}:
        return "RESOURCE_FAILURE"
    if answer == "INVALID_OUTPUT" or process["returncode"] not in codes.get(answer, []):
        return "INVALID_OUTPUT"
    return answer


def _recorded_paths(process: dict, directory: str) -> dict:
    """Resolve historical paths lexically; the archive may have been relocated."""
    working = process["working_directory"]
    path = PurePosixPath(working)
    _require(path.is_absolute() and path.as_posix() == working and ".." not in path.parts and
             working.endswith("/" + directory), "recorded solver working directory differs")
    return {"{cnf}": str(path / "input.cnf"), "{proof}": str(path / "proof.drat"),
            "{stats}": str(path / "stats.json"), "original": str(path.parent / "input.cnf"),
            "working": working}


def _substitute(command: list[str], replacements: dict) -> list[str]:
    result = []
    for argument in command:
        for token in ("{cnf}", "{proof}", "{stats}"):
            argument = argument.replace(token, replacements[token])
        result.append(argument)
    return result


def _solver_argv(process: dict, solver: dict, directory: str, config: dict) -> dict:
    replacements = _recorded_paths(process, directory)
    expected = _substitute(solver["command"], replacements)
    if "/" in expected[0] and not PurePosixPath(expected[0]).is_absolute():
        # The historical checkout root is not recorded as a separate field.
        # Require the resolved argv to be one of that solver's observed artifacts
        # and preserve the entire normalized relative suffix, not just basename.
        relative = os.path.normpath(expected[0])
        observed = process["command"][0]
        _require(not relative.startswith("../") and relative != ".." and
                 observed.endswith("/" + relative) and
                 observed in config["artifact_sha256"][solver["id"]], "relative solver executable cannot be bound to its recorded artifact")
        expected[0] = observed
    _same(process["command"], expected, "recorded solver argv and declared manifest")
    return replacements


def _checker_argv(process: dict, answer: str, solver: dict, paths: dict, config: dict) -> None:
    _same(process["working_directory"], paths["working"] + "/check", "recorded checker working directory")
    if answer == "SAT":
        expected = [process["command"][0], "-m", "experiments.general_competition.campaign", "check-model",
                    "--cnf", paths["original"], "--stdout", paths["working"] + "/solve/stdout.txt",
                    "--out", paths["working"] + "/complete.model"]
    elif solver.get("proof_checker"):
        expected = _substitute(solver["proof_checker"]["command"], {**paths, "{cnf}": paths["original"]})
    else:
        expected = [config["checker"], paths["original"], paths["{proof}"]]
    _same(process["command"], expected, "recorded checker argv and declared policy")


def _csv(cells: list[dict], timeout: float) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("solverid", "instanceid", "runtime", "status", "score", "vresult"))
    for cell in cells:
        runtime = Decimal(str(cell["elapsed_seconds"]))
        writer.writerow((cell["solver_id"], cell["instance_id"], runtime,
                         SOLVED.get(cell["status"], cell["status"].lower()),
                         runtime if cell["status"] in SOLVED else Decimal(str(timeout)) * 2,
                         "" if cell["expected"] == "UNKNOWN" else cell["expected"].lower()))
    return stream.getvalue().encode("utf-8")


def _cells(root: Path, count: int) -> list[dict]:
    result = []
    with (root / "cells.jsonl").open(encoding="utf-8") as stream:
        while line := stream.readline(MAX_METADATA_BYTES + 1):
            _require(len(line) <= MAX_METADATA_BYTES, "cell record exceeds byte limit")
            _require(len(result) < count, "campaign has extra cells")
            result.append(_parse_json(line, f"cell {len(result)}"))
    _require(len(result) == count, "campaign is missing declared cells")
    return result


def _original_source(source: Path, instance: dict, receipt: dict, maximum_bytes: int) -> None:
    # This path comes exclusively from a caller-supplied mapping, never an archive.
    _require(digest(source) == instance["sha256"], f"explicit original source hash differs: {instance['id']}")
    opener = lzma.open if source.suffix.lower() == ".xz" else gzip.open if source.suffix.lower() == ".gz" else open
    value, length = hashlib.sha256(), 0
    with opener(source, "rb") as stream:
        while chunk := stream.read(1 << 20):
            length += len(chunk)
            _require(length <= maximum_bytes, "explicit original source exceeds decompression limit")
            value.update(chunk)
    _require(value.hexdigest() == receipt["dimacs_sha256"] and length == receipt["dimacs_bytes"],
             f"explicit source decompression differs: {instance['id']}")


def _admissions(root: Path, config: dict, inventory: dict, archive: _Archive,
                original_sources: Mapping[str, Path]) -> tuple[dict, list, dict]:
    retained = _read_json(root / "admissions.json")
    admissions = retained.get("admissions")
    instances = inventory["instances"]
    _require(isinstance(admissions, list) and len(admissions) == len(instances), "admission rectangle differs")
    _require(set(original_sources) <= {item["id"] for item in instances}, "explicit source mapping has unknown instance ids")
    receipts, seen, duplicates = {}, {}, []
    counts = {"admitted": 0, "reparsed_retained_inputs": 0, "uncompressed_sources_bound": 0,
              "explicit_original_sources_rechecked": 0, "compressed_sources_not_available": 0}
    limit = config["maximum_dimacs_mib"] << 20
    for index, (instance, admission) in enumerate(zip(instances, admissions)):
        directory = root / "instances" / f"{index:05d}"
        process = _process(directory / "admission-process", archive)
        _same(admission.get("process"), process, "admission process record")
        _same(admission.get("instance_id"), instance["id"], "admission instance id")
        accepted = (process["returncode"] == 0 and not process["timed_out"] and
                    process["elapsed_seconds"] <= config["checker_timeout_seconds"] and
                    (directory / "admission.json").is_file())
        _same(admission.get("admitted"), accepted, "admission outcome")
        if not accepted:
            continue
        receipt = _read_json(directory / "admission.json")
        fresh = archive.admission(directory / "input.cnf", limit)
        for key, value in fresh.items():
            _same(receipt.get(key), value, f"admission receipt {instance['id']} {key}")
        _same(receipt.get("original_source_sha256"), instance["sha256"], "inventory original input hash")
        _require(type(receipt.get("original_source_bytes")) is int and receipt["original_source_bytes"] >= 0,
                 "invalid original source byte count")
        compressed = Path(instance["path"]).suffix.lower() in (".gz", ".xz")
        if not compressed:
            _same(receipt["dimacs_sha256"], instance["sha256"], "uncompressed original input hash")
            _same(receipt["dimacs_bytes"], receipt["original_source_bytes"], "uncompressed original input size")
            counts["uncompressed_sources_bound"] += 1
        if instance["id"] in original_sources:
            _original_source(Path(original_sources[instance["id"]]), instance, receipt, limit)
            counts["explicit_original_sources_rechecked"] += 1
        elif compressed:
            counts["compressed_sources_not_available"] += 1
        semantic = receipt["semantic_sha256"]
        if semantic in seen:
            duplicates.append([seen[semantic], instance["id"]])
        else:
            seen[semantic] = instance["id"]
        receipts[instance["id"]] = receipt
        counts["admitted"] += 1
        counts["reparsed_retained_inputs"] += 1
    _same(retained.get("exact_semantic_duplicate_pairs"), duplicates, "admission duplicate pairs")
    return receipts, duplicates, counts


def _input_integrity(value: object, label: str) -> dict:
    _require(isinstance(value, dict) and set(value) ==
             {"solver_input_sha256", "original_cnf_sha256", "source_sha256"}, f"invalid {label}")
    for key, item in value.items():
        _hash(item, label + " " + key, nullable=True)
    return value


def _cell(root: Path, cell: dict, solver: dict, instance: dict, directory: str,
          receipt: dict | None, config: dict, archive: _Archive) -> None:
    _same(cell.get("directory"), directory, "cell directory")
    path = root / directory
    _same(_read_json(path / "cell.json"), cell, "cell.json and cells.jsonl")
    _finite(cell.get("checking_seconds"), "checking runtime")
    _require(type(cell.get("proof_bytes")) is int and cell["proof_bytes"] >= 0, "invalid proof byte count")
    failures = config["artifact_identity_failures"].get(solver["id"])
    if failures or not (path / "solve").exists():
        _same(cell["status"], "SOURCE_IDENTITY_FAILURE" if failures else "ADMISSION_FAILURE", "unexecuted cell status")
        if failures:
            _same(cell.get("identity_failures"), failures, "source identity failures")
        for key, value in {"elapsed_seconds": 0.0, "peak_rss_kib": None, "checking_seconds": 0.0, "proof_bytes": 0}.items():
            _same(cell.get(key), value, f"unexecuted cell {key}")
        _require(not (path / "solve").exists() and not (path / "check").exists(), "unexecuted cell has process evidence")
        return
    _require(receipt is not None, "solver ran without an accepted input")
    process = _process(path / "solve", archive)
    historical_paths = _solver_argv(process, solver, directory, config)
    for key in ("elapsed_seconds", "peak_rss_kib", "returncode"):
        _same(cell.get(key), process.get(key), "solver process " + key)
    answer = _answer(path / "solve" / "stdout.txt", archive)
    _same(cell.get("reported_answer"), answer, "reported answer")
    status = _solver_status(process, answer, config["timeout_seconds"], solver)
    after_solve = _input_integrity(cell.get("input_integrity_after_solve"), "after-solve integrity")
    after_check = _input_integrity(cell.get("input_integrity_after_check"), "after-check integrity")
    expected_integrity = {"solver_input_sha256": receipt["dimacs_sha256"],
                          "original_cnf_sha256": receipt["dimacs_sha256"], "source_sha256": instance["sha256"]}
    unchanged = after_solve == expected_integrity
    if not unchanged:
        status = "INVALID_INPUT_MUTATION"
    custom = solver.get("proof_checker")
    unsupported = solver.get("proof_format", "DRAT").upper() not in ("DRAT", "DRUP") and custom is None
    needs_check = status in ("SAT", "UNSAT") and not (status == "UNSAT" and unsupported)
    _same((path / "check").is_dir(), needs_check, "checker presence")
    if status == "UNSAT" and unsupported:
        status = "UNAVAILABLE_CHECKER"
        _same(cell.get("proof_format"), solver["proof_format"], "unsupported proof format")
    elif needs_check:
        checked = _process(path / "check", archive)
        _checker_argv(checked, answer, solver, historical_paths, config)
        _same(cell["checking_seconds"], checked["elapsed_seconds"], "checker runtime")
        resource_diagnostic = (checker_resource_diagnostic(
            solver.get("proof_format", "DRAT"), checked,
            _lines(path / "check" / "stdout.txt", archive), _lines(path / "check" / "stderr.txt", archive))
            if status == "UNSAT" and config.get("scoring_policy", "legacy-v1") == "declared-inventory-v2" else None)
        applied_diagnostic = None
        if checked["timed_out"] or checked["elapsed_seconds"] > config["checker_timeout_seconds"]:
            status = "CHECKER_TIMEOUT"
        elif checked["spawn_error"] or checked["returncode"] is None or checked["returncode"] < 0:
            status = "CHECKER_RESOURCE_FAILURE"
        elif resource_diagnostic is not None:
            status = "CHECKER_RESOURCE_FAILURE"
            applied_diagnostic = resource_diagnostic
        elif status == "SAT":
            status = "SAT_VERIFIED" if checked["returncode"] == 0 else "INVALID_MODEL" if checked["returncode"] == 65 else "CHECKER_RESOURCE_FAILURE"
            if status == "SAT_VERIFIED":
                _require(archive.exists(path / "complete.model"), "verified SAT cell lacks complete model")
                check_receipt = _read_json(path / "check" / "stdout.txt")
                _same(check_receipt.get("status"), "SAT_VERIFIED", "SAT checker receipt status")
                _same(check_receipt.get("checked_clauses"), receipt["declared_clauses"], "SAT checker clause count")
                supplied, completed = check_receipt.get("supplied_assignments"), check_receipt.get("completed_false_variables")
                _require(type(supplied) is int and type(completed) is int and supplied >= 0 and completed >= 0 and
                         supplied + completed == receipt["declared_variables"], "SAT checker assignment counts differ")
        else:
            success = custom["success_line"] if custom else "s VERIFIED"
            codes = custom.get("accepted_exit_codes", [0]) if custom else [0]
            accepted = checked["returncode"] in codes and any(line.strip() == success for line in _lines(path / "check" / "stdout.txt", archive))
            status = "UNSAT_VERIFIED" if accepted and archive.exists(path / "proof.drat") else "INVALID_PROOF"
        _same(cell.get("checker_resource_diagnostic"), applied_diagnostic, "checker resource diagnostic")
        if status in SOLVED and instance["expected"] not in ("UNKNOWN", answer):
            _require(config.get("scoring_policy", "legacy-v1") == "legacy-v1",
                     "declared-inventory-v2 metadata conflict requires adjudication; no ranking accepted")
            status = "INVALID_VERDICT_CONFLICT"
    else:
        _same(cell["checking_seconds"], 0.0, "unchecked cell checking runtime")
    if after_check != expected_integrity:
        status = "INVALID_INPUT_MUTATION"
    _same(cell["status"], status, "derived cell status")
    _require(after_solve["original_cnf_sha256"] == after_check["original_cnf_sha256"] == receipt["dimacs_sha256"] and
             after_solve["source_sha256"] == after_check["source_sha256"] == instance["sha256"],
             "original input mutation prevents campaign acceptance")
    if unchanged and after_check == expected_integrity:
        _same(cell.get("identical_input_retained_at"), str(PurePosixPath(directory).parent / "input.cnf"), "retained identical input")
        _require(not archive.exists(path / "input.cnf"), "unexpected retained identical solver input")
    else:
        _same(archive.sha256(path / "input.cnf") if archive.exists(path / "input.cnf") else None,
              after_check["solver_input_sha256"], "retained altered solver input")
    for filename, key in (("proof.drat", "proof"), ("stats.json", "stats")):
        present = archive.exists(path / filename)
        _same(key + "_sha256" in cell, present, key + " hash presence")
        if present:
            _same(cell[key + "_sha256"], archive.sha256(path / filename), key + " hash")
    _same(cell["proof_bytes"], archive.size(path / "proof.drat") if archive.exists(path / "proof.drat") else 0, "proof byte count")


@dataclass(frozen=True)
class TrustedProofChecker:
    """An explicit local trust decision; never construct this from the archive."""
    executable: Path
    arguments: tuple[str, ...] = ("{cnf}", "{proof}")
    success_line: str = "s VERIFIED"
    accepted_exit_codes: tuple[int, ...] = (0,)
    expected_sha256: str | None = None


def _fresh_checks(root: Path, cells: list[dict], archive: _Archive, *, recheck_models: bool,
                  proof_checkers: Mapping[str, TrustedProofChecker], require_all_proofs: bool,
                  timeout: float, memory_mib: int, output_mib: int) -> dict:
    _finite(timeout, "fresh checker timeout", positive=True)
    for value, label in ((memory_mib, "fresh checker memory"), (output_mib, "fresh checker output")):
        _require(type(value) is int and value > 0, f"{label} must be a positive integer")
    result = {"models_rechecked": 0, "proofs_rechecked": 0, "verified_models_not_rechecked": 0,
              "verified_proofs_not_rechecked": 0, "trusted_checker_sha256": {}}
    trusted = {}
    for solver_id, checker in proof_checkers.items():
        executable = checker.executable.expanduser().resolve()
        _require(executable.is_file() and os.access(executable, os.X_OK), f"trusted checker is not executable: {solver_id}")
        _require(checker.arguments and all(isinstance(arg, str) and arg for arg in checker.arguments) and
                 any("{cnf}" in arg for arg in checker.arguments) and any("{proof}" in arg for arg in checker.arguments), "trusted checker needs explicit CNF and proof arguments")
        _require(checker.success_line and "\n" not in checker.success_line and "\r" not in checker.success_line,
                 "trusted checker needs a single exact success line")
        _require(checker.accepted_exit_codes and all(type(code) is int and 0 <= code <= 255 for code in checker.accepted_exit_codes),
                 "invalid trusted checker exit codes")
        observed = digest(executable)
        if checker.expected_sha256 is not None:
            _hash(checker.expected_sha256, "trusted checker expected hash")
            _same(observed, checker.expected_sha256, "trusted checker executable hash")
        result["trusted_checker_sha256"][solver_id] = observed
        trusted[solver_id] = (checker, executable)
    for cell in cells:
        if cell["status"] not in SOLVED:
            continue
        sat = cell["status"] == "SAT_VERIFIED"
        if (sat and not recheck_models) or (not sat and cell["solver_id"] not in trusted):
            result["verified_models_not_rechecked" if sat else "verified_proofs_not_rechecked"] += 1
            _require(sat or not require_all_proofs, f"no explicitly trusted proof checker for {cell['solver_id']}")
            continue
        # An explicit directory outside the archive avoids inheriting TMPDIR that
        # could itself point inside the evidence tree. All checker inputs are copies.
        with tempfile.TemporaryDirectory(prefix="spectra-recheck-", dir="/tmp") as temporary:
            work = Path(temporary)
            _require(not work.is_relative_to(root), "fresh checker workspace is inside evidence")
            cell_dir = root / cell["directory"]
            archive.copy(cell_dir.parent / "input.cnf", work / "input.cnf")
            inputs = {"input.cnf": archive.sha256(cell_dir.parent / "input.cnf")}
            if sat:
                archive.copy(cell_dir / "solve" / "stdout.txt", work / "stdout.txt")
                inputs["stdout.txt"] = archive.sha256(cell_dir / "solve" / "stdout.txt")
                command = [sys.executable, "-m", "experiments.general_competition.campaign", "check-model",
                           "--cnf", str(work / "input.cnf"), "--stdout", str(work / "stdout.txt"), "--out", str(work / "complete.model")]
            else:
                checker, executable = trusted[cell["solver_id"]]
                archive.copy(cell_dir / "proof.drat", work / "proof.drat")
                inputs["proof.drat"] = archive.sha256(cell_dir / "proof.drat")
                command = [str(executable), *[arg.replace("{cnf}", str(work / "input.cnf")).replace("{proof}", str(work / "proof.drat"))
                                             for arg in checker.arguments]]
            for filename, expected in inputs.items():
                _same(digest(work / filename), expected, "fresh checker input before checking")
            process = run_process(command, work / "check", timeout=timeout, memory_mib=memory_mib,
                                  output_mib=output_mib, workdir=work)
            for filename, expected in inputs.items():
                path = work / filename
                _require(path.is_file() and not path.is_symlink() and digest(path) == expected,
                         f"fresh checker mutated its input: {filename}")
            _require(not process["timed_out"] and process["elapsed_seconds"] <= timeout and not process["spawn_error"],
                     f"fresh checker did not complete: {cell['solver_id']} / {cell['instance_id']}")
            if sat:
                _require(process["returncode"] == 0 and (work / "complete.model").is_file(),
                         f"fresh model check failed: {cell['solver_id']} / {cell['instance_id']}")
                _same(digest(work / "complete.model"), archive.sha256(cell_dir / "complete.model"), "fresh complete model")
                result["models_rechecked"] += 1
            else:
                _require(process["returncode"] in checker.accepted_exit_codes and
                         any(line.strip() == checker.success_line for line in _lines(work / "check" / "stdout.txt")),
                         f"fresh proof check failed: {cell['solver_id']} / {cell['instance_id']}")
                _same(digest(executable), result["trusted_checker_sha256"][cell["solver_id"]], "trusted checker changed during replay")
                result["proofs_rechecked"] += 1
    return result


def verify_campaign(evidence: Path, *, expected_manifest_sha256: str | None = None,
                    original_sources: Mapping[str, Path] | None = None,
                    recheck_models: bool = False,
                    proof_checkers: Mapping[str, TrustedProofChecker] | None = None,
                    require_all_proofs: bool = False, checker_timeout: float = 120,
                    checker_memory_mib: int = 4096, checker_output_mib: int = 1024) -> dict:
    """Verify a portable archive without modifying it or trusting stored commands.

    Original compressed source files are checked only through explicit id-to-path
    mappings. Without those files the retained decompressed CNF is still parsed
    and hash-checked, but its compressed source hash remains a recorded claim.
    """
    implementations = {name: Path(__file__).with_name(name + ".py") for name in
                       ("campaign_verify", "campaign", "cnf_admission", "official_score")}
    implementation_hashes = {name: digest(path) for name, path in implementations.items()}
    root = _root(Path(evidence))
    hashes, directories = _verify_inventory(root, expected_manifest_sha256)
    manifest_hash = digest(root / MANIFEST)
    config = _read_json(root / "campaign.json")
    _same(config.get("schema"), "spectra.general_campaign.v1", "campaign schema")
    _same(config.get("evidence_class"), "EXPOSED_DEVELOPMENT", "campaign evidence class")
    _same(config.get("claim"), CLAIM, "campaign claim")
    scoring_policy = config.get("scoring_policy", "legacy-v1")
    _require(scoring_policy in ("legacy-v1", "declared-inventory-v2"), "unsupported campaign scoring policy")
    for key in ("complete_roadmap_field_verified", "near_duplicate_control", "source_to_binary_rebuild_link_verified", "exclusive_host"):
        _same(config.get(key), False, "unsupported campaign claim " + key)
    for key in ("timeout_seconds", "checker_timeout_seconds"):
        _finite(config.get(key), key, positive=True)
    for key in ("memory_mib", "maximum_output_file_mib", "maximum_dimacs_mib"):
        _require(type(config.get(key)) is int and config[key] > 0, f"invalid campaign {key}")
    for name, key in (("inventory.json", "inventory_sha256"), ("solvers.json", "solver_field_sha256")):
        _read_json(root / name)  # Reject duplicate/non-finite JSON before public manifest validation.
        _same(hashes.get(name), config.get(key), "copied " + name + " binding")
    inventory, field = load_inventory(root / "inventory.json"), load_solvers(root / "solvers.json")
    _same(config.get("solver_metadata"), field["solvers"], "solver metadata")
    solver_ids = {solver["id"] for solver in field["solvers"]}
    _require(isinstance(config.get("artifact_identity_failures"), dict) and
             set(config["artifact_identity_failures"]) <= solver_ids, "invalid artifact identity failures")
    for failures in config["artifact_identity_failures"].values():
        _require(isinstance(failures, list) and failures and all(isinstance(path, str) and path for path in failures),
                 "invalid artifact identity failure paths")
    artifacts = config.get("artifact_sha256")
    _require(isinstance(artifacts, dict) and set(artifacts) == solver_ids, "artifact solver field differs")
    for observed in artifacts.values():
        _require(isinstance(observed, dict), "invalid observed artifact hashes")
        for filename, observed_hash in observed.items():
            _require(isinstance(filename, str) and bool(filename), "invalid observed artifact path")
            _hash(observed_hash, "observed artifact hash", nullable=True)
    _hash(config.get("runner_sha256"), "recorded runner hash")
    _hash(config.get("checker_sha256"), "recorded checker hash", nullable=True)
    integrity = _read_json(root / "artifact-integrity.json")
    _same(integrity.get("initial_identity_failures"), config["artifact_identity_failures"], "initial artifact identity failures")
    _same(integrity.get("changed_artifacts"), [], "changed artifacts prevent campaign acceptance")
    _same(integrity.get("original_input_mutations"), [], "original input mutations prevent campaign acceptance")
    count = len(inventory["instances"]) * len(field["solvers"])
    _require(count <= MAX_CELLS, "declared rectangle exceeds verifier cell limit")
    archive = _Archive(root, hashes, config, inventory, field)
    cells = _cells(root, count)
    for cell in cells:
        _require(isinstance(cell.get("status"), str) and cell["status"] in FINAL_STATUSES, "unknown or invalid cell status")
    try:
        summary = score_campaign(cells, inventory, field, config["timeout_seconds"], scoring_policy=scoring_policy)
    except (KeyError, TypeError, ValueError) as error:
        raise VerificationError(f"cannot reconstruct declared rectangle and scores: {error}") from error
    receipts, duplicates, input_checks = _admissions(root, config, inventory, archive, original_sources or {})
    by_id = {(cell["solver_id"], cell["instance_id"]): cell for cell in cells}
    declared_directories = set()
    for index, instance in enumerate(inventory["instances"]):
        for solver_index, solver in enumerate(field["solvers"]):
            directory = f"instances/{index:05d}/solver-{solver_index:03d}"
            declared_directories.add(directory)
            _cell(root, by_id[solver["id"], instance["id"]], solver, instance, directory,
                  receipts.get(instance["id"]), config, archive)
    actual_directories = {name for name in directories if re.fullmatch(r"instances/[^/]+/solver-[^/]+", name)}
    _same(sorted(actual_directories), sorted(declared_directories), "cell directory rectangle")
    actual_instances = {name for name in directories if re.fullmatch(r"instances/[^/]+", name)}
    _same(sorted(actual_instances), [f"instances/{index:05d}" for index in range(len(inventory["instances"]))],
          "instance directory rectangle")
    expected_csv = _csv(cells, config["timeout_seconds"])
    _require((root / "scores.csv").stat().st_size == len(expected_csv) and (root / "scores.csv").read_bytes() == expected_csv,
             "scores.csv differs from recomputed cells")
    summary.update(evidence_class="EXPOSED_DEVELOPMENT", declared_field_complete=True,
                   complete_roadmap_field_verified=False, exact_semantic_duplicate_pairs=duplicates, claim=CLAIM)
    _same(_read_json(root / "summary.json"), summary, "recomputed campaign summary")
    proof_checkers = proof_checkers or {}
    _require(set(proof_checkers) <= solver_ids, "trusted checker mapping has unknown solver ids")
    fresh = _fresh_checks(root, cells, archive, recheck_models=recheck_models, proof_checkers=proof_checkers,
                          require_all_proofs=require_all_proofs, timeout=checker_timeout,
                          memory_mib=checker_memory_mib, output_mib=checker_output_mib)
    # Detect changes during verification, including a trusted checker unexpectedly
    # modifying the original archive through an unrelated absolute path.
    after_hashes, after_directories = _verify_inventory(root, manifest_hash)
    _same(after_hashes, hashes, "evidence changed during verification")
    _same(sorted(after_directories), sorted(directories), "evidence directories changed during verification")
    _same({name: digest(path) for name, path in implementations.items()}, implementation_hashes,
          "local verifier or checker implementation changed during verification")
    return {"schema": "spectra.general_campaign_verification.v1", "status": "CONSISTENT_EXPOSED_ARCHIVE",
            "implementation_sha256": implementation_hashes,
            "evidence_manifest_sha256": manifest_hash, "externally_pinned": expected_manifest_sha256 is not None,
            "files_verified": len(hashes), "compressed_files_decoded_and_verified": len(archive.storage),
            "cells_verified": len(cells), "scores_recomputed": True, "scoring_policy": scoring_policy,
            "input_checks": input_checks, "fresh_answer_checks": fresh,
            "stored_commands_executed": False, "campaign_evidence_modified": False,
            "evidence_class": "EXPOSED_DEVELOPMENT", "hidden_evidence_verified": False,
            "independent_replication_verified": False, "complete_roadmap_field_verified": False,
            "source_to_binary_rebuild_link_verified": False, "timings_independently_remeasured": False,
            "source_and_executable_identity": "Recorded hashes only; external artifacts and rebuild linkage are not inferred.",
            "claim": CLAIM}


def _mapping(values: list[str], label: str) -> dict[str, Path]:
    result = {}
    for value in values:
        key, separator, filename = value.partition("=")
        _require(bool(separator and key and filename) and key not in result, f"{label} must be unique ID=PATH pairs")
        result[key] = Path(filename)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--expected-manifest-sha256", help="independently retained evidence-sha256.json digest")
    parser.add_argument("--original-source", action="append", default=[], metavar="INSTANCE_ID=PATH",
                        help="explicit trusted local source file; archived source paths are never opened")
    parser.add_argument("--recheck-models", action="store_true", help="fresh bounded model checks on temporary copies")
    parser.add_argument("--proof-checker", action="append", default=[], metavar="SOLVER_ID=PATH",
                        help="explicit local checker invoked as PATH CNF PROOF; requires exit 0 and exact 's VERIFIED'")
    parser.add_argument("--require-all-proofs", action="store_true", help="fail unless every verified UNSAT cell receives a fresh proof check")
    parser.add_argument("--checker-timeout", type=float, default=120)
    parser.add_argument("--checker-memory-mib", type=int, default=4096)
    parser.add_argument("--checker-output-mib", type=int, default=1024)
    args = parser.parse_args(argv)
    try:
        result = verify_campaign(args.evidence, expected_manifest_sha256=args.expected_manifest_sha256,
                    original_sources=_mapping(args.original_source, "original sources"),
                    recheck_models=args.recheck_models,
                    proof_checkers={key: TrustedProofChecker(path) for key, path in _mapping(args.proof_checker, "proof checkers").items()},
                    require_all_proofs=args.require_all_proofs, checker_timeout=args.checker_timeout,
                    checker_memory_mib=args.checker_memory_mib, checker_output_mib=args.checker_output_mib)
    except (OSError, ValueError, KeyError, TypeError, MemoryError, lzma.LZMAError) as error:
        print(json.dumps({"schema": "spectra.general_campaign_verification.v1", "status": "REJECTED",
                          "error": str(error), "claim": "No ranking or scientific gate accepted"}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
