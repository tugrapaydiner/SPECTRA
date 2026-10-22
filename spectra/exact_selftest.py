"""Deterministic release/replication smoke test for the installed exact service.

Generation proves that the installed package can explicitly build its bundled
native runtime, return checked SAT and UNSAT answers, replay every answer without
native code, reject a mutated answer, and retain a portable hash manifest.

Audit verifies a retained self-test without loading native code. Neither operation
is a benchmark, an independent scientific replication, or a deployment/adoption
claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys

from . import __version__
from ._json import load_file, read_bounded
from .evidence import (
    FORMAT as EVIDENCE_FORMAT,
    capture,
    sha256_file,
    verify,
    write_new,
)
from .exact import (
    DEFAULT_BYTES,
    EXACT_API_VERSION,
    EXACT_NATIVE_ABI,
    SOURCE,
    ExactProblem,
    ExactRuntime,
    build_exact_runtime,
    decode_certified_output,
    pack_session_packet,
    unpack_session_packet,
    verify_session_output,
)
from .exact_proof import decode_proof

SCHEMA = "spectra.exact_selftest.v2"
AUDIT_SCHEMA = "spectra.exact_selftest_audit.v2"
_SCOPE = (
    "deterministic installed exact-service build and semantic self-test; "
    "not a benchmark, scientific replication, or deployment claim"
)
_JSON_MAX_BYTES = 16 << 20


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_max_bytes(max_bytes: int) -> None:
    if type(max_bytes) is not int or not 1 <= max_bytes <= sys.maxsize - 1:
        raise ValueError("max-bytes must be a positive bounded integer")


def _read_bounded(path: Path, max_bytes: int) -> bytes:
    return read_bounded(path, max_bytes=max_bytes)


def _load_json(path: Path, max_bytes: int) -> dict:
    value = load_file(path, max_bytes=min(max_bytes, _JSON_MAX_BYTES))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_no_native_module() -> None:
    _require(
        "_spectra_exact_binary" not in sys.modules,
        "exact audit must run in a fresh process without the native module loaded",
    )


def _write_bytes_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _case_specs() -> tuple[
    tuple[
        str,
        ExactProblem,
        tuple[tuple[tuple[int, int], ...], ...],
        tuple[int, ...],
        tuple[int | None, ...],
        str,
        str,
    ],
    ...,
]:
    mixed_problem = ExactProblem.create(
        2,
        2,
        ((0, 1),),
        (0b11, 0b11),
    )
    strong_compression_problem = ExactProblem.create(
        8,
        2,
        tuple((vertex, vertex + 1) for vertex in range(7)),
        (0b11,) * 8,
    )
    static_problem = ExactProblem.create(
        2,
        1,
        ((0, 1),),
        (0b1, 0b1),
    )
    return (
        (
            "mixed-current-query",
            mixed_problem,
            (
                (),
                ((0, 0b01),),
                ((0, 0b01), (1, 0b01)),
                ((0, 0),),
            ),
            (1, 1, 0, 0),
            (None, None, 2, 1),
            "raw",
            "short_weak_compression",
        ),
        (
            "strong-compression",
            strong_compression_problem,
            ((),) * 8,
            (1,) * 8,
            (None,) * 8,
            "scc",
            "scc_strong_compression",
        ),
        (
            "base-static-contradiction",
            static_problem,
            ((),),
            (0,),
            (3,),
            "raw",
            "base_infeasible",
        ),
    )


def _run_case(
    root: Path,
    runtime: ExactRuntime,
    *,
    name: str,
    problem: ExactProblem,
    queries: tuple[tuple[tuple[int, int], ...], ...],
    expected_codes: tuple[int, ...],
    expected_kinds: tuple[int | None, ...],
    expected_mode: str,
    expected_reason: str,
    max_bytes: int,
) -> dict:
    packet = pack_session_packet(problem, queries)
    result = runtime.solve_session_packet(packet, mode="auto", max_bytes=max_bytes)
    _require(result.query_count == len(queries), f"{name}: query count differs")
    _require(result.sat_count == expected_codes.count(1), f"{name}: SAT count differs")
    _require(result.unsat_count == expected_codes.count(0), f"{name}: UNSAT count differs")
    _require(result.info["selected_mode"] == expected_mode, f"{name}: selected mode differs")
    _require(result.info["auto_policy"] == "structural-v11", f"{name}: policy differs")
    _require(
        result.info["auto_policy_reason"] == expected_reason,
        f"{name}: policy reason differs",
    )
    _require(
        verify_session_output(packet, result.output),
        f"{name}: standard-library replay rejected native output",
    )

    decoded = decode_certified_output(result.output, problem.n, len(queries))
    codes = tuple(code for code, _labels, _proof in decoded)
    kinds = tuple(
        None if code else decode_proof(proof, problem.n)[0]
        for code, _labels, proof in decoded
    )
    _require(codes == expected_codes, f"{name}: decision sequence differs")
    _require(kinds == expected_kinds, f"{name}: proof-kind sequence differs")

    _require(bool(result.output), f"{name}: output is unexpectedly empty")
    tampered = bytes((result.output[0] ^ 1,)) + result.output[1:]
    _require(
        not verify_session_output(packet, tampered),
        f"{name}: mutated output was accepted",
    )

    packet_name = f"cases/{name}.sbc1"
    output_name = f"cases/{name}.scq2"
    _write_bytes_new(root / packet_name, packet)
    _write_bytes_new(root / output_name, result.output)

    return {
        "name": name,
        "vertices": problem.n,
        "edges": len(problem.edges),
        "queries": result.query_count,
        "sat": result.sat_count,
        "unsat": result.unsat_count,
        "proof_kinds": list(kinds),
        "selected_mode": result.info["selected_mode"],
        "auto_policy": result.info["auto_policy"],
        "auto_policy_reason": result.info["auto_policy_reason"],
        "packet": packet_name,
        "packet_bytes": len(packet),
        "packet_sha256": _sha256(packet),
        "output": output_name,
        "output_bytes": len(result.output),
        "output_sha256": _sha256(result.output),
        "independent_replay": True,
        "tampered_output_rejected": True,
    }


def run_exact_selftest(
    directory: str | Path,
    *,
    compiler: str | None = None,
    max_bytes: int = DEFAULT_BYTES,
) -> dict:
    """Create a new self-test evidence directory and return its final report.

    The destination must not exist. A failed run deliberately leaves its partial
    directory in place for diagnosis; callers must choose a fresh destination
    rather than silently replacing or completing earlier evidence.
    """
    _validate_max_bytes(max_bytes)
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)

    library = build_exact_runtime(root / "native", compiler=compiler)
    source_name = "native/exact_binary.cpp"
    _write_bytes_new(root / source_name, SOURCE.read_bytes())
    runtime = ExactRuntime(library)
    cases = [
        _run_case(
            root,
            runtime,
            name=name,
            problem=problem,
            queries=queries,
            expected_codes=codes,
            expected_kinds=kinds,
            expected_mode=mode,
            expected_reason=reason,
            max_bytes=max_bytes,
        )
        for name, problem, queries, codes, kinds, mode, reason in _case_specs()
    ]

    proof_kinds = sorted(
        {
            kind
            for case in cases
            for kind in case["proof_kinds"]
            if kind is not None
        }
    )
    _require(proof_kinds == [1, 2, 3], "self-test did not cover all SBP1 proof kinds")

    receipt_path = library.with_name("build.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    compiler_lines = [
        line.strip() for line in str(receipt.get("compiler", "")).splitlines()
        if line.strip()
    ]
    report = {
        "schema": SCHEMA,
        "status": "PASS",
        "scope": _SCOPE,
        "spectra_version": __version__,
        "exact_api_version": EXACT_API_VERSION,
        "native_abi": EXACT_NATIVE_ABI,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "module": str(Path(__file__).resolve()),
        "build": {
            "source": source_name,
            "library": library.relative_to(root).as_posix(),
            "receipt": receipt_path.relative_to(root).as_posix(),
            "source_sha256": receipt["source_sha256"],
            "library_sha256": receipt["library_sha256"],
            "compiler": compiler_lines[0] if compiler_lines else "",
            "sanitizer": receipt["sanitizer"],
        },
        "cases": cases,
        "checks": {
            "receipt_bound_native_load": True,
            "native_original_input_observer": True,
            "independent_python_replay": True,
            "tampered_output_rejected": True,
            "proof_kinds": proof_kinds,
        },
        "evidence_manifest": "SHA256.json",
        "evidence_format": EVIDENCE_FORMAT,
    }
    _write_json_new(root / "selftest.json", report)

    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "SHA256.json"
    )
    manifest = capture(root, names)
    verify(root, manifest, required=names)
    write_new(root / "SHA256.json", manifest)
    return report


def _artifact(root: Path, artifacts: dict, name: object) -> Path:
    _require(type(name) is str and name in artifacts, "report references an unfrozen artifact")
    relative = PurePosixPath(name)
    _require(
        bool(relative.parts)
        and not relative.is_absolute()
        and all(part not in ("", ".", "..") for part in relative.parts),
        "report references an unsafe artifact path",
    )
    path = root.joinpath(*relative.parts)
    _require(not path.is_symlink(), "self-test artifacts must not be symlinks")
    return path


def verify_exact_selftest(
    directory: str | Path,
    *,
    max_bytes: int = DEFAULT_BYTES,
) -> dict:
    """Audit a retained self-test without compiling or loading native code."""
    _validate_max_bytes(max_bytes)
    _require_no_native_module()
    supplied_root = Path(directory)
    _require(not supplied_root.is_symlink(), "self-test root must not be a symlink")
    root = supplied_root.resolve(strict=True)
    _require(root.is_dir(), "self-test root must be a directory")
    _require(
        not any(path.is_symlink() for path in root.rglob("*")),
        "self-test directory must not contain symlinks",
    )

    manifest_path = root / "SHA256.json"
    manifest = _load_json(manifest_path, max_bytes)
    entries = manifest.get("artifacts")
    _require(type(entries) is dict and entries, "self-test manifest is empty or malformed")
    for name, entry in entries.items():
        _require(
            type(entry) is dict
            and type(entry.get("bytes")) is int
            and 0 <= entry["bytes"] <= max_bytes,
            f"self-test artifact exceeds max-bytes or is malformed: {name}",
        )
    manifest_result = verify(root, manifest)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    _require(actual == set(entries), "self-test directory contains unbound or missing files")
    _require("selftest.json" in entries, "self-test report is not frozen")

    report = _load_json(root / "selftest.json", max_bytes)
    expected_report_fields = {
        "schema", "status", "scope", "spectra_version", "exact_api_version",
        "native_abi", "python", "python_executable", "platform", "machine",
        "module", "build", "cases", "checks", "evidence_manifest",
        "evidence_format",
    }
    _require(set(report) == expected_report_fields, "self-test report fields differ")
    _require(report["schema"] == SCHEMA, "unsupported self-test schema")
    _require(report["status"] == "PASS", "self-test report is not a PASS")
    _require(report["scope"] == _SCOPE, "self-test claim boundary differs")
    for field in (
        "spectra_version", "python", "python_executable", "platform", "machine", "module"
    ):
        _require(type(report[field]) is str and bool(report[field]), f"{field} is missing")
    _require(report["exact_api_version"] == EXACT_API_VERSION, "exact API version differs")
    _require(report["native_abi"] == EXACT_NATIVE_ABI, "native ABI differs")
    _require(report["evidence_manifest"] == "SHA256.json", "manifest name differs")
    _require(report["evidence_format"] == EVIDENCE_FORMAT, "manifest format differs")
    _require(
        report["checks"] == {
            "receipt_bound_native_load": True,
            "native_original_input_observer": True,
            "independent_python_replay": True,
            "tampered_output_rejected": True,
            "proof_kinds": [1, 2, 3],
        },
        "self-test checks differ",
    )

    build = report["build"]
    _require(
        type(build) is dict
        and set(build) == {
            "source", "library", "receipt", "source_sha256", "library_sha256",
            "compiler", "sanitizer",
        },
        "self-test build fields differ",
    )
    _require(build["source"] == "native/exact_binary.cpp", "retained source path differs")
    _require(build["receipt"] == "native/build.json", "build receipt path differs")
    source_path = _artifact(root, entries, build["source"])
    library_path = _artifact(root, entries, build["library"])
    receipt_path = _artifact(root, entries, build["receipt"])
    expected_artifacts = {
        "selftest.json",
        build["source"],
        build["library"],
        build["receipt"],
    }
    receipt = _load_json(receipt_path, max_bytes)
    expected_receipt_fields = {
        "command", "compiler", "elapsed_ns", "exact_api_version", "library_name",
        "library_sha256", "native_abi", "platform", "python_executable",
        "python_implementation", "python_version", "sanitizer", "soabi",
        "source_sha256", "stderr", "stdout",
    }
    _require(set(receipt) == expected_receipt_fields, "build receipt fields differ")
    _require(
        type(receipt["command"]) is list
        and bool(receipt["command"])
        and all(type(item) is str and bool(item) for item in receipt["command"]),
        "build command differs",
    )
    for field in (
        "compiler", "library_name", "platform", "python_executable",
        "python_implementation", "python_version", "sanitizer", "soabi",
        "source_sha256", "library_sha256", "stderr", "stdout",
    ):
        _require(type(receipt[field]) is str, f"receipt {field} differs")
    _require(
        type(receipt["elapsed_ns"]) is int and receipt["elapsed_ns"] >= 0,
        "receipt elapsed time differs",
    )
    _require(receipt["exact_api_version"] == EXACT_API_VERSION, "receipt API differs")
    _require(receipt["native_abi"] == EXACT_NATIVE_ABI, "receipt ABI differs")
    _require(receipt["python_implementation"] == "CPython", "receipt Python implementation differs")
    _require(report["python"] == receipt["python_version"], "reported Python version differs")
    _require(
        report["python_executable"] == receipt["python_executable"],
        "reported Python executable differs",
    )
    _require(report["platform"] == receipt["platform"], "reported platform differs")
    _require(receipt["library_name"] == library_path.name, "receipt library name differs")
    normalized_command = [item.replace("\\", "/") for item in receipt["command"]]
    _require(
        any(item.endswith("/exact_binary.cpp") for item in normalized_command),
        "build command does not identify the exact native source",
    )
    _require(
        any(receipt["library_name"] in item for item in receipt["command"]),
        "build command does not identify the retained library",
    )
    _require(receipt["sanitizer"] == "none", "self-test build must be unsanitized")
    _require(receipt["source_sha256"] == build["source_sha256"], "source hash differs")
    _require(receipt["library_sha256"] == build["library_sha256"], "receipt hash differs")
    _require(sha256_file(source_path) == build["source_sha256"], "retained source hash differs")
    _require(sha256_file(library_path) == build["library_sha256"], "native binary hash differs")
    compiler_lines = [
        line.strip() for line in receipt["compiler"].splitlines() if line.strip()
    ]
    _require(
        build["compiler"] == (compiler_lines[0] if compiler_lines else ""),
        "reported compiler differs",
    )
    _require(build["sanitizer"] == receipt["sanitizer"], "reported sanitizer differs")

    case_rows = report["cases"]
    specs = _case_specs()
    _require(type(case_rows) is list and len(case_rows) == len(specs), "case inventory differs")
    total_queries = total_sat = total_unsat = 0
    observed_kinds: set[int] = set()
    for row, spec in zip(case_rows, specs):
        (name, expected_problem, expected_queries, expected_codes,
         expected_kinds, expected_mode, expected_reason) = spec
        expected_case_fields = {
            "name", "vertices", "edges", "queries", "sat", "unsat",
            "proof_kinds", "selected_mode", "auto_policy", "auto_policy_reason",
            "packet", "packet_bytes", "packet_sha256", "output", "output_bytes",
            "output_sha256", "independent_replay", "tampered_output_rejected",
        }
        _require(type(row) is dict and set(row) == expected_case_fields, f"{name}: fields differ")
        _require(row["name"] == name, f"{name}: name differs")
        _require(row["packet"] == f"cases/{name}.sbc1", f"{name}: packet path differs")
        _require(row["output"] == f"cases/{name}.scq2", f"{name}: output path differs")
        packet_path = _artifact(root, entries, row["packet"])
        output_path = _artifact(root, entries, row["output"])
        expected_artifacts.update((row["packet"], row["output"]))
        packet = _read_bounded(packet_path, max_bytes)
        output = _read_bounded(output_path, max_bytes)
        _require(len(packet) == row["packet_bytes"], f"{name}: packet size differs")
        _require(_sha256(packet) == row["packet_sha256"], f"{name}: packet hash differs")
        _require(len(output) == row["output_bytes"], f"{name}: output size differs")
        _require(_sha256(output) == row["output_sha256"], f"{name}: output hash differs")
        _require(
            packet == pack_session_packet(expected_problem, expected_queries),
            f"{name}: deterministic request bank differs",
        )
        problem, queries = unpack_session_packet(packet)
        _require(problem == expected_problem and queries == expected_queries, f"{name}: packet differs")
        _require(verify_session_output(packet, output), f"{name}: independent replay failed")
        _require(bool(output), f"{name}: output is unexpectedly empty")
        tampered = bytes((output[0] ^ 1,)) + output[1:]
        _require(
            not verify_session_output(packet, tampered),
            f"{name}: audit accepted a mutated output",
        )
        decoded = decode_certified_output(output, problem.n, len(queries))
        codes = tuple(code for code, _labels, _proof in decoded)
        kinds = tuple(
            None if code else decode_proof(proof, problem.n)[0]
            for code, _labels, proof in decoded
        )
        _require(codes == expected_codes, f"{name}: decisions differ")
        _require(kinds == expected_kinds, f"{name}: proof kinds differ")
        _require(row["proof_kinds"] == list(kinds), f"{name}: reported proof kinds differ")
        _require(row["vertices"] == problem.n, f"{name}: vertex count differs")
        _require(row["edges"] == len(problem.edges), f"{name}: edge count differs")
        _require(row["queries"] == len(queries), f"{name}: query count differs")
        _require(row["sat"] == codes.count(1), f"{name}: SAT count differs")
        _require(row["unsat"] == codes.count(0), f"{name}: UNSAT count differs")
        _require(row["selected_mode"] == expected_mode, f"{name}: mode differs")
        _require(row["auto_policy"] == "structural-v11", f"{name}: policy differs")
        _require(
            row["auto_policy_reason"] == expected_reason,
            f"{name}: policy reason differs",
        )
        _require(row["independent_replay"] is True, f"{name}: replay receipt differs")
        _require(row["tampered_output_rejected"] is True, f"{name}: tamper receipt differs")
        total_queries += len(queries)
        total_sat += codes.count(1)
        total_unsat += codes.count(0)
        observed_kinds.update(kind for kind in kinds if kind is not None)

    _require(sorted(observed_kinds) == [1, 2, 3], "audited proof coverage differs")
    _require(set(entries) == expected_artifacts, "self-test artifact inventory differs")
    _require_no_native_module()
    return {
        "schema": AUDIT_SCHEMA,
        "verified": True,
        "selftest_schema": SCHEMA,
        "origin_spectra_version": report["spectra_version"],
        "auditor_spectra_version": __version__,
        "exact_api_version": EXACT_API_VERSION,
        "native_abi": EXACT_NATIVE_ABI,
        "cases": len(case_rows),
        "queries": total_queries,
        "sat": total_sat,
        "unsat": total_unsat,
        "proof_kinds": sorted(observed_kinds),
        "selected_modes": {
            row["name"]: row["selected_mode"] for row in case_rows
        },
        "selector_reasons": {
            row["name"]: row["auto_policy_reason"] for row in case_rows
        },
        "tamper_checks": len(case_rows),
        "manifest_artifacts": manifest_result["artifacts"],
        "source_sha256": build["source_sha256"],
        "library_sha256": build["library_sha256"],
        "native_execution": False,
        "native_module_loaded": False,
    }
