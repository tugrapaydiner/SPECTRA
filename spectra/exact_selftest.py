"""Deterministic release/replication smoke test for the installed exact service.

This module proves that the installed package can explicitly build its bundled
native runtime, return checked SAT and UNSAT answers, replay every answer without
native code, reject a mutated answer, and retain a portable hash manifest.

It is intentionally not a benchmark, an independent scientific replication, or a
deployment/adoption claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

from . import __version__
from .evidence import FORMAT as EVIDENCE_FORMAT, capture, verify, write_new
from .exact import (
    DEFAULT_BYTES,
    EXACT_API_VERSION,
    EXACT_NATIVE_ABI,
    ExactProblem,
    ExactRuntime,
    build_exact_runtime,
    decode_certified_output,
    pack_session_packet,
    verify_session_output,
)
from .exact_proof import decode_proof

SCHEMA = "spectra.exact_selftest.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _run_case(
    root: Path,
    runtime: ExactRuntime,
    *,
    name: str,
    problem: ExactProblem,
    queries: tuple[tuple[tuple[int, int], ...], ...],
    expected_codes: tuple[int, ...],
    expected_kinds: tuple[int | None, ...],
    max_bytes: int,
) -> dict:
    packet = pack_session_packet(problem, queries)
    result = runtime.solve_session_packet(packet, mode="auto", max_bytes=max_bytes)
    _require(result.query_count == len(queries), f"{name}: query count differs")
    _require(result.sat_count == expected_codes.count(1), f"{name}: SAT count differs")
    _require(result.unsat_count == expected_codes.count(0), f"{name}: UNSAT count differs")
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
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)

    library = build_exact_runtime(root / "native", compiler=compiler)
    runtime = ExactRuntime(library)

    mixed_problem = ExactProblem.create(
        2,
        2,
        ((0, 1),),
        (0b11, 0b11),
    )
    mixed_queries = (
        (),
        ((0, 0b01),),
        ((0, 0b01), (1, 0b01)),
        ((0, 0),),
    )
    mixed = _run_case(
        root,
        runtime,
        name="mixed-current-query",
        problem=mixed_problem,
        queries=mixed_queries,
        expected_codes=(1, 1, 0, 0),
        expected_kinds=(None, None, 2, 1),
        max_bytes=max_bytes,
    )

    static_problem = ExactProblem.create(
        2,
        1,
        ((0, 1),),
        (0b1, 0b1),
    )
    static = _run_case(
        root,
        runtime,
        name="base-static-contradiction",
        problem=static_problem,
        queries=((),),
        expected_codes=(0,),
        expected_kinds=(3,),
        max_bytes=max_bytes,
    )

    proof_kinds = sorted(
        {
            kind
            for case in (mixed, static)
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
        "scope": (
            "deterministic installed exact-service build and semantic self-test; "
            "not a benchmark, scientific replication, or deployment claim"
        ),
        "spectra_version": __version__,
        "exact_api_version": EXACT_API_VERSION,
        "native_abi": EXACT_NATIVE_ABI,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "module": str(Path(__file__).resolve()),
        "build": {
            "library": library.relative_to(root).as_posix(),
            "receipt": receipt_path.relative_to(root).as_posix(),
            "source_sha256": receipt["source_sha256"],
            "library_sha256": receipt["library_sha256"],
            "compiler": compiler_lines[0] if compiler_lines else "",
            "sanitizer": receipt["sanitizer"],
        },
        "cases": [mixed, static],
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
        if path.is_file() and path.name != "SHA256.json"
    )
    manifest = capture(root, names)
    verify(root, manifest, required=names)
    write_new(root / "SHA256.json", manifest)
    return report
