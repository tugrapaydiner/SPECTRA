"""Receipt-bound native learned-clause residual solver for SPECTRA-G.

The native kernel accepts arbitrary canonical CNF clauses, runs deterministic
CDCL with watched literals, first-UIP learning, phase saving and restarts, and
returns either a complete model, ``UNKNOWN`` at a resource budget, or a retained
DRUP-subset clause-addition proof. Python independently checks every SAT model
and every RUP proof. The engine can run directly on the exact original CNF or
behind the replayable binary frontend.

This is a development engine.  Passing these contracts does not imply SAT
Competition leadership, and the simple proof checker is not a substitute for an
official competition checker.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import tempfile
import time
from typing import Iterable

from .binary_core import (
    BinaryCore,
    build_binary_core,
    canonical_cnf,
    model_satisfies,
    verify_binary_core,
)

Clause = tuple[int, ...]
SOURCE = Path(__file__).with_name("native_cdcl.cpp")
BUILD_SCHEMA = "spectra.general_native_cdcl_build.v1"
RESULT_SCHEMA = "spectra.general_native_cdcl.v1"
PROOF_SCHEMA = "spectra.general_drup.v1"
_MAGIC = b"SGC1"
_HEADER = struct.Struct("<4sIIQQQQQQQQ")
_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")


@dataclass(frozen=True)
class NativeStats:
    decisions: int
    propagations: int
    conflicts: int
    restarts: int
    learned_clauses: int
    learned_literals: int
    reductions: int
    deleted_clauses: int


@dataclass(frozen=True)
class DRUPProof:
    schema: str
    additions: tuple[Clause, ...]


@dataclass(frozen=True)
class NativeSolveResult:
    schema: str
    status: str
    model: tuple[bool, ...]
    core: BinaryCore
    proof: DRUPProof | None
    reason: str
    max_conflicts: int
    max_learned_literals: int
    stats: NativeStats
    residual_variables: int
    residual_clauses: int
    proof_clauses: int
    proof_literals: int
    elapsed_ns: int


@dataclass(frozen=True)
class DirectNativeResult:
    """One checked solve of the exact original canonical CNF."""

    schema: str
    status: str
    model: tuple[bool, ...]
    proof: DRUPProof | None
    reason: str
    max_conflicts: int
    max_learned_literals: int
    stats: NativeStats
    variables: int
    clauses: int
    proof_clauses: int
    proof_literals: int
    elapsed_ns: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def build_native_cdcl(
    directory: str | Path,
    *,
    compiler: str | None = None,
    sanitizer: str = "none",
) -> Path:
    """Compile the native solver into a new receipt-bound directory."""
    if sanitizer not in ("none", "undefined", "address,undefined"):
        raise ValueError("unsupported sanitizer")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    suffix = ".dll" if os.name == "nt" else ".so"
    library = destination / ("spectra_g_cdcl" + suffix)
    receipt = destination / "build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("native CDCL outputs already exist")

    if os.name == "nt":
        compiler = compiler or "cl"
        resolved = shutil.which(compiler)
        if resolved is None:
            raise FileNotFoundError(compiler)
        if sanitizer != "none":
            raise NotImplementedError("sanitizer builds currently require clang++ or g++")
        command = [
            resolved,
            "/nologo",
            "/std:c++17",
            "/O2",
            "/EHsc",
            "/LD",
            str(SOURCE.resolve()),
            "/link",
            "/OUT:" + str(library),
        ]
        version_command = [resolved]
    else:
        compiler = compiler or "g++"
        resolved = shutil.which(compiler)
        if resolved is None:
            raise FileNotFoundError(compiler)
        flags = [
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-fvisibility=hidden",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fno-fast-math",
            "-ffp-contract=off",
        ]
        if sanitizer == "none":
            flags.extend(("-O3", "-DNDEBUG"))
        else:
            flags.extend((
                "-O1",
                "-g",
                "-fno-omit-frame-pointer",
                "-fsanitize=" + sanitizer,
                "-fno-sanitize-recover=all",
            ))
        command = [resolved, *flags, str(SOURCE.resolve()), "-o", str(library)]
        version_command = [resolved, "--version"]

    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        completed = _run(command, cwd=Path(temporary))
        if not library.is_file():
            raise RuntimeError("native CDCL library was not produced")
        try:
            version = _run(version_command, cwd=Path(temporary), timeout=30)
            compiler_text = version.stdout + version.stderr
        except RuntimeError as error:
            compiler_text = str(error)
    record = {
        "schema": BUILD_SCHEMA,
        "source": SOURCE.name,
        "source_sha256": _sha256(SOURCE),
        "library": library.name,
        "library_sha256": _sha256(library),
        "compiler": resolved,
        "compiler_version": compiler_text,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "sanitizer": sanitizer,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    try:
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except BaseException:
        library.unlink(missing_ok=True)
        receipt.unlink(missing_ok=True)
        raise
    return library


def _check_receipt(library: Path) -> dict:
    receipt = library.with_name("build.json")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    required = {
        "schema", "source", "source_sha256", "library", "library_sha256",
        "compiler", "compiler_version", "command", "stdout", "stderr",
        "sanitizer", "platform", "machine", "elapsed_ns",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("native CDCL build receipt differs")
    if value["schema"] != BUILD_SCHEMA or value["source"] != SOURCE.name:
        raise ValueError("native CDCL source identity differs")
    if value["source_sha256"] != _sha256(SOURCE):
        raise ValueError("native CDCL source hash differs")
    if value["library"] != library.name or value["library_sha256"] != _sha256(library):
        raise ValueError("native CDCL library hash differs")
    return value


def _uint64(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise ValueError(f"{label} must be an exact uint64")
    return value


def _pack_cnf(nvars: int, clauses: Iterable[Iterable[int]]):
    canonical = canonical_cnf(nvars, clauses)
    flattened: list[int] = []
    offsets = [0]
    for row in canonical:
        flattened.extend(row)
        offsets.append(len(flattened))
    literal_array = (ctypes.c_int32 * len(flattened))(*flattened)
    offset_array = (ctypes.c_uint64 * len(offsets))(*offsets)
    return canonical, literal_array, offset_array


def _read_u32(data: bytes, offset: int) -> tuple[int, int]:
    if offset + _U32.size > len(data):
        raise ValueError("truncated native result")
    return _U32.unpack_from(data, offset)[0], offset + _U32.size


def _decode(data: bytes) -> tuple[int, int, NativeStats, tuple[bool, ...], DRUPProof | None]:
    if len(data) < _HEADER.size:
        raise ValueError("native result is truncated")
    (
        magic, status, nvars, decisions, propagations, conflicts, restarts,
        learned_clauses, learned_literals, reductions, deleted_clauses,
    ) = _HEADER.unpack_from(data)
    if magic != _MAGIC or status not in (1, 2, 10, 20):
        raise ValueError("native result header differs")
    stats = NativeStats(
        decisions, propagations, conflicts, restarts,
        learned_clauses, learned_literals, reductions, deleted_clauses,
    )
    offset = _HEADER.size
    model: tuple[bool, ...] = ()
    proof: DRUPProof | None = None
    if status == 10:
        if offset + nvars != len(data):
            raise ValueError("native SAT model length differs")
        raw = data[offset:]
        if any(value not in (0, 1) for value in raw):
            raise ValueError("native SAT model contains a non-Boolean byte")
        model = tuple(bool(value) for value in raw)
        offset = len(data)
    elif status == 20:
        count, offset = _read_u32(data, offset)
        additions: list[Clause] = []
        literal_total = 0
        for _ in range(count):
            width, offset = _read_u32(data, offset)
            literal_total += width
            if literal_total > 100_000_000:
                raise ValueError("native proof literal count is excessive")
            size = width * _I32.size
            if offset + size > len(data):
                raise ValueError("native proof clause is truncated")
            row = tuple(_I32.unpack_from(data, offset + index * 4)[0]
                        for index in range(width))
            offset += size
            additions.append(row)
        if offset != len(data):
            raise ValueError("native UNSAT result has trailing bytes")
        proof = DRUPProof(PROOF_SCHEMA, tuple(additions))
    elif offset != len(data):
        raise ValueError("native UNKNOWN result has trailing bytes")
    return status, nvars, stats, model, proof


class NativeCDCLRuntime:
    """Receipt-bound ctypes wrapper for the native residual engine."""

    def __init__(self, library: str | Path):
        self.path = Path(library).resolve(strict=True)
        self.receipt = _check_receipt(self.path)
        self._library = ctypes.CDLL(str(self.path))
        self._library.spectra_g_native_signature.argtypes = []
        self._library.spectra_g_native_signature.restype = ctypes.c_char_p
        self._library.spectra_g_native_error.argtypes = []
        self._library.spectra_g_native_error.restype = ctypes.c_char_p
        self._library.spectra_g_native_free.argtypes = [ctypes.c_void_p]
        self._library.spectra_g_native_free.restype = None
        self._library.spectra_g_native_solve.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._library.spectra_g_native_solve.restype = ctypes.c_int
        raw = self._library.spectra_g_native_signature()
        self.signature = raw.decode("utf-8", "replace") if raw else ""
        if self.signature != "SPECTRA-G native CDCL v1":
            raise RuntimeError("native CDCL signature differs")

    def solve_residual(
        self,
        nvars: int,
        clauses: Iterable[Iterable[int]],
        *,
        max_conflicts: int = 1_000_000,
        max_learned_literals: int = 100_000_000,
    ) -> tuple[int, tuple[bool, ...], DRUPProof | None, NativeStats, tuple[Clause, ...]]:
        max_conflicts = _uint64(max_conflicts, "max_conflicts")
        max_learned_literals = _uint64(
            max_learned_literals, "max_learned_literals"
        )
        canonical, literals, offsets = _pack_cnf(nvars, clauses)
        pointer = ctypes.c_void_p()
        size = ctypes.c_size_t()
        code = self._library.spectra_g_native_solve(
            nvars,
            literals,
            len(literals),
            offsets,
            len(canonical),
            max_conflicts,
            max_learned_literals,
            ctypes.byref(pointer),
            ctypes.byref(size),
        )
        if code:
            raw = self._library.spectra_g_native_error()
            message = raw.decode("utf-8", "replace") if raw else "native failure"
            raise RuntimeError(message)
        try:
            data = ctypes.string_at(pointer, size.value)
        finally:
            if pointer.value:
                self._library.spectra_g_native_free(pointer)
        status, observed_nvars, stats, model, proof = _decode(data)
        if observed_nvars != nvars:
            raise RuntimeError("native result variable count differs")
        return status, model, proof, stats, canonical


def _assign(values: list[int], literal: int) -> bool:
    variable = abs(literal)
    sign = 1 if literal > 0 else -1
    if values[variable] == -sign:
        return False
    values[variable] = sign
    return True


def _rup_conflict(clauses: tuple[Clause, ...], assumption_clause: Clause, nvars: int) -> bool:
    values = [0] * (nvars + 1)
    for literal in assumption_clause:
        if not _assign(values, -literal):
            return True
    while True:
        forced: dict[int, int] = {}
        for row in clauses:
            satisfied = False
            open_literal = 0
            open_count = 0
            for literal in row:
                value = values[abs(literal)]
                if value and value == (1 if literal > 0 else -1):
                    satisfied = True
                    break
                if value == 0:
                    open_literal = literal
                    open_count += 1
            if satisfied:
                continue
            if open_count == 0:
                return True
            if open_count == 1:
                variable = abs(open_literal)
                sign = 1 if open_literal > 0 else -1
                previous = forced.get(variable)
                if previous is not None and previous != sign:
                    return True
                forced[variable] = sign
        if not forced:
            return False
        changed = False
        for variable in sorted(forced):
            sign = forced[variable]
            if values[variable] == -sign:
                return True
            if values[variable] == 0:
                values[variable] = sign
                changed = True
        if not changed:
            return False


def verify_drup_proof(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    proof: DRUPProof,
    *,
    max_clauses: int = 5_000_000,
    max_literals: int = 100_000_000,
) -> bool:
    """Replay a clause-addition DRUP subset; the final addition must be empty."""
    if (
        type(proof) is not DRUPProof
        or proof.schema != PROOF_SCHEMA
        or type(proof.additions) is not tuple
        or not proof.additions
        or type(max_clauses) is not int
        or type(max_literals) is not int
        or not 0 <= len(proof.additions) <= max_clauses
    ):
        return False
    try:
        database = list(canonical_cnf(nvars, clauses))
    except (TypeError, ValueError):
        return False
    literal_total = 0
    for index, raw in enumerate(proof.additions):
        if type(raw) is not tuple:
            return False
        literal_total += len(raw)
        if literal_total > max_literals:
            return False
        try:
            normalized = canonical_cnf(nvars, (raw,))
        except (TypeError, ValueError):
            return False
        if raw:
            if len(normalized) != 1 or len(normalized[0]) != len(raw):
                return False
            row = normalized[0]
        else:
            if normalized != ((),):
                return False
            row = ()
        if not _rup_conflict(tuple(database), row, nvars):
            return False
        database.append(row)
        if not row and index != len(proof.additions) - 1:
            return False
    return proof.additions[-1] == ()


def solve_cnf_native(
    runtime: NativeCDCLRuntime,
    nvars: int,
    clauses: Iterable[Iterable[int]],
    *,
    max_conflicts: int = 1_000_000,
    max_learned_literals: int = 100_000_000,
) -> DirectNativeResult:
    """Solve the exact original canonical CNF without preprocessing composition."""
    if type(runtime) is not NativeCDCLRuntime:
        raise TypeError("NativeCDCLRuntime required")
    max_conflicts = _uint64(max_conflicts, "max_conflicts")
    max_learned_literals = _uint64(
        max_learned_literals, "max_learned_literals"
    )
    canonical = canonical_cnf(nvars, clauses)
    started = time.perf_counter_ns()
    status_code, model, proof, stats, observed = runtime.solve_residual(
        nvars,
        canonical,
        max_conflicts=max_conflicts,
        max_learned_literals=max_learned_literals,
    )
    if observed != canonical:
        raise AssertionError("native original-CNF canonicalization differs")
    status = {
        1: "UNKNOWN",
        2: "UNKNOWN",
        10: "SAT_VERIFIED",
        20: "UNSAT_VERIFIED",
    }[status_code]
    reason = {
        1: "conflict_budget",
        2: "learned_literal_budget",
        10: "model",
        20: "drup",
    }[status_code]
    result = DirectNativeResult(
        schema="spectra.general_native_direct.v1",
        status=status,
        model=model if status_code == 10 else (),
        proof=proof,
        reason=reason,
        max_conflicts=max_conflicts,
        max_learned_literals=max_learned_literals,
        stats=stats,
        variables=nvars,
        clauses=len(canonical),
        proof_clauses=0 if proof is None else len(proof.additions),
        proof_literals=0 if proof is None else sum(map(len, proof.additions)),
        elapsed_ns=time.perf_counter_ns() - started,
    )
    if result.status != "UNKNOWN" and not verify_direct_native_result(
        nvars, canonical, result
    ):
        raise AssertionError("direct native SPECTRA-G result failed independent replay")
    return result


def _valid_native_evidence(result: DirectNativeResult | NativeSolveResult) -> bool:
    """Validate receipt containers and ABI counters before inspecting payloads."""
    if (
        type(result.stats) is not NativeStats
        or type(result.model) is not tuple
        or any(type(value) is not bool for value in result.model)
    ):
        return False
    if any(type(value) is not int or not 0 <= value < 1 << 64 for value in (
        result.max_conflicts, result.max_learned_literals,
        result.stats.decisions, result.stats.propagations, result.stats.conflicts,
        result.stats.restarts, result.stats.learned_clauses,
        result.stats.learned_literals, result.stats.reductions,
        result.stats.deleted_clauses,
    )):
        return False
    if (
        any(type(value) is not int or value < 0 for value in (
            result.proof_clauses, result.proof_literals, result.elapsed_ns,
        ))
        or result.stats.learned_literals > result.max_learned_literals
    ):
        return False
    proof = result.proof
    if proof is None:
        return result.proof_clauses == result.proof_literals == 0
    if (
        type(proof) is not DRUPProof
        or proof.schema != PROOF_SCHEMA
        or type(proof.additions) is not tuple
        or any(type(row) is not tuple for row in proof.additions)
    ):
        return False
    return (
        result.proof_clauses == len(proof.additions)
        and result.proof_literals == sum(map(len, proof.additions))
    )


def _valid_solved_native_statistics(
    result: DirectNativeResult | NativeSolveResult,
) -> bool:
    if result.status == "SAT_VERIFIED":
        return result.stats.conflicts <= result.max_conflicts
    # Root conflicts terminate before the budget check. An UNSAT proof retains
    # every learned clause and adds exactly one final empty clause.
    return (
        result.stats.conflicts <= result.max_conflicts + 1
        and result.proof_clauses == result.stats.learned_clauses + 1
        and result.proof_literals == result.stats.learned_literals
    )


def _valid_unknown_budget(
    result: DirectNativeResult | NativeSolveResult, native_variables: int,
) -> bool:
    """Check consistency with native exit conditions, not a replay of the search."""
    if result.model != () or result.proof is not None:
        return False
    if result.reason == "conflict_budget":
        # The kernel stops at the first conflict exceeding the configured limit.
        return result.stats.conflicts == result.max_conflicts + 1
    if result.reason == "learned_literal_budget":
        # This check follows conflict admission and precedes the learned addition.
        # A learned clause contains at most one literal per native variable.
        return (
            0 < result.stats.conflicts <= result.max_conflicts
            and result.max_learned_literals - result.stats.learned_literals
            < native_variables
        )
    return False


def verify_direct_native_result(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    result: DirectNativeResult,
) -> bool:
    """Independently check a direct native result against the original CNF."""
    if type(result) is not DirectNativeResult:
        return False
    try:
        canonical = canonical_cnf(nvars, clauses)
    except (TypeError, ValueError):
        return False
    if (
        result.schema != "spectra.general_native_direct.v1"
        or result.status not in ("SAT_VERIFIED", "UNSAT_VERIFIED", "UNKNOWN")
        or not _valid_native_evidence(result)
        or type(result.variables) is not int
        or type(result.clauses) is not int
        or result.variables != nvars
        or result.clauses != len(canonical)
    ):
        return False
    if result.status == "SAT_VERIFIED":
        return (
            result.reason == "model"
            and _valid_solved_native_statistics(result)
            and result.proof is None
            and len(result.model) == nvars
            and model_satisfies(nvars, canonical, result.model)
        )
    if result.status == "UNSAT_VERIFIED":
        return (
            result.reason == "drup"
            and _valid_solved_native_statistics(result)
            and result.model == ()
            and result.proof is not None
            and verify_drup_proof(
                nvars,
                canonical,
                result.proof,
                max_clauses=max(result.proof_clauses, 1),
                max_literals=max(result.proof_literals, 1),
            )
        )
    return _valid_unknown_budget(result, nvars)


def write_drup(path: str | Path, proof: DRUPProof) -> Path:
    """Write the retained addition-only proof in standard text DRAT syntax."""
    if (
        type(proof) is not DRUPProof
        or proof.schema != PROOF_SCHEMA
        or not proof.additions
        or proof.additions[-1] != ()
    ):
        raise ValueError("a complete DRUP proof ending in the empty clause is required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="ascii", newline="\n") as stream:
        for row in proof.additions:
            if row:
                stream.write(" ".join(map(str, row)) + " ")
            stream.write("0\n")
    return destination


def solve_general_native(
    runtime: NativeCDCLRuntime,
    nvars: int,
    clauses: Iterable[Iterable[int]],
    *,
    max_conflicts: int = 1_000_000,
    max_learned_literals: int = 100_000_000,
) -> NativeSolveResult:
    """Apply the verified binary core and solve the arbitrary-width residual."""
    if type(runtime) is not NativeCDCLRuntime:
        raise TypeError("NativeCDCLRuntime required")
    max_conflicts = _uint64(max_conflicts, "max_conflicts")
    max_learned_literals = _uint64(
        max_learned_literals, "max_learned_literals"
    )
    canonical = canonical_cnf(nvars, clauses)
    started = time.perf_counter_ns()
    core = build_binary_core(nvars, canonical)
    zero = NativeStats(0, 0, 0, 0, 0, 0, 0, 0)
    if core.contradiction:
        result = NativeSolveResult(
            RESULT_SCHEMA, "UNSAT_VERIFIED", (), core, None, "binary_core",
            max_conflicts, max_learned_literals, zero, 0, 1, 0, 0,
            time.perf_counter_ns() - started,
        )
    elif not core.residual_clauses:
        model = core.lift((False,) * core.reduced_nvars)
        result = NativeSolveResult(
            RESULT_SCHEMA, "SAT_VERIFIED", model, core, None,
            "binary_core_model", max_conflicts, max_learned_literals, zero,
            core.reduced_nvars, 0, 0, 0, time.perf_counter_ns() - started,
        )
    else:
        status_code, residual_model, proof, stats, residual = runtime.solve_residual(
            core.reduced_nvars,
            core.residual_clauses,
            max_conflicts=max_conflicts,
            max_learned_literals=max_learned_literals,
        )
        if residual != core.residual_clauses:
            raise AssertionError("native residual canonicalization differs")
        status = {
            1: "UNKNOWN",
            2: "UNKNOWN",
            10: "SAT_VERIFIED",
            20: "UNSAT_VERIFIED",
        }[status_code]
        model = core.lift(residual_model) if status_code == 10 else ()
        reason = {
            1: "conflict_budget",
            2: "learned_literal_budget",
            10: "model",
            20: "drup",
        }[status_code]
        result = NativeSolveResult(
            RESULT_SCHEMA,
            status,
            model,
            core,
            proof,
            reason,
            max_conflicts,
            max_learned_literals,
            stats,
            core.reduced_nvars,
            len(core.residual_clauses),
            0 if proof is None else len(proof.additions),
            0 if proof is None else sum(map(len, proof.additions)),
            time.perf_counter_ns() - started,
        )
    if result.status != "UNKNOWN" and not verify_native_result(nvars, canonical, result):
        raise AssertionError("native SPECTRA-G result failed independent replay")
    return result


def verify_native_result(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    result: NativeSolveResult,
) -> bool:
    if type(result) is not NativeSolveResult:
        return False
    try:
        canonical = canonical_cnf(nvars, clauses)
    except (TypeError, ValueError):
        return False
    if (
        result.schema != RESULT_SCHEMA
        or result.status not in ("SAT_VERIFIED", "UNSAT_VERIFIED", "UNKNOWN")
        or not _valid_native_evidence(result)
        or any(type(value) is not int or value < 0 for value in (
            result.residual_variables,
            result.residual_clauses,
        ))
        or not verify_binary_core(nvars, canonical, result.core)
        or result.residual_variables != result.core.reduced_nvars
        or result.residual_clauses != len(result.core.residual_clauses)
    ):
        return False
    if result.core.contradiction or not result.core.residual_clauses:
        if (
            result.stats != NativeStats(0, 0, 0, 0, 0, 0, 0, 0)
            or result.proof is not None
        ):
            return False
        if result.core.contradiction:
            return (
                result.status == "UNSAT_VERIFIED"
                and result.reason == "binary_core"
                and result.model == ()
            )
        return (
            result.status == "SAT_VERIFIED"
            and result.reason == "binary_core_model"
            and len(result.model) == nvars
            and model_satisfies(nvars, canonical, result.model)
        )
    if result.status == "SAT_VERIFIED":
        return (
            result.reason == "model"
            and _valid_solved_native_statistics(result)
            and result.proof is None
            and len(result.model) == nvars
            and model_satisfies(nvars, canonical, result.model)
        )
    if result.status == "UNSAT_VERIFIED":
        if result.model != ():
            return False
        return (
            result.reason == "drup"
            and _valid_solved_native_statistics(result)
            and result.proof is not None
            and verify_drup_proof(
                result.core.reduced_nvars,
                result.core.residual_clauses,
                result.proof,
                max_clauses=max(result.proof_clauses, 1),
                max_literals=max(result.proof_literals, 1),
            )
        )
    return _valid_unknown_budget(result, result.residual_variables)
