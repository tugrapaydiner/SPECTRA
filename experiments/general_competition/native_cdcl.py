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

from array import array
import ctypes
from dataclasses import dataclass
import hashlib
from itertools import chain
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

from spectra._native_build import native_load_guard, publish_build, snapshot_source

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


@dataclass(frozen=True)
class OriginalRUPComposition:
    """Unverified original-CNF proof, with disjoint frontend/backend counts.

    Constructing this object is not proof verification. The complete stream must
    be checked against the original CNF under an independent resource budget.
    """

    proof: DRUPProof
    prefix_clauses: int
    prefix_literals: int
    backend_clauses: int
    backend_literals: int
    frontend_terminal_clauses: int
    proof_bytes: int


class _OriginalProofBudgetExceeded(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _compose_native_original_rup(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    reduction,
    backend_proof: DRUPProof | None = None,
    *,
    max_proof_clauses: int = 5_000_000,
    max_proof_literals: int = 100_000_000,
    max_proof_bytes: int = 1_073_741_824,
) -> OriginalRUPComposition:
    """Compose new S/R clauses and retained native RUP additions without certifying.

    This private helper is for the own native backend only, not arbitrary RAT
    composition. No native solve, proof replay, file write, or fallback occurs.
    An endpoint using it must supervise construction and independent checking.
    """
    from .native_binary_core import NativeReduction

    canonical = canonical_cnf(nvars, clauses)
    limits = (
        _uint64(max_proof_clauses, "max_proof_clauses"),
        _uint64(max_proof_literals, "max_proof_literals"),
        _uint64(max_proof_bytes, "max_proof_bytes"),
    )
    if type(reduction) is not NativeReduction:
        raise TypeError("NativeReduction required")
    if (
        reduction.schema != "spectra.general_native_binary_core.v1"
        or type(reduction.contradiction) is not bool
        or reduction.conflict_kind not in ("", "binary", "unit")
        or reduction.contradiction != bool(reduction.conflict_kind)
        or reduction.original_nvars != nvars
        or any(type(value) is not int or value < 0 for value in (
            reduction.original_nvars, reduction.normalized_clause_count,
            reduction.unit_input_clauses, reduction.binary_input_clauses,
            reduction.long_input_clauses, reduction.class_count,
            reduction.reduced_nvars, reduction.conflict_variable,
            reduction.equivalent_variables_removed, reduction.propagated_classes,
        ))
        or reduction.normalized_clause_count != len(canonical)
        or reduction.unit_input_clauses != sum(len(row) == 1 for row in canonical)
        or reduction.binary_input_clauses != sum(len(row) == 2 for row in canonical)
        or reduction.long_input_clauses != sum(len(row) >= 3 for row in canonical)
        or type(reduction.mapping) is not tuple
        or len(reduction.mapping) != nvars
        or any(type(value) is not int for value in reduction.mapping)
        or type(reduction.fixed) is not tuple
        or len(reduction.fixed) != nvars
        or any(type(value) is not bool for value in reduction.fixed)
        or type(reduction.class_literals) is not tuple
        or type(reduction.residual_clauses) is not tuple
    ):
        raise ValueError("native reduction structure differs")
    if reduction.contradiction:
        if (
            backend_proof is not None
            or reduction.reduced_nvars != 0
            or reduction.residual_clauses != ((),)
            or any(reduction.mapping)
            or any(reduction.fixed)
        ):
            raise ValueError("native contradiction evidence differs")
    elif reduction.conflict_variable != 0:
        raise ValueError("noncontradictory reduction has a conflict variable")

    additions: list[Clause] = []
    available_prefix = set(canonical)
    literal_count = byte_count = 0

    def append(row: Clause, *, prefix: bool = False) -> None:
        nonlocal literal_count, byte_count
        if (
            type(row) is not tuple
            or any(type(lit) is not int or lit == 0 or abs(lit) > nvars for lit in row)
        ):
            raise ValueError("original proof clause structure differs")
        values = set(row)
        if len(values) != len(row) or any(-lit in values for lit in row):
            raise ValueError("original proof clause contains duplicates or a tautology")
        if prefix:
            # Signed residual embeddings may reorder variables. Membership is
            # by clause contents; backend rows bypass this filter entirely.
            key = tuple(sorted(row, key=lambda lit: (abs(lit), lit)))
            if key in available_prefix:
                return
        next_literals = literal_count + len(row)
        next_bytes = byte_count + sum(len(str(lit)) + 1 for lit in row) + 2
        for observed, maximum, reason in (
            (len(additions) + 1, limits[0], "proof_clause_budget"),
            (next_literals, limits[1], "proof_literal_budget"),
            (next_bytes, limits[2], "proof_byte_budget"),
        ):
            if observed > maximum:
                raise _OriginalProofBudgetExceeded(reason)
        additions.append(row)
        literal_count, byte_count = next_literals, next_bytes
        if prefix:
            available_prefix.add(key)

    def frontend_result(prefix_count: int, prefix_literals: int) -> OriginalRUPComposition:
        append(())
        return OriginalRUPComposition(
            DRUPProof(PROOF_SCHEMA, tuple(additions)), prefix_count,
            prefix_literals, 0, 0, 1, byte_count,
        )

    if () in canonical:
        if not reduction.contradiction:
            raise ValueError("native reduction ignored an original empty clause")
        return frontend_result(0, 0)
    if reduction.conflict_kind == "binary":
        if (
            not 1 <= reduction.conflict_variable <= nvars
            or reduction.class_literals != ()
            or reduction.class_count != 0
        ):
            raise ValueError("native SCC contradiction witness differs")
        append((reduction.conflict_variable,), prefix=True)
        return frontend_result(len(additions), literal_count)

    labels = reduction.class_literals
    count = reduction.class_count
    if (
        len(labels) != nvars
        or not 0 <= count <= nvars
        or any(type(label) is not int or not 1 <= abs(label) <= count for label in labels)
        or reduction.equivalent_variables_removed != nvars - count
        or not 0 <= reduction.propagated_classes <= count
    ):
        raise ValueError("native class labels differ")
    representatives = [0] * (count + 1)
    for variable, label in enumerate(labels, 1):
        index = abs(label)
        if representatives[index] == 0:
            if label < 0:
                raise ValueError("native representative orientation differs")
            representatives[index] = variable
    if (
        any(rep == 0 for rep in representatives[1:])
        or representatives[1:] != sorted(representatives[1:])
    ):
        raise ValueError("native representative coverage differs")

    def substitute(literal: int) -> int:
        label = labels[abs(literal) - 1]
        target = representatives[abs(label)]
        return target if (literal > 0) == (label > 0) else -target

    def append_substituted() -> None:
        # Validate all branch evidence before calling this function: a tiny
        # certificate budget must never mask malformed native evidence.
        # Substitute the full row first; fixed literals are never omitted from
        # S. Original and already emitted clauses remain available because the
        # composed RUP stream has no deletions, so their copies need not be sent.
        for row in canonical:
            normalized = canonical_cnf(nvars, (tuple(map(substitute, row)),))
            if normalized:
                append(normalized[0], prefix=True)
        additions.sort(key=lambda row: (len(row), row))

    if reduction.conflict_kind == "unit":
        if reduction.conflict_variable != 0:
            raise ValueError("native quotient conflict variable differs")
        append_substituted()
        return frontend_result(len(additions), literal_count)

    reduced_nvars = reduction.reduced_nvars
    if (
        not 0 <= reduced_nvars <= count
        or reduction.propagated_classes != count - reduced_nvars
    ):
        raise ValueError("native residual variable count differs")
    back = [0] * (reduced_nvars + 1)
    for representative in representatives[1:]:
        target = reduction.mapping[representative - 1]
        if abs(target) > reduced_nvars:
            raise ValueError("native mapping target is out of range")
        if target:
            if back[abs(target)]:
                raise ValueError("native residual mapping is not injective")
            back[abs(target)] = representative if target > 0 else -representative
    if any(target == 0 for target in back[1:]):
        raise ValueError("native residual mapping is incomplete")
    for variable, label in enumerate(labels, 1):
        representative = representatives[abs(label)]
        target = reduction.mapping[representative - 1]
        expected = target if label > 0 else -target
        if reduction.mapping[variable - 1] != expected:
            raise ValueError("native mapping disagrees within a class")
        if target:
            if reduction.fixed[variable - 1]:
                raise ValueError("native free-class placeholder differs")
        elif reduction.fixed[variable - 1] != (
            reduction.fixed[representative - 1] == (label > 0)
        ):
            raise ValueError("native fixed values disagree within a class")
    residual = reduction.residual_clauses
    if canonical_cnf(reduced_nvars, residual) != residual or not residual or () in residual:
        raise ValueError("noncontradictory native residual differs")
    if (
        type(backend_proof) is not DRUPProof
        or backend_proof.schema != PROOF_SCHEMA
        or type(backend_proof.additions) is not tuple
        or not backend_proof.additions
        or backend_proof.additions[-1] != ()
    ):
        raise ValueError("complete retained native RUP proof required")
    for index, row in enumerate(backend_proof.additions):
        if (
            type(row) is not tuple
            or any(type(lit) is not int or lit == 0 or abs(lit) > reduced_nvars for lit in row)
        ):
            raise ValueError("native residual proof literal differs")
        values = set(row)
        if len(values) != len(row) or any(-lit in values for lit in row):
            raise ValueError("native proof contains duplicates or a tautology")
        if not row and index != len(backend_proof.additions) - 1:
            raise ValueError("native proof contains an early empty clause")

    def embed(row: Clause) -> Clause:
        if (
            type(row) is not tuple
            or any(type(lit) is not int or lit == 0 or abs(lit) > reduced_nvars for lit in row)
        ):
            raise ValueError("native residual proof literal differs")
        return tuple(back[abs(lit)] if lit > 0 else -back[abs(lit)] for lit in row)

    append_substituted()
    for row in residual:
        append(embed(row), prefix=True)
    prefix_count, prefix_literals = len(additions), literal_count
    for index, row in enumerate(backend_proof.additions):
        if not row and index != len(backend_proof.additions) - 1:
            raise ValueError("native proof contains an early empty clause")
        append(embed(row))
    return OriginalRUPComposition(
        DRUPProof(PROOF_SCHEMA, tuple(additions)), prefix_count, prefix_literals,
        len(backend_proof.additions), literal_count - prefix_literals, 0, byte_count,
    )


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
        command_prefix = [
            resolved,
            "/nologo",
            "/std:c++17",
            "/O2",
            "/EHsc",
            "/LD",
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
        command_prefix = [resolved, *flags]
        version_command = [resolved, "--version"]

    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        target = Path(temporary) / library.name
        source, source_sha256 = snapshot_source(SOURCE, Path(temporary))
        command = [*command_prefix, str(source)]
        command += ["/link", "/OUT:" + str(target)] if os.name == "nt" else ["-o", str(target)]
        completed = _run(command, cwd=Path(temporary))
        if not target.is_file():
            raise RuntimeError("native CDCL library was not produced")
        try:
            version = _run(version_command, cwd=Path(temporary), timeout=30)
            compiler_text = version.stdout + version.stderr
        except RuntimeError as error:
            compiler_text = str(error)
        record = {
            "schema": BUILD_SCHEMA,
            "source": SOURCE.name,
            "source_sha256": source_sha256,
            "library": library.name,
            "library_sha256": _sha256(target),
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
        publish_build(target, library, record)
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
    return _pack_canonical(canonical_cnf(nvars, clauses))


def _pack_canonical(canonical: tuple[Clause, ...]):
    literals = array("i")
    offsets = array("Q", [0])
    if (literals.itemsize != ctypes.sizeof(ctypes.c_int32)
            or offsets.itemsize != ctypes.sizeof(ctypes.c_uint64)):
        raise RuntimeError("native CDCL input scalar widths differ")
    literals.extend(chain.from_iterable(canonical))
    for row in canonical:
        offsets.append(offsets[-1] + len(row))
    # Each ctypes view keeps its backing array alive through the native call.
    literal_array = (ctypes.c_int32 * len(literals)).from_buffer(literals)
    offset_array = (ctypes.c_uint64 * len(offsets)).from_buffer(offsets)
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
        with native_load_guard(self.path, self.receipt["library_sha256"]):
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
        canonical = canonical_cnf(nvars, clauses)
        return self._solve_canonical(
            nvars, canonical, max_conflicts=max_conflicts,
            max_learned_literals=max_learned_literals,
        )

    def _solve_canonical(
        self, nvars: int, canonical: tuple[Clause, ...], *,
        max_conflicts: int, max_learned_literals: int,
    ) -> tuple[int, tuple[bool, ...], DRUPProof | None, NativeStats, tuple[Clause, ...]]:
        canonical, literals, offsets = _pack_canonical(canonical)
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


class _RUPDatabase:
    """Incremental original-clause propagation for independent DRUP replay."""

    def __init__(self, nvars: int, clauses: Iterable[Clause]):
        self.nvars = nvars
        self.clauses: list[list[int]] = []
        self.watches: dict[int, list[int]] = {}
        self.units: list[int] = []
        self.has_empty_clause = False
        for row in clauses:
            self.append(row)

    def append(self, row: Clause) -> None:
        if not row:
            self.has_empty_clause = True
        elif len(row) == 1:
            self.units.append(row[0])
        else:
            index = len(self.clauses)
            self.clauses.append(list(row))
            self.watches.setdefault(row[0], []).append(index)
            self.watches.setdefault(row[1], []).append(index)

    def conflict(self, assumptions: Clause) -> bool:
        if self.has_empty_clause:
            return True
        # Watch positions remain valid when each check resets the assignment.
        values = [0] * (self.nvars + 1)
        queue: list[int] = []

        def enqueue(literal: int) -> bool:
            variable = abs(literal)
            sign = 1 if literal > 0 else -1
            if values[variable]:
                return values[variable] == sign
            values[variable] = sign
            queue.append(literal)
            return True

        for literal in assumptions:
            if not enqueue(-literal):
                return True
        for literal in self.units:
            if not enqueue(literal):
                return True
        cursor = 0
        while cursor < len(queue):
            false_literal = -queue[cursor]
            cursor += 1
            watched = self.watches.get(false_literal, [])
            index = 0
            while index < len(watched):
                clause_id = watched[index]
                row = self.clauses[clause_id]
                if row[0] == false_literal:
                    row[0], row[1] = row[1], row[0]
                assert row[1] == false_literal
                other = row[0]
                if values[abs(other)] == (1 if other > 0 else -1):
                    index += 1
                    continue
                for offset in range(2, len(row)):
                    literal = row[offset]
                    if values[abs(literal)] != (-1 if literal > 0 else 1):
                        row[1], row[offset] = literal, false_literal
                        watched[index] = watched[-1]
                        watched.pop()
                        self.watches.setdefault(literal, []).append(clause_id)
                        break
                else:
                    if not enqueue(other):
                        return True
                    index += 1
        return False


def _rup_conflict(clauses: tuple[Clause, ...], assumption_clause: Clause, nvars: int) -> bool:
    return _RUPDatabase(nvars, clauses).conflict(assumption_clause)


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
        database = _RUPDatabase(nvars, canonical_cnf(nvars, clauses))
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
        if not database.conflict(row):
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
    status_code, model, proof, stats, observed = runtime._solve_canonical(
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
