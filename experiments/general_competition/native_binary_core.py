"""Explicit native acceleration for the replayable general-CNF binary core.

The generic entry point validates and canonicalizes arbitrary clauses. Repeated
matched comparisons can instead prepare an already-canonical formula once and
use the receipt-bound canonical entry point without re-packing or re-canonicalizing it.
The Python binary core remains the proof-producing authority for retained claims.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass, field
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import Iterable

from spectra._native_build import native_load_guard, publish_build, snapshot_source

SOURCE = Path(__file__).with_name("native_binary_core.cpp")
NATIVE_ABI = 1
_HEADER = struct.Struct("<4s14I2Q")
_MAGIC = b"SGC1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_native_binary_core(
    directory: str | Path,
    *,
    compiler: str | None = None,
    sanitizer: str = "none",
) -> Path:
    """Build a receipt-bound Linux shared library without importing it."""
    if sys.platform != "linux":
        raise NotImplementedError("native general-CNF builder currently supports Linux")
    if sanitizer not in ("none", "undefined", "address,undefined"):
        raise ValueError("unsupported sanitizer")
    compiler = compiler or "g++"
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError(compiler)
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / "libspectra_general_core.so"
    receipt = destination / "build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("native general-CNF build outputs already exist")

    flags = [
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fno-fast-math",
        "-ffp-contract=off",
    ]
    if sanitizer == "none":
        flags += ["-O3", "-DNDEBUG"]
    else:
        flags += [
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            f"-fsanitize={sanitizer}",
            "-fno-sanitize-recover=all",
        ]
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        target = Path(temporary) / library.name
        source, source_sha256 = snapshot_source(SOURCE, Path(temporary))
        command = [executable, *flags, str(source), "-o", str(target)]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        version = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        record = {
            "schema": "spectra.general_native_build.v1",
            "native_abi": NATIVE_ABI,
            "library_name": library.name,
            "source_sha256": source_sha256,
            "library_sha256": _sha256(target),
            "command": command,
            "sanitizer": sanitizer,
            "compiler": version.stdout + version.stderr,
            "platform": platform.platform(),
            "python": sys.version,
            "elapsed_ns": time.perf_counter_ns() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        publish_build(target, library, record)
    return library


@dataclass(frozen=True)
class PreparedNativeFormula:
    """Canonical CNF packed once for the receipt-bound canonical entry point."""

    nvars: int
    clause_count: int
    literal_count: int
    unit_clauses: int
    binary_clauses: int
    long_clauses: int
    empty_clauses: int
    clauses: tuple[tuple[int, ...], ...] = field(repr=False, compare=False)
    _offset_storage: array = field(repr=False, compare=False)
    _literal_storage: array = field(repr=False, compare=False)
    _offset_view: object = field(repr=False, compare=False)
    _literal_view: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class NativeReduction:
    schema: str
    contradiction: bool
    conflict_kind: str
    conflict_variable: int
    original_nvars: int
    normalized_clause_count: int
    unit_input_clauses: int
    binary_input_clauses: int
    long_input_clauses: int
    class_count: int
    class_literals: tuple[int, ...]
    reduced_nvars: int
    residual_clauses: tuple[tuple[int, ...], ...]
    mapping: tuple[int, ...]
    fixed: tuple[bool, ...]
    equivalent_variables_removed: int
    propagated_classes: int

    def lift(self, model: tuple[bool, ...]) -> tuple[bool, ...]:
        if self.contradiction:
            raise ValueError("a contradictory reduction has no model to lift")
        if (
            type(model) is not tuple
            or len(model) != self.reduced_nvars
            or any(type(value) is not bool for value in model)
        ):
            raise ValueError("model must be a complete tuple of exact bool values")
        return tuple(
            self.fixed[index]
            if target == 0
            else model[abs(target) - 1] == (target > 0)
            for index, target in enumerate(self.mapping)
        )


class NativeBinaryCore:
    """One receipt-checked native reducer instance."""

    def __init__(self, library: str | Path) -> None:
        path = Path(library).resolve(strict=True)
        receipt_path = path.with_name("build.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != "spectra.general_native_build.v1":
            raise ValueError("native general-CNF build receipt schema differs")
        if receipt.get("native_abi") != NATIVE_ABI:
            raise ValueError("native general-CNF ABI differs")
        if receipt.get("library_name") != path.name:
            raise ValueError("native general-CNF library name differs")
        if receipt.get("source_sha256") != _sha256(SOURCE):
            raise ValueError("native general-CNF source hash differs")
        if receipt.get("library_sha256") != _sha256(path):
            raise ValueError("native general-CNF library hash differs")

        with native_load_guard(path, receipt["library_sha256"]):
            module = ctypes.CDLL(str(path))
        module.spectra_general_abi.argtypes = []
        module.spectra_general_abi.restype = ctypes.c_uint32
        if module.spectra_general_abi() != NATIVE_ABI:
            raise ValueError("loaded native general-CNF ABI differs")
        signature = [
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_uint64,
        ]
        for name in ("spectra_general_reduce", "spectra_general_reduce_canonical"):
            function = getattr(module, name)
            function.argtypes = signature
            function.restype = ctypes.c_int
        module.spectra_general_free.argtypes = [ctypes.c_void_p]
        module.spectra_general_free.restype = None
        self.path = path
        self.receipt = receipt
        self._module = module

    @staticmethod
    def _validate_nvars(nvars: int) -> None:
        if type(nvars) is not int or not 0 <= nvars <= 10_000_000:
            raise ValueError("nvars must be an exact integer in [0, 10000000]")

    @classmethod
    def _flatten(
        cls, nvars: int, clauses: Iterable[Iterable[int]]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        cls._validate_nvars(nvars)
        try:
            iterator = iter(clauses)
        except TypeError as error:
            raise ValueError("clauses must be iterable") from error
        offsets = [0]
        literals: list[int] = []
        for clause in iterator:
            try:
                row = tuple(clause)
            except TypeError as error:
                raise ValueError("each clause must be iterable") from error
            for literal in row:
                if (
                    type(literal) is not int
                    or literal == 0
                    or abs(literal) > nvars
                ):
                    raise ValueError(
                        "CNF literal must be a nonzero in-range exact integer"
                    )
                literals.append(literal)
            offsets.append(len(literals))
        return tuple(offsets), tuple(literals)

    @classmethod
    def prepare_canonical(
        cls,
        nvars: int,
        clauses: Iterable[Iterable[int]],
    ) -> PreparedNativeFormula:
        """Validate canonical order and pack one immutable repeated-use input."""
        cls._validate_nvars(nvars)
        try:
            iterator = iter(clauses)
        except TypeError as error:
            raise ValueError("clauses must be iterable") from error

        offsets = array("Q", [0])
        literals = array("i")
        previous: tuple[int, ...] | None = None
        frozen_rows: list[tuple[int, ...]] = []
        units = binaries = longs = empties = 0
        for clause in iterator:
            if type(clause) is not tuple:
                try:
                    row = tuple(clause)
                except TypeError as error:
                    raise ValueError("each clause must be iterable") from error
            else:
                row = clause
            previous_key: tuple[int, int] | None = None
            previous_abs = 0
            for literal in row:
                if (
                    type(literal) is not int
                    or literal == 0
                    or abs(literal) > nvars
                ):
                    raise ValueError(
                        "CNF literal must be a nonzero in-range exact integer"
                    )
                key = (abs(literal), literal)
                if previous_key is not None and not previous_key < key:
                    raise ValueError("canonical CNF clause is not strictly normalized")
                if previous_key is not None and previous_abs == abs(literal):
                    raise ValueError("canonical CNF clause is tautological")
                previous_key = key
                previous_abs = abs(literal)
                literals.append(literal)
            if previous is not None and not (len(previous), previous) < (len(row), row):
                raise ValueError("canonical CNF clauses are not strictly ordered")
            previous = row
            frozen_rows.append(row)
            offsets.append(len(literals))
            if len(row) == 0:
                empties += 1
            elif len(row) == 1:
                units += 1
            elif len(row) == 2:
                binaries += 1
            elif len(row) >= 3:
                longs += 1

        if offsets.itemsize != ctypes.sizeof(ctypes.c_uint64):
            raise RuntimeError("platform unsigned-long-long width differs")
        if literals.itemsize != ctypes.sizeof(ctypes.c_int32):
            raise RuntimeError("platform integer width differs")
        literal_count = len(literals)
        if literal_count == 0:
            literals.append(0)  # stable non-null buffer; count remains zero
        offset_view = (ctypes.c_uint64 * len(offsets)).from_buffer(offsets)
        literal_view = (ctypes.c_int32 * len(literals)).from_buffer(literals)
        return PreparedNativeFormula(
            nvars=nvars,
            clause_count=len(offsets) - 1,
            literal_count=literal_count,
            unit_clauses=units,
            binary_clauses=binaries,
            long_clauses=longs,
            empty_clauses=empties,
            clauses=tuple(frozen_rows),
            _offset_storage=offsets,
            _literal_storage=literals,
            _offset_view=offset_view,
            _literal_view=literal_view,
        )

    @staticmethod
    def _parse(payload: bytes) -> NativeReduction:
        if len(payload) < _HEADER.size:
            raise RuntimeError("native general-CNF payload is truncated")
        (
            magic,
            version,
            flags,
            conflict_code,
            conflict_variable,
            original_nvars,
            normalized,
            units,
            binaries,
            longs,
            class_count,
            class_literal_count,
            reduced_nvars,
            removed,
            propagated,
            residual_clause_count,
            residual_literal_count,
        ) = _HEADER.unpack_from(payload)
        if magic != _MAGIC or version != NATIVE_ABI or flags & ~1:
            raise RuntimeError("native general-CNF payload header differs")
        conflict_names = {0: "", 1: "binary", 2: "unit"}
        if conflict_code not in conflict_names:
            raise RuntimeError("native general-CNF conflict code differs")
        if class_literal_count not in (0, original_nvars):
            raise RuntimeError("native general-CNF class-literal count differs")
        if residual_clause_count > sys.maxsize or residual_literal_count > sys.maxsize:
            raise RuntimeError("native general-CNF payload exceeds platform limits")

        offset = _HEADER.size

        def read_i32(count: int) -> tuple[int, ...]:
            nonlocal offset
            size = 4 * count
            if offset + size > len(payload):
                raise RuntimeError("native general-CNF payload is truncated")
            values = struct.unpack_from(f"<{count}i", payload, offset) if count else ()
            offset += size
            return tuple(values)

        class_literals = read_i32(class_literal_count)
        mapping = read_i32(original_nvars)
        if offset + original_nvars > len(payload):
            raise RuntimeError("native general-CNF payload is truncated")
        fixed_bytes = payload[offset : offset + original_nvars]
        if any(value not in (0, 1) for value in fixed_bytes):
            raise RuntimeError("native general-CNF fixed-value bytes differ")
        fixed = tuple(bool(value) for value in fixed_bytes)
        offset += original_nvars

        count = int(residual_clause_count) + 1
        size = 8 * count
        if offset + size > len(payload):
            raise RuntimeError("native general-CNF payload is truncated")
        boundaries = struct.unpack_from(f"<{count}Q", payload, offset)
        offset += size
        if (
            not boundaries
            or boundaries[0] != 0
            or boundaries[-1] != residual_literal_count
            or any(left > right for left, right in zip(boundaries, boundaries[1:]))
        ):
            raise RuntimeError("native general-CNF residual offsets differ")
        residual_literals = read_i32(int(residual_literal_count))
        if offset != len(payload):
            raise RuntimeError("native general-CNF payload has trailing bytes")
        residual = tuple(
            residual_literals[boundaries[index] : boundaries[index + 1]]
            for index in range(int(residual_clause_count))
        )
        contradiction = bool(flags & 1)
        if contradiction != bool(conflict_code):
            raise RuntimeError("native general-CNF contradiction flags differ")
        return NativeReduction(
            schema="spectra.general_native_binary_core.v1",
            contradiction=contradiction,
            conflict_kind=conflict_names[conflict_code],
            conflict_variable=conflict_variable,
            original_nvars=original_nvars,
            normalized_clause_count=normalized,
            unit_input_clauses=units,
            binary_input_clauses=binaries,
            long_input_clauses=longs,
            class_count=class_count,
            class_literals=class_literals,
            reduced_nvars=reduced_nvars,
            residual_clauses=residual,
            mapping=mapping,
            fixed=fixed,
            equivalent_variables_removed=removed,
            propagated_classes=propagated,
        )

    def _invoke(
        self,
        function,
        *,
        nvars: int,
        clause_count: int,
        offsets,
        literals,
        literal_count: int,
    ) -> NativeReduction:
        output = ctypes.POINTER(ctypes.c_uint8)()
        output_size = ctypes.c_uint64()
        error = ctypes.create_string_buffer(1024)
        status = function(
            nvars,
            clause_count,
            offsets,
            literals,
            literal_count,
            ctypes.byref(output),
            ctypes.byref(output_size),
            error,
            len(error),
        )
        if status:
            message = error.value.decode("utf-8", "replace") or "native reduction failed"
            if status == 1:
                raise ValueError(message)
            if status == 2 and "memory" in message.lower():
                raise MemoryError(message)
            raise RuntimeError(message)
        if not output or output_size.value == 0:
            raise RuntimeError("native general-CNF reduction returned no payload")
        try:
            payload = ctypes.string_at(output, output_size.value)
        finally:
            self._module.spectra_general_free(output)
        reduction = self._parse(payload)
        if reduction.original_nvars != nvars:
            raise RuntimeError("native general-CNF result belongs to another formula")
        return reduction

    def reduce(
        self, nvars: int, clauses: Iterable[Iterable[int]]
    ) -> NativeReduction:
        offsets, literals = self._flatten(nvars, clauses)
        offset_array = (ctypes.c_uint64 * len(offsets))(*offsets)
        literal_array = (ctypes.c_int32 * max(1, len(literals)))(*(literals or (0,)))
        return self._invoke(
            self._module.spectra_general_reduce,
            nvars=nvars,
            clause_count=len(offsets) - 1,
            offsets=offset_array,
            literals=literal_array,
            literal_count=len(literals),
        )

    def reduce_prepared(self, prepared: PreparedNativeFormula) -> NativeReduction:
        if type(prepared) is not PreparedNativeFormula:
            raise ValueError("prepared formula must come from prepare_canonical")
        if (
            prepared.empty_clauses == 0
            and prepared.unit_clauses == 0
            and prepared.binary_clauses == 0
        ):
            identity = tuple(range(1, prepared.nvars + 1))
            return NativeReduction(
                schema="spectra.general_native_binary_core.v1",
                contradiction=False,
                conflict_kind="",
                conflict_variable=0,
                original_nvars=prepared.nvars,
                normalized_clause_count=prepared.clause_count,
                unit_input_clauses=0,
                binary_input_clauses=0,
                long_input_clauses=prepared.long_clauses,
                class_count=prepared.nvars,
                class_literals=identity,
                reduced_nvars=prepared.nvars,
                residual_clauses=prepared.clauses,
                mapping=identity,
                fixed=(False,) * prepared.nvars,
                equivalent_variables_removed=0,
                propagated_classes=0,
            )
        return self._invoke(
            self._module.spectra_general_reduce_canonical,
            nvars=prepared.nvars,
            clause_count=prepared.clause_count,
            offsets=prepared._offset_view,
            literals=prepared._literal_view,
            literal_count=prepared.literal_count,
        )


def assert_matches_python(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    native: NativeReduction,
    reference,
) -> None:
    """Reject a native reduction unless every solver-facing field matches."""
    fields = {
        "contradiction": native.contradiction,
        "conflict_kind": native.conflict_kind,
        "normalized_clause_count": native.normalized_clause_count,
        "unit_input_clauses": native.unit_input_clauses,
        "binary_input_clauses": native.binary_input_clauses,
        "long_input_clauses": native.long_input_clauses,
        "class_count": native.class_count,
        "class_literals": native.class_literals,
        "reduced_nvars": native.reduced_nvars,
        "residual_clauses": native.residual_clauses,
        "mapping": native.mapping,
        "fixed": native.fixed,
        "equivalent_variables_removed": native.equivalent_variables_removed,
        "propagated_classes": native.propagated_classes,
    }
    for name, value in fields.items():
        expected = reference.contradiction if name == "contradiction" else getattr(reference, name)
        if value != expected:
            raise AssertionError(
                f"native binary core differs in {name}: {value!r} != {expected!r}"
            )
    if native.conflict_kind == "binary" and not 1 <= native.conflict_variable <= nvars:
        raise AssertionError("native binary conflict variable is invalid")
    if native.conflict_kind != "binary" and native.conflict_variable != 0:
        raise AssertionError("native non-binary result has a conflict variable")


def reduce_and_verify(
    runtime: NativeBinaryCore,
    nvars: int,
    clauses: Iterable[Iterable[int]],
) -> NativeReduction:
    """Run both implementations and return the native result only on equality."""
    from experiments.general_competition.binary_core import (
        build_binary_core,
        verify_binary_core,
    )

    frozen = tuple(tuple(row) for row in clauses)
    reference = build_binary_core(nvars, frozen)
    if not verify_binary_core(nvars, frozen, reference):
        raise AssertionError("Python binary core failed its own independent replay")
    native = runtime.reduce(nvars, frozen)
    assert_matches_python(nvars, frozen, native, reference)
    return native
