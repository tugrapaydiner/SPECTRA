"""Independent native 2-SAT challenger for the proof-carrying exact service.

The implementation consumes the public ``SBC1`` packet and emits a complete
``SCQ2`` answer bank. It shares no native data structures, SCC identifiers,
compiled closure, cached answers, or proof search with ``spectra.exact``.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

SOURCE = Path(__file__).with_name("twosat_baseline.cpp")
ABI = 1
DEFAULT_BYTES = 512 << 20
MAX_BYTES = (1 << 64) - 1


def build_twosat_baseline(
    directory: str | Path,
    *,
    compiler: str | None = None,
    sanitizer: str = "none",
) -> Path:
    """Explicitly build the independent C++17 baseline on Linux or Windows."""
    windows = sys.platform == "win32"
    if sys.platform not in ("linux", "win32"):
        raise NotImplementedError("direct 2-SAT baseline supports Linux and Windows")
    if sanitizer not in ("none", "undefined", "address,undefined"):
        raise ValueError("unsupported sanitizer")
    if windows and sanitizer != "none":
        raise ValueError("sanitizer builds currently require Linux")
    compiler = compiler or ("cl" if windows else "g++")
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError(compiler)

    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / ("spectra_twosat.dll" if windows else "spectra_twosat.so")
    receipt = destination / "build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("direct 2-SAT build destination is already occupied")

    if windows:
        flags = [
            "/nologo", "/std:c++17", "/LD", "/EHsc", "/W4", "/WX",
            "/O2", "/DNDEBUG",
        ]
    else:
        flags = [
            "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
            "-fno-fast-math", "-ffp-contract=off",
        ]
        flags += (
            ["-O3", "-DNDEBUG"]
            if sanitizer == "none"
            else [
                "-O1", "-g", "-fno-omit-frame-pointer", f"-fsanitize={sanitizer}",
                "-fno-sanitize-recover=all",
            ]
        )

    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix=".twosat-", dir=destination) as temporary:
        target = Path(temporary) / library.name
        if windows:
            command = [executable, *flags, str(SOURCE), f"/Fe:{target}"]
        else:
            command = [executable, *flags, str(SOURCE), "-o", str(target)]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        version_command = [executable] if windows else [executable, "--version"]
        version = subprocess.run(
            version_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=not windows,
            timeout=10,
        )
        record = {
            "schema": "spectra.twosat_baseline.build.v1",
            "abi": ABI,
            "library_name": library.name,
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "command": command,
            "compiler": version.stdout + version.stderr,
            "platform": platform.platform(),
            "sanitizer": sanitizer,
            "elapsed_ns": time.perf_counter_ns() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        os.link(target, library)
        try:
            with receipt.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, sort_keys=True)
                stream.write("\n")
        except BaseException:
            library.unlink(missing_ok=True)
            receipt.unlink(missing_ok=True)
            raise
    return library


def _read_receipt(library: Path) -> dict:
    library = Path(library).resolve()
    try:
        record = json.loads(library.with_name("build.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("direct 2-SAT baseline requires a valid adjacent build.json") from error
    if type(record) is not dict:
        raise ValueError("direct 2-SAT build receipt must be an object")
    expected = {
        "schema": "spectra.twosat_baseline.build.v1",
        "abi": ABI,
        "library_name": library.name,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"direct 2-SAT build receipt {key} differs")
    if record.get("source_sha256") != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
        raise ValueError("direct 2-SAT source hash differs from this implementation")
    if record.get("library_sha256") != hashlib.sha256(library.read_bytes()).hexdigest():
        raise ValueError("direct 2-SAT library hash differs from its receipt")
    return record


@dataclass(frozen=True)
class TwoSatAnswer:
    output: bytes
    query_count: int
    sat_count: int
    unsat_count: int
    proof_bytes: int
    literal_count: int
    implication_arcs: int
    components: int
    closure_bytes: int
    dense_closure: bool
    assignment_fallbacks: int
    proof_vertices_visited: int
    proof_arcs_scanned: int


class TwoSatBaseline:
    """Receipt-bound independent baseline with one full-cost ``SBC1`` entry point."""

    def __init__(self, library: str | Path):
        path = Path(library).resolve()
        self.build_receipt = _read_receipt(path)
        native = C.CDLL(str(path))
        native.spectra_twosat_abi.argtypes = []
        native.spectra_twosat_abi.restype = C.c_uint32
        if native.spectra_twosat_abi() != ABI:
            raise ValueError("unsupported direct 2-SAT ABI")
        native.spectra_twosat_solve.argtypes = [
            C.POINTER(C.c_uint8), C.c_uint64, C.c_uint64,
            C.POINTER(C.POINTER(C.c_uint8)), C.POINTER(C.c_uint64),
            C.POINTER(C.c_uint64), C.c_uint32, C.c_char_p, C.c_uint64,
        ]
        native.spectra_twosat_solve.restype = C.c_int
        native.spectra_twosat_free.argtypes = [C.c_void_p]
        native.spectra_twosat_free.restype = None
        self._native = native

    def solve_session_packet(
        self,
        packet: bytes,
        *,
        max_bytes: int = DEFAULT_BYTES,
    ) -> TwoSatAnswer:
        if type(packet) is not bytes:
            raise ValueError("session packet must be immutable bytes")
        if type(max_bytes) is not int or not (1 << 20) <= max_bytes <= MAX_BYTES:
            raise ValueError("max_bytes must be an exact uint64 of at least one MiB")
        payload = (C.c_uint8 * len(packet)).from_buffer_copy(packet)
        output = C.POINTER(C.c_uint8)()
        output_bytes = C.c_uint64()
        stats = (C.c_uint64 * 12)()
        error = C.create_string_buffer(512)
        code = self._native.spectra_twosat_solve(
            payload,
            len(packet),
            max_bytes,
            C.byref(output),
            C.byref(output_bytes),
            stats,
            len(stats),
            error,
            len(error),
        )
        message = error.value.decode("utf-8", errors="replace")
        if code:
            if code == 2:
                raise MemoryError(message)
            if code == 1:
                raise ValueError(message)
            raise RuntimeError(message)
        try:
            result = C.string_at(output, output_bytes.value)
        finally:
            self._native.spectra_twosat_free(output)
        return TwoSatAnswer(
            result,
            stats[0], stats[1], stats[2], stats[3], stats[4], stats[5],
            stats[6], stats[7], bool(stats[8]), stats[9], stats[10], stats[11],
        )
