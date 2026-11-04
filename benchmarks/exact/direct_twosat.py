"""Independent native 2-SAT baseline for the proof-carrying SBC1 -> SCQ2 contract.

The native implementation shares no solver, SCC, proof-construction, parser, or
observer code with :mod:`spectra.exact`. It is a deliberately strong challenge,
not a fallback used by the public package.
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
import subprocess
import tempfile
import time

NATIVE_SOURCE = Path(__file__).with_name("direct_twosat_native.cpp")
MODE_SOURCE = Path(__file__).with_name("direct_twosat_modes.cpp")
SOURCE_FILES = (NATIVE_SOURCE, MODE_SOURCE)
SCHEMA = "spectra.direct_twosat_build.v3"
DEFAULT_BYTES = 512 << 20


class _Stats(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_uint32),
        ("edges", ctypes.c_uint32),
        ("queries", ctypes.c_uint32),
        ("pins", ctypes.c_uint32),
        ("components", ctypes.c_uint32),
        ("dag_edges", ctypes.c_uint32),
        ("sat", ctypes.c_uint32),
        ("unsat", ctypes.c_uint32),
        ("engine", ctypes.c_uint32),
        ("closure_bytes", ctypes.c_uint64),
        ("output_bytes", ctypes.c_uint64),
        ("proof_bytes", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class Direct2SATResult:
    output: bytes
    info: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_FILES:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_direct_twosat_runtime(
    directory: str | Path,
    *,
    compiler: str | None = None,
    sanitizer: str = "none",
) -> Path:
    """Build the standalone Linux shared library and write a bound receipt."""
    if os.name != "posix" or platform.system() != "Linux":
        raise NotImplementedError("the direct 2-SAT challenge currently builds on Linux")
    if sanitizer not in ("none", "undefined", "address,undefined"):
        raise ValueError("unsupported sanitizer")
    compiler = compiler or "g++"
    resolved = shutil.which(compiler)
    if resolved is None:
        raise FileNotFoundError(compiler)
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / "libdirect_twosat.so"
    receipt = destination / "build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("direct 2-SAT build outputs already exist")
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
        flags += ["-O3", "-DNDEBUG"]
    else:
        flags += [
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-fsanitize=" + sanitizer,
            "-fno-sanitize-recover=all",
        ]
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        target = Path(temporary) / library.name
        command = [resolved, *flags, str(MODE_SOURCE.resolve()), "-o", str(target)]
        completed = subprocess.run(
            command,
            cwd=temporary,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        version = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=10,
        )
        record = {
            "schema": SCHEMA,
            "sources": [path.name for path in SOURCE_FILES],
            "source_sha256": _source_digest(),
            "library": library.name,
            "library_sha256": _sha256(target),
            "compiler": version.stdout + version.stderr,
            "command": command,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "sanitizer": sanitizer,
            "elapsed_ns": time.perf_counter_ns() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        os.link(target, library)
        try:
            with receipt.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, sort_keys=True, indent=2)
                stream.write("\n")
        except BaseException:
            library.unlink(missing_ok=True)
            receipt.unlink(missing_ok=True)
            raise
    return library


def _check_receipt(library: Path) -> dict:
    receipt = library.with_name("build.json")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "sources",
        "source_sha256",
        "library",
        "library_sha256",
        "compiler",
        "command",
        "platform",
        "machine",
        "sanitizer",
        "elapsed_ns",
        "stdout",
        "stderr",
    }
    if type(value) is not dict or set(value) != expected or value["schema"] != SCHEMA:
        raise ValueError("direct 2-SAT build receipt differs")
    if value["sources"] != [path.name for path in SOURCE_FILES]:
        raise ValueError("direct 2-SAT source inventory differs")
    if value["library"] != library.name:
        raise ValueError("direct 2-SAT library identity differs")
    if value["source_sha256"] != _source_digest():
        raise ValueError("direct 2-SAT source hash differs")
    if value["library_sha256"] != _sha256(library):
        raise ValueError("direct 2-SAT library hash differs")
    return value


def _configure_solver(library, name: str):
    function = getattr(library, name)
    function.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(_Stats),
    ]
    function.restype = ctypes.c_int
    return function


class Direct2SATRuntime:
    """Receipt-bound wrapper around the independent C ABI."""

    def __init__(self, library: str | Path):
        self.path = Path(library).resolve()
        self.receipt = _check_receipt(self.path)
        self._library = ctypes.CDLL(str(self.path))
        self._solvers = {
            "auto": _configure_solver(self._library, "direct_twosat_solve"),
            "dag": _configure_solver(self._library, "direct_twosat_solve_dag"),
            "tile": _configure_solver(self._library, "direct_twosat_solve_tile"),
        }
        self._library.direct_twosat_error.argtypes = []
        self._library.direct_twosat_error.restype = ctypes.c_char_p
        self._library.direct_twosat_free.argtypes = [ctypes.c_void_p]
        self._library.direct_twosat_free.restype = None

    def solve_session_packet(
        self,
        packet: bytes,
        *,
        engine: str = "auto",
        max_bytes: int = DEFAULT_BYTES,
    ) -> Direct2SATResult:
        if type(packet) is not bytes:
            raise TypeError("direct 2-SAT packet must be bytes")
        if engine not in self._solvers:
            raise ValueError("direct 2-SAT engine must be auto, dag or tile")
        if type(max_bytes) is not int or not 1 <= max_bytes < 1 << 64:
            raise ValueError("max_bytes must be a positive uint64")
        buffer = (ctypes.c_uint8 * len(packet)).from_buffer_copy(packet)
        pointer = ctypes.c_void_p()
        size = ctypes.c_size_t()
        stats = _Stats()
        code = self._solvers[engine](
            buffer,
            len(packet),
            max_bytes,
            ctypes.byref(pointer),
            ctypes.byref(size),
            ctypes.byref(stats),
        )
        if code:
            raw = self._library.direct_twosat_error()
            message = raw.decode("utf-8", "replace") if raw else "direct 2-SAT failed"
            if code == 1:
                raise ValueError(message)
            if code == 2:
                raise MemoryError(message)
            raise RuntimeError(message)
        try:
            output = ctypes.string_at(pointer, size.value)
        finally:
            self._library.direct_twosat_free(pointer)
        selected_engine = {1: "closure", 2: "dag", 3: "tile"}.get(stats.engine)
        if selected_engine is None:
            raise RuntimeError("direct 2-SAT returned an unknown engine")
        return Direct2SATResult(
            output=output,
            info={
                "requested_engine": engine,
                "engine": selected_engine,
                "vertices": stats.n,
                "edges": stats.edges,
                "queries": stats.queries,
                "pins": stats.pins,
                "components": stats.components,
                "dag_edges": stats.dag_edges,
                "sat": stats.sat,
                "unsat": stats.unsat,
                "closure_bytes": stats.closure_bytes,
                "output_bytes": stats.output_bytes,
                "proof_bytes": stats.proof_bytes,
                "source_sha256": self.receipt["source_sha256"],
                "library_sha256": self.receipt["library_sha256"],
            },
        )
