"""Receipt-bound official CaDiCaL adapter for proof-carrying exact banks."""
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

ADAPTER_SOURCE = Path(__file__).with_name("cadical_exact_native.cpp")
PROOF_SUPPORT_SOURCE = Path(__file__).with_name("direct_twosat_native.cpp")
CADICAL_COMMIT = "c60730422e758ef1cebe7aeddf2dda31c996bf04"
CADICAL_TAG = "rel-3.0.1"
SCHEMA = "spectra.cadical_exact_build.v1"
DEFAULT_BYTES = 512 << 20


class _Stats(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_uint32),
        ("edges", ctypes.c_uint32),
        ("queries", ctypes.c_uint32),
        ("pins", ctypes.c_uint32),
        ("clauses", ctypes.c_uint32),
        ("components", ctypes.c_uint32),
        ("sat", ctypes.c_uint32),
        ("unsat", ctypes.c_uint32),
        ("solves", ctypes.c_uint32),
        ("failed_assumptions", ctypes.c_uint32),
        ("output_bytes", ctypes.c_uint64),
        ("proof_bytes", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class CadicalExactResult:
    output: bytes
    info: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
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


def _head(source: Path) -> str:
    completed = _run(
        ["git", "rev-parse", "HEAD"], cwd=source, timeout=30
    )
    return completed.stdout.strip()


def build_cadical_exact_runtime(
    directory: str | Path,
    cadical_source: str | Path,
    *,
    compiler: str | None = None,
    sanitizer: str = "none",
) -> Path:
    """Build the pinned official CaDiCaL core and the project-owned adapter."""
    if os.name != "posix" or platform.system() != "Linux":
        raise NotImplementedError("the official CaDiCaL challenge currently builds on Linux")
    if sanitizer not in ("none", "undefined", "address,undefined"):
        raise ValueError("unsupported sanitizer")
    source = Path(cadical_source).resolve(strict=True)
    if _head(source) != CADICAL_COMMIT:
        raise ValueError("CaDiCaL source commit differs from the frozen challenge")
    header = source / "src" / "cadical.hpp"
    configure = source / "configure"
    if not header.is_file() or not configure.is_file():
        raise ValueError("CaDiCaL source tree is incomplete")

    compiler = compiler or "g++"
    resolved = shutil.which(compiler)
    make = shutil.which("make")
    if resolved is None:
        raise FileNotFoundError(compiler)
    if make is None:
        raise FileNotFoundError("make")

    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / "libcadical_exact.so"
    receipt = destination / "build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("CaDiCaL adapter build outputs already exist")

    started = time.perf_counter_ns()
    configure_result = _run([str(configure)], cwd=source)
    build_result = _run(
        [make, "-C", "build", "-j2", "libcadical.a"], cwd=source
    )
    static_library = source / "build" / "libcadical.a"
    if not static_library.is_file():
        raise RuntimeError("official CaDiCaL build did not produce libcadical.a")

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

    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        target = Path(temporary) / library.name
        command = [
            resolved,
            *flags,
            "-I" + str(source / "src"),
            str(ADAPTER_SOURCE.resolve()),
            str(static_library),
            "-pthread",
            "-ldl",
            "-o",
            str(target),
        ]
        compiled = _run(command, cwd=Path(temporary))
        version = _run([resolved, "--version"], cwd=Path(temporary), timeout=30)
        record = {
            "schema": SCHEMA,
            "cadical_repository": "arminbiere/cadical",
            "cadical_tag": CADICAL_TAG,
            "cadical_commit": CADICAL_COMMIT,
            "cadical_header_sha256": _sha256(header),
            "cadical_static_library_sha256": _sha256(static_library),
            "adapter_source": ADAPTER_SOURCE.name,
            "adapter_source_sha256": _sha256(ADAPTER_SOURCE),
            "proof_support_source": PROOF_SUPPORT_SOURCE.name,
            "proof_support_source_sha256": _sha256(PROOF_SUPPORT_SOURCE),
            "library": library.name,
            "library_sha256": _sha256(target),
            "compiler": version.stdout + version.stderr,
            "configure_command": [str(configure)],
            "configure_stdout": configure_result.stdout,
            "configure_stderr": configure_result.stderr,
            "make_command": [make, "-C", "build", "-j2", "libcadical.a"],
            "make_stdout": build_result.stdout,
            "make_stderr": build_result.stderr,
            "adapter_command": command,
            "adapter_stdout": compiled.stdout,
            "adapter_stderr": compiled.stderr,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "sanitizer": sanitizer,
            "elapsed_ns": time.perf_counter_ns() - started,
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
    required = {
        "schema",
        "cadical_repository",
        "cadical_tag",
        "cadical_commit",
        "cadical_header_sha256",
        "cadical_static_library_sha256",
        "adapter_source",
        "adapter_source_sha256",
        "proof_support_source",
        "proof_support_source_sha256",
        "library",
        "library_sha256",
        "compiler",
        "configure_command",
        "configure_stdout",
        "configure_stderr",
        "make_command",
        "make_stdout",
        "make_stderr",
        "adapter_command",
        "adapter_stdout",
        "adapter_stderr",
        "platform",
        "machine",
        "sanitizer",
        "elapsed_ns",
    }
    if type(value) is not dict or set(value) != required or value["schema"] != SCHEMA:
        raise ValueError("CaDiCaL adapter build receipt differs")
    if (
        value["cadical_repository"] != "arminbiere/cadical"
        or value["cadical_tag"] != CADICAL_TAG
        or value["cadical_commit"] != CADICAL_COMMIT
    ):
        raise ValueError("CaDiCaL external source identity differs")
    if value["adapter_source"] != ADAPTER_SOURCE.name:
        raise ValueError("CaDiCaL adapter source name differs")
    if value["proof_support_source"] != PROOF_SUPPORT_SOURCE.name:
        raise ValueError("CaDiCaL proof support source name differs")
    if value["adapter_source_sha256"] != _sha256(ADAPTER_SOURCE):
        raise ValueError("CaDiCaL adapter source hash differs")
    if value["proof_support_source_sha256"] != _sha256(PROOF_SUPPORT_SOURCE):
        raise ValueError("CaDiCaL proof support source hash differs")
    if value["library"] != library.name or value["library_sha256"] != _sha256(library):
        raise ValueError("CaDiCaL adapter library hash differs")
    return value


class CadicalExactRuntime:
    """Receipt-bound wrapper around the official CaDiCaL exact adapter."""

    def __init__(self, library: str | Path):
        self.path = Path(library).resolve(strict=True)
        self.receipt = _check_receipt(self.path)
        self._library = ctypes.CDLL(str(self.path))
        self._library.cadical_exact_signature.argtypes = []
        self._library.cadical_exact_signature.restype = ctypes.c_char_p
        self._library.cadical_exact_solve.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(_Stats),
        ]
        self._library.cadical_exact_solve.restype = ctypes.c_int
        self._library.cadical_exact_error.argtypes = []
        self._library.cadical_exact_error.restype = ctypes.c_char_p
        self._library.cadical_exact_free.argtypes = [ctypes.c_void_p]
        self._library.cadical_exact_free.restype = None
        raw_signature = self._library.cadical_exact_signature()
        self.signature = (
            raw_signature.decode("utf-8", "replace") if raw_signature else ""
        )
        if "cadical" not in self.signature.lower():
            raise RuntimeError("combined library does not report a CaDiCaL signature")

    def solve_session_packet(
        self, packet: bytes, *, max_bytes: int = DEFAULT_BYTES
    ) -> CadicalExactResult:
        if type(packet) is not bytes:
            raise TypeError("CaDiCaL adapter packet must be bytes")
        if type(max_bytes) is not int or not 1 <= max_bytes < 1 << 64:
            raise ValueError("max_bytes must be a positive uint64")
        buffer = (ctypes.c_uint8 * len(packet)).from_buffer_copy(packet)
        pointer = ctypes.c_void_p()
        size = ctypes.c_size_t()
        stats = _Stats()
        code = self._library.cadical_exact_solve(
            buffer,
            len(packet),
            max_bytes,
            ctypes.byref(pointer),
            ctypes.byref(size),
            ctypes.byref(stats),
        )
        if code:
            raw = self._library.cadical_exact_error()
            message = raw.decode("utf-8", "replace") if raw else "CaDiCaL adapter failed"
            if code == 1:
                raise ValueError(message)
            if code == 2:
                raise MemoryError(message)
            raise RuntimeError(message)
        try:
            output = ctypes.string_at(pointer, size.value)
        finally:
            self._library.cadical_exact_free(pointer)
        return CadicalExactResult(
            output=output,
            info={
                "backend": "official-cadical-proof-adapter",
                "cadical_repository": self.receipt["cadical_repository"],
                "cadical_tag": self.receipt["cadical_tag"],
                "cadical_commit": self.receipt["cadical_commit"],
                "cadical_signature": self.signature,
                "vertices": stats.n,
                "edges": stats.edges,
                "queries": stats.queries,
                "pins": stats.pins,
                "clauses": stats.clauses,
                "components": stats.components,
                "sat": stats.sat,
                "unsat": stats.unsat,
                "solves": stats.solves,
                "failed_assumptions": stats.failed_assumptions,
                "output_bytes": stats.output_bytes,
                "proof_bytes": stats.proof_bytes,
                "adapter_source_sha256": self.receipt["adapter_source_sha256"],
                "proof_support_source_sha256": self.receipt[
                    "proof_support_source_sha256"
                ],
                "library_sha256": self.receipt["library_sha256"],
                "query_answer_cache": False,
                "portable_unsat_certificate": True,
                "prepared_input": True,
                "input_schema": "SBC1",
                "output_schema": "SCQ2",
            },
        )
