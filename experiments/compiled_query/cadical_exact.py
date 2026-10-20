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
SCHEMA = "spectra.cadical_exact_build.v3"
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


def _source_records(paths: list[Path]) -> tuple[list[dict], str]:
    records = []
    manifest = hashlib.sha256()
    for path in paths:
        observed = _sha256(path)
        records.append({"name": path.name, "sha256": observed})
        manifest.update(path.name.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(observed.encode("ascii"))
        manifest.update(b"\n")
    return records, manifest.hexdigest()


def _run(
    command: list[str], *, cwd: Path, timeout: int = 900
) -> subprocess.CompletedProcess:
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
    return _run(
        ["git", "rev-parse", "HEAD"], cwd=source, timeout=30
    ).stdout.strip()


def build_cadical_exact_runtime(
    directory: str | Path,
    cadical_source: str | Path,
    *,
    compiler: str | None = None,
    c_compiler: str | None = None,
    sanitizer: str = "none",
) -> Path:
    """Build the exact pinned official sources as PIC and link the adapter."""
    if os.name != "posix" or platform.system() != "Linux":
        raise NotImplementedError(
            "the official CaDiCaL challenge currently builds on Linux"
        )
    if sanitizer not in ("none", "undefined", "address,undefined"):
        raise ValueError("unsupported sanitizer")
    source = Path(cadical_source).resolve(strict=True)
    if _head(source) != CADICAL_COMMIT:
        raise ValueError("CaDiCaL source commit differs from the frozen challenge")
    source_dir = source / "src"
    header = source_dir / "cadical.hpp"
    if not header.is_file():
        raise ValueError("CaDiCaL source tree is incomplete")

    official_cpp_sources = sorted(
        path
        for path in source_dir.glob("*.cpp")
        if not path.name.endswith("ical.cpp")
    )
    official_c_sources = sorted(source_dir.glob("*.c"))
    official_sources = [*official_cpp_sources, *official_c_sources]
    if len(official_cpp_sources) < 40 or not official_c_sources:
        raise ValueError("CaDiCaL source inventory is unexpectedly small")
    if not any(path.name == "kitten.c" for path in official_c_sources):
        raise ValueError("CaDiCaL Kitten source is missing")
    source_records, source_manifest = _source_records(official_sources)

    compiler = compiler or "g++"
    c_compiler = c_compiler or "gcc"
    resolved = shutil.which(compiler)
    resolved_c = shutil.which(c_compiler)
    archiver = shutil.which("ar")
    if resolved is None:
        raise FileNotFoundError(compiler)
    if resolved_c is None:
        raise FileNotFoundError(c_compiler)
    if archiver is None:
        raise FileNotFoundError("ar")

    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / "libcadical_exact.so"
    receipt = destination / "build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("CaDiCaL adapter build outputs already exist")

    started = time.perf_counter_ns()
    common = [
        "-fPIC",
        "-DNBUILD",
        "-I" + str(source_dir),
    ]
    if sanitizer == "none":
        optimization = ["-O3", "-DNDEBUG"]
    else:
        optimization = [
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-fsanitize=" + sanitizer,
            "-fno-sanitize-recover=all",
        ]
    cpp_flags = ["-std=c++11", *common, *optimization]
    c_flags = ["-std=c11", *common, *optimization]

    adapter_flags = [
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fno-fast-math",
        "-ffp-contract=off",
        *optimization,
    ]

    with tempfile.TemporaryDirectory(dir=destination) as temporary_name:
        temporary = Path(temporary_name)
        object_dir = temporary / "cadical-objects"
        object_dir.mkdir()
        objects = []
        compile_stdout = []
        compile_stderr = []
        for index, source_file in enumerate(official_cpp_sources):
            target_object = object_dir / f"cpp-{index:03d}-{source_file.stem}.o"
            command = [
                resolved,
                *cpp_flags,
                "-c",
                str(source_file),
                "-o",
                str(target_object),
            ]
            completed = _run(command, cwd=temporary)
            compile_stdout.append(completed.stdout)
            compile_stderr.append(completed.stderr)
            objects.append(target_object)
        for index, source_file in enumerate(official_c_sources):
            target_object = object_dir / f"c-{index:03d}-{source_file.stem}.o"
            command = [
                resolved_c,
                *c_flags,
                "-c",
                str(source_file),
                "-o",
                str(target_object),
            ]
            completed = _run(command, cwd=temporary)
            compile_stdout.append(completed.stdout)
            compile_stderr.append(completed.stderr)
            objects.append(target_object)

        archive = temporary / "libcadical-pic.a"
        archive_command = [archiver, "rcs", str(archive), *(str(x) for x in objects)]
        archive_result = _run(archive_command, cwd=temporary)
        if not archive.is_file():
            raise RuntimeError("PIC CaDiCaL archive was not produced")

        target = temporary / library.name
        adapter_command = [
            resolved,
            *adapter_flags,
            "-I" + str(source_dir),
            str(ADAPTER_SOURCE.resolve()),
            str(archive),
            "-pthread",
            "-ldl",
            "-o",
            str(target),
        ]
        adapter_result = _run(adapter_command, cwd=temporary)
        cxx_version = _run([resolved, "--version"], cwd=temporary, timeout=30)
        c_version = _run([resolved_c, "--version"], cwd=temporary, timeout=30)
        record = {
            "schema": SCHEMA,
            "cadical_repository": "arminbiere/cadical",
            "cadical_tag": CADICAL_TAG,
            "cadical_commit": CADICAL_COMMIT,
            "cadical_header_sha256": _sha256(header),
            "cadical_source_files": source_records,
            "cadical_source_manifest_sha256": source_manifest,
            "cadical_cpp_compile_flags": cpp_flags,
            "cadical_c_compile_flags": c_flags,
            "cadical_archive_sha256": _sha256(archive),
            "archiver": archiver,
            "archive_command": archive_command,
            "archive_stdout": archive_result.stdout,
            "archive_stderr": archive_result.stderr,
            "official_compile_stdout": "".join(compile_stdout),
            "official_compile_stderr": "".join(compile_stderr),
            "adapter_source": ADAPTER_SOURCE.name,
            "adapter_source_sha256": _sha256(ADAPTER_SOURCE),
            "proof_support_source": PROOF_SUPPORT_SOURCE.name,
            "proof_support_source_sha256": _sha256(PROOF_SUPPORT_SOURCE),
            "library": library.name,
            "library_sha256": _sha256(target),
            "cxx_compiler": cxx_version.stdout + cxx_version.stderr,
            "c_compiler": c_version.stdout + c_version.stderr,
            "adapter_command": adapter_command,
            "adapter_stdout": adapter_result.stdout,
            "adapter_stderr": adapter_result.stderr,
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
        "cadical_source_files",
        "cadical_source_manifest_sha256",
        "cadical_cpp_compile_flags",
        "cadical_c_compile_flags",
        "cadical_archive_sha256",
        "archiver",
        "archive_command",
        "archive_stdout",
        "archive_stderr",
        "official_compile_stdout",
        "official_compile_stderr",
        "adapter_source",
        "adapter_source_sha256",
        "proof_support_source",
        "proof_support_source_sha256",
        "library",
        "library_sha256",
        "cxx_compiler",
        "c_compiler",
        "adapter_command",
        "adapter_stdout",
        "adapter_stderr",
        "platform",
        "machine",
        "sanitizer",
        "elapsed_ns",
    }
    if (
        type(value) is not dict
        or set(value) != required
        or value["schema"] != SCHEMA
    ):
        raise ValueError("CaDiCaL adapter build receipt differs")
    if (
        value["cadical_repository"] != "arminbiere/cadical"
        or value["cadical_tag"] != CADICAL_TAG
        or value["cadical_commit"] != CADICAL_COMMIT
    ):
        raise ValueError("CaDiCaL external source identity differs")
    files = value["cadical_source_files"]
    if (
        type(files) is not list
        or len(files) < 41
        or not any(row.get("name") == "kitten.c" for row in files if type(row) is dict)
        or any(
            type(row) is not dict
            or set(row) != {"name", "sha256"}
            or type(row["name"]) is not str
            or type(row["sha256"]) is not str
            or len(row["sha256"]) != 64
            for row in files
        )
    ):
        raise ValueError("CaDiCaL source manifest differs")
    manifest = hashlib.sha256()
    for row in files:
        manifest.update(row["name"].encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(row["sha256"].encode("ascii"))
        manifest.update(b"\n")
    if manifest.hexdigest() != value["cadical_source_manifest_sha256"]:
        raise ValueError("CaDiCaL source manifest hash differs")
    if value["adapter_source"] != ADAPTER_SOURCE.name:
        raise ValueError("CaDiCaL adapter source name differs")
    if value["proof_support_source"] != PROOF_SUPPORT_SOURCE.name:
        raise ValueError("CaDiCaL proof support source name differs")
    if value["adapter_source_sha256"] != _sha256(ADAPTER_SOURCE):
        raise ValueError("CaDiCaL adapter source hash differs")
    if value["proof_support_source_sha256"] != _sha256(PROOF_SUPPORT_SOURCE):
        raise ValueError("CaDiCaL proof support source hash differs")
    if (
        value["library"] != library.name
        or value["library_sha256"] != _sha256(library)
    ):
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
            raise RuntimeError(
                "combined library does not report a CaDiCaL signature"
            )

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
            message = (
                raw.decode("utf-8", "replace")
                if raw
                else "CaDiCaL adapter failed"
            )
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
                "cadical_source_manifest_sha256": self.receipt[
                    "cadical_source_manifest_sha256"
                ],
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
                "adapter_source_sha256": self.receipt[
                    "adapter_source_sha256"
                ],
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
