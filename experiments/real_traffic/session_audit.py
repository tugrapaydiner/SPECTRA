"""Native independent checking for complete trace-driven solver sessions."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sysconfig
import tempfile
import threading
import time
from typing import Sequence

SOURCE = Path(__file__).with_name("session_audit.cpp")
DEFAULT_BYTES = 64 * 1024 * 1024


def build_session_audit(directory: str | Path, *, compiler: str = "g++",
                        sanitize: bool = False) -> Path:
    if platform.system() != "Linux":
        raise NotImplementedError("Linux builder only; other platforms unvalidated")
    if type(sanitize) is not bool:
        raise TypeError("sanitize must be bool")
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError(compiler)
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / ("_spectra_session_audit" + sysconfig.get_config_var("EXT_SUFFIX"))
    receipt = destination / "build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("build outputs already exist")
    flags = ["-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
             "-fno-fast-math", "-ffp-contract=off"]
    flags += (["-O1", "-g", "-fsanitize=undefined", "-fno-sanitize-recover=all"]
              if sanitize else ["-O3", "-DNDEBUG"])
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=destination, prefix=".audit-") as temporary:
        target = Path(temporary) / library.name
        command = [executable, *flags, "-I" + sysconfig.get_path("include"),
                   str(SOURCE), "-o", str(target)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr)
        record = {
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "command": command,
            "compiler": subprocess.run([executable, "--version"], check=True,
                                       capture_output=True, text=True).stdout,
            "elapsed_ns": time.perf_counter_ns() - started,
            "sanitize": sanitize,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        os.link(target, library)
        with receipt.open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return library


@dataclass(frozen=True)
class AuditResult:
    sat: int
    unsat: int
    sat_check_ns: int
    proof_check_ns: int
    proof_edges: int
    query_ns: tuple[int, ...]
    elapsed_ns: int


class SessionAuditRuntime:
    def __init__(self, library: str | Path):
        spec = importlib.util.spec_from_file_location(
            "_spectra_session_audit", Path(library).resolve())
        if spec is None or spec.loader is None:
            raise ValueError("invalid trusted audit extension path")
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        if self._module.abi() != 1:
            raise ValueError("unsupported session-audit ABI")

    def prepare(self, edges, masks, queries, statuses, contradictions, *,
                max_bytes: int = DEFAULT_BYTES):
        return PreparedSessionAudit(self, edges, masks, queries, statuses,
                                    contradictions, max_bytes=max_bytes)


class PreparedSessionAudit:
    def __init__(self, runtime: SessionAuditRuntime, edges, masks, queries,
                 statuses, contradictions, *, max_bytes: int):
        self._module = runtime._module
        self._lock = threading.RLock()
        self._handle = self._module.create(
            tuple(tuple(edge) for edge in edges), tuple(masks),
            tuple(tuple(tuple(item) for item in query) for query in queries),
            tuple(statuses), tuple(contradictions), max_bytes,
        )

    @property
    def info(self) -> dict:
        with self._lock:
            if self._handle is None:
                raise RuntimeError("session audit is closed")
            values = self._module.info(self._handle)
        return dict(zip(("vertices", "queries", "edges", "proof_literals",
                         "payload_bytes"), values))

    def verify(self, statuses: bytes, labels: bytes) -> AuditResult:
        started = time.perf_counter_ns()
        with self._lock:
            if self._handle is None:
                raise RuntimeError("session audit is closed")
            values = self._module.verify_batch(self._handle, statuses, labels)
        return AuditResult(*values[:-1], tuple(values[-1]),
                           time.perf_counter_ns() - started)

    def close(self) -> None:
        with self._lock:
            self._handle = None

    def __enter__(self):
        with self._lock:
            if self._handle is None:
                raise RuntimeError("session audit is closed")
        return self

    def __exit__(self, *_):
        self.close()
