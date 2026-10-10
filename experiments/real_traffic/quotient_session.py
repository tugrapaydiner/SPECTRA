"""Exact quotient-compiled sessions for repeated binary-list support queries.

The runtime consumes the public integer-only quotient certificate, not private
search state.  It evaluates each query as a 2-SAT instance over the residual
implication graph, returns complete original-address byte witnesses, and reports
UNSAT without treating exhaustion or a timeout as a proof.
"""
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

SOURCE = Path(__file__).with_name("quotient_session.cpp")
DEFAULT_BYTES = 64 * 1024 * 1024


def build_quotient_session(directory: str | Path, *, compiler: str = "g++",
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
    library = destination / ("_spectra_quotient_session" + sysconfig.get_config_var("EXT_SUFFIX"))
    receipt = destination / "build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("quotient session outputs already exist")
    flags = ["-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
             "-fno-fast-math", "-ffp-contract=off"]
    flags += (["-O1", "-g", "-fsanitize=undefined", "-fno-sanitize-recover=all"]
              if sanitize else ["-O3", "-DNDEBUG"])
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=destination, prefix=".quotient-session-") as temporary:
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
class QuotientBatchResult:
    statuses: tuple[str, ...]
    labels: tuple[bytes | None, ...]
    engine_ns: tuple[int, ...]
    traversed_edges: tuple[int, ...]
    elapsed_ns: int


class QuotientSessionRuntime:
    def __init__(self, library: str | Path):
        spec = importlib.util.spec_from_file_location(
            "_spectra_quotient_session", Path(library).resolve())
        if spec is None or spec.loader is None:
            raise ValueError("invalid trusted native extension path")
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        if self._module.abi() != 1:
            raise ValueError("unsupported quotient-session ABI")

    def prepare(self, certificate: dict, *, max_bytes: int = DEFAULT_BYTES):
        return PreparedQuotientSession(self, certificate, max_bytes=max_bytes)


class PreparedQuotientSession:
    def __init__(self, runtime: QuotientSessionRuntime, certificate: dict,
                 *, max_bytes: int):
        required = {"node", "colour0", "colour1", "wide", "palettes", "initial",
                    "offsets", "arcs"}
        if type(certificate) is not dict or not required <= set(certificate):
            raise ValueError("quotient certificate is incomplete")
        if any(value != 0 for value in certificate["wide"]):
            raise ValueError("quotient session currently requires binary original vertices")
        self._module = runtime._module
        self._lock = threading.RLock()
        self._n = len(certificate["node"])
        self._handle = self._module.create(
            tuple(certificate["node"]), tuple(certificate["colour0"]),
            tuple(certificate["colour1"]), tuple(certificate["palettes"]),
            tuple(certificate["initial"]), tuple(certificate["offsets"]),
            tuple(tuple(row) for row in certificate["arcs"]), max_bytes,
        )

    @property
    def info(self) -> dict:
        with self._lock:
            if self._handle is None:
                raise RuntimeError("quotient session is closed")
            values = self._module.info(self._handle)
        return dict(zip(("vertices", "quotient_vertices", "implication_edges",
                         "payload_bytes"), values))

    def solve_raw(self, queries: Sequence[tuple[tuple[int, int], ...]], *,
                  compiled: bool = True, max_bytes: int = DEFAULT_BYTES):
        """Return the exact native status and witness buffers without slicing them."""
        exact_queries = tuple(tuple(tuple(item) for item in query) for query in queries)
        started = time.perf_counter_ns()
        with self._lock:
            if self._handle is None:
                raise RuntimeError("quotient session is closed")
            native = (self._module.solve_batch_closure if compiled
                      else self._module.solve_batch)
            statuses, blob, engine_ns, traversed = native(
                self._handle, exact_queries, max_bytes)
        if len(statuses) != len(exact_queries) or len(blob) != len(exact_queries) * self._n:
            raise AssertionError("quotient session output geometry differs")
        if any(status not in (0, 1) for status in statuses):
            raise AssertionError("quotient session returned an unknown status")
        return statuses, blob, tuple(engine_ns), tuple(traversed), (
            time.perf_counter_ns() - started)

    def _solve_batch(self, method: str,
                     queries: Sequence[tuple[tuple[int, int], ...]], *,
                     max_bytes: int) -> QuotientBatchResult:
        statuses, blob, engine_ns, traversed, elapsed = self.solve_raw(
            queries, compiled=method == "solve_batch_closure", max_bytes=max_bytes)
        outputs = []
        named = []
        for index, status in enumerate(statuses):
            if status == 0:
                named.append("SAT")
                outputs.append(blob[index * self._n:(index + 1) * self._n])
            else:
                named.append("UNSAT")
                outputs.append(None)
        return QuotientBatchResult(tuple(named), tuple(outputs), engine_ns,
                                  traversed, elapsed)

    def solve_batch(self, queries: Sequence[tuple[tuple[int, int], ...]], *,
                    max_bytes: int = DEFAULT_BYTES) -> QuotientBatchResult:
        """Evaluate each query with a fresh exact SCC pass."""
        return self._solve_batch("solve_batch", queries, max_bytes=max_bytes)

    def solve_compiled(self, queries: Sequence[tuple[tuple[int, int], ...]], *,
                       max_bytes: int = DEFAULT_BYTES) -> QuotientBatchResult:
        """Evaluate queries from precompiled implication closures."""
        return self._solve_batch("solve_batch_closure", queries, max_bytes=max_bytes)

    def close(self) -> None:
        with self._lock:
            self._handle = None

    def __enter__(self):
        with self._lock:
            if self._handle is None:
                raise RuntimeError("quotient session is closed")
        return self

    def __exit__(self, *_):
        self.close()
