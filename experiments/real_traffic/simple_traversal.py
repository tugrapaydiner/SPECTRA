"""Cheap exact implication traversal used as a non-promotional baseline."""
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

SOURCE = Path(__file__).with_name("simple_traversal.cpp")
DEFAULT_BYTES = 64 * 1024 * 1024


def build_simple_traversal(directory: str | Path, *, compiler: str = "g++",
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
    library = destination / (
        "_spectra_simple_traversal" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    receipt = destination / "build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("build outputs already exist")
    flags = [
        "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
        "-fno-fast-math", "-ffp-contract=off",
    ]
    flags += (
        ["-O1", "-g", "-fsanitize=undefined", "-fno-sanitize-recover=all"]
        if sanitize else ["-O3", "-DNDEBUG"]
    )
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(
            dir=destination, prefix=".traversal-") as temporary:
        target = Path(temporary) / library.name
        command = [
            executable, *flags, "-I" + sysconfig.get_path("include"),
            str(SOURCE), "-o", str(target),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        record = {
            "schema": "spectra.simple_traversal.build.v1",
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "command": command,
            "compiler": subprocess.run(
                [executable, "--version"], capture_output=True,
                text=True, check=True
            ).stdout,
            "sanitize": sanitize,
            "elapsed_ns": time.perf_counter_ns() - started,
        }
        os.link(target, library)
        with receipt.open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return library


@dataclass(frozen=True)
class TraversalResult:
    status: str
    labels: bytes
    implications_visited: int
    literals_visited: int
    elapsed_ns: int


class SimpleTraversalRuntime:
    def __init__(self, library: str | Path):
        spec = importlib.util.spec_from_file_location(
            "_spectra_simple_traversal", Path(library).resolve()
        )
        if spec is None or spec.loader is None:
            raise ValueError("invalid native extension path")
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)

    def prepare(self, edges: tuple, masks: tuple, *,
                max_bytes: int = DEFAULT_BYTES) -> "PreparedTraversal":
        return PreparedTraversal(self._module, edges, masks, max_bytes)


class PreparedTraversal:
    def __init__(self, module, edges, masks, max_bytes):
        self._module = module
        self._lock = threading.RLock()
        self._handle = module.create(edges, masks, max_bytes)

    @property
    def info(self) -> dict:
        with self._lock:
            if self._handle is None:
                raise RuntimeError("traversal is closed")
            vertices, implications, payload = self._module.info(self._handle)
        return {
            "vertices": vertices,
            "implications": implications,
            "payload_bytes": payload,
        }

    def solve(self, restrictions: tuple = ()) -> TraversalResult:
        started = time.perf_counter_ns()
        with self._lock:
            if self._handle is None:
                raise RuntimeError("traversal is closed")
            labels, reason, implications, literals = self._module.solve(
                self._handle, restrictions
            )
        return TraversalResult(
            "UNSAT" if reason else "SAT",
            labels,
            implications,
            literals,
            time.perf_counter_ns() - started,
        )

    def close(self) -> None:
        with self._lock:
            self._handle = None

    def __enter__(self):
        with self._lock:
            if self._handle is None:
                raise RuntimeError("traversal is closed")
        return self

    def __exit__(self, *_):
        self.close()
