"""Explicitly built implication-closure engine for repeated binary support queries."""
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

SOURCE = Path(__file__).with_name("compiled_support.cpp")
DEFAULT_BYTES = 512 * 1024 * 1024


def build_compiled_support(directory: str | Path, *, compiler: str = "g++",
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
    library = destination / ("_spectra_compiled_support" + sysconfig.get_config_var("EXT_SUFFIX"))
    receipt = destination / "build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("compiled support outputs already exist")
    flags = [
        "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
        "-fno-fast-math", "-ffp-contract=off",
    ]
    flags += (["-O1", "-g", "-fsanitize=undefined", "-fno-sanitize-recover=all"]
              if sanitize else ["-O3", "-DNDEBUG"])
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=destination, prefix=".compiled-support-") as temporary:
        target = Path(temporary) / library.name
        include = sysconfig.get_path("include")
        command = [executable, *flags, "-I" + include, str(SOURCE), "-o", str(target)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        record = {
            "schema": "spectra.real_traffic.compiled_support_build.v1",
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "command": command,
            "compiler": subprocess.run(
                [executable, "--version"], capture_output=True, text=True,
                check=True, timeout=10,
            ).stdout,
            "elapsed_ns": time.perf_counter_ns() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "python_include": include,
            "sanitize": sanitize,
        }
        os.link(target, library)
        with receipt.open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return library


@dataclass(frozen=True)
class CompiledAnswer:
    status: str
    labels: bytes
    elapsed_ns: int


class CompiledSupportRuntime:
    """Load one explicitly built trusted extension; never compile on import."""

    def __init__(self, library: str | Path):
        spec = importlib.util.spec_from_file_location(
            "_spectra_compiled_support", Path(library).resolve())
        if spec is None or spec.loader is None:
            raise ValueError("invalid compiled support extension path")
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        if self._module.abi() != 1:
            raise ValueError("unsupported compiled support ABI")

    def prepare(self, variables: int, colors: int,
                edges: tuple[tuple[int, int], ...], *,
                masks: tuple[int, ...],
                max_bytes: int = DEFAULT_BYTES) -> "PreparedCompiledSupport":
        return PreparedCompiledSupport(
            self, variables, colors, edges, masks, max_bytes)


class PreparedCompiledSupport:
    def __init__(self, runtime: CompiledSupportRuntime, variables: int,
                 colors: int, edges: tuple[tuple[int, int], ...],
                 masks: tuple[int, ...], max_bytes: int):
        self._lock = threading.RLock()
        self._module = runtime._module
        self._handle = self._module.create(
            variables, colors, edges, masks, max_bytes)

    @property
    def info(self) -> dict:
        with self._lock:
            if self._handle is None:
                raise RuntimeError("compiled support index is closed")
            variables, components, payload, build_bound, words = self._module.info(
                self._handle)
        return {
            "variables": variables,
            "components": components,
            "payload_bytes": payload,
            "build_payload_bound": build_bound,
            "closure_words": words,
        }

    def solve(self, restrictions: tuple[tuple[int, int], ...] = ()) -> CompiledAnswer:
        started = time.perf_counter_ns()
        with self._lock:
            if self._handle is None:
                raise RuntimeError("compiled support index is closed")
            labels, sat = self._module.solve(self._handle, restrictions)
        if type(labels) is not bytes or type(sat) is not bool:
            raise AssertionError("compiled support returned an invalid result shape")
        if sat and not labels:
            raise AssertionError("compiled support returned SAT without a witness")
        if not sat and labels:
            raise AssertionError("compiled support returned UNSAT with a witness")
        return CompiledAnswer(
            "SAT_VERIFIED" if sat else "UNSAT_REPORTED",
            labels,
            time.perf_counter_ns() - started,
        )

    def close(self) -> None:
        with self._lock:
            self._handle = None

    def __enter__(self) -> "PreparedCompiledSupport":
        with self._lock:
            if self._handle is None:
                raise RuntimeError("compiled support index is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
