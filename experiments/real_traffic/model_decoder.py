"""Explicitly built native model decoding for the MiniCard control arm."""
from __future__ import annotations

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

SOURCE = Path(__file__).with_name("model_decoder.cpp")
DEFAULT_BYTES = 8 * 1024 * 1024


def build_model_decoder(directory: str | Path, *, compiler: str = "g++",
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
    library = destination / ("_spectra_model_decoder" + sysconfig.get_config_var("EXT_SUFFIX"))
    receipt = destination / "build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("decoder build outputs already exist")
    flags = [
        "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
        "-fno-fast-math", "-ffp-contract=off",
    ]
    flags += (["-O1", "-g", "-fsanitize=undefined", "-fno-sanitize-recover=all"]
              if sanitize else ["-O3", "-DNDEBUG"])
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=destination, prefix=".decoder-") as temporary:
        target = Path(temporary) / library.name
        include = sysconfig.get_path("include")
        command = [executable, *flags, "-I" + include, str(SOURCE), "-o", str(target)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        record = {
            "schema": "spectra.real_traffic.model_decoder_build.v1",
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


class ModelDecoderRuntime:
    """Load one explicitly built trusted extension; never compile on import."""

    def __init__(self, library: str | Path):
        spec = importlib.util.spec_from_file_location(
            "_spectra_model_decoder", Path(library).resolve())
        if spec is None or spec.loader is None:
            raise ValueError("invalid model decoder extension path")
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        if self._module.abi() != 1:
            raise ValueError("unsupported model decoder ABI")

    def prepare(self, masks: tuple[int, ...], *,
                max_bytes: int = DEFAULT_BYTES) -> "PreparedModelDecoder":
        return PreparedModelDecoder(self, masks, max_bytes)


class PreparedModelDecoder:
    def __init__(self, runtime: ModelDecoderRuntime, masks: tuple[int, ...],
                 max_bytes: int):
        self._lock = threading.RLock()
        self._module = runtime._module
        self._handle = self._module.create(masks, max_bytes)

    @property
    def info(self) -> dict:
        with self._lock:
            if self._handle is None:
                raise RuntimeError("model decoder is closed")
            variables, payload = self._module.info(self._handle)
        return {"variables": variables, "payload_bytes": payload}

    def decode(self, model: list[int] | tuple[int, ...]) -> bytes:
        with self._lock:
            if self._handle is None:
                raise RuntimeError("model decoder is closed")
            result = self._module.decode(self._handle, model)
        if type(result) is not bytes:
            raise AssertionError("native model decoder returned a non-byte witness")
        return result

    def close(self) -> None:
        with self._lock:
            self._handle = None

    def __enter__(self) -> "PreparedModelDecoder":
        with self._lock:
            if self._handle is None:
                raise RuntimeError("model decoder is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
