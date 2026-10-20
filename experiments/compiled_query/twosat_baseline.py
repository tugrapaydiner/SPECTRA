"""Independent native 2-SAT challenger for the exact SBC1 -> SCQ2 contract.

This module intentionally shares no solver index, SCC labels, cached answers, or
native implementation with :mod:`spectra.exact`.  It consumes only the public
packet format and emits the public certified-answer format.  The common Python
verifier remains the final semantic judge in experiments.
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
import sys
import sysconfig
import tempfile
import time

SOURCE = Path(__file__).with_name("native_twosat.cpp")
TWOSAT_BASELINE_API = 1
TWOSAT_NATIVE_ABI = 1
DEFAULT_BYTES = 512 << 20
MAX_BYTES = (1 << 64) - 1


def build_twosat_runtime(
    directory: str | Path,
    *,
    compiler: str | None = None,
    sanitizer: str = "none",
) -> Path:
    """Compile the independent challenger and bind its source and binary hashes."""
    windows = sys.platform == "win32"
    if sys.platform not in ("linux", "win32"):
        raise NotImplementedError("native 2-SAT challenger supports Linux and Windows x64")
    if sys.implementation.name != "cpython" or sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise ValueError("GIL-enabled CPython is required")
    if sanitizer not in ("none", "undefined", "address,undefined"):
        raise ValueError("unsupported sanitizer")
    if windows and sanitizer != "none":
        raise ValueError("sanitizer builds currently require Linux")
    compiler = compiler or ("cl" if windows else "g++")
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise FileNotFoundError(compiler)

    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / (
        "_spectra_twosat_baseline" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    receipt = destination / "twosat-build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("2-SAT build outputs already exist")

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
    flags += (
        ["-O3", "-DNDEBUG"]
        if sanitizer == "none"
        else [
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-fsanitize=" + sanitizer,
            "-fno-sanitize-recover=all",
        ]
    )

    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        target = Path(temporary) / library.name
        if windows:
            from spectra.svm_build import command_for

            if sysconfig.get_config_var("Py_DEBUG"):
                raise ValueError("Windows debug Python is not supported")
            import_name = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
            import_library = Path(sys.base_prefix) / "libs" / import_name
            if not import_library.is_file():
                raise ValueError("CPython import library missing: " + str(import_library))
            command = command_for(
                SOURCE.resolve(),
                target,
                target="portable",
                compiler=compiler_path,
                include=Path(sysconfig.get_path("include")),
                import_library=import_library,
            )
        else:
            command = [
                compiler_path,
                *flags,
                "-I" + sysconfig.get_path("include"),
                str(SOURCE),
                "-o",
                str(target),
            ]
        result = subprocess.run(
            command,
            cwd=temporary,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        version = subprocess.run(
            [compiler_path] if windows else [compiler_path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=not windows,
        )
        record = {
            "baseline_api_version": TWOSAT_BASELINE_API,
            "native_abi": TWOSAT_NATIVE_ABI,
            "library_name": library.name,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "soabi": sysconfig.get_config_var("SOABI"),
            "platform": platform.platform(),
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "command": command,
            "sanitizer": sanitizer,
            "elapsed_ns": time.perf_counter_ns() - started,
            "compiler": version.stdout + version.stderr,
            "stdout": result.stdout,
            "stderr": result.stderr,
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
    receipt = library.with_name("twosat-build.json")
    try:
        record = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("2-SAT runtime requires a valid adjacent build receipt") from error
    required = {
        "baseline_api_version": TWOSAT_BASELINE_API,
        "native_abi": TWOSAT_NATIVE_ABI,
        "library_name": library.name,
        "python_implementation": platform.python_implementation(),
        "soabi": sysconfig.get_config_var("SOABI"),
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(f"2-SAT build receipt {key} differs")
    if record.get("source_sha256") != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
        raise ValueError("2-SAT source hash differs from the challenger wrapper")
    try:
        binary = library.read_bytes()
    except OSError as error:
        raise ValueError("2-SAT runtime library is not readable") from error
    if record.get("library_sha256") != hashlib.sha256(binary).hexdigest():
        raise ValueError("2-SAT runtime library hash differs from its receipt")
    return record


@dataclass(frozen=True)
class TwoSatPacketAnswer:
    output: bytes
    query_count: int
    sat_count: int
    unsat_count: int
    proof_bytes: int
    visited: int
    arcs_scanned: int
    info: dict


class TwoSatRuntime:
    """Receipt-bound independent challenger loaded from an explicit native build."""

    def __init__(self, library: str | Path, *, verify_receipt: bool = True):
        self.build_receipt = _read_receipt(Path(library)) if verify_receipt else None
        spec = importlib.util.spec_from_file_location(
            "_spectra_twosat_baseline", Path(library).resolve()
        )
        if spec is None or spec.loader is None:
            raise ValueError("invalid 2-SAT native extension path")
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        if self._module.abi() != TWOSAT_NATIVE_ABI:
            raise ValueError("unsupported 2-SAT challenger ABI")

    def solve_session_packet(
        self,
        packet: bytes,
        *,
        mode: str = "auto",
        max_bytes: int = DEFAULT_BYTES,
    ) -> TwoSatPacketAnswer:
        if type(packet) is not bytes:
            raise ValueError("session packet must be immutable exact bytes")
        if mode not in ("raw", "scc", "auto"):
            raise ValueError("mode must be raw, scc or auto")
        if type(max_bytes) is not int or not (1 << 20) <= max_bytes <= MAX_BYTES:
            raise ValueError("max_bytes must be an exact uint64 of at least 1 MiB")
        values = self._module.solve_session_packet(
            packet, {"raw": 0, "scc": 1, "auto": 2}[mode], max_bytes
        )
        output, count, sat, unsat, proof_bytes, visits, arcs, info = values
        info = {
            **info,
            "backend": "independent-native-2sat",
            "requested_mode": mode,
            "baseline_api_version": TWOSAT_BASELINE_API,
            "native_abi": TWOSAT_NATIVE_ABI,
            "prepared_input": True,
            "input_schema": "SBC1",
            "output_schema": "SCQ2",
            "portable_unsat_certificate": True,
            "query_answer_cache": False,
            "common_verifier_required": True,
        }
        return TwoSatPacketAnswer(
            output,
            count,
            sat,
            unsat,
            proof_bytes,
            visits,
            arcs,
            info,
        )
