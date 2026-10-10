"""Direct native session adapter for a pinned, unmodified MiniCard solver."""
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

SOURCE = Path(__file__).with_name("minicard_session.cpp")
PINNED_COMMIT = "79776615ddc8803dd86803ea6f00838fc74349b1"
DEFAULT_BYTES = 64 * 1024 * 1024
REQUIRED = (
    "minicard/Solver.cc", "minicard/Solver.h", "minicard/SolverTypes.h",
    "mtl/Alg.h", "mtl/Alloc.h", "mtl/Heap.h", "mtl/IntTypes.h", "mtl/Map.h",
    "mtl/Queue.h", "mtl/Sort.h", "mtl/Vec.h", "mtl/XAlloc.h",
    "utils/Options.cc", "utils/Options.h", "utils/ParseUtils.h",
    "utils/System.cc", "utils/System.h",
)


def source_manifest(source: str | Path) -> dict[str, str]:
    root = Path(source).resolve()
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("incomplete MiniCard source: " + ", ".join(missing))
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in REQUIRED}


def build_minicard_session(directory: str | Path, *, source: str | Path,
                           compiler: str = "g++", sanitize: bool = False) -> Path:
    if platform.system() != "Linux":
        raise NotImplementedError("Linux builder only; other platforms unvalidated")
    if type(sanitize) is not bool:
        raise TypeError("sanitize must be bool")
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError(compiler)
    upstream = Path(source).resolve()
    manifest = source_manifest(upstream)
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / ("_spectra_minicard_session" + sysconfig.get_config_var("EXT_SUFFIX"))
    receipt = destination / "build.json"
    if library.exists() or receipt.exists():
        raise FileExistsError("build outputs already exist")
    flags = ["-std=c++17", "-shared", "-fPIC", "-fno-fast-math", "-ffp-contract=off",
             "-Wno-parentheses", "-Wno-sign-compare", "-Wno-unused-parameter"]
    flags += (["-O1", "-g", "-fsanitize=undefined", "-fno-sanitize-recover=all"]
              if sanitize else ["-O3", "-DNDEBUG"])
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=destination, prefix=".minicard-") as temporary:
        target = Path(temporary) / library.name
        command = [
            executable, *flags, "-I" + sysconfig.get_path("include"), "-I" + str(upstream),
            str(SOURCE), str(upstream / "minicard/Solver.cc"),
            str(upstream / "utils/Options.cc"), str(upstream / "utils/System.cc"),
            "-lz", "-o", str(target),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(result.stderr)
        record = {
            "adapter_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "upstream_commit": PINNED_COMMIT,
            "upstream_manifest": manifest,
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
class MiniCardBatchResult:
    statuses: tuple[str, ...]
    labels: tuple[bytes | None, ...]
    engine_ns: tuple[int, ...]
    decisions: tuple[int, ...]
    conflicts: tuple[int, ...]
    elapsed_ns: int


class MiniCardSessionRuntime:
    def __init__(self, library: str | Path):
        spec = importlib.util.spec_from_file_location(
            "_spectra_minicard_session", Path(library).resolve())
        if spec is None or spec.loader is None:
            raise ValueError("invalid trusted MiniCard extension path")
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        if self._module.abi() != 1:
            raise ValueError("unsupported MiniCard-session ABI")

    def prepare(self, edges: Sequence[tuple[int, int]], masks: Sequence[int], *,
                max_bytes: int = DEFAULT_BYTES):
        return PreparedMiniCardSession(self, edges, masks, max_bytes=max_bytes)


class PreparedMiniCardSession:
    def __init__(self, runtime: MiniCardSessionRuntime,
                 edges: Sequence[tuple[int, int]], masks: Sequence[int], *, max_bytes: int):
        self._module = runtime._module
        self._lock = threading.RLock()
        self._n = len(masks)
        self._handle = self._module.create(tuple(tuple(edge) for edge in edges),
                                           tuple(masks), max_bytes)

    @property
    def info(self) -> dict:
        with self._lock:
            if self._handle is None:
                raise RuntimeError("MiniCard session is closed")
            values = self._module.info(self._handle)
        return dict(zip(("variables", "palette", "clauses", "adapter_payload_bytes"), values))

    def solve_raw(self, queries: Sequence[tuple[tuple[int, int], ...]], *,
                  max_bytes: int = DEFAULT_BYTES):
        """Return native status and witness buffers without per-answer slicing."""
        exact_queries = tuple(tuple(tuple(item) for item in query) for query in queries)
        started = time.perf_counter_ns()
        with self._lock:
            if self._handle is None:
                raise RuntimeError("MiniCard session is closed")
            statuses, blob, times, decisions, conflicts = self._module.solve_batch(
                self._handle, exact_queries, max_bytes)
        if len(statuses) != len(exact_queries) or len(blob) != len(exact_queries) * self._n:
            raise AssertionError("MiniCard session output geometry differs")
        if any(status not in (0, 1) for status in statuses):
            raise AssertionError("MiniCard session returned unknown status")
        return (statuses, blob, tuple(times), tuple(decisions), tuple(conflicts),
                time.perf_counter_ns() - started)

    def solve_batch(self, queries: Sequence[tuple[tuple[int, int], ...]], *,
                    max_bytes: int = DEFAULT_BYTES) -> MiniCardBatchResult:
        statuses, blob, times, decisions, conflicts, elapsed = self.solve_raw(
            queries, max_bytes=max_bytes)
        named = []
        outputs = []
        for index, status in enumerate(statuses):
            if status == 0:
                named.append("SAT")
                outputs.append(blob[index * self._n:(index + 1) * self._n])
            else:
                named.append("UNSAT")
                outputs.append(None)
        return MiniCardBatchResult(tuple(named), tuple(outputs), times,
                                   decisions, conflicts, elapsed)

    def close(self) -> None:
        with self._lock:
            self._handle = None

    def __enter__(self):
        with self._lock:
            if self._handle is None:
                raise RuntimeError("MiniCard session is closed")
        return self

    def __exit__(self, *_):
        self.close()
