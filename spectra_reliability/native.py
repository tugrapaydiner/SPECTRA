"""Explicit lazy CPU build; immutable validated weights and native exact checks.

AVX2 keeps an EXTRA int8 transposed layout. Its payload cost is disclosed; this
is NOT packed-only execution or a cache-residency claim. No silent SIMD fallback.
"""
from __future__ import annotations
import ctypes as C
import hashlib
import os
import platform
import shutil
import subprocess
import threading
import weakref
from pathlib import Path
import numpy as np
from .identity import canonical_json, file_sha256, positive_int, strict_json, write_json


class NativeBuildError(RuntimeError):
    pass


def build(cache_dir: str | Path) -> Path:
    source = Path(__file__).with_name("native_cpu.cpp")
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None: raise NativeBuildError("local g++/clang++ required; no download attempted")
    if platform.system() != "Linux": raise NativeBuildError("retained native build supports Linux only")
    import fcntl
    flags = ["-O3", "-std=c++17", "-fPIC", "-shared", "-ffp-contract=off", "-fno-fast-math", "-fno-tree-vectorize"]
    version = subprocess.run([compiler, "--version"], check=True, capture_output=True, text=True).stdout
    descriptor = {"source_sha256": file_sha256(source), "compiler": compiler, "compiler_version": version,
                  "flags": flags, "machine": platform.machine(), "platform": platform.platform()}
    key = hashlib.sha256(canonical_json(descriptor)).hexdigest()[:24]
    directory = Path(cache_dir) / key; directory.mkdir(parents=True, exist_ok=True)
    target = directory / "spectra_cpu.so"; meta = directory / "build.json"
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists() and meta.exists():
            saved = strict_json(meta.read_bytes())
            if saved.get("descriptor") == descriptor and saved.get("binary_sha256") == file_sha256(target): return target
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        try:
            command = [compiler, *flags, str(source), "-o", str(temporary)]
            done = subprocess.run(command, capture_output=True, text=True, timeout=180)
            (directory / "compiler.log").write_text(done.stdout + done.stderr)
            if done.returncode: raise NativeBuildError(f"compilation failed: {directory / 'compiler.log'}")
            temporary.replace(target)
            write_json(meta, {"descriptor": descriptor, "binary_sha256": file_sha256(target), "command": command})
        finally:
            temporary.unlink(missing_ok=True)
    return target


def _array(value, dtype, ndim, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.ndim != ndim or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous {np.dtype(dtype)} rank {ndim}; implicit conversion prohibited")
    return value


def _p(a): return C.c_void_p(int(a.ctypes.data))


class NativeCPU:
    def __init__(self, library: str | Path):
        self.path = Path(library).resolve(); self.lib = C.CDLL(str(self.path)); lib = self.lib
        lib.spectra_abi_version.argtypes = []; lib.spectra_abi_version.restype = C.c_int
        if lib.spectra_abi_version() != 1: raise NativeBuildError("native ABI mismatch")
        lib.spectra_has_avx2.argtypes = []; lib.spectra_has_avx2.restype = C.c_int
        lib.spectra_weight_create.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int, C.c_int]
        lib.spectra_weight_create.restype = C.c_void_p
        lib.spectra_weight_destroy.argtypes = [C.c_void_p]; lib.spectra_weight_destroy.restype = None
        lib.spectra_weight_linear.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int]
        lib.spectra_weight_linear.restype = C.c_int
        lib.spectra_checked_linear.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_void_p]
        lib.spectra_checked_linear.restype = C.c_int
        lib.spectra_sudoku_valid.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
        lib.spectra_sudoku_valid.restype = C.c_int

    @property
    def has_avx2(self) -> bool: return bool(self.lib.spectra_has_avx2())

    def sudoku_valid(self, puzzle: np.ndarray, answer: np.ndarray, box: int) -> bool:
        x = _array(puzzle, np.int64, 1, "puzzle"); y = _array(answer, np.int64, 1, "answer")
        box = positive_int(box, "box")
        if box > 7 or x.shape != y.shape or len(x) != box**4: raise ValueError("invalid Sudoku geometry")
        result = self.lib.spectra_sudoku_valid(_p(x), _p(y), len(x), box)
        if result not in (0, 1): raise RuntimeError("native checker rejected wrapper-validated input")
        return bool(result)

    def weight(self, packed, scales, bias, hidden: int):
        return ValidatedWeight(self, packed, scales, bias, hidden)

    def checked_linear(self, x, packed, scales, bias, *, hidden: int):
        xx = _array(x, np.float32, 2, "x"); w, s, b, out = _weight_arrays(packed, scales, bias, hidden)
        _linear_shape(xx, hidden, out); y = np.empty((len(xx), out), dtype=np.float32)
        result = self.lib.spectra_checked_linear(_p(xx), len(xx), _p(w), len(w), _p(s), len(s), _p(b), len(b), out, hidden, _p(y))
        if result: raise ValueError("invalid packed codes/padding/scales/bias")
        return y


def _weight_arrays(packed, scales, bias, hidden):
    hidden = positive_int(hidden, "hidden")
    w = _array(packed, np.uint8, 1, "packed"); s = _array(scales, np.float32, 1, "scales")
    b = np.empty(0, dtype=np.float32) if bias is None else _array(bias, np.float32, 1, "bias"); out = len(s)
    if out <= 0 or hidden > 100000 or out > 100000 or out * hidden > 1 << 28: raise ValueError("unsupported dimensions")
    if len(w) != out * ((hidden + 3)//4) or len(b) not in (0, out): raise ValueError("weight inventory mismatch")
    return w, s, b, out


def _linear_shape(x, hidden, out):
    if len(x) < 1 or len(x) > 100000 or x.shape[1] != hidden or len(x) * max(hidden, out) > 1 << 28:
        raise ValueError("invalid linear input/output dimensions")


class ValidatedWeight:
    def __init__(self, native: NativeCPU, packed, scales, bias, hidden: int):
        w, s, b, out = _weight_arrays(packed, scales, bias, hidden)
        ptr = native.lib.spectra_weight_create(_p(w), len(w), _p(s), len(s), _p(b), len(b), out, hidden)
        if not ptr: raise ValueError("native construction failed: invalid payload or allocation failure")
        self._native = native; self._ptr = ptr; self._lock = threading.RLock(); self.hidden = int(hidden); self.out = out
        self._finalizer = weakref.finalize(self, native.lib.spectra_weight_destroy, C.c_void_p(ptr))
        self.memory = {"packed_bytes": int(w.nbytes), "unpacked_transposed_int8_bytes": out * hidden,
            "scales_bytes": int(s.nbytes), "bias_bytes": int(b.nbytes),
            "tensor_payload_bytes": int(w.nbytes + out * hidden + s.nbytes + b.nbytes),
            "scope": "owned payloads only; excludes allocator, handle, library and Python overhead"}

    def close(self):
        with self._lock:
            if self._finalizer.alive: self._finalizer()
            self._ptr = None

    def __enter__(self):
        if self._ptr is None: raise RuntimeError("weight is closed")
        return self

    def __exit__(self, *args): self.close()

    def linear(self, x: np.ndarray, *, avx2: bool = False) -> np.ndarray:
        xx = _array(x, np.float32, 2, "x"); _linear_shape(xx, self.hidden, self.out)
        if type(avx2) is not bool: raise ValueError("avx2 must be boolean")
        with self._lock:
            if self._ptr is None: raise RuntimeError("weight is closed")
            if avx2 and not self._native.has_avx2: raise RuntimeError("AVX2 unavailable; no silent scalar fallback")
            y = np.empty((len(xx), self.out), dtype=np.float32)
            result = self._native.lib.spectra_weight_linear(C.c_void_p(self._ptr), _p(xx), len(xx), self.hidden, _p(y), int(avx2))
            if result: raise RuntimeError(f"native linear failure {result}")
            return y


def pack_ternary(values: np.ndarray) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 2 or values.dtype.kind not in "iu":
        raise ValueError("ternary weights must be a rank-2 integer array")
    if not np.isin(values, (-1, 0, 1)).all(): raise ValueError("weights must be exactly -1, 0, 1")
    out, hidden = values.shape
    if min(out, hidden) < 1: raise ValueError("empty weights")
    packed = np.zeros((out, (hidden+3)//4), dtype=np.uint8); codes = np.where(values == -1, 2, values).astype(np.uint8)
    for d in range(hidden): packed[:, d//4] |= codes[:, d] << (2*(d%4))
    return packed.reshape(-1)
