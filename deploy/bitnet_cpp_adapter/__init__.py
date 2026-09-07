"""Adapters to SPECTRA native W1.58A8 CPU backends.

The fused sparse ctypes path uses the Milestone-02 checked C ABI. It rejects
invalid NumPy dtype/shape/contiguity/buffer contracts before entering native code,
and the native function repeats the structural/content checks.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

_KERNEL_DIR = Path(__file__).resolve().parents[1] / "cpp_sparse_kernel"


def _shared_lib_name() -> str:
    if sys.platform == "win32":
        return "spectra_kernel.dll"
    if sys.platform == "darwin":
        return "libspectra_kernel.dylib"
    return "libspectra_kernel.so"


def sparse_kernel_path() -> Path:
    return _KERNEL_DIR / "build" / _shared_lib_name()


def is_sparse_kernel_available() -> bool:
    return sparse_kernel_path().exists()


def _cpu_supports_avx2() -> bool:
    try:
        from deploy.torch_kernel import cpu_supports_avx2
        return cpu_supports_avx2()
    except Exception:
        return False


def load_sparse_kernel() -> ctypes.CDLL:
    path = sparse_kernel_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Sparse kernel not built at {path}. Build with `cmake -B build && "
            f"cmake --build build` in {_KERNEL_DIR}."
        )
    lib = ctypes.CDLL(str(path))
    I8 = ctypes.POINTER(ctypes.c_int8)
    I32 = ctypes.POINTER(ctypes.c_int32)
    U8 = ctypes.POINTER(ctypes.c_uint8)
    SZ = ctypes.c_size_t
    lib.spectra_sparse_ternary_gemv.restype = ctypes.c_int
    lib.spectra_sparse_ternary_gemv.argtypes = [
        I8, SZ, I32, SZ, U8, SZ, I32, SZ,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, I8, SZ,
    ]
    lib.spectra_status_string.restype = ctypes.c_char_p
    lib.spectra_status_string.argtypes = [ctypes.c_int]
    lib.spectra_compiled_with_avx2.restype = ctypes.c_int
    lib.spectra_compiled_with_avx2.argtypes = []
    if lib.spectra_compiled_with_avx2() and not _cpu_supports_avx2():
        raise RuntimeError(
            "loaded SPECTRA library was compiled with AVX2 but this CPU does not "
            "advertise AVX2; rebuild with -DSPECTRA_AVX2=OFF for scalar fallback"
        )
    return lib


def _require_array(a, dtype, ndim, name):
    if not isinstance(a, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if a.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}, got {a.dtype}")
    if a.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {a.shape}")
    if not a.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return a


def sparse_ternary_gemv(x, active_idx, w_packed, requant_mult, shift: int, out_dim: int):
    """Checked NumPy/ctypes sparse GEMV wrapper; returns int8 [tokens,out_dim]."""
    x = _require_array(x, np.int8, 2, "x")
    active_idx = _require_array(active_idx, np.int32, 1, "active_idx")
    w_packed = _require_array(w_packed, np.uint8, 1, "w_packed")
    requant_mult = _require_array(requant_mult, np.int32, 1, "requant_mult")
    if not isinstance(shift, int) or isinstance(shift, bool) or not 0 <= shift <= 62:
        raise ValueError("shift must be an integer in [0, 62]")
    if not isinstance(out_dim, int) or isinstance(out_dim, bool) or out_dim <= 0:
        raise ValueError("out_dim must be a positive integer")
    num_tokens, hidden = map(int, x.shape)
    if num_tokens <= 0 or hidden <= 0 or hidden > (2**31 - 1) // 128:
        raise ValueError("invalid x dimensions for int32 accumulation")
    if active_idx.size:
        if int(active_idx.min()) < 0 or int(active_idx.max()) >= num_tokens:
            raise ValueError(f"active_idx entries must be in [0, {num_tokens})")
    expected_w = out_dim * ((hidden + 3) // 4)
    if w_packed.size != expected_w:
        raise ValueError(f"w_packed length must be exactly {expected_w}")
    if requant_mult.size != out_dim:
        raise ValueError(f"requant_mult length must be exactly {out_dim}")
    if (requant_mult < 0).any():
        raise ValueError("requant_mult entries must be non-negative")

    y = np.zeros((num_tokens, out_dim), dtype=np.int8)
    lib = load_sparse_kernel()
    status = lib.spectra_sparse_ternary_gemv(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), x.size,
        active_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), active_idx.size,
        w_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), w_packed.size,
        requant_mult.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), requant_mult.size,
        shift, num_tokens, hidden, out_dim,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), y.size,
    )
    if status != 0:
        msg = lib.spectra_status_string(status).decode("utf-8", errors="replace")
        raise ValueError(f"native sparse kernel rejected input: {msg} (status={status})")
    return y


def bitnet_cpp_root() -> Path | None:
    root = os.environ.get("SPECTRA_BITNET_CPP")
    return Path(root) if root and Path(root).exists() else None


def is_bitnet_cpp_available() -> bool:
    return bitnet_cpp_root() is not None
