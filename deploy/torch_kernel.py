"""Loader for the SPECTRA PyTorch C++ extension with graceful fallback.

Prefers the native torch op (no Python/ctypes overhead in the recursion loop). If
the extension cannot be JIT-built (e.g. the host toolchain is ABI-incompatible
with torch), callers can fall back to the ctypes path in ``bitnet_cpp_adapter``
(validated for correctness) -- accepting the per-call overhead.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import torch

_SRC = Path(__file__).resolve().parent / "cpp_sparse_kernel" / "extension.cpp"


@functools.lru_cache(maxsize=1)
def load_extension() -> Any | None:
    """JIT-build and return the extension module, or ``None`` if unavailable."""
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="spectra_kernel_ext",
            sources=[str(_SRC)],
            extra_cflags=["-O3", "-mavx2", "-std=c++20"],
            verbose=False,
        )
    except Exception:  # toolchain / ABI mismatch -> caller falls back to ctypes
        return None


def available() -> bool:
    return load_extension() is not None


def _require_dtype(t: torch.Tensor, dtype: torch.dtype, name: str) -> torch.Tensor:
    """Hard W1.58A8 boundary guard: refuse to silently upcast across the bridge.

    The kernel reads raw bytes via ``data_ptr<int8/uint8/int32>``; a forgotten
    ``.to(torch.int8)`` upstream would reinterpret an FP32 buffer as garbage. We
    fail loudly instead -- the explicit "tensor upcast detector" the blueprint's
    strict dtype mandate requires.
    """
    if t.dtype is not dtype:
        raise TypeError(
            f"{name} must be {dtype} at the C++ kernel boundary, got {t.dtype}. "
            f"Quantize explicitly (e.g. .to({dtype})); the bridge will not upcast."
        )
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous (the kernel reads raw bytes)")
    return t


def sparse_ternary_gemv(
    x: torch.Tensor,
    active_idx: torch.Tensor,
    w_packed: torch.Tensor,
    requant_mult: torch.Tensor,
    shift: int,
    out_dim: int,
) -> torch.Tensor:
    """Native fused sparse ternary GEMV (raises if the extension is not built)."""
    _require_dtype(x, torch.int8, "x (INT8 activations)")
    _require_dtype(active_idx, torch.int32, "active_idx")
    _require_dtype(w_packed, torch.uint8, "w_packed (2-bit ternary)")
    _require_dtype(requant_mult, torch.int32, "requant_mult")
    ext = load_extension()
    if ext is None:
        raise RuntimeError("spectra_kernel_ext is not built on this host")
    return ext.sparse_ternary_gemv(x, active_idx, w_packed, requant_mult, shift, out_dim)


def fused_ternary_ffn(
    x: torch.Tensor,
    active_idx: torch.Tensor,
    w1_packed: torch.Tensor,
    mult1: torch.Tensor,
    w2_packed: torch.Tensor,
    mult2: torch.Tensor,
    shift: int,
    inter_dim: int,
    out_dim: int,
) -> torch.Tensor:
    """Fused two-layer ternary FFN; the loop runs entirely in C++."""
    _require_dtype(x, torch.int8, "x (INT8 activations)")
    _require_dtype(active_idx, torch.int32, "active_idx")
    _require_dtype(w1_packed, torch.uint8, "w1_packed")
    _require_dtype(w2_packed, torch.uint8, "w2_packed")
    _require_dtype(mult1, torch.int32, "mult1")
    _require_dtype(mult2, torch.int32, "mult2")
    ext = load_extension()
    if ext is None:
        raise RuntimeError("spectra_kernel_ext is not built on this host")
    return ext.fused_ternary_ffn(
        x, active_idx, w1_packed, mult1, w2_packed, mult2, shift, inter_dim, out_dim
    )
