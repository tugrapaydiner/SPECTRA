"""Milestone 10 native packed-ternary FP32 CPU primitive.

This module intentionally does not reuse the historical fused INT8 FFN because
that operator is not the trained GELU graph. M10 compiles a separate correctness-
first pybind extension whose only operation is a dense packed-ternary FP32 linear.
"""
from __future__ import annotations

import functools
import platform
import sys
from pathlib import Path
from typing import Any

import torch

_SRC = Path(__file__).resolve().parent / "m10_dense_extension.cpp"


def _compile_flags() -> list[str]:
    if sys.platform == "win32":
        return ["/O2", "/std:c++20"]
    return ["-O3", "-std=c++20"]


@functools.lru_cache(maxsize=1)
def load_extension() -> Any:
    """Build/load the M10 correctness-first native extension."""
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="spectra_m10_dense_v1",
            sources=[str(_SRC)],
            extra_cflags=_compile_flags(),
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to build/load M10 native dense backend with flags {_compile_flags()}: {exc}"
        ) from exc


def available() -> bool:
    try:
        load_extension()
        return True
    except RuntimeError:
        return False


def operator_identity() -> str:
    return str(load_extension().operator_identity())


def backend_identity() -> dict[str, object]:
    return {
        "backend": "spectra_m10_native",
        "operator": operator_identity(),
        "implementation": "scalar_cpp_fp32_accumulation",
        "vectorized": False,
        "packed_weight_bits": 2,
        "input_dtype": "float32",
        "accumulator_dtype": "float32",
        "output_dtype": "float32",
        "compile_flags": _compile_flags(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _require(t: torch.Tensor, dtype: torch.dtype, ndim: int, name: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if t.device.type != "cpu":
        raise ValueError(f"{name} must be CPU, got {t.device}")
    if t.dtype is not dtype:
        raise TypeError(f"{name} must be {dtype}, got {t.dtype}")
    if t.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {tuple(t.shape)}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return t


def dense_ternary_linear_fp32(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    row_scale: torch.Tensor,
    bias: torch.Tensor,
    out_dim: int,
) -> torch.Tensor:
    """Checked M10 native linear over row-padded packed ternary weights."""
    _require(x, torch.float32, 2, "x")
    _require(packed_weight, torch.uint8, 1, "packed_weight")
    _require(row_scale, torch.float32, 1, "row_scale")
    _require(bias, torch.float32, 1, "bias")
    if not isinstance(out_dim, int) or isinstance(out_dim, bool) or out_dim <= 0:
        raise ValueError("out_dim must be a positive integer")
    hidden = int(x.shape[1])
    expected = out_dim * ((hidden + 3) // 4)
    if packed_weight.numel() != expected:
        raise ValueError(f"packed_weight must contain exactly {expected} bytes")
    if row_scale.numel() != out_dim:
        raise ValueError("row_scale must contain out_dim values")
    if bias.numel() not in {0, out_dim}:
        raise ValueError("bias must contain zero or out_dim values")
    if not bool(torch.isfinite(row_scale).all()) or bool((row_scale < 0).any()):
        raise ValueError("row_scale must be finite and non-negative")
    if bias.numel() and not bool(torch.isfinite(bias).all()):
        raise ValueError("bias must be finite")
    with torch.profiler.record_function("spectra::dense_ternary_linear_fp32"):
        return load_extension().dense_ternary_linear_fp32(
            x, packed_weight, row_scale, bias, out_dim
        )
