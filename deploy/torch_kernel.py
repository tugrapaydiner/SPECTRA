"""Checked PyTorch bridge for SPECTRA's native ternary CPU kernels.

Milestone 02 makes the public contract explicit. Inputs are CPU-only, contiguous,
shape-checked tensors; the complete INT8 activation range is supported. The JIT
loader detects AVX2 and builds either the AVX2 implementation or the real scalar
implementation. Backend build failures are raised with the underlying exception
instead of being silently converted into a fake success.
"""

from __future__ import annotations

import functools
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch

_SRC = Path(__file__).resolve().parent / "cpp_sparse_kernel" / "extension.cpp"
_MAX_DOT_WIDTH = (2**31 - 1) // 128


def cpu_supports_avx2() -> bool:
    """Conservatively detect AVX2 on the current CPU.

    Unknown/non-x86 hosts return False and therefore use the scalar backend.
    """
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64", "i386", "i686", "x86"}:
        return False
    try:
        cap = str(torch.backends.cpu.get_cpu_capability()).upper()
        if "AVX2" in cap or "AVX512" in cap:
            return True
        if cap in {"DEFAULT", "NO AVX", "AVX"}:
            return False
    except Exception:
        pass
    try:
        flags = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").lower()
        return " avx2 " in f" {flags.replace(chr(10), ' ')} "
    except OSError:
        return False


def _compile_flags(use_avx2: bool) -> list[str]:
    if sys.platform == "win32":
        flags = ["/O2", "/std:c++20"]
        if use_avx2:
            flags.append("/arch:AVX2")
        return flags
    flags = ["-O3", "-std=c++20"]
    if use_avx2:
        flags.append("-mavx2")
    return flags


@functools.lru_cache(maxsize=1)
def load_extension() -> Any:
    """Build/load the host-safe native extension or raise a visible RuntimeError."""
    use_avx2 = cpu_supports_avx2()
    backend = "avx2" if use_avx2 else "scalar"
    flags = _compile_flags(use_avx2)
    try:
        from torch.utils.cpp_extension import load

        return load(
            name=f"spectra_kernel_ext_{backend}",
            sources=[str(_SRC)],
            extra_cflags=flags,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to build/load SPECTRA native {backend} backend with flags {flags}: {exc}"
        ) from exc


def available() -> bool:
    """Whether a native backend can be loaded; use load_extension() for the error."""
    try:
        load_extension()
        return True
    except RuntimeError:
        return False


def backend_name() -> str:
    """Return the backend actually compiled into the loaded extension."""
    return str(load_extension().backend_name())


def _require_tensor(
    t: torch.Tensor, dtype: torch.dtype, ndim: int, name: str
) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if t.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU, got {t.device}")
    if t.dtype is not dtype:
        raise TypeError(f"{name} must be {dtype}, got {t.dtype}")
    if t.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {tuple(t.shape)}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return t


def _require_shift(shift: int) -> int:
    if not isinstance(shift, int) or isinstance(shift, bool):
        raise TypeError(f"shift must be an int in [0, 62], got {type(shift).__name__}")
    if not 0 <= shift <= 62:
        raise ValueError(f"shift must be in [0, 62], got {shift}")
    return shift


def _require_positive_dim(value: int, name: str, *, dot_width: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    max_value = _MAX_DOT_WIDTH if dot_width else 2**31 - 1
    if not 1 <= value <= max_value:
        raise ValueError(f"{name} must be in [1, {max_value}], got {value}")
    return value


def _packed_row_bytes(width: int) -> int:
    return (width + 3) // 4


def _check_active_bounds(active_idx: torch.Tensor, num_tokens: int) -> None:
    if active_idx.numel() == 0:
        return
    lo = int(active_idx.min().item())
    hi = int(active_idx.max().item())
    if lo < 0 or hi >= num_tokens:
        raise ValueError(
            f"active_idx entries must be inside [0, {num_tokens}); observed min={lo}, max={hi}"
        )
    # Duplicates are intentionally valid: the native kernel recomputes and
    # overwrites the same row, with no accumulation.


def _check_nonnegative_mult(mult: torch.Tensor, name: str) -> None:
    if mult.numel() and bool((mult < 0).any().item()):
        raise ValueError(
            f"{name} must contain non-negative int32 fixed-point multipliers; "
            "scales must be finite/non-negative with a positive output scale"
        )


def requant_multipliers_from_scales(
    weight_scale: torch.Tensor, act_scale: float, out_scale: float, shift: int
) -> torch.Tensor:
    """Convert physical scales to the native non-negative int32 multiplier.

    ``mult[o] = round(weight_scale[o] * act_scale / out_scale * 2**shift)``.
    Weight scales and activation scale must be finite and non-negative; output
    scale must be finite and strictly positive. Overflow beyond int32 is rejected.
    """
    _require_shift(shift)
    if not isinstance(weight_scale, torch.Tensor):
        raise TypeError("weight_scale must be a torch.Tensor")
    if weight_scale.device.type != "cpu":
        raise ValueError(f"weight_scale must be on CPU, got {weight_scale.device}")
    if weight_scale.ndim != 1:
        raise ValueError(f"weight_scale must be rank 1, got {tuple(weight_scale.shape)}")
    if not weight_scale.is_contiguous():
        raise ValueError("weight_scale must be contiguous")
    if not weight_scale.dtype.is_floating_point:
        raise TypeError(f"weight_scale must be floating point, got {weight_scale.dtype}")
    if not math.isfinite(float(act_scale)) or float(act_scale) < 0:
        raise ValueError("act_scale must be finite and non-negative")
    if not math.isfinite(float(out_scale)) or float(out_scale) <= 0:
        raise ValueError("out_scale must be finite and strictly positive")
    ws = weight_scale.to(torch.float64)
    if not bool(torch.isfinite(ws).all().item()) or bool((ws < 0).any().item()):
        raise ValueError("weight_scale entries must be finite and non-negative")
    raw = torch.round(ws * float(act_scale) / float(out_scale) * float(1 << shift))
    if bool((raw > (2**31 - 1)).any().item()):
        raise OverflowError("requantization multiplier exceeds int32 range")
    return raw.to(torch.int32).contiguous()


def sparse_ternary_gemv(
    x: torch.Tensor,
    active_idx: torch.Tensor,
    w_packed: torch.Tensor,
    requant_mult: torch.Tensor,
    shift: int,
    out_dim: int,
) -> torch.Tensor:
    """Run checked sparse ternary GEMV.

    Contract:
      * x: CPU contiguous int8 [num_tokens, hidden], hidden >= 1.
      * active_idx: CPU contiguous int32 [N]; N may be zero. Duplicates are
        allowed and cause deterministic recomputation/overwrite, not accumulation.
      * w_packed: CPU contiguous uint8 [out_dim * ceil(hidden/4)] with per-row
        zero padding; reserved 2-bit code 11 is rejected natively.
      * requant_mult: CPU contiguous int32 [out_dim], values >= 0.
      * shift: integer [0, 62].
    """
    _require_tensor(x, torch.int8, 2, "x")
    _require_tensor(active_idx, torch.int32, 1, "active_idx")
    _require_tensor(w_packed, torch.uint8, 1, "w_packed")
    _require_tensor(requant_mult, torch.int32, 1, "requant_mult")
    _require_shift(shift)
    _require_positive_dim(out_dim, "out_dim")
    num_tokens, hidden = map(int, x.shape)
    _require_positive_dim(num_tokens, "num_tokens")
    _require_positive_dim(hidden, "hidden", dot_width=True)
    _check_active_bounds(active_idx, num_tokens)
    _check_nonnegative_mult(requant_mult, "requant_mult")
    expected_w = out_dim * _packed_row_bytes(hidden)
    if w_packed.numel() != expected_w:
        raise ValueError(
            f"w_packed must contain exactly out_dim*ceil(hidden/4)={expected_w} bytes, "
            f"got {w_packed.numel()}"
        )
    if requant_mult.numel() != out_dim:
        raise ValueError(f"requant_mult must contain exactly {out_dim} entries")
    return load_extension().sparse_ternary_gemv(
        x, active_idx, w_packed, requant_mult, shift, out_dim
    )


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
    """Run the checked fused two-layer ternary FFN on CPU."""
    _require_tensor(x, torch.int8, 2, "x")
    _require_tensor(active_idx, torch.int32, 1, "active_idx")
    _require_tensor(w1_packed, torch.uint8, 1, "w1_packed")
    _require_tensor(mult1, torch.int32, 1, "mult1")
    _require_tensor(w2_packed, torch.uint8, 1, "w2_packed")
    _require_tensor(mult2, torch.int32, 1, "mult2")
    _require_shift(shift)
    _require_positive_dim(inter_dim, "inter_dim", dot_width=True)
    _require_positive_dim(out_dim, "out_dim")
    num_tokens, hidden = map(int, x.shape)
    _require_positive_dim(num_tokens, "num_tokens")
    _require_positive_dim(hidden, "hidden", dot_width=True)
    _check_active_bounds(active_idx, num_tokens)
    _check_nonnegative_mult(mult1, "mult1")
    _check_nonnegative_mult(mult2, "mult2")
    expected_w1 = inter_dim * _packed_row_bytes(hidden)
    expected_w2 = out_dim * _packed_row_bytes(inter_dim)
    if w1_packed.numel() != expected_w1:
        raise ValueError(f"w1_packed must contain exactly {expected_w1} bytes")
    if mult1.numel() != inter_dim:
        raise ValueError(f"mult1 must contain exactly {inter_dim} entries")
    if w2_packed.numel() != expected_w2:
        raise ValueError(f"w2_packed must contain exactly {expected_w2} bytes")
    if mult2.numel() != out_dim:
        raise ValueError(f"mult2 must contain exactly {out_dim} entries")
    return load_extension().fused_ternary_ffn(
        x, active_idx, w1_packed, mult1, w2_packed, mult2,
        shift, inter_dim, out_dim,
    )
