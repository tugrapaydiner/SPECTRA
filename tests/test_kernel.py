"""Fix 4: compile the AVX2 kernel and validate AVX2 == scalar == PyTorch.

The previous kernel was never compiled. Here we actually build it (AVX2 and a
scalar reference), run both via ctypes, and prove the pshufb unpack + integer
multiply-shift requantize are bit-for-bit correct against a NumPy reference and
agree with the PyTorch fake-quant forward (BLUEPRINT sections 12, 26.3).
"""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from deploy.pack_ternary import pack_ternary
from model.bitlinear import FakeBitLinear

_SRC = Path(__file__).resolve().parents[1] / "deploy" / "cpp_sparse_kernel" / "spectra_kernel.cpp"
_SHIFT = 15


def _compiler() -> str | None:
    for cc in ("clang++", "g++"):
        if shutil.which(cc):
            return cc
    return None


def _build(cc: str, tmp: Path, avx2: bool) -> ctypes.CDLL:
    out = tmp / (f"spectra_{'avx2' if avx2 else 'scalar'}.dll")
    flags = [cc, "-O3", "-std=c++17", "-shared", "-fPIC", str(_SRC), "-o", str(out)]
    if avx2:
        flags.insert(1, "-mavx2")
    subprocess.run(flags, check=True, capture_output=True)
    lib = ctypes.CDLL(str(out))
    lib.spectra_sparse_ternary_gemv.restype = None
    lib.spectra_sparse_ternary_gemv.argtypes = [
        ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int8),
    ]
    lib.spectra_fused_ternary_ffn.restype = None
    lib.spectra_fused_ternary_ffn.argtypes = [
        ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int8),
    ]
    lib.spectra_weight_stationary_gemv.restype = None
    lib.spectra_weight_stationary_gemv.argtypes = [
        ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int8),
    ]
    return lib


def _requant(acc, mult, shift):
    v = (acc.astype(np.int64) * mult + (1 << (shift - 1))) >> shift
    return np.clip(v, -128, 127)


def _pack_layer(out_dim, in_dim, rng):
    """A ternary layer: returns packed weights, per-channel int32 requant mult."""
    from model.bitlinear import FakeBitLinear
    import torch as _t

    layer = FakeBitLinear(in_dim, out_dim)
    with _t.no_grad():
        w_q, scale = layer._ternarize_hard(layer.weight)
    w_q = w_q.numpy().astype(np.int8)
    scale = scale.squeeze(1).numpy().astype(np.float64)
    packed = np.concatenate([pack_ternary(w_q[o])[0] for o in range(out_dim)]).astype(np.uint8)
    return w_q, packed, scale


def _run(lib, X, active_idx, W_packed, mult, hidden, out_dim, num_tokens):
    Y = np.zeros((num_tokens, out_dim), dtype=np.int8)
    lib.spectra_sparse_ternary_gemv(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        active_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(len(active_idx)),
        W_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        mult.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(_SHIFT),
        ctypes.c_int(hidden), ctypes.c_int(out_dim),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    return Y


@pytest.mark.skipif(_compiler() is None, reason="no C++ compiler available")
def test_kernel_avx2_matches_scalar_and_pytorch(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    hidden, out_dim, num_tokens = 96, 64, 8
    act_scale, out_scale = 0.05, 0.1

    # Ternary weights + per-channel scales straight from a FakeBitLinear.
    layer = FakeBitLinear(hidden, out_dim)
    with torch.no_grad():
        w_q, scale = layer._ternarize_hard(layer.weight)    # {-1,0,1} [O,H], scale [O,1]
    w_q_np = w_q.numpy().astype(np.int8)
    scale_np = scale.squeeze(1).numpy().astype(np.float64)

    W_packed = np.concatenate([pack_ternary(w_q_np[o])[0] for o in range(out_dim)]).astype(np.uint8)
    # Fixed-point requant multiplier folding w_scale*act_scale/out_scale.
    mult = np.round(scale_np * act_scale / out_scale * (1 << _SHIFT)).astype(np.int32)

    X = rng.integers(-40, 40, size=(num_tokens, hidden)).astype(np.int8)
    active_idx = np.array([1, 3, 4, 6], dtype=np.int32)

    cc = _compiler()
    avx2 = _build(cc, tmp_path, avx2=True)
    scalar = _build(cc, tmp_path, avx2=False)
    Y_avx2 = _run(avx2, X, active_idx, W_packed, mult, hidden, out_dim, num_tokens)
    Y_scalar = _run(scalar, X, active_idx, W_packed, mult, hidden, out_dim, num_tokens)

    # --- AVX2 path is bit-for-bit identical to the scalar reference. ---
    assert np.array_equal(Y_avx2, Y_scalar), "AVX2 pshufb unpack disagrees with scalar"

    # --- Both match an independent NumPy reference (integer requant). ---
    Y_ref = np.zeros_like(Y_avx2)
    for t in active_idx:
        acc = w_q_np.astype(np.int32) @ X[t].astype(np.int32)  # [O]
        v = (acc.astype(np.int64) * mult + (1 << (_SHIFT - 1))) >> _SHIFT
        Y_ref[t] = np.clip(v, -128, 127)
    assert np.array_equal(Y_avx2, Y_ref)

    # --- Agrees with the PyTorch fake-quant forward within int8 rounding. ---
    for t in active_idx:
        x_float = torch.from_numpy(X[t].astype(np.float32)) * act_scale
        out_real = torch.nn.functional.linear(x_float, w_q * scale)  # ternarized fwd
        y_torch = torch.clamp(torch.round(out_real / out_scale), -128, 127).numpy()
        assert np.abs(Y_avx2[t].astype(np.int32) - y_torch).max() <= 1

    # --- Frozen tokens are never written (lazy routing scatter is sparse). ---
    untouched = [t for t in range(num_tokens) if t not in active_idx]
    assert (Y_avx2[untouched] == 0).all()


@pytest.mark.skipif(_compiler() is None, reason="no C++ compiler available")
def test_fused_ternary_ffn_matches_reference(tmp_path):
    """The fused (C++-resident) two-layer FFN matches a NumPy 2-layer reference."""
    rng = np.random.default_rng(1)
    hidden, inter, out_dim, num_tokens = 64, 96, 48, 6
    s1, s2 = 0.05, 0.05  # act_scale-like folds; only relative correctness matters

    w1_q, W1_packed, scale1 = _pack_layer(inter, hidden, rng)
    w2_q, W2_packed, scale2 = _pack_layer(out_dim, inter, rng)
    mult1 = np.round(scale1 * s1 * (1 << _SHIFT)).astype(np.int32)
    mult2 = np.round(scale2 * s2 * (1 << _SHIFT)).astype(np.int32)

    X = rng.integers(-40, 40, size=(num_tokens, hidden)).astype(np.int8)
    active_idx = np.array([0, 2, 5], dtype=np.int32)

    lib = _build(_compiler(), tmp_path, avx2=True)
    Y = np.zeros((num_tokens, out_dim), dtype=np.int8)
    lib.spectra_fused_ternary_ffn(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        active_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int(len(active_idx)),
        W1_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        mult1.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        W2_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        mult2.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(_SHIFT), ctypes.c_int(hidden), ctypes.c_int(inter),
        ctypes.c_int(out_dim), Y.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )

    Y_ref = np.zeros_like(Y)
    for t in active_idx:
        tmp = _requant(w1_q.astype(np.int32) @ X[t].astype(np.int32), mult1, _SHIFT)
        tmp = np.clip(tmp, 0, 127).astype(np.int8)  # fused int8 ReLU
        Y_ref[t] = _requant(w2_q.astype(np.int32) @ tmp.astype(np.int32), mult2, _SHIFT)
    assert np.array_equal(Y, Y_ref)


@pytest.mark.skipif(_compiler() is None, reason="no C++ compiler available")
def test_weight_stationary_recursion_gemv_matches_reference(tmp_path):
    """The weight-stationary recursion kernel (decode-once, reuse across K steps)
    is numerically identical to applying the matrix to each activation."""
    rng = np.random.default_rng(2)
    hidden, out_dim, K = 64, 48, 5  # K = recursion-step activations
    w_q, W_packed, scale = _pack_layer(out_dim, hidden, rng)
    mult = np.round(scale * 0.05 / 0.1 * (1 << _SHIFT)).astype(np.int32)
    X = rng.integers(-40, 40, size=(K, hidden)).astype(np.int8)

    lib = _build(_compiler(), tmp_path, avx2=True)
    Y = np.zeros((K, out_dim), dtype=np.int8)
    lib.spectra_weight_stationary_gemv(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        W_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        mult.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int(_SHIFT),
        ctypes.c_int(K), ctypes.c_int(hidden), ctypes.c_int(out_dim),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    Y_ref = np.zeros_like(Y)
    for k in range(K):
        Y_ref[k] = _requant(w_q.astype(np.int32) @ X[k].astype(np.int32), mult, _SHIFT)
    assert np.array_equal(Y, Y_ref)


def test_torch_extension_loader_is_graceful():
    """The torch C++ extension loader never crashes; degrades to a clear error.

    On a host whose toolchain is ABI-incompatible with torch (e.g. no MSVC on
    Windows) ``available()`` is False and the op raises a clear RuntimeError --
    callers fall back to the validated ctypes kernel.
    """
    from deploy import torch_kernel

    assert isinstance(torch_kernel.available(), bool)
    if not torch_kernel.available():
        with pytest.raises(RuntimeError):
            torch_kernel.sparse_ternary_gemv(
                torch.zeros(1, 4, dtype=torch.int8), torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.int32), 15, 1,
            )
