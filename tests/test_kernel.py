"""Native SPECTRA kernel correctness regression tests.

Milestone 02 keeps the original functional gates while adapting them to the
checked native ABI and row-padded packed-weight contract.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from deploy.pack_ternary import pack_ternary_rows
from model.bitlinear import FakeBitLinear

_SRC = Path(__file__).resolve().parents[1] / "deploy" / "cpp_sparse_kernel" / "spectra_kernel.cpp"
_SHIFT = 15
I8 = ctypes.POINTER(ctypes.c_int8)
I32 = ctypes.POINTER(ctypes.c_int32)
U8 = ctypes.POINTER(ctypes.c_uint8)
SZ = ctypes.c_size_t


def _compiler() -> str | None:
    for cc in ("clang++", "g++"):
        if shutil.which(cc):
            return cc
    return None


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.spectra_sparse_ternary_gemv.restype = ctypes.c_int
    lib.spectra_sparse_ternary_gemv.argtypes = [
        I8, SZ, I32, SZ, U8, SZ, I32, SZ,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, I8, SZ,
    ]
    lib.spectra_fused_ternary_ffn.restype = ctypes.c_int
    lib.spectra_fused_ternary_ffn.argtypes = [
        I8, SZ, I32, SZ,
        U8, SZ, I32, SZ,
        U8, SZ, I32, SZ,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        I8, SZ,
    ]
    lib.spectra_weight_stationary_gemv.restype = ctypes.c_int
    lib.spectra_weight_stationary_gemv.argtypes = [
        I8, SZ, U8, SZ, I32, SZ,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, I8, SZ,
    ]
    return lib


def _build(cc: str, tmp: Path, avx2: bool) -> ctypes.CDLL:
    suffix = "dll" if __import__("sys").platform == "win32" else "so"
    out = tmp / f"spectra_{'avx2' if avx2 else 'scalar'}.{suffix}"
    flags = [cc, "-O3", "-std=c++17", "-shared", "-fPIC", str(_SRC), "-o", str(out)]
    if avx2:
        flags.insert(1, "-mavx2")
    subprocess.run(flags, check=True, capture_output=True)
    return _bind(ctypes.CDLL(str(out)))


def _requant(acc, mult, shift):
    product = acc.astype(np.int64) * mult
    if shift == 0:
        v = product
    else:
        v = (product + (1 << (shift - 1))) >> shift
    return np.clip(v, -128, 127)


def _pack_layer(out_dim, in_dim):
    layer = FakeBitLinear(in_dim, out_dim)
    with torch.no_grad():
        w_q, scale = layer._ternarize_hard(layer.weight)
    w_q = w_q.numpy().astype(np.int8)
    scale = scale.squeeze(1).numpy().astype(np.float64)
    packed, _ = pack_ternary_rows(w_q)
    return w_q, packed, scale


def _run_sparse(lib, X, active_idx, W_packed, mult, hidden, out_dim, num_tokens, shift=_SHIFT):
    X = np.ascontiguousarray(X, dtype=np.int8)
    active_idx = np.ascontiguousarray(active_idx, dtype=np.int32)
    W_packed = np.ascontiguousarray(W_packed, dtype=np.uint8)
    mult = np.ascontiguousarray(mult, dtype=np.int32)
    Y = np.zeros((num_tokens, out_dim), dtype=np.int8)
    status = lib.spectra_sparse_ternary_gemv(
        X.ctypes.data_as(I8), X.size,
        active_idx.ctypes.data_as(I32), active_idx.size,
        W_packed.ctypes.data_as(U8), W_packed.size,
        mult.ctypes.data_as(I32), mult.size,
        shift, num_tokens, hidden, out_dim,
        Y.ctypes.data_as(I8), Y.size,
    )
    assert status == 0
    return Y


@pytest.mark.skipif(_compiler() is None, reason="no C++ compiler available")
def test_kernel_avx2_matches_scalar_and_pytorch(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    hidden, out_dim, num_tokens = 96, 64, 8
    act_scale, out_scale = 0.05, 0.1

    layer = FakeBitLinear(hidden, out_dim)
    with torch.no_grad():
        w_q, scale = layer._ternarize_hard(layer.weight)
    w_q_np = w_q.numpy().astype(np.int8)
    scale_np = scale.squeeze(1).numpy().astype(np.float64)
    W_packed, _ = pack_ternary_rows(w_q_np)
    mult = np.round(scale_np * act_scale / out_scale * (1 << _SHIFT)).astype(np.int32)

    X = rng.integers(-128, 128, size=(num_tokens, hidden), dtype=np.int16).astype(np.int8)
    active_idx = np.array([1, 3, 4, 6], dtype=np.int32)

    cc = _compiler()
    avx2 = _build(cc, tmp_path, avx2=True)
    scalar = _build(cc, tmp_path, avx2=False)
    Y_avx2 = _run_sparse(avx2, X, active_idx, W_packed, mult, hidden, out_dim, num_tokens)
    Y_scalar = _run_sparse(scalar, X, active_idx, W_packed, mult, hidden, out_dim, num_tokens)

    assert np.array_equal(Y_avx2, Y_scalar), "AVX2 path disagrees with scalar"

    Y_ref = np.zeros_like(Y_avx2)
    for t in active_idx:
        acc = w_q_np.astype(np.int32) @ X[t].astype(np.int32)
        Y_ref[t] = _requant(acc, mult, _SHIFT)
    assert np.array_equal(Y_avx2, Y_ref)

    for t in active_idx:
        x_float = torch.from_numpy(X[t].astype(np.float32)) * act_scale
        out_real = torch.nn.functional.linear(x_float, w_q * scale)
        y_torch = torch.clamp(torch.round(out_real / out_scale), -128, 127).numpy()
        assert np.abs(Y_avx2[t].astype(np.int32) - y_torch).max() <= 1

    untouched = [t for t in range(num_tokens) if t not in active_idx]
    assert (Y_avx2[untouched] == 0).all()


@pytest.mark.skipif(_compiler() is None, reason="no C++ compiler available")
def test_fused_ternary_ffn_matches_reference(tmp_path):
    rng = np.random.default_rng(1)
    hidden, inter, out_dim, num_tokens = 65, 67, 49, 6
    s1, s2 = 0.05, 0.05

    w1_q, W1_packed, scale1 = _pack_layer(inter, hidden)
    w2_q, W2_packed, scale2 = _pack_layer(out_dim, inter)
    mult1 = np.round(scale1 * s1 * (1 << _SHIFT)).astype(np.int32)
    mult2 = np.round(scale2 * s2 * (1 << _SHIFT)).astype(np.int32)

    X = rng.integers(-128, 128, size=(num_tokens, hidden), dtype=np.int16).astype(np.int8)
    active_idx = np.array([0, 2, 5], dtype=np.int32)

    lib = _build(_compiler(), tmp_path, avx2=True)
    Y = np.zeros((num_tokens, out_dim), dtype=np.int8)
    status = lib.spectra_fused_ternary_ffn(
        X.ctypes.data_as(I8), X.size,
        active_idx.ctypes.data_as(I32), active_idx.size,
        W1_packed.ctypes.data_as(U8), W1_packed.size,
        mult1.ctypes.data_as(I32), mult1.size,
        W2_packed.ctypes.data_as(U8), W2_packed.size,
        mult2.ctypes.data_as(I32), mult2.size,
        _SHIFT, num_tokens, hidden, inter, out_dim,
        Y.ctypes.data_as(I8), Y.size,
    )
    assert status == 0

    Y_ref = np.zeros_like(Y)
    for t in active_idx:
        tmp = _requant(w1_q.astype(np.int32) @ X[t].astype(np.int32), mult1, _SHIFT)
        tmp = np.clip(tmp, 0, 127).astype(np.int8)
        Y_ref[t] = _requant(w2_q.astype(np.int32) @ tmp.astype(np.int32), mult2, _SHIFT)
    assert np.array_equal(Y, Y_ref)


@pytest.mark.skipif(_compiler() is None, reason="no C++ compiler available")
def test_weight_stationary_recursion_gemv_matches_reference(tmp_path):
    rng = np.random.default_rng(2)
    hidden, out_dim, K = 65, 49, 5
    w_q, W_packed, scale = _pack_layer(out_dim, hidden)
    mult = np.round(scale * 0.05 / 0.1 * (1 << _SHIFT)).astype(np.int32)
    X = rng.integers(-128, 128, size=(K, hidden), dtype=np.int16).astype(np.int8)

    lib = _build(_compiler(), tmp_path, avx2=True)
    Y = np.zeros((K, out_dim), dtype=np.int8)
    status = lib.spectra_weight_stationary_gemv(
        X.ctypes.data_as(I8), X.size,
        W_packed.ctypes.data_as(U8), W_packed.size,
        mult.ctypes.data_as(I32), mult.size,
        _SHIFT, K, hidden, out_dim,
        Y.ctypes.data_as(I8), Y.size,
    )
    assert status == 0
    Y_ref = np.zeros_like(Y)
    for k in range(K):
        Y_ref[k] = _requant(w_q.astype(np.int32) @ X[k].astype(np.int32), mult, _SHIFT)
    assert np.array_equal(Y, Y_ref)


def test_torch_extension_loader_is_graceful():
    from deploy import torch_kernel

    assert isinstance(torch_kernel.cpu_supports_avx2(), bool)
    assert isinstance(torch_kernel.available(), bool)
    if torch_kernel.available():
        assert torch_kernel.backend_name() in {"avx2", "scalar"}
    else:
        x = torch.zeros((1, 4), dtype=torch.int8)
        idx = torch.zeros(1, dtype=torch.int32)
        w = torch.zeros(1, dtype=torch.uint8)
        m = torch.ones(1, dtype=torch.int32)
        with pytest.raises(RuntimeError, match="failed to build/load"):
            torch_kernel.sparse_ternary_gemv(x, idx, w, m, 15, 1)
