"""Milestone 02 exhaustive native numerical/input-contract tests."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from deploy.pack_ternary import pack_ternary_rows, unpack_ternary_rows
from deploy import torch_kernel

_SRC = Path(__file__).resolve().parents[1] / "deploy" / "cpp_sparse_kernel" / "spectra_kernel.cpp"
I8 = ctypes.POINTER(ctypes.c_int8)
I32 = ctypes.POINTER(ctypes.c_int32)
U8 = ctypes.POINTER(ctypes.c_uint8)
SZ = ctypes.c_size_t

OK = 0
INVALID_DIM = 2
INVALID_SHIFT = 3
INVALID_LENGTH = 4
ACTIVE_OOB = 5
INVALID_PACKED = 6
INVALID_MULT = 7


def _compiler():
    return shutil.which("g++") or shutil.which("clang++")


def _bind(lib):
    lib.spectra_sparse_ternary_gemv.restype = ctypes.c_int
    lib.spectra_sparse_ternary_gemv.argtypes = [
        I8, SZ, I32, SZ, U8, SZ, I32, SZ,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, I8, SZ,
    ]
    lib.spectra_weight_stationary_gemv.restype = ctypes.c_int
    lib.spectra_weight_stationary_gemv.argtypes = [
        I8, SZ, U8, SZ, I32, SZ,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, I8, SZ,
    ]
    return lib


def _build(tmp_path, avx2):
    cc = _compiler()
    if cc is None:
        pytest.skip("no C++ compiler")
    out = tmp_path / ("libavx2.so" if avx2 else "libscalar.so")
    cmd = [cc, "-O3", "-std=c++17", "-shared", "-fPIC"]
    if avx2:
        cmd.append("-mavx2")
    subprocess.run(cmd + [str(_SRC), "-o", str(out)], check=True, capture_output=True)
    return _bind(ctypes.CDLL(str(out)))


def _requant_wide(acc: int, mult: int, shift: int) -> int:
    product = int(acc) * int(mult)
    value = product if shift == 0 else (product + (1 << (shift - 1))) >> shift
    return max(-128, min(127, value))


def _raw_sparse(lib, x, w, active=None, mult=None, shift=7, y_fill=0,
                x_len_delta=0, w_len_delta=0, mult_len_delta=0, y_len_delta=0):
    x = np.ascontiguousarray(x, dtype=np.int8)
    w = np.ascontiguousarray(w, dtype=np.int8)
    nt, h = x.shape
    od = w.shape[0]
    packed, _ = pack_ternary_rows(w)
    if active is None:
        active = np.arange(nt, dtype=np.int32)
    active = np.ascontiguousarray(active, dtype=np.int32)
    if mult is None:
        mult = np.ones(od, dtype=np.int32)
    mult = np.ascontiguousarray(mult, dtype=np.int32)
    y = np.full((nt, od), y_fill, dtype=np.int8)
    status = lib.spectra_sparse_ternary_gemv(
        x.ctypes.data_as(I8), x.size + x_len_delta,
        active.ctypes.data_as(I32), active.size,
        packed.ctypes.data_as(U8), packed.size + w_len_delta,
        mult.ctypes.data_as(I32), mult.size + mult_len_delta,
        shift, nt, h, od,
        y.ctypes.data_as(I8), y.size + y_len_delta,
    )
    return status, y, packed


def _raw_ws(lib, X, w, mult=None, shift=7):
    X = np.ascontiguousarray(X, dtype=np.int8)
    w = np.ascontiguousarray(w, dtype=np.int8)
    K, h = X.shape
    od = w.shape[0]
    packed, _ = pack_ternary_rows(w)
    if mult is None:
        mult = np.ones(od, dtype=np.int32)
    mult = np.ascontiguousarray(mult, dtype=np.int32)
    y = np.zeros((K, od), dtype=np.int8)
    status = lib.spectra_weight_stationary_gemv(
        X.ctypes.data_as(I8), X.size,
        packed.ctypes.data_as(U8), packed.size,
        mult.ctypes.data_as(I32), mult.size,
        shift, K, h, od,
        y.ctypes.data_as(I8), y.size,
    )
    return status, y


@pytest.mark.parametrize("avx2", [False, True])
def test_reported_int8_min_negation_case(tmp_path, avx2):
    lib = _build(tmp_path, avx2)
    x = np.zeros((1, 32), dtype=np.int8); x[0, 0] = -128
    w = np.zeros((1, 32), dtype=np.int8); w[0, 0] = -1
    status, y, _ = _raw_sparse(lib, x, w, shift=7)
    assert status == OK
    assert y[0, 0] == 1
    status, y_ws = _raw_ws(lib, x, w, shift=7)
    assert status == OK
    assert y_ws[0, 0] == 1


@pytest.mark.parametrize("avx2", [False, True])
def test_all_int8_values_all_ternary_signs_against_wide_reference(tmp_path, avx2):
    lib = _build(tmp_path, avx2)
    vals = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
    for width in (1, 3, 4, 31, 32, 33, 63, 64, 65):
        X = np.zeros((256, width), dtype=np.int8)
        X[:, 0] = vals
        for sign in (-1, 0, 1):
            w = np.zeros((1, width), dtype=np.int8); w[0, 0] = sign
            status, y, _ = _raw_sparse(lib, X, w, shift=7)
            assert status == OK
            expected = np.array([_requant_wide(int(v) * sign, 1, 7) for v in vals], dtype=np.int8)
            assert np.array_equal(y[:, 0], expected), (avx2, width, sign)
            status, y_ws = _raw_ws(lib, X, w, shift=7)
            assert status == OK
            assert np.array_equal(y_ws[:, 0], expected), ("decoded", avx2, width, sign)


@pytest.mark.parametrize("avx2", [False, True])
def test_saturation_tails_frozen_zero_active_and_duplicates(tmp_path, avx2):
    lib = _build(tmp_path, avx2)
    width = 65
    w = np.ones((1, width), dtype=np.int8)
    x_pos = np.full((1, width), 127, dtype=np.int8)
    x_neg = np.full((1, width), -128, dtype=np.int8)
    assert _raw_sparse(lib, x_pos, w, mult=np.array([100000], np.int32), shift=0)[1][0, 0] == 127
    assert _raw_sparse(lib, x_neg, w, mult=np.array([100000], np.int32), shift=0)[1][0, 0] == -128

    X = np.arange(4 * 33, dtype=np.int16).reshape(4, 33).astype(np.int8)
    W = np.ones((2, 33), dtype=np.int8)
    status, y, _ = _raw_sparse(lib, X, W, active=np.array([], np.int32), y_fill=77)
    assert status == OK and (y == 77).all()

    status, y, _ = _raw_sparse(lib, X, W, active=np.array([1, 3], np.int32), y_fill=77)
    assert status == OK
    assert (y[0] == 77).all() and (y[2] == 77).all()
    single = y[1].copy()
    status, y_dup, _ = _raw_sparse(lib, X, W, active=np.array([1, 1], np.int32), y_fill=77)
    assert status == OK and np.array_equal(y_dup[1], single)


@pytest.mark.parametrize("avx2", [False, True])
def test_native_contract_rejects_bad_lengths_indices_shift_weights_and_multiplier(tmp_path, avx2):
    lib = _build(tmp_path, avx2)
    X = np.zeros((2, 5), dtype=np.int8)
    W = np.zeros((1, 5), dtype=np.int8)

    assert _raw_sparse(lib, X, W, shift=-1)[0] == INVALID_SHIFT
    assert _raw_sparse(lib, X, W, shift=63)[0] == INVALID_SHIFT
    assert _raw_sparse(lib, X, W, active=np.array([2], np.int32))[0] == ACTIVE_OOB
    assert _raw_sparse(lib, X, W, mult=np.array([-1], np.int32))[0] == INVALID_MULT
    assert _raw_sparse(lib, X, W, x_len_delta=-1)[0] == INVALID_LENGTH
    assert _raw_sparse(lib, X, W, w_len_delta=-1)[0] == INVALID_LENGTH
    assert _raw_sparse(lib, X, W, mult_len_delta=-1)[0] == INVALID_LENGTH
    assert _raw_sparse(lib, X, W, y_len_delta=-1)[0] == INVALID_LENGTH

    status, _, packed = _raw_sparse(lib, X, W)
    assert status == OK
    active = np.array([0], dtype=np.int32)
    mult = np.array([1], dtype=np.int32)
    y = np.zeros((2, 1), dtype=np.int8)

    bad_reserved = packed.copy(); bad_reserved[0] = (bad_reserved[0] & ~0x3) | 0x3
    status = lib.spectra_sparse_ternary_gemv(
        X.ctypes.data_as(I8), X.size, active.ctypes.data_as(I32), active.size,
        bad_reserved.ctypes.data_as(U8), bad_reserved.size,
        mult.ctypes.data_as(I32), mult.size, 7, 2, 5, 1,
        y.ctypes.data_as(I8), y.size)
    assert status == INVALID_PACKED

    bad_padding = packed.copy(); bad_padding[-1] |= 0b00000100
    status = lib.spectra_sparse_ternary_gemv(
        X.ctypes.data_as(I8), X.size, active.ctypes.data_as(I32), active.size,
        bad_padding.ctypes.data_as(U8), bad_padding.size,
        mult.ctypes.data_as(I32), mult.size, 7, 2, 5, 1,
        y.ctypes.data_as(I8), y.size)
    assert status == INVALID_PACKED


def test_row_packer_has_independent_zero_padding():
    W = np.array([[1, -1, 0, 1, -1], [-1, 1, 1, 0, 1]], dtype=np.int8)
    packed, shape = pack_ternary_rows(W)
    assert shape == (2, 5)
    assert packed.size == 2 * 2
    assert (packed.reshape(2, 2)[:, 1] & 0b11111100 == 0).all()
    assert np.array_equal(unpack_ternary_rows(packed, 2, 5), W)


def _valid_torch_inputs(width=5, out_dim=2):
    x = torch.zeros((3, width), dtype=torch.int8)
    idx = torch.tensor([0, 2], dtype=torch.int32)
    W = np.zeros((out_dim, width), dtype=np.int8)
    packed, _ = pack_ternary_rows(W)
    wp = torch.from_numpy(packed.copy())
    m = torch.ones(out_dim, dtype=torch.int32)
    return x, idx, wp, m


def test_public_python_contract_rejects_device_dtype_shape_contiguity_lengths_and_scales():
    x, idx, wp, m = _valid_torch_inputs()
    with pytest.raises(TypeError):
        torch_kernel.sparse_ternary_gemv(x.float(), idx, wp, m, 7, 2)
    with pytest.raises(ValueError):
        torch_kernel.sparse_ternary_gemv(x[:, ::2], idx, wp, m, 7, 2)
    with pytest.raises(ValueError):
        torch_kernel.sparse_ternary_gemv(x, idx.reshape(1, -1), wp, m, 7, 2)
    with pytest.raises(ValueError):
        torch_kernel.sparse_ternary_gemv(x, idx, wp[:-1], m, 7, 2)
    with pytest.raises(ValueError):
        torch_kernel.sparse_ternary_gemv(x, torch.tensor([3], dtype=torch.int32), wp, m, 7, 2)
    with pytest.raises(ValueError):
        torch_kernel.sparse_ternary_gemv(x, idx, wp, m, 63, 2)
    with pytest.raises(ValueError):
        torch_kernel.sparse_ternary_gemv(x, idx, wp, -m, 7, 2)
    meta = torch.empty((3, 5), dtype=torch.int8, device="meta")
    with pytest.raises(ValueError, match="CPU"):
        torch_kernel.sparse_ternary_gemv(meta, idx, wp, m, 7, 2)

    ws = torch.tensor([0.1, 0.2], dtype=torch.float32)
    mult = torch_kernel.requant_multipliers_from_scales(ws, 0.05, 0.1, 15)
    assert mult.dtype == torch.int32 and (mult >= 0).all()
    with pytest.raises(ValueError):
        torch_kernel.requant_multipliers_from_scales(ws, -0.1, 0.1, 15)
    with pytest.raises(ValueError):
        torch_kernel.requant_multipliers_from_scales(ws, 0.1, 0.0, 15)


def test_native_extension_boundary_rechecks_inputs_when_available():
    if not torch_kernel.available():
        pytest.skip("native PyTorch extension unavailable")
    ext = torch_kernel.load_extension()
    x, idx, wp, m = _valid_torch_inputs()
    assert ext.backend_name() in {"avx2", "scalar"}
    with pytest.raises(RuntimeError, match="X has wrong dtype"):
        ext.sparse_ternary_gemv(x.float(), idx, wp, m, 7, 2)
    with pytest.raises(RuntimeError, match="W_packed must contain exactly"):
        ext.sparse_ternary_gemv(x, idx, wp[:-1], m, 7, 2)
    with pytest.raises(RuntimeError, match="outside"):
        ext.sparse_ternary_gemv(x, torch.tensor([3], dtype=torch.int32), wp, m, 7, 2)


def test_backend_detection_and_failure_visibility(monkeypatch):
    assert isinstance(torch_kernel.cpu_supports_avx2(), bool)
    assert "-mavx2" not in torch_kernel._compile_flags(False)

    import torch.utils.cpp_extension as cpp
    torch_kernel.load_extension.cache_clear()
    monkeypatch.setattr(torch_kernel, "cpu_supports_avx2", lambda: False)
    monkeypatch.setattr(cpp, "load", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="scalar backend.*boom"):
        torch_kernel.load_extension()
    torch_kernel.load_extension.cache_clear()
