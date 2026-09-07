"""Executable acceptance gate for Milestone 02 native correctness/contracts."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from deploy.pack_ternary import pack_ternary_rows
from deploy import torch_kernel

_SRC = Path(__file__).resolve().parents[1] / "deploy" / "cpp_sparse_kernel" / "spectra_kernel.cpp"
I8 = ctypes.POINTER(ctypes.c_int8)
I32 = ctypes.POINTER(ctypes.c_int32)
U8 = ctypes.POINTER(ctypes.c_uint8)
SZ = ctypes.c_size_t


def _build(tmp_path: Path, avx2: bool):
    cc = shutil.which("g++") or shutil.which("clang++")
    if cc is None:
        pytest.skip("no C++ compiler")
    out = tmp_path / ("gate_avx2.so" if avx2 else "gate_scalar.so")
    cmd = [cc, "-O3", "-std=c++17", "-shared", "-fPIC"]
    if avx2:
        cmd.append("-mavx2")
    subprocess.run(cmd + [str(_SRC), "-o", str(out)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(out))
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


def _requant_ref(acc: int, mult: int, shift: int) -> int:
    product = int(acc) * int(mult)
    value = product if shift == 0 else (product + (1 << (shift - 1))) >> shift
    return max(-128, min(127, value))


def _reference(X: np.ndarray, W: np.ndarray, mult: np.ndarray, shift: int) -> np.ndarray:
    out = np.empty((X.shape[0], W.shape[0]), dtype=np.int8)
    for k in range(X.shape[0]):
        for o in range(W.shape[0]):
            acc = sum(int(X[k, d]) * int(W[o, d]) for d in range(X.shape[1]))
            out[k, o] = _requant_ref(acc, int(mult[o]), shift)
    return out


@pytest.mark.parametrize("avx2", [False, True])
def test_acceptance_native_paths_match_independent_reference(tmp_path, avx2):
    """Exercise interacting lanes/tails/shifts, not only one-hot products."""
    lib = _build(tmp_path, avx2)
    rng = np.random.default_rng(20260906)
    for width in (1, 3, 4, 31, 32, 33, 63, 64, 65):
        X = rng.integers(-128, 128, size=(9, width), dtype=np.int16).astype(np.int8)
        # Force multiple exceptional INT8_MIN * -1 lanes when width permits.
        X[:, 0] = -128
        if width > 32:
            X[:, 32] = -128
        W = rng.integers(-1, 2, size=(3, width), dtype=np.int8)
        W[:, 0] = -1
        if width > 32:
            W[:, 32] = -1
        packed, _ = pack_ternary_rows(W)
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        active = np.ascontiguousarray(np.arange(X.shape[0], dtype=np.int32))
        for shift in (0, 1, 7, 15, 31, 62):
            mult = np.ascontiguousarray(np.array([0, 1, 100000], dtype=np.int32))
            expected = _reference(X, W, mult, shift)

            y = np.full(expected.shape, -99, dtype=np.int8)
            status = lib.spectra_sparse_ternary_gemv(
                X.ctypes.data_as(I8), X.size,
                active.ctypes.data_as(I32), active.size,
                packed.ctypes.data_as(U8), packed.size,
                mult.ctypes.data_as(I32), mult.size,
                shift, X.shape[0], width, W.shape[0],
                y.ctypes.data_as(I8), y.size,
            )
            assert status == 0
            assert np.array_equal(y, expected), ("packed", avx2, width, shift)

            y_ws = np.full(expected.shape, -99, dtype=np.int8)
            status = lib.spectra_weight_stationary_gemv(
                X.ctypes.data_as(I8), X.size,
                packed.ctypes.data_as(U8), packed.size,
                mult.ctypes.data_as(I32), mult.size,
                shift, X.shape[0], width, W.shape[0],
                y_ws.ctypes.data_as(I8), y_ws.size,
            )
            assert status == 0
            assert np.array_equal(y_ws, expected), ("decoded", avx2, width, shift)


@pytest.mark.parametrize("avx2", [False, True])
def test_acceptance_invalid_native_calls_do_no_compute_or_scatter(tmp_path, avx2):
    """A rejected raw-C call must leave its output buffer untouched."""
    lib = _build(tmp_path, avx2)
    X = np.zeros((2, 5), dtype=np.int8)
    W = np.zeros((1, 5), dtype=np.int8)
    packed, _ = pack_ternary_rows(W)
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    active = np.ascontiguousarray(np.array([0], dtype=np.int32))
    mult = np.ascontiguousarray(np.array([1], dtype=np.int32))

    for bad_shift, x_len_delta in ((-1, 0), (63, 0), (7, -1)):
        y = np.full((2, 1), 77, dtype=np.int8)
        status = lib.spectra_sparse_ternary_gemv(
            X.ctypes.data_as(I8), X.size + x_len_delta,
            active.ctypes.data_as(I32), active.size,
            packed.ctypes.data_as(U8), packed.size,
            mult.ctypes.data_as(I32), mult.size,
            bad_shift, 2, 5, 1,
            y.ctypes.data_as(I8), y.size,
        )
        assert status != 0
        assert (y == 77).all()

    y = np.full((2, 1), 77, dtype=np.int8)
    bad_idx = np.ascontiguousarray(np.array([2], dtype=np.int32))
    status = lib.spectra_sparse_ternary_gemv(
        X.ctypes.data_as(I8), X.size,
        bad_idx.ctypes.data_as(I32), bad_idx.size,
        packed.ctypes.data_as(U8), packed.size,
        mult.ctypes.data_as(I32), mult.size,
        7, 2, 5, 1,
        y.ctypes.data_as(I8), y.size,
    )
    assert status != 0
    assert (y == 77).all()


def test_acceptance_public_invalid_calls_never_reach_native_backend(monkeypatch):
    """Public structural validation must fail before extension loading/execution."""
    def trap():
        raise AssertionError("native backend must not be reached for invalid public input")

    monkeypatch.setattr(torch_kernel, "load_extension", trap)
    x = torch.zeros((2, 5), dtype=torch.int8)
    idx = torch.tensor([0], dtype=torch.int32)
    W = np.zeros((1, 5), dtype=np.int8)
    packed, _ = pack_ternary_rows(W)
    wp = torch.from_numpy(packed.copy())
    mult = torch.ones(1, dtype=torch.int32)

    invalid_calls = [
        lambda: torch_kernel.sparse_ternary_gemv(x.float(), idx, wp, mult, 7, 1),
        lambda: torch_kernel.sparse_ternary_gemv(x[:, ::2], idx, wp, mult, 7, 1),
        lambda: torch_kernel.sparse_ternary_gemv(x, torch.tensor([2], dtype=torch.int32), wp, mult, 7, 1),
        lambda: torch_kernel.sparse_ternary_gemv(x, idx, wp[:-1], mult, 7, 1),
        lambda: torch_kernel.sparse_ternary_gemv(x, idx, wp, -mult, 7, 1),
        lambda: torch_kernel.sparse_ternary_gemv(x, idx, wp, mult, 63, 1),
    ]
    for call in invalid_calls:
        with pytest.raises((TypeError, ValueError)):
            call()


def test_acceptance_reported_edge_case_is_frozen_regression():
    """Guard the exact acceptance case at source level in the dedicated test suite."""
    text = (Path(__file__).with_name("test_kernel_contract.py")).read_text(encoding="utf-8")
    assert "def test_reported_int8_min_negation_case" in text
    assert "x[0, 0] = -128" in text
    assert "w[0, 0] = -1" in text
    assert "shift=7" in text
