"""Phase 11 tests: ternary packing, export reconstruction, ONNX/kernel adapters."""

import numpy as np
import pytest

from deploy.bitnet_cpp_adapter import is_sparse_kernel_available, load_sparse_kernel
from deploy.export_bitnet import (
    export_summary,
    export_ternary_weights,
    reconstruct_weight,
    save_export,
)
from deploy.export_onnx import export_to_onnx, onnx_available
from deploy.pack_ternary import pack_ternary, packed_size_bytes, unpack_ternary
from model.stability import iter_bitlinears
from model.trm import TRM


@pytest.mark.parametrize("n", [4, 7, 16, 81, 1000])
def test_pack_unpack_roundtrip(n):
    rng = np.random.default_rng(n)
    w = rng.integers(-1, 2, size=n).astype(np.int8)
    packed, count = pack_ternary(w)
    assert count == n
    assert packed.size == packed_size_bytes(n)
    restored = unpack_ternary(packed, count, shape=(n,))
    assert np.array_equal(restored, w)


def test_pack_preserves_2d_shape():
    rng = np.random.default_rng(0)
    w = rng.integers(-1, 2, size=(8, 12)).astype(np.int8)
    packed, n = pack_ternary(w)
    restored = unpack_ternary(packed, n, shape=(8, 12))
    assert np.array_equal(restored, w)


def test_pack_rejects_non_ternary():
    with pytest.raises(ValueError):
        pack_ternary(np.array([0, 1, 2], dtype=np.int8))


def test_export_reconstruction_is_lossless():
    """Reconstructed gamma*W_q exactly equals the fake-quant hard forward weight."""
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=2, ternary=True, max_grid_size=8)
    export = export_ternary_weights(model)
    assert export, "ternary model should have ternary layers"
    for name, module in iter_bitlinears(model):
        w_q, scale = module._ternarize_hard(module.weight)
        ref = (w_q * scale).detach().cpu().numpy()
        rec = reconstruct_weight(export[name])
        assert np.allclose(rec, ref, atol=1e-5)


def test_export_summary_two_bits_per_weight():
    model = TRM(dim=64, num_tokens=10, seq_len=81, N_sup=2, ternary=True)
    summary = export_summary(model)
    assert summary["ternary_params"] > 0
    # Packing is exactly 2 bits/weight (4 weights per byte).
    assert summary["bits_per_weight"] == pytest.approx(2.0, abs=0.01)
    assert summary["packed_weight_mb"] > 0


def test_save_export_writes_npz(tmp_path):
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=2, ternary=True, max_grid_size=8)
    summary = save_export(model, tmp_path / "core.npz")
    assert (tmp_path / "core.npz").exists()
    assert summary["ternary_params"] > 0


def test_sparse_kernel_adapter_graceful():
    # Not built on the dev box -> unavailable, and loading raises a helpful error.
    assert is_sparse_kernel_available() is False
    with pytest.raises(FileNotFoundError):
        load_sparse_kernel()


def test_onnx_export_graceful_when_unavailable():
    if onnx_available():
        pytest.skip("onnx is installed; graceful-failure path not exercised")
    import torch

    from model.system1_student import System1Student

    model = System1Student(dim=32, num_tokens=5, seq_len=16, n_layers=1, max_grid_size=8)
    with pytest.raises(RuntimeError):
        export_to_onnx(model, torch.randint(0, 5, (1, 16)), "out.onnx")
