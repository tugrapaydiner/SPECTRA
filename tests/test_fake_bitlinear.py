"""FakeBitLinear unit tests (BLUEPRINT section 18) + warmup behaviour."""

import torch

from model.bitlinear import FakeBitLinear


def test_fake_bitlinear_backward():
    """Gradient flows to the full-precision master weight through the STE."""
    layer = FakeBitLinear(32, 64)
    x = torch.randn(4, 10, 32)
    y = layer(x).sum()
    y.backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.abs().sum() > 0


def test_fake_bitlinear_ternary_distribution():
    """The hard-quantized forward weights are exactly in {-1, 0, +1}."""
    layer = FakeBitLinear(32, 64)
    w_q, _ = layer._ternarize_hard(layer.weight)
    assert set(w_q.unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_quant_strength_zero_is_full_precision():
    """rho = 0 reproduces the full-precision linear forward exactly."""
    layer = FakeBitLinear(16, 8, quant_strength=0.0)
    x = torch.randn(3, 16)
    out = layer(x)
    ref = torch.nn.functional.linear(x, layer.weight)
    assert torch.allclose(out, ref, atol=1e-6)


def test_quant_strength_one_is_hard_ternary():
    """rho = 1 reproduces the section 18 hard ternary forward (gamma * W_q)."""
    layer = FakeBitLinear(16, 8, quant_strength=1.0)
    x = torch.randn(3, 16)
    out = layer(x)
    w_q, scale = layer._ternarize_hard(layer.weight)
    ref = torch.nn.functional.linear(x, w_q * scale)
    assert torch.allclose(out, ref, atol=1e-6)


def test_ternary_stats_sum_to_one():
    layer = FakeBitLinear(64, 64)
    stats = layer.ternary_stats()
    assert abs(stats["neg"] + stats["zero"] + stats["pos"] - 1.0) < 1e-9
