"""Phase 5 gate: INT8 activation fake-quant and the W1.58A8 recursive loop.

Fast: FakeActQuant is a correct STE quantizer and the W1.58A8 core trains a step.
Slow: the full W1.58A8 core (ternary weights + INT8 activations) overfits one
batch without INT8 quantization destroying the recursion (section 19).
"""

import pytest
import torch

from common.seed import resolve_device, set_seed
from eval.metrics import per_step_accuracy
from model.fake_quant import FakeActQuant
from model.stability import QuantWarmup
from model.trm import TRM
from train.losses import deep_supervision_loss


def test_fake_act_quant_ste_gradient():
    """The STE passes an identity gradient through the (rounding) quantizer."""
    q = FakeActQuant()
    x = torch.randn(2, 8, requires_grad=True)
    q(x).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_fake_act_quant_error_bounded_and_on_grid():
    """Quantization error is <= half a step and levels stay within INT8 range."""
    q = FakeActQuant()
    x = torch.randn(4, 64) * 5.0
    y = q(x).detach()
    scale = x.abs().amax(dim=-1, keepdim=True) / 127
    assert ((y - x).abs() <= scale * 0.5 + 1e-6).all()
    levels = torch.round(y / scale)
    assert levels.min() >= -128 and levels.max() <= 127


def test_w1a8_builds_and_trains_step():
    """W1.58A8 core builds, logs an activation scale, and back-propagates."""
    m = TRM(dim=64, num_tokens=5, seq_len=16, N_sup=3, ternary=True, act8=True, max_grid_size=8)
    assert isinstance(m.act_quant, FakeActQuant)
    x = torch.randint(0, 5, (4, 16))
    y = torch.randint(1, 5, (4, 16))
    _, steps = m(x, height=4, width=4)
    deep_supervision_loss(steps, y).backward()
    assert m.blocks[0].ff[0].weight.grad is not None
    assert m.act_quant.last_scale_mean.item() > 0  # activation scale logged


@pytest.mark.slow
def test_w1a8_overfit_single_batch():
    """W1.58A8 overfits one batch and the recursion still improves across steps."""
    set_seed(0)
    device = resolve_device("auto")

    model = TRM(
        dim=128, num_tokens=10, seq_len=81, N_sup=8, ternary=True, act8=True
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    warmup = QuantWarmup(80)

    x = torch.randint(0, 10, (4, 81), device=device)
    y = torch.randint(0, 10, (4, 81), device=device)

    steps = None
    for step in range(200):
        warmup.apply(model, step)
        logits, steps = model(x, height=9, width=9)
        loss = deep_supervision_loss(steps, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    acc = (logits.argmax(-1) == y).float().mean().item()
    accs = per_step_accuracy(steps, y)
    print(f"\nW1.58A8 overfit acc={acc:.3f} per-step={[round(a, 2) for a in accs]}")

    assert acc > 0.95, f"W1.58A8 cannot overfit one batch: acc={acc}"
    # INT8 activation quant must not destroy the recursion's monotonic improvement.
    assert accs[-1] >= accs[0] - 0.02, accs
