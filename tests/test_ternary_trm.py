"""Phase 4 gate: a fully ternary (W1.58) recursive core trains.

Structure test (fast): the ternary core ternarizes FFN, attention, and output
head while keeping embeddings / norms / halting head higher precision (section
11.5). Overfit gate (slow): the ternary core overfits one batch with the
quant-strength warmup and shows no ternary saturation collapse (section 11.7).
"""

import pytest
import torch
import torch.nn as nn

from common.seed import resolve_device, set_seed
from model.bitlinear import FakeBitLinear
from model.stability import QuantWarmup, is_ternary_saturated, ternary_report
from model.trm import TRM
from train.losses import deep_supervision_loss


def test_ternary_core_structure():
    """Ternary core ternarizes the right parts and trains end-to-end."""
    m = TRM(dim=64, num_tokens=10, seq_len=81, N_sup=2, ternary=True)

    # Ternarized: FFN, attention projections, output head.
    assert isinstance(m.out_head, FakeBitLinear)
    assert m.blocks[0].ternary_attn
    assert isinstance(m.blocks[0].ff[0], FakeBitLinear)
    assert isinstance(m.blocks[0].attn.q, FakeBitLinear)
    # Higher precision: embeddings, norms, halting head (section 11.5).
    assert isinstance(m.halt_head, nn.Linear) and not isinstance(m.halt_head, FakeBitLinear)
    assert isinstance(m.token_embed, nn.Embedding)

    x = torch.randint(0, 10, (2, 81))
    y = torch.randint(0, 10, (2, 81))
    _, steps = m(x, height=9, width=9)
    deep_supervision_loss(steps, y).backward()
    assert m.blocks[0].attn.q.weight.grad is not None
    assert m.out_head.weight.grad is not None


@pytest.mark.slow
def test_ternary_overfit_single_batch():
    """Fully ternary core overfits one batch; ternary distribution stays healthy."""
    set_seed(0)
    device = resolve_device("auto")

    model = TRM(dim=128, num_tokens=10, seq_len=81, N_sup=8, ternary=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    warmup = QuantWarmup(80)  # ramp rho 0 -> 1 over the first 80 steps

    x = torch.randint(0, 10, (4, 81), device=device)
    y = torch.randint(0, 10, (4, 81), device=device)

    logits = None
    for step in range(150):
        warmup.apply(model, step)
        logits, steps = model(x, height=9, width=9)
        loss = deep_supervision_loss(steps, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    acc = (logits.argmax(-1) == y).float().mean().item()
    report = ternary_report(model)
    print(f"\nternary overfit acc={acc:.3f} zero%={report['zero']:.3f}")

    assert acc > 0.95, f"ternary core cannot overfit one batch: acc={acc}"
    assert not is_ternary_saturated(report), report
