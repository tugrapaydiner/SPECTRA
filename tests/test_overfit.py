"""Phase 1 gate 3: single-batch overfit > 95% (BLUEPRINT section 16).

This is the most important MVP gate: if the recursive core cannot memorise one
batch, the algorithm is broken and no amount of quantization/optimization will
help (section 36). Runs on CUDA when available purely for speed; the test is
otherwise the section 16 reference.
"""

import pytest
import torch

from common.seed import resolve_device, set_seed
from model.trm import TRM
from train.losses import deep_supervision_loss


@pytest.mark.slow
def test_overfit_single_batch():
    set_seed(0)
    device = resolve_device("auto")

    model = TRM(dim=128, num_tokens=10, seq_len=81, N_sup=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, 10, (4, 81), device=device)
    y = torch.randint(0, 10, (4, 81), device=device)

    logits = None
    for _ in range(300):
        logits, steps_out = model(x, height=9, width=9, y_target=y)
        loss = deep_supervision_loss(steps_out, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    acc = (logits.argmax(-1) == y).float().mean().item()
    assert acc > 0.95, f"Cannot overfit one batch: acc={acc}"
