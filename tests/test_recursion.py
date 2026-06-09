"""Phase 1 gates 1, 2, 5: shapes/backward, effective depth, no output collapse.

(BLUEPRINT section 16.)
"""

import torch

from common.seed import set_seed
from model.trm import TRM
from train.losses import deep_supervision_loss, policy_improvement_loss


def test_shapes_and_backward():
    """Gate 1: forward shapes are correct and gradients reach the blocks."""
    model = TRM(dim=64, num_tokens=10, seq_len=81, N_sup=4)
    x = torch.randint(0, 10, (2, 81))
    y = torch.randint(0, 10, (2, 81))

    logits, steps = model(x, height=9, width=9, y_target=y)
    assert logits.shape == (2, 81, 10)
    assert len(steps) == 4
    assert steps[-1]["halt_logit"].shape == (2,)

    loss = deep_supervision_loss(steps, y)
    loss.backward()

    # Gradient must flow into the shared operator's FFN.
    assert model.blocks[0].ff[0].weight.grad is not None
    # ... and into the embeddings and output head.
    assert model.token_embed.weight.grad is not None
    assert model.out_head.weight.grad is not None


def test_effective_depth():
    """Gate 2: effective depth = T * (n + 1) * n_layers = 3 * 7 * 2 = 42."""
    m = TRM(dim=64, num_tokens=10, seq_len=81, n=6, T=3, n_layers=2)
    effective_depth = m.T * (m.n + 1) * len(m.blocks)
    assert effective_depth == 42
    # The convenience property must agree.
    assert m.effective_depth == 42


def test_no_output_collapse():
    """Gate 5: predictions are not dominated by a single token (mode < 90%)."""
    set_seed(0)
    model = TRM(dim=64, num_tokens=10, seq_len=81, N_sup=4)
    x = torch.randint(0, 10, (4, 81))
    logits, _ = model(x, height=9, width=9)

    pred = logits.argmax(-1).flatten()
    mode_ratio = torch.bincount(pred, minlength=10).max().float() / pred.numel()
    assert mode_ratio < 0.90, f"Output collapsed: mode_ratio={mode_ratio:.3f}"


def test_states_reset_each_forward():
    """Two forward passes on the same input give identical logits (y,z reset)."""
    set_seed(0)
    model = TRM(dim=64, num_tokens=10, seq_len=81, N_sup=3)
    model.eval()
    x = torch.randint(0, 10, (2, 81))
    with torch.no_grad():
        logits_a, _ = model(x, height=9, width=9)
        logits_b, _ = model(x, height=9, width=9)
    assert torch.allclose(logits_a, logits_b)


def test_policy_improvement_loss_has_gradient():
    """The corrected policy-improvement loss keeps a live gradient (section 12)."""
    set_seed(0)
    model = TRM(dim=64, num_tokens=10, seq_len=81, N_sup=3)
    x = torch.randint(0, 10, (2, 81))
    y = torch.randint(0, 10, (2, 81))
    _, steps = model(x, height=9, width=9)

    loss = policy_improvement_loss([s["logits"] for s in steps], y)
    loss.backward()
    assert model.out_head.weight.grad is not None
    assert model.out_head.weight.grad.abs().sum() > 0
