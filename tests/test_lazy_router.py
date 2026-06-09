"""Phase 7 gate: lazy active-token routing.

Fast: masked recursion freezes tokens, the confidence/RL routers behave, and the
REINFORCE loss back-propagates. Slow gate: applying the confidence router to a
trained model drops mean active-token density below 40% with acceptable accuracy
loss (BLUEPRINT section 7.6).
"""

import pytest
import torch

from eval.metrics import active_token_density, cell_accuracy
from model.lazy_router import ConfidenceRouter, RLTokenRouter, router_reinforce_loss
from model.trm import TRM


def test_masked_recursion_freezes_tokens():
    """Frozen tokens (mask=0) keep their state; active tokens (mask=1) update."""
    torch.manual_seed(0)
    m = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=2, max_grid_size=8)
    x_emb = torch.randn(2, 16, 32)
    y = torch.randn(2, 16, 32)
    z = torch.randn(2, 16, 32)
    mask = torch.ones(2, 16, 1)
    mask[:, 8:, :] = 0.0  # freeze the second half of tokens

    y2, z2 = m.recursive_cycle(x_emb, y, z, mask)
    assert torch.allclose(z2[:, 8:], z[:, 8:])  # frozen unchanged
    assert torch.allclose(y2[:, 8:], y[:, 8:])
    assert not torch.allclose(z2[:, :8], z[:, :8])  # active changed


def test_confidence_router_freezes_confident_tokens():
    cr = ConfidenceRouter(threshold=0.9, warmup_steps=0)
    logits = torch.zeros(1, 2, 5)
    logits[0, 0, 3] = 10.0  # token 0 is very confident; token 1 is uniform
    mask, logprob = cr(1, None, torch.zeros(1, 2, 4), logits)
    assert logprob is None
    assert mask[0, 0, 0] == 0.0  # confident -> frozen
    assert mask[0, 1, 0] == 1.0  # uncertain -> active


def test_rl_router_actions_and_reinforce_gradient():
    rl = RLTokenRouter(dim=32, device_dim=6)
    rl.train()
    z = torch.randn(4, 16, 32)
    mask, logprob = rl(0, None, z, None, None)
    assert mask.shape == (4, 16, 1)
    assert logprob.shape == (4, 16)
    assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    loss = router_reinforce_loss([logprob], reward=torch.randn(4), baseline=0.0)
    loss.backward()
    assert rl.policy[0].weight.grad is not None


@pytest.mark.slow
def test_lazy_routing_reduces_active_density(trained_sudoku_4x4):
    d = trained_sudoku_4x4
    model, vx, vy = d["model"], d["vx"], d["vy"]

    with torch.no_grad():
        base_logits, _ = model(vx, height=d["height"], width=d["width"])
    base_cell = cell_accuracy(base_logits.argmax(-1), vy)

    router = ConfidenceRouter(threshold=0.95, warmup_steps=1)
    with torch.no_grad():
        logits, steps = model(vx, height=d["height"], width=d["width"], router=router)
    density = active_token_density(steps)
    routed_cell = cell_accuracy(logits.argmax(-1), vy)
    print(f"\nlazy routing: mean_density={density:.3f} "
          f"cell {base_cell:.3f} -> {routed_cell:.3f}")

    # Gate: active-token density falls well below the 30-40% kernel-go threshold...
    assert density < 0.4, f"density={density:.3f}"
    # ...without an unacceptable accuracy hit.
    assert routed_cell >= base_cell - 0.05, (base_cell, routed_cell)
