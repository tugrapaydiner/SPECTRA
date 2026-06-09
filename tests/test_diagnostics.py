"""Phase 3 gate: recursion diagnostics.

Using the shared trained 4x4 model: later recursion steps must preserve or
improve accuracy (Gate 4), there must be no collapse, latent states must actually
evolve, and gradients must reach every recursive block.
"""

import pytest
import torch

from eval.metrics import per_step_accuracy
from train.collapse_watch import (
    check_collapse,
    per_block_grad_norm,
    state_delta_norms,
    total_grad_norm,
)
from train.losses import deep_supervision_loss


@pytest.mark.slow
def test_per_step_accuracy_preserves_or_improves(trained_sudoku_4x4):
    """Gate 4: accs[-1] >= accs[0] - 0.02 (deep supervision does not regress)."""
    d = trained_sudoku_4x4
    with torch.no_grad():
        _, steps = d["model"](d["vx"], height=d["height"], width=d["width"])
    accs = per_step_accuracy(steps, d["vy"])
    print(f"\nper-step accuracy: {[round(a, 3) for a in accs]}")
    assert accs[-1] >= accs[0] - 0.02, accs


@pytest.mark.slow
def test_no_collapse(trained_sudoku_4x4):
    """No output collapse, no NaN/Inf, no explosion; latent state evolves."""
    d = trained_sudoku_4x4
    with torch.no_grad():
        _, steps = d["model"](d["vx"], height=d["height"], width=d["width"])
    acc = per_step_accuracy(steps, d["vy"])[-1]
    report = check_collapse(steps, accuracy=acc)
    print(
        f"\nmode_ratio={report.most_common_ratio:.3f} "
        f"dz={[round(x, 3) for x in report.delta_z]}"
    )
    assert not report.has_nan
    assert not report.output_collapse, f"mode_ratio={report.most_common_ratio}"
    assert not report.explosion
    assert not report.fixed_point_collapse
    # The latent reasoning state z must actually change across steps.
    assert max(report.delta_z) > 0.0


@pytest.mark.slow
def test_gradients_reach_all_blocks(trained_sudoku_4x4):
    """Total grad norm is finite/positive and every recursive block gets gradient."""
    d = trained_sudoku_4x4
    model = d["model"]
    model.train()
    idx = torch.randint(0, d["tx"].shape[0], (64,), device=d["device"])
    _, steps = model(d["tx"][idx], height=d["height"], width=d["width"])
    loss = deep_supervision_loss(steps, d["ty"][idx])
    model.zero_grad()
    loss.backward()

    total = total_grad_norm(model)
    per_block = per_block_grad_norm(model.blocks)
    dy, dz = state_delta_norms(steps)

    model.zero_grad()
    model.eval()

    import math

    assert total > 0 and math.isfinite(total)
    assert all(g > 0 for g in per_block), per_block
    assert all(math.isfinite(v) for v in dy + dz)
