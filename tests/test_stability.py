"""Phase 6 gate: the Stability Shield.

Fast: EMA tracking/restore, teacher->student distillation gradient routing, and a
Trainer smoke run. Slow: a multi-epoch W1.58A8 training run via the Trainer stays
stable -- accuracy improves and no collapse signal fires at any evaluation.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from data.datasets import GridDataset, build_sudoku_arrays
from model.trm import TRM
from train.distill import trajectory_distillation_loss
from train.ema import EMA
from train.trainer import Trainer, TrainConfig


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
def test_ema_tracks_and_restores():
    torch.manual_seed(0)
    m = nn.Linear(4, 4, bias=False)
    orig = m.weight.detach().clone()
    ema = EMA(m, decay=0.9)

    with torch.no_grad():
        m.weight.add_(1.0)  # move the live weights
    ema.update(m)  # shadow = 0.9*orig + 0.1*(orig+1) = orig + 0.1
    assert torch.allclose(ema.shadow["weight"], orig + 0.1, atol=1e-6)

    live = m.weight.detach().clone()
    with ema.average_parameters(m):  # swap EMA in
        assert torch.allclose(m.weight, orig + 0.1, atol=1e-6)
    assert torch.allclose(m.weight, live, atol=1e-6)  # and restore


# --------------------------------------------------------------------------- #
# Distillation
# --------------------------------------------------------------------------- #
def test_distillation_updates_student_only():
    teacher = TRM(dim=64, num_tokens=5, seq_len=16, N_sup=2, max_grid_size=8)
    student = TRM(dim=64, num_tokens=5, seq_len=16, N_sup=2, ternary=True, max_grid_size=8)
    x = torch.randint(0, 5, (4, 16))
    y = torch.randint(1, 5, (4, 16))

    with torch.no_grad():
        _, t_steps = teacher(x, height=4, width=4)
    _, s_steps = student(x, height=4, width=4)

    loss, comp = trajectory_distillation_loss(s_steps, t_steps, y)
    loss.backward()

    assert torch.isfinite(loss)
    assert {"task", "logit", "state", "total"} <= set(comp)
    assert student.blocks[0].ff[0].weight.grad is not None
    # Teacher is frozen: it must receive no gradient.
    assert teacher.blocks[0].ff[0].weight.grad is None


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
def _tiny_datasets(n=64, box=2):
    rng = np.random.default_rng(0)
    tx, ty, h, w = build_sudoku_arrays(box, n, 8, rng, require_unique=True, augment=True)
    vx, vy, _, _ = build_sudoku_arrays(box, n, 8, rng, require_unique=True, augment=True)
    return GridDataset(tx, ty, h, w), GridDataset(vx, vy, h, w)


def test_trainer_smoke_runs_without_collapse():
    train_ds, val_ds = _tiny_datasets(n=64)
    model = TRM(dim=48, num_tokens=5, seq_len=16, N_sup=2, max_grid_size=8)
    cfg = TrainConfig(
        batch_size=32, max_steps=10, lr_warmup_steps=2,
        eval_every=5, log_every=5, device="cpu",
    )
    out = Trainer(model, train_ds, val_ds, cfg).fit()
    assert len(out["history"]) >= 1
    assert not out["final"]["dashboard"]["has_nan"]
    assert not out["final"]["dashboard"]["output_collapse"]


@pytest.mark.slow
def test_stable_multi_epoch_w1a8_training():
    rng = np.random.default_rng(0)
    tx, ty, h, w = build_sudoku_arrays(2, 768, 8, rng, require_unique=True, augment=True)
    vx, vy, _, _ = build_sudoku_arrays(2, 256, 8, rng, require_unique=True, augment=True)
    train_ds, val_ds = GridDataset(tx, ty, h, w), GridDataset(vx, vy, h, w)

    model = TRM(
        dim=96, num_tokens=5, seq_len=16, N_sup=4, ternary=True, act8=True, max_grid_size=8
    )
    cfg = TrainConfig(
        lr=2e-3, batch_size=128, max_steps=400,
        lr_warmup_steps=20, quant_warmup_steps=60,
        ema_decay=0.95,  # short run -> faster-tracking EMA so validation is meaningful
        eval_every=80, log_every=80, device="auto",
    )
    out = Trainer(model, train_ds, val_ds, cfg).fit()

    accs = [row["cell_acc"] for row in out["history"]]
    final = out["final"]
    print(
        f"\nW1.58A8 stable training: cell_acc trajectory={[round(a, 3) for a in accs]} "
        f"final board_acc={final['board_acc']:.3f}"
    )

    # Stability: no collapse signal fires at any evaluation.
    for row in out["history"]:
        assert not row["dash_has_nan"]
        assert not row["dash_output_collapse"]
        assert not row["dash_fixed_point_collapse"]
        assert not row["dash_explosion"]
    # Multi-epoch progress: accuracy improves and ends up solidly learned.
    assert accs[-1] >= accs[0]
    assert final["cell_acc"] > 0.7, final
    assert not final["dashboard"]["fixed_point_collapse"]
