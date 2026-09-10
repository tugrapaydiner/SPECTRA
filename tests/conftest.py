"""Shared test fixtures.

The 4x4 Sudoku training run is the most expensive thing in the suite, so we do it
once per session and reuse the trained model for both the MVP accuracy gate
(Phase 2) and the recursion-diagnostics gate (Phase 3).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from common.seed import resolve_device, set_seed
from data.datasets import build_sudoku_arrays
from model.trm import TRM
from train.losses import deep_supervision_loss
from train.accumulation import backward_mean_loss


@pytest.fixture(scope="session")
def trained_sudoku_4x4():
    """Train a small TRM on 4x4 Sudoku and return the model + data tensors."""
    set_seed(0)
    device = resolve_device("auto")
    box, side = 2, 4

    rng = np.random.default_rng(0)
    train_x, train_y, h, w = build_sudoku_arrays(
        box, n=768, num_clues=8, rng=rng, require_unique=True, augment=True
    )
    val_x, val_y, _, _ = build_sudoku_arrays(
        box, n=256, num_clues=8, rng=rng, require_unique=True, augment=True
    )

    tx = torch.from_numpy(train_x).to(device)
    ty = torch.from_numpy(train_y).to(device)
    vx = torch.from_numpy(val_x).to(device)
    vy = torch.from_numpy(val_y).to(device)

    model = TRM(
        dim=96, num_tokens=side + 1, seq_len=side * side, N_sup=4, max_grid_size=8
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(500):
        idx = torch.randint(0, tx.shape[0], (128,), device=device)
        opt.zero_grad()
        microbatch = int(os.environ.get("SPECTRA_TEST_MICROBATCH", "128"))
        # Same 128 sampled examples and one optimizer update; only backward
        # partitioning changes. Opt-in for memory-constrained CPU environments.
        backward_mean_loss(
            lambda xx, yy: deep_supervision_loss(model(xx, height=h, width=w)[1], yy),
            tx[idx], ty[idx], microbatch)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()

    return {
        "model": model,
        "tx": tx, "ty": ty, "vx": vx, "vy": vy,
        "height": h, "width": w, "device": device,
    }
