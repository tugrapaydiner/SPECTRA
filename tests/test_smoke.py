"""Phase 0 gate: pytest runs and a dummy model trains one step.

This is the minimal "the harness is wired up" check from BLUEPRINT section 28
(Phase 0). It exercises imports of the shared utilities and confirms that a
trivial model can take a single optimisation step (loss decreases, gradients
flow).
"""

import torch
import torch.nn as nn

from common import DotDict, load_config, resolve_device, set_seed


def test_imports_and_seed():
    """Shared utilities import and seeding runs without error."""
    set_seed(0)
    device = resolve_device("cpu")
    assert device.type == "cpu"


def test_config_loads_with_inheritance():
    """A task config inherits base fields and applies its own overrides."""
    cfg = load_config("config/sudoku.yaml")
    assert isinstance(cfg, DotDict)
    # Inherited from base.yaml:
    assert cfg.model.n == 6
    assert cfg.seed == 42
    # Defined in sudoku.yaml:
    assert cfg.task == "sudoku"
    assert cfg.data.seq_len == 81


def test_config_cli_override():
    """Dotted-key overrides parse with YAML scalar semantics."""
    cfg = load_config("config/sudoku.yaml", overrides=["model.dim=256", "seed=7"])
    assert cfg.model.dim == 256
    assert cfg.seed == 7


def test_dummy_model_trains_one_step():
    """A trivial linear model takes one optimiser step and the loss drops."""
    set_seed(0)
    model = nn.Linear(8, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x = torch.randn(16, 8)
    target = torch.randn(16, 4)

    loss_before = nn.functional.mse_loss(model(x), target)
    opt.zero_grad()
    loss_before.backward()
    # Gradients must reach parameters.
    assert model.weight.grad is not None
    opt.step()

    loss_after = nn.functional.mse_loss(model(x), target)
    assert loss_after.item() < loss_before.item()
