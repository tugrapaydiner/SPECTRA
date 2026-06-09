"""Phase 9 gate: RL-driven hardware-aware halting.

Fast: device-state encoding, compute-penalty monotonicity, halting policy /
episode / REINFORCE mechanics, adaptive-halt inference, telemetry. Slow gate: a
budget-conditioned halter rations compute -- it stops early under tight device
budgets and thinks longer when plugged in, smoothly across intermediate budgets
(BLUEPRINT section 8).

Note on the value signal: deep-supervised TRMs have near-flat per-step accuracy
(every step is trained to be correct), so there is no per-step accuracy curve to
trade against on these tasks. The gate therefore exercises the halting controller
against recursive reasoning's *diminishing-returns* value (more thinking -> better
answer, with falling marginal gain), which is exactly the premise of section 8.3.
"""

import numpy as np
import pytest
import torch

from common.seed import resolve_device, set_seed
from data.datasets import build_sudoku_arrays
from eval.telemetry import read_device_state
from model.halting import (
    DEVICE_DIM,
    DeviceState,
    HaltingPolicy,
    compute_penalty_lambda,
    device_state_to_tensor,
    halting_episode,
    halting_reinforce_loss,
    halting_reward,
    make_device_states,
    run_with_halting,
)
from model.trm import TRM
from train.losses import deep_supervision_loss


# --------------------------------------------------------------------------- #
# Device state + compute penalty
# --------------------------------------------------------------------------- #
def test_device_state_to_tensor_normalises():
    d = DeviceState(0.5, 0.2, 800, 1.0, 4096, 1)
    t = device_state_to_tensor(d)
    assert t.shape == (DEVICE_DIM,)
    expected = [0.5, 0.2, 0.8, 1.0, 0.5, 0.1]
    assert torch.allclose(t, torch.tensor(expected), atol=1e-6)


def test_compute_penalty_higher_under_low_budget():
    low = make_device_states(*[torch.tensor([v]) for v in (0.1, 0.9, 100.0, 0.0, 512.0, 0.0)])
    high = make_device_states(*[torch.tensor([v]) for v in (1.0, 0.0, 2000.0, 1.0, 8192.0, 2.0)])
    assert compute_penalty_lambda(low).item() > compute_penalty_lambda(high).item()
    # Penalty is >= lambda0 (=1) since exponent args are non-negative.
    assert compute_penalty_lambda(high).item() == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Policy / episode / REINFORCE mechanics
# --------------------------------------------------------------------------- #
def test_halting_policy_and_episode():
    pol = HaltingPolicy(dim=16)
    y_steps = torch.randn(4, 8, 5, 16)  # [K, B, L, D]
    d = torch.rand(8, DEVICE_DIM)
    halt_step, logp = halting_episode(pol, y_steps, d, sample=True)
    assert halt_step.shape == (8,) and logp.shape == (8,)
    assert (halt_step >= 0).all() and (halt_step <= 3).all()  # within K
    assert torch.isfinite(logp).all()


def test_halting_reinforce_gradient():
    pol = HaltingPolicy(dim=16)
    y_steps = torch.randn(4, 8, 5, 16)
    d = torch.rand(8, DEVICE_DIM)
    halt_step, logp = halting_episode(pol, y_steps, d, sample=True)
    correct = (torch.rand(4, 8) > 0.3).float()
    reward = halting_reward(correct, halt_step, d, max_step=3)
    halting_reinforce_loss(logp, reward).backward()
    assert pol.net[0].weight.grad is not None


def test_run_with_halting_inference():
    torch.manual_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=4, max_grid_size=8)
    pol = HaltingPolicy(dim=32)
    x = torch.randint(0, 5, (6, 16))
    d = device_state_to_tensor(DeviceState(0.5, 0.3, 500, 1.0, 4096, 1)).expand(6, -1)
    answer, halt_step = run_with_halting(model, x, pol, d, height=4, width=4)
    assert answer.shape == (6, 16)
    assert (halt_step >= 0).all() and (halt_step <= 3).all()


def test_telemetry_returns_valid_device_state():
    d = read_device_state(latency_budget_ms=750, device_class=1)
    assert 0.0 <= d.battery_level <= 1.0
    assert 0.0 <= d.thermal_level <= 1.0
    assert d.available_ram_mb > 0
    # Encodable to the model's device-state tensor.
    assert device_state_to_tensor(d).shape == (DEVICE_DIM,)


# --------------------------------------------------------------------------- #
# Slow gate: budget-conditioned compute rationing
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_halter_rations_compute_by_budget():
    set_seed(0)
    device = resolve_device("auto")
    box, K = 2, 8

    rng = np.random.default_rng(0)
    tx, ty, h, w = build_sudoku_arrays(box, 512, 8, rng, require_unique=True, augment=True)
    vx, _, _, _ = build_sudoku_arrays(box, 256, 8, rng, require_unique=True, augment=True)
    tx = torch.from_numpy(tx).to(device)
    ty = torch.from_numpy(ty).to(device)
    vx = torch.from_numpy(vx).to(device)

    # Lightly train a TRM so the per-step states fed to the halter are real.
    model = TRM(dim=64, num_tokens=5, seq_len=16, N_sup=K, T=1, n=2, max_grid_size=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(60):
        idx = torch.randint(0, tx.shape[0], (128,), device=device)
        _, steps = model(tx[idx], height=h, width=w)
        deep_supervision_loss(steps, ty[idx]).backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        _, vsteps = model(vx, height=h, width=w)
    y_steps = torch.stack([vsteps[k]["y"] for k in range(K)]).detach()
    b = y_steps.shape[1]

    # Diminishing-returns value of recursive thinking (section 8.3 premise).
    set_seed(1)
    floor = torch.rand(b, device=device) * 0.2 + 0.3
    ceil = torch.rand(b, device=device) * 0.2 + 0.75
    ks = torch.arange(K, device=device).float() / (K - 1)
    value = floor[None] + (ceil - floor)[None] * (1 - (1 - ks[:, None]) ** 2)

    # Train the budget-conditioned halter with REINFORCE across device budgets.
    set_seed(2)
    policy = HaltingPolicy(dim=64).to(device)
    popt = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    for _ in range(1000):
        battery = torch.rand(b, device=device)
        dstate = make_device_states(
            battery, 1 - battery, battery * 1900 + 100, battery,
            torch.full((b,), 8192.0, device=device), torch.ones(b, device=device),
        )
        halt_step, logp = halting_episode(policy, y_steps, dstate, sample=True)
        reward = halting_reward(value, halt_step, dstate, max_step=K - 1, lambda_step=0.15)
        popt.zero_grad()
        halting_reinforce_loss(logp, reward).backward()
        popt.step()

    def mean_halt(battery: float) -> float:
        full = lambda v: torch.full((b,), v, device=device)
        dstate = make_device_states(
            full(battery), full(1 - battery), full(battery * 1900 + 100),
            full(battery), full(8192.0), full(1.0),
        )
        halt_step, _ = halting_episode(policy, y_steps, dstate, sample=False)
        return halt_step.float().mean().item()

    low, mid, high = mean_halt(0.05), mean_halt(0.5), mean_halt(1.0)
    print(f"\nhalt steps by budget: low={low:.2f} mid={mid:.2f} high={high:.2f}")

    # Reduces steps under low budget, thinks longer when plugged in, smoothly.
    assert low < mid < high
    assert high - low > 2.0, (low, mid, high)
