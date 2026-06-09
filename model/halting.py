"""RL-driven hardware-aware halting (BLUEPRINT section 8).

The halter decides, at each supervision step, whether to keep thinking or stop.
The final controller is a *learned policy* conditioned on a device-state vector
``d`` (battery, thermal, latency budget, ...), trained so it stops earlier under
tight budgets and thinks longer when plugged in / cool:

    pi_eta(halt | pool(y), z, d) -> {continue, halt}
    R_halt = 1[correct] - lambda_J * MeasuredJoules - lambda_T * LatencyMs

Here compute is rationed via the dynamic penalty ``lambda_compute(d)`` of section
8.3 (grows as battery/latency-budget fall and thermal rises). Heuristic halt
heads remain useful for warm-starting, but the gate target is learned rationing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Device state (section 21)
# --------------------------------------------------------------------------- #
DEVICE_DIM = 6


@dataclass
class DeviceState:
    """Normalised-ish device telemetry fed to the router/halter (section 21)."""

    battery_level: float  # 0 (empty) .. 1 (full)
    thermal_level: float  # 0 (cool) .. 1 (severe throttle)
    latency_budget_ms: float
    power_mode: float  # 0 (power-saver) .. 1 (performance)
    available_ram_mb: float
    device_class: int


def device_state_to_tensor(d: DeviceState, device: torch.device | str = "cpu") -> torch.Tensor:
    """Encode a :class:`DeviceState` to a normalised ``[DEVICE_DIM]`` tensor."""
    return torch.tensor(
        [
            d.battery_level,
            d.thermal_level,
            min(d.latency_budget_ms / 1000.0, 1.0),
            d.power_mode,
            min(d.available_ram_mb / 8192.0, 1.0),
            d.device_class / 10.0,
        ],
        device=device,
        dtype=torch.float32,
    )


def make_device_states(
    battery: torch.Tensor,
    thermal: torch.Tensor,
    latency_budget_ms: torch.Tensor,
    power_mode: torch.Tensor,
    available_ram_mb: torch.Tensor,
    device_class: torch.Tensor,
) -> torch.Tensor:
    """Build a batched device-state tensor ``[B, DEVICE_DIM]`` from columns."""
    return torch.stack(
        [
            battery,
            thermal,
            (latency_budget_ms / 1000.0).clamp(max=1.0),
            power_mode,
            (available_ram_mb / 8192.0).clamp(max=1.0),
            device_class / 10.0,
        ],
        dim=-1,
    ).float()


def compute_penalty_lambda(
    device_state: torch.Tensor,
    lambda0: float = 1.0,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
) -> torch.Tensor:
    """Dynamic compute penalty ``lambda_compute(d)`` (section 8.3).

    ``lambda0 * exp(a*(1-battery) + b*thermal + c*(1-latency_budget))`` -- larger
    (stop sooner) when battery/latency-budget are low or thermal is high.
    """
    battery = device_state[..., 0]
    thermal = device_state[..., 1]
    latency = device_state[..., 2]
    return lambda0 * torch.exp(a * (1.0 - battery) + b * thermal + c * (1.0 - latency))


# --------------------------------------------------------------------------- #
# Halting policy (section 8.2)
# --------------------------------------------------------------------------- #
class HaltingPolicy(nn.Module):
    """Budget-conditioned halting head ``p_halt = sigmoid(g(pool(y), d))``."""

    def __init__(self, dim: int, device_dim: int = DEVICE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + device_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, y: torch.Tensor, device_state: torch.Tensor) -> torch.Tensor:
        """Halt logit ``[B]`` from answer state ``y`` ``[B, L, D]`` and device ``d``."""
        pooled = y.mean(dim=1) if y.dim() == 3 else y
        return self.net(torch.cat([pooled, device_state], dim=-1)).squeeze(-1)

    def halt_prob(self, y: torch.Tensor, device_state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(y, device_state))


# --------------------------------------------------------------------------- #
# REINFORCE halting episode (section 8)
# --------------------------------------------------------------------------- #
def halting_episode(
    policy: HaltingPolicy,
    y_steps: torch.Tensor,
    device_state: torch.Tensor,
    sample: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out the halt/continue policy over precomputed per-step states.

    The episode halts at the first sampled "halt" (forced at the last step). Only
    decisions up to the halt step contribute to the trajectory log-prob.

    Args:
        policy: The halting policy.
        y_steps: Per-step answer states ``[K, B, L, D]`` (or pooled ``[K, B, D]``).
        device_state: Device-state vector ``[B, DEVICE_DIM]``.
        sample: Sample halts (training) vs. threshold at 0.5 (eval).

    Returns:
        ``(halt_step [B] long, traj_logprob [B])``.
    """
    k_steps, b = y_steps.shape[0], y_steps.shape[1]
    device = y_steps.device
    active = torch.ones(b, dtype=torch.bool, device=device)
    halt_step = torch.full((b,), k_steps - 1, dtype=torch.long, device=device)
    traj_logprob = torch.zeros(b, device=device)

    for k in range(k_steps):
        logit = policy(y_steps[k], device_state)
        p = torch.sigmoid(logit).clamp(1e-6, 1.0 - 1e-6)
        if k == k_steps - 1:
            action = torch.ones(b, device=device)  # must halt by the last step
        elif sample:
            action = torch.bernoulli(p)
        else:
            action = (p > 0.5).float()

        logp_k = torch.where(action.bool(), torch.log(p), torch.log(1.0 - p))
        traj_logprob = traj_logprob + active.float() * logp_k

        halts_now = active & action.bool()
        halt_step = torch.where(halts_now, torch.full_like(halt_step, k), halt_step)
        active = active & ~action.bool()

    return halt_step, traj_logprob


def halting_reinforce_loss(
    traj_logprob: torch.Tensor,
    reward: torch.Tensor,
    baseline: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """REINFORCE loss for the halting policy (section 8 / 20.2)."""
    if baseline is None:
        baseline = reward.mean()
    advantage = (reward - baseline).detach()
    return -(advantage * traj_logprob).mean()


def halting_reward(
    correct_at_step: torch.Tensor,
    halt_step: torch.Tensor,
    device_state: torch.Tensor,
    max_step: int,
    lambda_step: float = 0.5,
) -> torch.Tensor:
    """Hardware-aware halting reward (section 8.3).

    ``1[correct at halt] - lambda_compute(d) * lambda_step * (halt_step / max_step)``
    -- correctness minus a budget-scaled compute cost. Low-budget device states
    have a larger ``lambda_compute(d)``, so each extra step is punished more.
    """
    correct = correct_at_step.gather(0, halt_step[None]).squeeze(0).float()
    lam = compute_penalty_lambda(device_state)
    norm_steps = halt_step.float() / max(1, max_step)
    return correct - lam * lambda_step * norm_steps


# --------------------------------------------------------------------------- #
# Adaptive-halt inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_with_halting(
    model: nn.Module,
    x: torch.Tensor,
    policy: HaltingPolicy,
    device_state: torch.Tensor,
    height: int,
    width: int,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the recursive core, stopping each example when the policy says halt.

    Returns ``(answer [B, L], halt_step [B])``. All examples run the full
    recursion (PyTorch has no ragged batch), but each example's answer is frozen
    at the step the policy first chose to halt -- the metric the gate cares about.
    """
    model.eval()
    _, steps = model(x, height=height, width=width)
    b = x.shape[0]
    device = x.device

    answer = steps[-1]["logits"].argmax(dim=-1)
    halt_step = torch.full((b,), len(steps) - 1, dtype=torch.long, device=device)
    decided = torch.zeros(b, dtype=torch.bool, device=device)

    for k, step in enumerate(steps):
        halt = policy.halt_prob(step["y"], device_state) > threshold
        take = halt & ~decided
        if take.any():
            answer = torch.where(take[:, None], step["logits"].argmax(dim=-1), answer)
            halt_step = torch.where(take, torch.full_like(halt_step, k), halt_step)
            decided = decided | take

    return answer, halt_step
