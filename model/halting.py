"""RL-driven hardware-aware halting plus M11 real online inference.

The training helpers still support precomputed trajectories.  Inference no longer
runs the full TRM before choosing an answer: ``run_with_halting`` advances the
explicit TRM execution state one supervision step at a time and returns as soon
as a policy/budget/model stop is reached.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
import torch.nn as nn

DEVICE_DIM = 6


@dataclass
class DeviceState:
    battery_level: float
    thermal_level: float
    latency_budget_ms: float
    power_mode: float
    available_ram_mb: float
    device_class: int


def device_state_to_tensor(d: DeviceState, device: torch.device | str = "cpu") -> torch.Tensor:
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
    battery = device_state[..., 0]
    thermal = device_state[..., 1]
    latency = device_state[..., 2]
    return lambda0 * torch.exp(a * (1.0 - battery) + b * thermal + c * (1.0 - latency))


class HaltingPolicy(nn.Module):
    def __init__(self, dim: int, device_dim: int = DEVICE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + device_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, y: torch.Tensor, device_state: torch.Tensor) -> torch.Tensor:
        pooled = y.mean(dim=1) if y.dim() == 3 else y
        return self.net(torch.cat([pooled, device_state], dim=-1)).squeeze(-1)

    def halt_prob(self, y: torch.Tensor, device_state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(y, device_state))


def halting_episode(
    policy: HaltingPolicy,
    y_steps: torch.Tensor,
    device_state: torch.Tensor,
    sample: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_steps, b = y_steps.shape[0], y_steps.shape[1]
    device = y_steps.device
    active = torch.ones(b, dtype=torch.bool, device=device)
    halt_step = torch.full((b,), k_steps - 1, dtype=torch.long, device=device)
    traj_logprob = torch.zeros(b, device=device)

    for k in range(k_steps):
        logit = policy(y_steps[k], device_state)
        p = torch.sigmoid(logit).clamp(1e-6, 1.0 - 1e-6)
        if k == k_steps - 1:
            action = torch.ones(b, device=device)
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
    correct = correct_at_step.gather(0, halt_step[None]).squeeze(0).float()
    lam = compute_penalty_lambda(device_state)
    norm_steps = halt_step.float() / max(1, max_step)
    return correct - lam * lambda_step * norm_steps


@dataclass
class HaltingRunResult:
    """Observable result of online adaptive execution."""

    answer: torch.Tensor
    halt_step: torch.Tensor
    executed_steps: int
    stop_reason: str | list[str]
    logits: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    step_records: list[dict[str, Any]]
    work: dict[str, Any]
    timing: dict[str, float]
    reactivation_policy: str


@torch.no_grad()
def run_truncated_reference(
    model: nn.Module,
    x: torch.Tensor,
    *,
    steps: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Dense reference that executes exactly ``steps`` supervision steps."""
    if steps < 1 or steps > int(model.N_sup):
        raise ValueError("steps must be in [1, model.N_sup]")
    model.eval()
    state = model.init_execution_state(x, height=height, width=width)
    out = None
    for _ in range(steps):
        out = model.run_execution_step(state)
    assert out is not None
    return {
        "answer": out["logits"].argmax(dim=-1),
        "logits": out["logits"],
        "y": state.y,
        "z": state.z,
        "executed_steps": steps,
        "work": dict(state.work),
    }


@torch.no_grad()
def run_with_halting(
    model: nn.Module,
    x: torch.Tensor,
    policy: HaltingPolicy | None,
    device_state: torch.Tensor,
    height: int,
    width: int,
    threshold: float = 0.5,
    *,
    router: Any | None = None,
    max_steps: int | None = None,
    reactivation_policy: str = "allow",
    return_details: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | HaltingRunResult:
    """Execute supervision steps online and stop before later recursion is computed.

    ``max_steps`` is an executed-step budget (1 means only supervision step 0 can
    run).  For batch size one, stopping is a physical early exit.  Larger batches
    remain supported for compatibility, but no ragged-compaction throughput claim
    is made: the loop stops when all examples have decided.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if max_steps is None:
        max_steps = int(model.N_sup)
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= int(model.N_sup):
        raise ValueError("max_steps must be an integer in [1, model.N_sup]")
    if reactivation_policy not in {"allow", "sticky"}:
        raise ValueError("reactivation_policy must be 'allow' or 'sticky'")

    model.eval()
    b = int(x.shape[0])
    if device_state.ndim == 1:
        device_state = device_state.unsqueeze(0).expand(b, -1)
    if tuple(device_state.shape) != (b, DEVICE_DIM):
        raise ValueError(f"device_state must have shape [{b},{DEVICE_DIM}]")

    timing = {
        "initialization_seconds": 0.0,
        "core_seconds": 0.0,
        "router_input_seconds": 0.0,
        "router_seconds": 0.0,
        "halter_seconds": 0.0,
        "decode_seconds": 0.0,
        "total_seconds": 0.0,
    }
    total_t0 = time.perf_counter()
    t0 = time.perf_counter()
    state = model.init_execution_state(x, height=height, width=width)
    timing["initialization_seconds"] = time.perf_counter() - t0

    decided = torch.zeros(b, dtype=torch.bool, device=x.device)
    halt_step = torch.full((b,), -1, dtype=torch.long, device=x.device)
    answer = torch.zeros_like(x, dtype=torch.long)
    reasons = ["model_exhausted"] * b
    step_records: list[dict[str, Any]] = []
    previous_logits: torch.Tensor | None = None

    router_calls = 0
    halter_calls = 0
    router_prep_output_head_vectors = 0

    for step_index in range(max_steps):
        proposed_mask = None
        if router is not None:
            if previous_logits is None:
                t0 = time.perf_counter()
                previous_logits = model.out_head(state.y)
                timing["router_input_seconds"] += time.perf_counter() - t0
                router_prep_output_head_vectors += int(state.y.shape[0] * state.y.shape[1])
            t0 = time.perf_counter()
            proposed_mask, _ = router(
                step_index, state.y, state.z, previous_logits, device_state
            )
            timing["router_seconds"] += time.perf_counter() - t0
            router_calls += 1

        t0 = time.perf_counter()
        step = model.run_execution_step(
            state,
            active_mask=proposed_mask,
            reactivation_policy=reactivation_policy,
        )
        timing["core_seconds"] += time.perf_counter() - t0
        previous_logits = step["logits"]

        if policy is None:
            p_halt = torch.zeros(b, device=x.device)
            halt_now = torch.zeros(b, dtype=torch.bool, device=x.device)
        else:
            t0 = time.perf_counter()
            p_halt = policy.halt_prob(step["y"], device_state)
            timing["halter_seconds"] += time.perf_counter() - t0
            halter_calls += 1
            halt_now = p_halt > threshold

        newly = (~decided) & halt_now
        if newly.any():
            t0 = time.perf_counter()
            decoded = step["logits"].argmax(dim=-1)
            timing["decode_seconds"] += time.perf_counter() - t0
            answer = torch.where(newly[:, None], decoded, answer)
            halt_step = torch.where(newly, torch.full_like(halt_step, step_index), halt_step)
            for i in torch.nonzero(newly, as_tuple=False).flatten().tolist():
                reasons[int(i)] = "policy_halt"
            decided |= newly

        is_last_budget = step_index + 1 >= max_steps
        is_last_model = step_index + 1 >= int(model.N_sup)
        if is_last_budget or is_last_model:
            remaining = ~decided
            if remaining.any():
                t0 = time.perf_counter()
                decoded = step["logits"].argmax(dim=-1)
                timing["decode_seconds"] += time.perf_counter() - t0
                answer = torch.where(remaining[:, None], decoded, answer)
                halt_step = torch.where(remaining, torch.full_like(halt_step, step_index), halt_step)
                reason = "model_exhausted" if is_last_model and max_steps == int(model.N_sup) else "budget_exhausted"
                for i in torch.nonzero(remaining, as_tuple=False).flatten().tolist():
                    reasons[int(i)] = reason
                decided |= remaining

        step_records.append(
            {
                "step": step_index,
                "halt_probability": p_halt.detach().cpu().tolist(),
                "active_tokens": int(step["active_tokens"]),
                "active_density": float(step["active_density"]),
                "work_delta": dict(step["work_delta"]),
            }
        )
        if bool(decided.all()):
            break

    timing["total_seconds"] = time.perf_counter() - total_t0
    work: dict[str, Any] = dict(state.work)
    work.update(
        {
            "router_calls": router_calls,
            "halter_calls": halter_calls,
            "router_prep_output_head_vectors": router_prep_output_head_vectors,
            "executed_steps": len(step_records),
        }
    )
    stop_reason: str | list[str] = reasons[0] if b == 1 else reasons
    result = HaltingRunResult(
        answer=answer,
        halt_step=halt_step,
        executed_steps=len(step_records),
        stop_reason=stop_reason,
        logits=previous_logits if previous_logits is not None else model.out_head(state.y),
        y=state.y,
        z=state.z,
        step_records=step_records,
        work=work,
        timing=timing,
        reactivation_policy=reactivation_policy,
    )
    if return_details:
        return result
    return result.answer, result.halt_step
