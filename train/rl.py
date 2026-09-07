"""Actor-critic utilities for SPECTRA adaptive execution.

The pre-M12 helpers are retained for compatibility, but M12 adds the contracts
needed by a real router/halter episode: per-example valid/terminal/truncation
masks, policy-invariant grounded potential shaping, and explicit decision masks
so forced environment actions never receive policy-gradient credit.

M12's optimized cost is a *logical step/token proxy*.  Nothing in this module
turns those proxies into measured energy; joules require the separate physical
measurement protocol.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def dense_step_rewards(
    verifier_values: torch.Tensor,
    active_density: torch.Tensor,
    energy_delta: torch.Tensor | None = None,
    terminal_reward: torch.Tensor | None = None,
    lambda_tokens: float = 0.05,
    lambda_step: float = 0.01,
    lambda_energy: float = 0.0,
) -> torch.Tensor:
    """Legacy dense reward helper retained for existing experiments.

    This historical helper uses ``Phi(next)-Phi(current)`` and therefore is only
    the canonical potential-based form when the return discount is one.  M12 uses
    :func:`grounded_step_rewards`, which explicitly implements
    ``gamma*Phi(next)-Phi(current)`` plus terminal treatment.
    """
    verifier_delta = verifier_values[1:] - verifier_values[:-1]
    rewards = verifier_delta - lambda_tokens * active_density - lambda_step
    if energy_delta is not None:
        rewards = rewards - lambda_energy * energy_delta
    if terminal_reward is not None:
        rewards = rewards.clone()
        rewards[-1] = rewards[-1] + terminal_reward
    return rewards


def _shape_kb(name: str, value: torch.Tensor, shape: tuple[int, int]) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def grounded_step_rewards(
    potentials: torch.Tensor,
    active_density: torch.Tensor,
    valid_mask: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    success: torch.Tensor,
    voluntary_halt_unsolved: torch.Tensor,
    *,
    gamma: float = 0.99,
    success_reward: float = 1.0,
    halt_failure_penalty: float = 0.25,
    lambda_step: float = 0.01,
    lambda_token: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """M12 grounded reward with explicit terminal/truncation semantics.

    ``potentials`` is ``[K+1,B]`` from the *frozen* grounded verifier.  All other
    tensors are ``[K,B]``.  For a true terminal, the effective next potential is
    zero.  A time-limit truncation keeps the real next potential.  The shaping
    term is therefore exactly ``gamma*Phi(next_effective)-Phi(current)``.

    Step/token charges are declared logical compute proxies, not energy.
    """
    if potentials.ndim != 2:
        raise ValueError("potentials must have shape [K+1,B]")
    k_steps, batch = potentials.shape[0] - 1, potentials.shape[1]
    if k_steps <= 0:
        raise ValueError("potentials must contain at least two states")
    shape = (k_steps, batch)
    for name, tensor in (
        ("active_density", active_density), ("valid_mask", valid_mask),
        ("terminated", terminated), ("truncated", truncated), ("success", success),
        ("voluntary_halt_unsolved", voluntary_halt_unsolved),
    ):
        _shape_kb(name, tensor, shape)
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    if bool((active_density < 0).any()) or bool((active_density > 1).any()):
        raise ValueError("active_density must lie in [0,1]")

    valid = valid_mask.bool()
    term = terminated.bool()
    trunc = truncated.bool()
    succ = success.bool()
    bad_halt = voluntary_halt_unsolved.bool()
    if bool((term & trunc & valid).any()):
        raise ValueError("a valid transition cannot be both terminated and truncated")
    if bool((succ & ~term & valid).any()):
        raise ValueError("exact success must be a terminal transition")
    if bool((bad_halt & ~term & valid).any()):
        raise ValueError("voluntary unsolved halt must be a terminal transition")

    v = valid.to(potentials.dtype)
    terminal_f = term.to(potentials.dtype)
    success_f = succ.to(potentials.dtype)
    bad_halt_f = bad_halt.to(potentials.dtype)

    step_proxy = float(lambda_step) * v
    token_proxy = float(lambda_token) * active_density.to(potentials.dtype) * v
    base = (
        float(success_reward) * success_f
        - float(halt_failure_penalty) * bad_halt_f
        - step_proxy
        - token_proxy
    )

    next_phi = potentials[1:]
    effective_next = next_phi * (1.0 - terminal_f)
    shaping = (float(gamma) * effective_next - potentials[:-1]) * v
    rewards = (base + shaping) * v
    return rewards, {
        "base_reward": base.detach(),
        "potential_shaping": shaping.detach(),
        "step_proxy_cost": step_proxy.detach(),
        "token_proxy_cost": token_proxy.detach(),
        "total_proxy_cost": (step_proxy + token_proxy).detach(),
    }


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_is_terminal: bool = True,
    *,
    valid_mask: torch.Tensor | None = None,
    terminated: torch.Tensor | None = None,
    truncated: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation with per-example episode boundaries.

    ``rewards`` is ``[K,B]`` and ``values`` is ``[K+1,B]``.  M12 semantics:

    - a true ``terminated`` transition does **not** bootstrap;
    - a ``truncated`` transition **does** bootstrap from ``values[k+1]``;
    - both termination and truncation stop the GAE trace because no later sampled
      transition belongs to that rollout segment;
    - invalid post-episode slots contribute exactly zero.

    The historical ``last_is_terminal`` argument remains as a compatibility
    shorthand when no explicit masks are supplied.
    """
    if rewards.ndim != 2 or values.ndim != 2:
        raise ValueError("rewards/values must be rank-2 [K,B]/[K+1,B]")
    k_steps, batch = rewards.shape
    if tuple(values.shape) != (k_steps + 1, batch):
        raise ValueError(
            f"values must have shape {(k_steps + 1, batch)}, got {tuple(values.shape)}"
        )
    if not 0.0 <= float(gamma) <= 1.0 or not 0.0 <= float(lam) <= 1.0:
        raise ValueError("gamma and lam must lie in [0,1]")

    shape = (k_steps, batch)
    explicit = any(x is not None for x in (valid_mask, terminated, truncated))
    if valid_mask is None:
        valid = torch.ones(shape, dtype=torch.bool, device=rewards.device)
    else:
        _shape_kb("valid_mask", valid_mask, shape)
        valid = valid_mask.bool()
    if terminated is None:
        term = torch.zeros(shape, dtype=torch.bool, device=rewards.device)
        if not explicit and last_is_terminal and k_steps:
            term[-1] = True
    else:
        _shape_kb("terminated", terminated, shape)
        term = terminated.bool()
    if truncated is None:
        trunc = torch.zeros(shape, dtype=torch.bool, device=rewards.device)
    else:
        _shape_kb("truncated", truncated, shape)
        trunc = truncated.bool()
    if bool((term & trunc & valid).any()):
        raise ValueError("a valid transition cannot be both terminated and truncated")

    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(rewards[0]) if k_steps else rewards.new_zeros(batch)
    for k in reversed(range(k_steps)):
        vk = valid[k]
        bootstrap = vk & ~term[k]
        trace = vk & ~term[k] & ~trunc[k]
        delta = (
            rewards[k]
            + float(gamma) * values[k + 1] * bootstrap.to(values.dtype)
            - values[k]
        )
        delta = delta * vk.to(delta.dtype)
        candidate = delta + float(gamma) * float(lam) * trace.to(delta.dtype) * last_adv
        last_adv = torch.where(vk, candidate, torch.zeros_like(candidate))
        advantages[k] = last_adv

    returns = (advantages + values[:k_steps]) * valid.to(values.dtype)
    return advantages, returns


def _masked_standardize(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = values.detach().clone()
    selected = out[mask]
    if selected.numel() > 1:
        mean = selected.mean()
        std = selected.std(unbiased=False)
        if float(std) > 0.0:
            out = (out - mean) / (std + 1e-8)
        else:
            out = out - mean
    return out


def masked_policy_objective(
    logprobs: torch.Tensor,
    entropies: torch.Tensor,
    advantages: torch.Tensor,
    decision_mask: torch.Tensor,
    *,
    normalize_adv: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Policy loss/entropy over decisions the policy actually made.

    Forced actions are represented by ``decision_mask=False`` and therefore have
    exactly zero gradient contribution even if a caller stores arbitrary synthetic
    log-probabilities in those slots.
    """
    if not (logprobs.shape == entropies.shape == advantages.shape == decision_mask.shape):
        raise ValueError("policy tensors and decision_mask must have identical [K,B] shape")
    mask = decision_mask.bool()
    count = int(mask.sum().item())
    if count == 0:
        zero = (logprobs * 0.0).sum()
        return zero, (entropies * 0.0).sum(), 0
    adv = _masked_standardize(advantages, mask) if normalize_adv else advantages.detach()
    denom = decision_mask.to(logprobs.dtype).sum().clamp_min(1.0)
    policy = -(adv * logprobs * decision_mask.to(logprobs.dtype)).sum() / denom
    entropy = (entropies * decision_mask.to(entropies.dtype)).sum() / denom
    return policy, entropy, count


def masked_value_loss(
    critic_values: torch.Tensor,
    returns: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if not (critic_values.shape == returns.shape == valid_mask.shape):
        raise ValueError("critic_values/returns/valid_mask must have identical [K,B] shape")
    mask = valid_mask.bool()
    if int(mask.sum()) == 0:
        return (critic_values * 0.0).sum()
    error = (critic_values - returns.detach()).square()
    return (error * valid_mask.to(error.dtype)).sum() / valid_mask.to(error.dtype).sum()


def grounded_actor_critic_loss(
    *,
    router_logprobs: torch.Tensor,
    router_entropies: torch.Tensor,
    router_decisions: torch.Tensor,
    halter_logprobs: torch.Tensor,
    halter_entropies: torch.Tensor,
    halter_decisions: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    critic_values: torch.Tensor,
    valid_mask: torch.Tensor,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    normalize_adv: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """M12 joint actor-critic objective with separate router/halter telemetry."""
    router_loss, router_entropy, router_n = masked_policy_objective(
        router_logprobs, router_entropies, advantages, router_decisions,
        normalize_adv=normalize_adv,
    )
    halter_loss, halter_entropy, halter_n = masked_policy_objective(
        halter_logprobs, halter_entropies, advantages, halter_decisions,
        normalize_adv=normalize_adv,
    )
    value_loss = masked_value_loss(critic_values, returns, valid_mask)
    entropy_bonus = router_entropy + halter_entropy
    total = router_loss + halter_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy_bonus
    return total, {
        "router_policy": float(router_loss.detach()),
        "halter_policy": float(halter_loss.detach()),
        "value": float(value_loss.detach()),
        "router_entropy": float(router_entropy.detach()),
        "halter_entropy": float(halter_entropy.detach()),
        "router_decisions": float(router_n),
        "halter_decisions": float(halter_n),
        "total": float(total.detach()),
    }


def dense_actor_critic_loss(
    step_logprobs: torch.Tensor,
    step_entropies: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    critic_values: torch.Tensor,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    normalize_adv: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Backward-compatible dense single-actor objective."""
    mask = torch.ones_like(step_logprobs, dtype=torch.bool)
    policy_loss, entropy, _ = masked_policy_objective(
        step_logprobs, step_entropies, advantages, mask,
        normalize_adv=normalize_adv,
    )
    value_loss = masked_value_loss(critic_values, returns, mask)
    total = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy
    return total, {
        "policy": float(policy_loss.detach()),
        "value": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "total": float(total.detach()),
    }


# value_fn(x [B, L], z [B, L, D]) -> legacy verifier value [B]
VerifierValueFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def make_target_critic(value_head: nn.Module) -> nn.Module:
    """Frozen target copy of the online critic."""
    target = copy.deepcopy(value_head)
    for p in target.parameters():
        p.requires_grad_(False)
    return target.eval()


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float = 0.01) -> None:
    """Polyak update: ``target <- (1-tau)*target + tau*online``."""
    if not 0.0 <= float(tau) <= 1.0:
        raise ValueError("tau must lie in [0,1]")
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.mul_(1.0 - float(tau)).add_(op.detach(), alpha=float(tau))


def rollout_router_gae(
    model: Any,
    router: Any,
    value_head: Any,
    x: torch.Tensor,
    height: int,
    width: int,
    verifier_value_fn: VerifierValueFn,
    target_value_head: Any | None = None,
    device_state: torch.Tensor | None = None,
    terminal_reward: torch.Tensor | None = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    lambda_tokens: float = 0.05,
    lambda_step: float = 0.01,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Legacy dense router-only rollout retained for prior tests/experiments.

    M12's real training entry point uses explicit M11 execution state, the full
    grounded `(x,y,z)` verifier and per-example terminal/truncation masks instead.
    """
    x_emb = model.token_embed(x) + model.encode_positions(x, height, width)
    y = torch.zeros_like(x_emb)
    z = torch.zeros_like(x_emb)
    k_steps = model.N_sup

    z_states = [z]
    logprobs, entropies, densities = [], [], []
    for k in range(k_steps):
        route_logits = model.out_head(y)
        mask, logprob, entropy = router.act(k, y, z, route_logits, device_state)
        for _ in range(model.T):
            y, z = model.recursive_cycle(x_emb, y, z, mask)
        logprobs.append(logprob.sum(dim=1))
        entropies.append(entropy.mean(dim=1))
        densities.append(mask.mean(dim=(1, 2)))
        z_states.append(z)

    z_detached = [zk.detach() for zk in z_states]
    target_head = target_value_head if target_value_head is not None else value_head
    with torch.no_grad():
        bootstrap_values = torch.stack([target_head(zk, device_state) for zk in z_detached])
        verifier_values = torch.stack([verifier_value_fn(x, zk) for zk in z_detached])
    online_values = torch.stack([value_head(z_detached[k], device_state) for k in range(k_steps)])

    density = torch.stack(densities)
    rewards = dense_step_rewards(
        verifier_values, density, terminal_reward=terminal_reward,
        lambda_tokens=lambda_tokens, lambda_step=lambda_step,
    )
    advantages, returns = compute_gae(rewards, bootstrap_values, gamma, lam)
    loss, comp = dense_actor_critic_loss(
        torch.stack(logprobs), torch.stack(entropies), advantages, returns,
        online_values, value_coef=value_coef, entropy_coef=entropy_coef,
    )
    telemetry = {
        "advantages": advantages.detach(),
        "rewards": rewards.detach(),
        "density": density.detach(),
        "verifier_values": verifier_values,
    }
    return loss, comp, telemetry
