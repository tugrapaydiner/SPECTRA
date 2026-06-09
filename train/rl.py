"""Dense, verifier-bootstrapped RL for the router/halter (BLUEPRINT section 5.5).

This replaces the terminal-only REINFORCE objective (``router_reinforce_loss``)
with a proper actor-critic that assigns *per-step* credit:

    r_k  = (V^psi_{k+1} - V^psi_k)            # dense verifier value delta (shaping)
           - lambda_tokens * |A_k|/L          # active-token compute penalty
           - lambda_step                      # per-step latency penalty
           - lambda_E * dE_k                   # measured/estimated energy penalty
    delta_k = r_k + gamma * V_phi(z^{k+1}) - V_phi(z^k)     # TD residual
    A_k     = sum_l (gamma*lambda)^l delta_{k+l}            # GAE-lambda
    R_k     = A_k + V_phi(z^k)                              # bootstrapped return

where ``V^psi_k = -E_psi(x, z^k)`` is the (frozen) neural energy verifier's value
of the latent state, and ``V_phi`` is the *learned* critic (``LatentValueHead``).
The policy gradient weights each step's routing log-prob by its own advantage --
the dense credit assignment the blueprint's section 5.5 actually specifies.
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
    """Per-step rewards ``r_k`` ``[K, B]`` (BLUEPRINT section 5.5).

    Args:
        verifier_values: ``V^psi_k = -E_psi(x, z^k)`` for each latent, shape
            ``[K+1, B]`` (the extra entry bootstraps the final transition).
        active_density: Active-token fraction ``|A_k|/L`` per step ``[K, B]``.
        energy_delta: Optional incremental energy ``dE_k`` per step ``[K, B]``.
        terminal_reward: Optional sparse final reward ``[B]`` added at the last
            step (e.g. ``1[correct] - lambda_E * E_measured``).
        lambda_tokens / lambda_step / lambda_energy: Penalty weights.

    Returns:
        Per-step reward tensor ``[K, B]``.
    """
    # Dense potential-based shaping from the verifier value (V_{k+1} - V_k).
    verifier_delta = verifier_values[1:] - verifier_values[:-1]  # [K, B]
    rewards = verifier_delta - lambda_tokens * active_density - lambda_step
    if energy_delta is not None:
        rewards = rewards - lambda_energy * energy_delta
    if terminal_reward is not None:
        rewards = rewards.clone()
        rewards[-1] = rewards[-1] + terminal_reward
    return rewards


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_is_terminal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (Schulman et al., 2016).

    Args:
        rewards: Per-step rewards ``[K, B]``.
        values: Critic values ``V_phi(z^k)`` for ``k = 0..K``, shape ``[K+1, B]``
            (``values[K]`` bootstraps the return past the last step).
        gamma: Discount factor.
        lam: GAE lambda (bias/variance trade-off).
        last_is_terminal: If True, the episode ends after step ``K-1`` so the
            bootstrap past the final step is masked out.

    Returns:
        ``(advantages [K, B], returns [K, B])``. ``returns = advantages +
        values[:K]`` are the critic regression targets.
    """
    k_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(rewards[0])
    for k in reversed(range(k_steps)):
        # Non-terminal mask for the bootstrap of step k -> k+1.
        nonterminal = 0.0 if (k == k_steps - 1 and last_is_terminal) else 1.0
        delta = rewards[k] + gamma * values[k + 1] * nonterminal - values[k]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        advantages[k] = last_adv
    returns = advantages + values[:k_steps]
    return advantages, returns


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
    """Actor-critic loss with per-step GAE advantages.

    Args:
        step_logprobs: Summed action log-prob per step ``[K, B]`` (e.g. routing
            log-probs summed over tokens, or the halt-decision log-prob).
        step_entropies: Policy entropy per step ``[K, B]`` (exploration bonus).
        advantages: GAE advantages ``[K, B]`` (detached from the critic).
        returns: Critic regression targets ``[K, B]``.
        critic_values: ``V_phi(z^k)`` predictions for ``k=0..K-1`` ``[K, B]``
            (these carry gradient to the value head).
        value_coef / entropy_coef: Loss weights.
        normalize_adv: Standardize advantages across the batch (stabilises PG).

    Returns:
        ``(total_loss, components)``.
    """
    adv = advantages.detach()
    if normalize_adv and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Per-step policy gradient: each step weighted by ITS OWN advantage.
    policy_loss = -(adv * step_logprobs).mean()
    value_loss = F.mse_loss(critic_values, returns.detach())
    entropy_loss = -step_entropies.mean()

    total = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    return total, {
        "policy": float(policy_loss.detach()),
        "value": float(value_loss.detach()),
        "entropy": float((-entropy_loss).detach()),
        "total": float(total.detach()),
    }


# value_fn(x [B, L], z [B, L, D]) -> verifier value [B]  (V^psi = -E_psi)
VerifierValueFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def make_target_critic(value_head: nn.Module) -> nn.Module:
    """Frozen target copy of the critic for stationary TD bootstrapping.

    Recursive cores share weights across steps, so the critic's input latent ``z``
    moves as ``f_theta`` updates -- a violently non-stationary regression target.
    Bootstrapping GAE from a slowly-tracking *target* critic (Polyak-updated via
    :func:`soft_update`) decouples the targets from the actor's shifting latent
    space, the standard DQN/actor-critic stabilisation.
    """
    target = copy.deepcopy(value_head)
    for p in target.parameters():
        p.requires_grad_(False)
    return target.eval()


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float = 0.01) -> None:
    """Polyak update: ``target <- (1 - tau) * target + tau * online``."""
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.mul_(1.0 - tau).add_(op.detach(), alpha=tau)


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
    """Run one instrumented recursive rollout and return the dense AC loss.

    Drives the TRM recursion with ``router`` sampling per-step masks, the learned
    critic ``value_head`` (``V_phi``) scoring each latent, and the frozen
    ``verifier_value_fn`` (``V^psi = -E_psi``) supplying the dense reward shaping.
    Computes GAE advantages and the actor-critic loss. This is the production dense
    training step the blueprint's section 5.5 describes.

    Returns ``(loss, components, telemetry)`` where ``telemetry`` holds the
    per-step advantages / rewards / densities for logging.
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
        logprobs.append(logprob.sum(dim=1))            # [B] summed over tokens
        entropies.append(entropy.mean(dim=1))          # [B] mean token entropy
        densities.append(mask.mean(dim=(1, 2)))        # [B] active-token fraction
        z_states.append(z)

    # STOP-GRADIENT the latent into the critic: the value-regression loss must not
    # warp the shared recursive trunk f_theta (only the actor's policy gradient
    # shapes the representation). The actor path keeps its gradient through the
    # routing log-probs, which still depend on z -> f_theta.
    z_detached = [zk.detach() for zk in z_states]
    target_head = target_value_head if target_value_head is not None else value_head
    with torch.no_grad():
        # Stationary bootstrap from the (frozen/EMA) TARGET critic -- defeats the
        # non-stationary-target trap. Plus the frozen verifier value for shaping.
        bootstrap_values = torch.stack([target_head(zk, device_state) for zk in z_detached])  # [K+1, B]
        verifier_values = torch.stack([verifier_value_fn(x, zk) for zk in z_detached])         # [K+1, B]
    # Online critic predictions for the regression loss (gradient to critic only).
    online_values = torch.stack(
        [value_head(z_detached[k], device_state) for k in range(k_steps)]
    )  # [K, B]

    density = torch.stack(densities)  # [K, B]
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
