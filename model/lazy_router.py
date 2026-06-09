"""Lazy active-token routing (BLUEPRINT section 7 / 20).

Dense recursion updates every token every step, but many tokens become confident
early and updating them wastes compute. A *router* decides, per token per step,
whether to recompute (active) or freeze it. Two routers are provided:

  * :class:`ConfidenceRouter` -- the heuristic prototype baseline of section 7.3:
    freeze a token once the answer head is confident about it. Useful for
    measuring how far active-token density can fall (the Phase 7 gate), but *not*
    the final method.
  * :class:`RLTokenRouter` -- the learned policy of section 20.1: a small network
    that maps the latent state ``z`` (+ device state ``d``) to a freeze/compute
    action, trained with REINFORCE (section 20.2).

Both share the call signature ``router(step, y, z, logits, device_state) ->
(mask, logprob)`` where ``mask`` is ``[B, L, 1]`` (1 = compute, 0 = freeze) and
``logprob`` is ``[B, L]`` for the RL router (``None`` for the heuristic one).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceRouter:
    """Heuristic router: freeze tokens whose answer confidence exceeds a threshold.

    Args:
        threshold: Freeze a token when ``max softmax prob >= threshold``.
        warmup_steps: Keep all tokens active for the first ``warmup_steps``
            supervision steps (the latent state has not formed yet, so early
            confidence is meaningless).
        min_active_frac: Always keep at least this fraction of tokens active per
            example (lowest-confidence tokens), so recursion never fully stalls.
    """

    def __init__(
        self,
        threshold: float = 0.9,
        warmup_steps: int = 1,
        min_active_frac: float = 0.0,
    ):
        self.threshold = threshold
        self.warmup_steps = warmup_steps
        self.min_active_frac = min_active_frac

    def __call__(
        self,
        step: int,
        y: torch.Tensor,
        z: torch.Tensor,
        logits: torch.Tensor,
        device_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        b, n, _ = logits.shape
        if step < self.warmup_steps:
            return logits.new_ones(b, n, 1), None

        confidence = F.softmax(logits, dim=-1).max(dim=-1).values  # [B, L]
        active = confidence < self.threshold  # uncertain tokens stay active

        if self.min_active_frac > 0.0:
            # Force the k least-confident tokens active even if "confident".
            k = max(1, int(self.min_active_frac * n))
            keep = confidence.argsort(dim=-1)[:, :k]  # lowest confidence indices
            forced = torch.zeros_like(active)
            forced.scatter_(1, keep, True)
            active = active | forced

        return active.float().unsqueeze(-1), None


class RLTokenRouter(nn.Module):
    """Learned per-token freeze/compute policy (BLUEPRINT section 20.1).

    Maps each token's latent state ``z`` concatenated with the device-state vector
    ``d`` to a Bernoulli action. Samples while ``training`` (for REINFORCE);
    takes the argmax in eval.
    """

    def __init__(self, dim: int, device_dim: int = 6):
        super().__init__()
        self.device_dim = device_dim
        self.policy = nn.Sequential(
            nn.Linear(dim + device_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2),  # logits for {0: freeze, 1: compute}
        )

    def action_distribution(
        self, z: torch.Tensor, device_state: torch.Tensor | None
    ) -> torch.distributions.Categorical:
        """Per-token freeze/compute action distribution from latent + device state."""
        b, n, _ = z.shape
        if device_state is None:
            device_state = z.new_zeros(b, self.device_dim)
        d = device_state[:, None, :].expand(b, n, self.device_dim)
        action_logits = self.policy(torch.cat([z, d], dim=-1))  # [B, L, 2]
        return torch.distributions.Categorical(logits=action_logits)

    def __call__(
        self,
        step: int,
        y: torch.Tensor,
        z: torch.Tensor,
        logits: torch.Tensor | None = None,
        device_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask, logprob, _ = self.act(step, y, z, logits, device_state)
        return mask, logprob

    def act(
        self,
        step: int,
        y: torch.Tensor,
        z: torch.Tensor,
        logits: torch.Tensor | None = None,
        device_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample routing actions; returns ``(mask [B,L,1], logprob [B,L], entropy [B,L])``.

        Samples while training (for the actor-critic estimator), argmax in eval.
        Entropy is returned for the exploration bonus in ``dense_actor_critic_loss``.
        """
        dist = self.action_distribution(z, device_state)
        actions = dist.sample() if self.training else dist.probs.argmax(dim=-1)
        logprob = dist.log_prob(actions)  # [B, L]
        entropy = dist.entropy()  # [B, L]
        mask = actions.float().unsqueeze(-1)  # 1 = compute, 0 = freeze
        return mask, logprob, entropy


class LatentValueHead(nn.Module):
    """Critic ``V_phi(z^k, d)`` for dense value bootstrapping (BLUEPRINT 5.5).

    Estimates the value of a latent reasoning state so the router/halter can be
    trained with per-step temporal-difference advantages (GAE) instead of a single
    terminal REINFORCE signal. Conditioned on the device-state ``d`` so value is
    budget-aware.
    """

    def __init__(self, dim: int, device_dim: int = 6):
        super().__init__()
        self.device_dim = device_dim
        self.net = nn.Sequential(
            nn.Linear(dim + device_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, z: torch.Tensor, device_state: torch.Tensor | None = None) -> torch.Tensor:
        """Return value ``[B]`` for latent ``z`` (``[B, L, D]`` pooled, or ``[B, D]``)."""
        pooled = z.mean(dim=1) if z.dim() == 3 else z
        if device_state is None:
            device_state = pooled.new_zeros(pooled.shape[0], self.device_dim)
        return self.net(torch.cat([pooled, device_state], dim=-1)).squeeze(-1)


def router_reinforce_loss(
    logprobs: list[torch.Tensor],
    reward: torch.Tensor,
    baseline: torch.Tensor | float,
) -> torch.Tensor:
    """Vanilla episodic REINFORCE (BLUEPRINT section 20.2).

    NOTE: kept only as the *baseline* ablation. The production training objective
    is the GAE-lambda actor-critic in ``train/rl.py`` (``dense_actor_critic_loss``),
    which assigns per-step credit via verifier-bootstrapped advantages. This single
    terminal-advantage estimator is high variance and must not be used for final
    routing/halting claims.

    Args:
        logprobs: Per-step routing log-probs, each ``[B, L]``.
        reward: Per-example terminal reward ``[B]``.
        baseline: Variance-reduction baseline (scalar or ``[B]``).
    """
    advantage = (reward - baseline).detach()  # [B]
    total_logprob = sum(lp.sum(dim=1) for lp in logprobs)  # [B]
    return -(advantage * total_logprob).mean()
