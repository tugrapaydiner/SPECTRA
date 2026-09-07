"""Discrete latent actions for cache-resident search.

The legacy :class:`LatentActionCodebook` stores trainable residual directions and
a *global* PUCT prior.  Search runs under ``torch.no_grad()`` and therefore never
updates those parameters by itself.

Milestone 09 adds :class:`StateConditionedLatentActionCodebook`, an explicitly
offline-trained mechanism whose prior depends on the complete search state
``(x, y, z)``.  Optimizer ownership lives in the training path, not in MCTS.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentActionCodebook(nn.Module):
    """Legacy learned action set with a global prior (action 0 = identity)."""

    def __init__(self, dim: int, n_actions: int = 4, scale: float = 0.5):
        super().__init__()
        if not isinstance(n_actions, int) or isinstance(n_actions, bool) or n_actions < 1:
            raise ValueError("n_actions must be >= 1 (index 0 is the identity)")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if not isinstance(scale, (int, float)) or not torch.isfinite(torch.tensor(float(scale))):
            raise ValueError("scale must be finite")
        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.scale = float(scale)
        # Keep the historical state-dict shape for n_actions=1 compatibility.
        self.directions = nn.Parameter(torch.randn(max(1, n_actions - 1), dim) * 0.02)
        self.prior_logits = nn.Parameter(torch.zeros(n_actions))

    def apply_action(self, z: torch.Tensor, action: int) -> torch.Tensor:
        """Return ``z'`` after applying discrete ``action`` (0 = identity)."""
        if not isinstance(action, int) or isinstance(action, bool):
            raise TypeError("action must be an integer")
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions - 1}]")
        if action == 0:
            return z
        return z + self.scale * self.directions[action - 1]

    def priors(self) -> torch.Tensor:
        """Global softmax action prior ``[n_actions]`` used by legacy PUCT."""
        return torch.softmax(self.prior_logits, dim=-1)


class StateConditionedLatentActionCodebook(nn.Module):
    """M09 action directions plus a state-conditioned ``P(a|x,y,z)`` policy.

    This class intentionally has no global ``prior_logits`` parameter.  The prior
    is produced by a small full-state network and can therefore differ across
    problems and search nodes.  ``directions`` remain explicit trainable tensors so
    their optimizer ownership and movement can be tested independently of policy
    fitting.
    """

    PRIOR_SEMANTICS = "state_conditioned_xyz_v1"
    STATE_REPRESENTATION = "search_state_xyz_v1"

    def __init__(
        self,
        *,
        dim: int,
        num_tokens: int,
        n_actions: int = 4,
        scale: float = 0.5,
        hidden_dim: int = 64,
    ):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool) or num_tokens <= 1:
            raise ValueError("num_tokens must be an integer > 1")
        if not isinstance(n_actions, int) or isinstance(n_actions, bool) or n_actions < 2:
            raise ValueError("state-conditioned codebook requires n_actions >= 2")
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        scale_t = torch.tensor(float(scale))
        if not torch.isfinite(scale_t) or float(scale) <= 0.0:
            raise ValueError("scale must be finite and positive")

        self.dim = int(dim)
        self.num_tokens = int(num_tokens)
        self.n_actions = int(n_actions)
        self.scale = float(scale)
        self.hidden_dim = int(hidden_dim)

        self.directions = nn.Parameter(torch.randn(n_actions - 1, dim) * 0.02)
        self.x_embed = nn.Embedding(num_tokens, dim)
        self.x_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.y_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.z_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, n_actions)

    def apply_action(self, z: torch.Tensor, action: int) -> torch.Tensor:
        if not isinstance(action, int) or isinstance(action, bool):
            raise TypeError("action must be an integer")
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions - 1}]")
        if action == 0:
            return z
        return z + self.scale * self.directions[action - 1]

    def policy_logits_for_state(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Return state-conditioned logits ``[B, A]`` for ``(x,y,z)``."""
        if x.ndim != 2 or y.ndim != 3 or z.ndim != 3:
            raise ValueError("expected x[B,L], y[B,L,D], z[B,L,D]")
        if y.shape != z.shape:
            raise ValueError("y and z shapes must match")
        if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1] or y.shape[2] != self.dim:
            raise ValueError("state shapes are incompatible with action policy")
        if x.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
            raise TypeError("x must contain integer token ids")
        if x.numel() and (int(x.min()) < 0 or int(x.max()) >= self.num_tokens):
            raise ValueError("x contains token ids outside the configured vocabulary")

        x_pool = self.x_embed(x).mean(dim=1)
        y_pool = y.mean(dim=1)
        z_pool = z.mean(dim=1)
        h = self.x_proj(x_pool) + self.y_proj(y_pool) + self.z_proj(z_pool)
        h = torch.tanh(self.norm(h))
        return self.policy_head(h)

    def priors_for_state(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        width: int | None = None,
    ) -> torch.Tensor:
        """Return ``P(a|x,y,z)``; ``width`` is accepted for search API symmetry."""
        del width
        return torch.softmax(self.policy_logits_for_state(x, y, z), dim=-1)
