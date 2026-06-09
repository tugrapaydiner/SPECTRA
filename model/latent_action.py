"""Discrete latent action codebook for Cache-Resident Latent MCTS (BLUEPRINT 9.6).

The previous MCTS "expanded" nodes with ``torch.randn`` -- random search wearing a
tree. This replaces that with a *principled, learned discrete action set*: a small
codebook of residual directions in latent space. Expanding a node with action
``a`` deterministically nudges the latent ``z' = z + scale * direction[a]`` (action
0 is the identity / pure recursive step), giving the search a structured,
learnable branching factor and an action prior for PUCT.

The codebook is trainable end-to-end from successful trajectories (the flywheel's
"Latent MCTS policy prior" distillation target, section 10.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentActionCodebook(nn.Module):
    """A learned set of ``n_actions`` discrete latent moves (action 0 = identity).

    Args:
        dim: Latent dimension.
        n_actions: Number of discrete actions (including the identity at index 0).
        scale: Magnitude of the residual nudge applied per action.
    """

    def __init__(self, dim: int, n_actions: int = 4, scale: float = 0.5):
        super().__init__()
        if n_actions < 1:
            raise ValueError("n_actions must be >= 1 (index 0 is the identity)")
        self.n_actions = n_actions
        self.scale = scale
        # n_actions - 1 learned directions; index 0 is the identity (no residual).
        self.directions = nn.Parameter(torch.randn(max(1, n_actions - 1), dim) * 0.02)
        self.prior_logits = nn.Parameter(torch.zeros(n_actions))  # PUCT action prior

    def apply_action(self, z: torch.Tensor, action: int) -> torch.Tensor:
        """Return ``z'`` after applying discrete ``action`` (0 = identity)."""
        if action == 0:
            return z
        return z + self.scale * self.directions[action - 1]

    def priors(self) -> torch.Tensor:
        """Softmax action prior ``[n_actions]`` used for PUCT exploration."""
        return torch.softmax(self.prior_logits, dim=-1)
