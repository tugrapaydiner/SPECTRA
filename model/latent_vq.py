"""Latent Vector Quantization -- bounds compounding recursion/INT8 error (DT #1).

THE CHAOS-THEORY ARGUMENT, AND WHY VQ DEFEATS IT
------------------------------------------------
A recursive map with per-step perturbation ``epsilon`` and Lipschitz constant
``L`` accumulates error ``~ epsilon * (L^d - 1)/(L - 1)`` over depth ``d``. If
``L > 1`` (the generic case for an unconstrained transition) this is exponential
and a depth-20 MCTS rollout turns the INT8 latent into noise the verifier can no
longer score.

Vector Quantization removes the exponent. After each step we *project* the latent
onto a finite learned codebook ``C = {c_1, ..., c_K}``:

    z_q = argmin_{c in C} || z - c ||.

Two consequences make the error bounded by a CONSTANT, independent of depth:

  1. **Bounded per-step error.** ``|| z - z_q || <= r``, the covering radius of the
     codebook. The map's output always lands exactly on a codebook point.
  2. **No compounding.** Because every state is a codebook point, the trajectory
     lives on a finite set; ``snap(snap(z)) = snap(z)`` (idempotent), so noise that
     does not change the nearest-neighbour assignment is annihilated rather than
     accumulated. The reachable error is ``<= r`` for ANY depth ``d``.

So VQ converts ``O(epsilon * L^d)`` compounding into ``O(r)`` constant error -- the
latent topology survives arbitrarily deep search. The codebook is trained with the
standard VQ-VAE straight-through estimator + commitment loss; MCTS inference uses
the pure projection ``snap``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentVQ(nn.Module):
    """Vector quantizer for the recursive latent (VQ-VAE style, per-token).

    Args:
        dim: Latent dimension D.
        codebook_size: Number of code vectors K.
        commitment: Weight of the commitment loss (encoder -> codebook).
        decay: EMA decay for the codebook (0 disables EMA -> plain learnable codebook).
    """

    def __init__(self, dim: int, codebook_size: int = 256, commitment: float = 0.25, decay: float = 0.99):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment = commitment
        self.decay = decay

        codebook = torch.randn(codebook_size, dim)
        if decay > 0.0:  # EMA codebook (not a leaf parameter), more stable
            self.register_buffer("codebook", codebook)
            self.register_buffer("_cluster_size", torch.zeros(codebook_size))
            self.register_buffer("_ema_sum", codebook.clone())
        else:
            self.codebook = nn.Parameter(codebook)

    def _nearest(self, flat: torch.Tensor) -> torch.Tensor:
        """Nearest codebook index for each row of ``flat`` ``[N, D]``."""
        # ||z - c||^2 = ||z||^2 - 2 z.c + ||c||^2  (argmin over c)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )
        return dist.argmin(dim=1)

    @torch.no_grad()
    def snap(self, z: torch.Tensor) -> torch.Tensor:
        """Project ``z`` onto the codebook (no grad) -- the MCTS inference path."""
        b, n, d = z.shape
        idx = self._nearest(z.reshape(-1, d))
        return self.codebook[idx].reshape(b, n, d)

    @torch.no_grad()
    def covering_radius(self, z: torch.Tensor) -> float:
        """Max per-token projection error ``max ||z - snap(z)||`` (the error bound)."""
        return (z - self.snap(z)).reshape(-1, z.shape[-1]).norm(dim=1).max().item()

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize with the straight-through estimator (training path).

        Returns ``(z_q_ste, indices [B, L], vq_loss)``.
        """
        b, n, d = z.shape
        flat = z.reshape(-1, d)
        idx = self._nearest(flat)
        z_q = self.codebook[idx].reshape(b, n, d)

        # Commitment loss pulls the encoder output toward its chosen code.
        commit = F.mse_loss(z, z_q.detach())
        if self.decay > 0.0:
            if self.training:
                self._ema_update(flat, idx)
            codebook_loss = torch.zeros((), device=z.device)  # EMA updates the codebook
        else:
            codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment * commit

        z_q_ste = z + (z_q - z).detach()  # straight-through: identity gradient to encoder
        return z_q_ste, idx.reshape(b, n), vq_loss

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, idx: torch.Tensor) -> None:
        onehot = F.one_hot(idx, self.codebook_size).type_as(flat)  # [N, K]
        counts = onehot.sum(0)  # [K]
        embed_sum = onehot.t() @ flat  # [K, D]
        self._cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self._ema_sum.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        n = self._cluster_size.sum()
        cluster_size = (self._cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
        self.codebook.copy_(self._ema_sum / cluster_size.unsqueeze(1))
