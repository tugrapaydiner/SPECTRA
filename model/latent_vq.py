"""Latent vector quantization helpers.

The quantizer projects each per-token latent onto a finite learned codebook.  Two
facts are mechanical:

* ``snap(z)`` is a nearest-code projection and is idempotent on exact codebook
  points;
* for any observed pre-projection state, ``||z - snap(z)||`` is a measurable local
  projection error.

Those facts do **not** imply a depth-independent error bound relative to an
unquantized recursive trajectory.  After projection, the next transition is
applied to a different state, so the projected and unprojected trajectories can
separate over time.  A bound on that trajectory divergence would require extra
assumptions (for example contraction/stability of the transition, codebook
coverage of the reference trajectory, and stable nearest-neighbour assignments)
and evidence that those assumptions hold.

Likewise, finite codebook membership does not by itself guarantee preserved
reasoning, topology, rank, isometry, calibration, or OOD detection.  Milestone 15
therefore measures local projection error, trajectory divergence, code usage,
decoded-answer agreement, and task success across depth instead of inferring those
properties from bounded codebook states.

The codebook uses the standard VQ-VAE straight-through/commitment construction;
MCTS inference may use the pure ``snap`` projection.
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
        """Project ``z`` onto the codebook (no grad) -- the optional MCTS path."""
        b, n, d = z.shape
        idx = self._nearest(z.reshape(-1, d))
        return self.codebook[idx].reshape(b, n, d)

    @torch.no_grad()
    def covering_radius(self, z: torch.Tensor) -> float:
        """Observed max local projection error on ``z``; not a trajectory bound."""
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