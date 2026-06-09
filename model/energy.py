"""Neural energy verifier (BLUEPRINT section 9.5).

A small network ``E_psi(x, y) -> R`` that scores a candidate answer; *lower energy
is better*. Trained contrastively to put correct answers below wrong ones, using
**hard negatives** (near-miss boards) rather than only random wrong answers.

The energy verifier doubles as the per-step value network for dense RL and Latent
MCTS (sections 5.5, 9.6): ``V_k = -E_psi(x, z^(k))``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.fake_quant import FakeActQuant
from model.operators import SwapBlock
from model.spatial_encoder import SpatialEncoder


class EnergyVerifier(nn.Module):
    """Scores ``(input, candidate answer)`` pairs with a scalar energy.

    Args:
        num_tokens: Vocabulary size.
        dim: Hidden dimension.
        n_layers: Number of attention/FFN blocks.
        heads: Attention heads.
        max_grid_size: Max grid extent for positional embeddings.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int = 64,
        n_layers: int = 2,
        heads: int = 8,
        max_grid_size: int = 32,
    ):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.x_embed = nn.Embedding(num_tokens, dim)
        self.y_embed = nn.Embedding(num_tokens, dim)
        self.pos_encoder = SpatialEncoder(dim, max_grid_size)
        self.blocks = nn.ModuleList([SwapBlock(dim, heads=heads) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, height: int = 9, width: int = 9) -> torch.Tensor:
        """Return energy ``[B]`` for each ``(x, y)`` pair (lower = better)."""
        h = self.x_embed(x) + self.y_embed(y) + self.pos_encoder(x.shape[1], width, x.device)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h).mean(dim=1)).squeeze(-1)


class LatentEnergyVerifier(nn.Module):
    """Energy verifier over the **INT8 latent** ``E_psi(x, z)`` (BLUEPRINT 9.6).

    Unlike :class:`EnergyVerifier` (which scores decoded token answers), this scores
    the continuous reasoning latent ``z`` *without decoding* -- the value network
    Cache-Resident Latent MCTS needs. The latent is INT8 fake-quantized internally,
    so the verifier operates in the same INT8 latent space the search runs in.

    Args:
        num_tokens: Vocabulary size (for the frozen input-context embedding).
        dim: Hidden dimension (matches the recursive core's latent dim).
        n_layers / heads: Verifier transformer depth/width.
        max_grid_size: For positional embeddings.
        act_bits: Latent quantization bits (8 for the A8 target).
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        n_layers: int = 2,
        heads: int = 8,
        max_grid_size: int = 32,
        act_bits: int = 8,
    ):
        super().__init__()
        self.x_embed = nn.Embedding(num_tokens, dim)
        self.pos_encoder = SpatialEncoder(dim, max_grid_size)
        self.z_proj = nn.Linear(dim, dim)  # project the latent into verifier space
        self.act_quant = FakeActQuant(bits=act_bits)
        self.blocks = nn.ModuleList([SwapBlock(dim, heads=heads) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, z: torch.Tensor, width: int = 9) -> torch.Tensor:
        """Energy ``[B]`` of latent ``z`` ``[B, L, D]`` given problem ``x`` ``[B, L]``."""
        z_q = self.act_quant(z)  # INT8 latent -- scored without ever decoding
        h = self.x_embed(x) + self.pos_encoder(x.shape[1], width, x.device) + self.z_proj(z_q)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h).mean(dim=1)).squeeze(-1)

    def value(self, x: torch.Tensor, z: torch.Tensor, width: int = 9) -> torch.Tensor:
        """Latent value ``V^psi = -E_psi(x, z)`` (higher = better)."""
        return -self.forward(x, z, width)


class EnsembleLatentEnergyVerifier(nn.Module):
    """Deep ensemble of latent energy verifiers for epistemic uncertainty (GT #3).

    A single ``E_psi`` treated as an oracle lets Latent MCTS infinitely expand OOD
    latent branches it falsely scores high. An ensemble exposes *epistemic*
    uncertainty as member disagreement: in-distribution latents -> low variance;
    OOD latents (reachable via codebook actions) -> high variance. The MCTS then
    uses a pessimistic lower-confidence-bound value (``value - beta * std``), so it
    will not chase latents whose high value is statistically ungrounded.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        n_members: int = 3,
        n_layers: int = 2,
        heads: int = 8,
        max_grid_size: int = 32,
        act_bits: int = 8,
    ):
        super().__init__()
        self.members = nn.ModuleList(
            [
                LatentEnergyVerifier(num_tokens, dim, n_layers, heads, max_grid_size, act_bits)
                for _ in range(n_members)
            ]
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor, width: int = 9) -> torch.Tensor:
        """Stacked member energies ``[n_members, B]``."""
        return torch.stack([m(x, z, width) for m in self.members])

    def value_with_uncertainty(
        self, x: torch.Tensor, z: torch.Tensor, width: int = 9
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean value = -mean energy [B], epistemic std [B])``."""
        energies = self.forward(x, z, width)  # [M, B]
        return -energies.mean(dim=0), energies.std(dim=0)

    def value(self, x: torch.Tensor, z: torch.Tensor, width: int = 9) -> torch.Tensor:
        """Mean latent value (uncertainty-agnostic; LCB applied in the searcher)."""
        return -self.forward(x, z, width).mean(dim=0)


def contrastive_energy_loss(
    e_pos: torch.Tensor, e_neg: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Margin ranking loss: push ``E(x, y+)`` below ``E(x, y-)`` (section 9.5)."""
    return F.relu(margin + e_pos - e_neg).mean()


def mine_hard_negatives(
    answers: torch.Tensor, num_tokens: int, n_changes: int = 2, blank: int = 0
) -> torch.Tensor:
    """Create near-miss negatives by changing a few cells to different values.

    Args:
        answers: Correct answers ``[B, L]`` (values ``1..num_tokens-1``; ``blank``
            reserved).
        num_tokens: Vocabulary size (values ``1..num_tokens-1`` are usable).
        n_changes: Number of cells to corrupt per example.
        blank: Reserved blank token id to avoid producing.

    Returns:
        Hard-negative answers ``[B, L]`` (each differs from the input in
        ``n_changes`` cells, never to the blank token).
    """
    neg = answers.clone()
    b, length = neg.shape
    m = num_tokens - 1  # number of usable non-blank values (1..m)
    for _ in range(n_changes):
        pos = torch.randint(0, length, (b, 1), device=neg.device)
        cur = neg.gather(1, pos)
        delta = torch.randint(1, max(2, m), (b, 1), device=neg.device)
        # New value in 1..m, guaranteed different from cur.
        new_val = ((cur - 1 + delta) % m) + 1
        neg.scatter_(1, pos, new_val)
    return neg
