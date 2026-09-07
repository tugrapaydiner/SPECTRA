"""Grounded verifier over the complete recursive search state.

M07 deliberately separates this independently grounded verifier from the legacy
``LatentEnergyVerifier(x, z)``.  The recursive transition consumes both ``y`` and
``z``; this module therefore scores the versioned state representation
``search_state_xyz_v1 = (x, y, z)``.

The primary M07 target is *not* eventual solve probability.  The sigmoid output
estimates the probability that one deterministic frozen-reasoner recursive cycle
improves the independently computed Sudoku structural score.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.fake_quant import FakeActQuant
from model.operators import SwapBlock
from model.spatial_encoder import SpatialEncoder

STATE_REPRESENTATION_VERSION = "search_state_xyz_v1"
GROUNDED_TARGET_ID = "sudoku_one_cycle_improvement_v1"


class GroundedStateVerifier(nn.Module):
    """Predict one-cycle oracle-score improvement from ``(x, y, z)``.

    ``y`` remains FP because native MCTS stores the answer accumulator in FP.
    ``z`` is fake-quantized internally to match the native INT8 search-state
    boundary before projection.

    ``include_y=False`` exists only for the M07 z-only ablation.  Such a model is
    not a valid complete search-state representation and must not be accepted as
    the grounded native-search verifier.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        n_layers: int = 1,
        heads: int = 4,
        max_grid_size: int = 32,
        act_bits: int = 8,
        *,
        include_y: bool = True,
        state_representation: str = STATE_REPRESENTATION_VERSION,
    ):
        super().__init__()
        if state_representation != STATE_REPRESENTATION_VERSION:
            raise ValueError(
                f"unsupported grounded state representation {state_representation!r}"
            )
        self.num_tokens = int(num_tokens)
        self.dim = int(dim)
        self.n_layers = int(n_layers)
        self.heads = int(heads)
        self.max_grid_size = int(max_grid_size)
        self.act_bits = int(act_bits)
        self.include_y = bool(include_y)
        self.state_representation = state_representation

        self.x_embed = nn.Embedding(self.num_tokens, self.dim)
        self.pos_encoder = SpatialEncoder(self.dim, self.max_grid_size)
        self.y_proj = nn.Linear(self.dim, self.dim) if self.include_y else None
        self.z_proj = nn.Linear(self.dim, self.dim)
        self.z_quant = FakeActQuant(bits=self.act_bits)
        self.blocks = nn.ModuleList(
            [SwapBlock(self.dim, heads=self.heads) for _ in range(self.n_layers)]
        )
        self.norm = nn.RMSNorm(self.dim)
        self.head = nn.Linear(self.dim, 1)

    def forward_logits(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        width: int = 9,
    ) -> torch.Tensor:
        """Return improvement logits ``[B]`` for full search states."""
        if x.ndim != 2 or y.ndim != 3 or z.ndim != 3:
            raise ValueError("grounded verifier expects x[B,L], y[B,L,D], z[B,L,D]")
        if y.shape != z.shape:
            raise ValueError("grounded verifier y/z shapes must match")
        if x.shape[:2] != y.shape[:2] or y.shape[-1] != self.dim:
            raise ValueError("grounded verifier state shape disagrees with architecture")

        h = self.x_embed(x) + self.pos_encoder(x.shape[1], width, x.device)
        if self.y_proj is not None:
            h = h + self.y_proj(y)
        h = h + self.z_proj(self.z_quant(z))
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h).mean(dim=1)).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        width: int = 9,
    ) -> torch.Tensor:
        """Sigmoid probability of the declared one-cycle improvement event."""
        return torch.sigmoid(self.forward_logits(x, y, z, width))

    def value_state(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        width: int = 9,
    ) -> torch.Tensor:
        """Higher-is-better grounded state value used by native MCTS."""
        return self.forward(x, y, z, width)


class EnsembleGroundedStateVerifier(nn.Module):
    """Deep ensemble used for M07 probability mean + disagreement diagnostics.

    Member disagreement is only a heuristic.  M07 separately measures its
    relationship to held-out error and does not enable an MCTS uncertainty
    penalty by default.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        n_members: int = 3,
        n_layers: int = 1,
        heads: int = 4,
        max_grid_size: int = 32,
        act_bits: int = 8,
        *,
        include_y: bool = True,
        state_representation: str = STATE_REPRESENTATION_VERSION,
    ):
        super().__init__()
        if int(n_members) < 2:
            raise ValueError("grounded verifier ensemble requires at least two members")
        self.num_tokens = int(num_tokens)
        self.dim = int(dim)
        self.n_members = int(n_members)
        self.n_layers = int(n_layers)
        self.heads = int(heads)
        self.max_grid_size = int(max_grid_size)
        self.act_bits = int(act_bits)
        self.include_y = bool(include_y)
        self.state_representation = state_representation
        self.members = nn.ModuleList(
            [
                GroundedStateVerifier(
                    self.num_tokens,
                    self.dim,
                    self.n_layers,
                    self.heads,
                    self.max_grid_size,
                    self.act_bits,
                    include_y=self.include_y,
                    state_representation=self.state_representation,
                )
                for _ in range(self.n_members)
            ]
        )

    def member_logits(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, width: int = 9
    ) -> torch.Tensor:
        """Stacked member logits ``[M,B]``."""
        return torch.stack([m.forward_logits(x, y, z, width) for m in self.members])

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, width: int = 9
    ) -> torch.Tensor:
        """Stacked member probabilities ``[M,B]``."""
        return torch.sigmoid(self.member_logits(x, y, z, width))

    def value_state(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, width: int = 9
    ) -> torch.Tensor:
        """Mean grounded improvement probability ``[B]``."""
        return self.forward(x, y, z, width).mean(dim=0)

    def value_with_uncertainty_state(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, width: int = 9
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean probability and member standard deviation ``[B]``."""
        p = self.forward(x, y, z, width)
        return p.mean(dim=0), p.std(dim=0, unbiased=False)


def grounded_architecture_metadata(module: nn.Module) -> dict[str, object]:
    """Versioned architecture metadata for strict auxiliary checkpoints."""
    if isinstance(module, EnsembleGroundedStateVerifier):
        return {
            "class": "EnsembleGroundedStateVerifier",
            "num_tokens": module.num_tokens,
            "dim": module.dim,
            "n_members": module.n_members,
            "n_layers": module.n_layers,
            "heads": module.heads,
            "max_grid_size": module.max_grid_size,
            "act_bits": module.act_bits,
            "include_y": module.include_y,
            "state_representation": module.state_representation,
        }
    if isinstance(module, GroundedStateVerifier):
        return {
            "class": "GroundedStateVerifier",
            "num_tokens": module.num_tokens,
            "dim": module.dim,
            "n_members": 1,
            "n_layers": module.n_layers,
            "heads": module.heads,
            "max_grid_size": module.max_grid_size,
            "act_bits": module.act_bits,
            "include_y": module.include_y,
            "state_representation": module.state_representation,
        }
    raise TypeError(f"unsupported grounded verifier module {type(module).__name__}")
