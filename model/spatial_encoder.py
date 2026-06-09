"""Tiny 2D spatial/symbolic positional encoder (BLUEPRINT section 4, 15.2).

Grids are flattened to sequences, so the model needs to know each token's (row,
col). This shared module adds learned row + column embeddings and is reused by the
recursive core, the energy verifier, and the System 1 student (DRY -- previously
each inlined the same computation).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialEncoder(nn.Module):
    """Learned row+column positional embeddings for a flattened HxW grid.

    Args:
        dim: Embedding dimension (matches the model hidden size).
        max_grid_size: Largest supported grid extent (row/col table size).
    """

    def __init__(self, dim: int, max_grid_size: int = 32):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.row_embed = nn.Embedding(max_grid_size, dim)
        self.col_embed = nn.Embedding(max_grid_size, dim)

    def forward(self, length: int, width: int, device: torch.device) -> torch.Tensor:
        """Return positional embedding ``[1, length, dim]`` (broadcasts over batch).

        Token ``i`` maps to ``row = i // width``, ``col = i % width``.
        """
        positions = torch.arange(length, device=device)
        rows = positions // width
        cols = positions % width
        if int(rows.max()) >= self.max_grid_size or int(cols.max()) >= self.max_grid_size:
            raise ValueError(
                f"grid index exceeds max_grid_size={self.max_grid_size} "
                f"(length={length}, width={width})"
            )
        return self.row_embed(rows)[None, :, :] + self.col_embed(cols)[None, :, :]
