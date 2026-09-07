"""Symbolic task verifiers with explicit milestone-03 contracts.

Scores are soft diagnostics.  Boolean ``*_correct`` functions are the semantic
success gates and intentionally do not substitute exact reference matching.
"""
from __future__ import annotations

import numpy as np
import torch

from data import maze as maze_task


# --------------------------------------------------------------------------- #
# Sudoku
# --------------------------------------------------------------------------- #
def _require_sudoku_shape(puzzle: torch.Tensor, candidate: torch.Tensor, box: int) -> int:
    if not isinstance(box, int) or isinstance(box, bool) or box <= 0:
        raise ValueError("box must be a positive integer")
    n = box * box
    if puzzle.ndim != 2 or candidate.ndim != 2:
        raise ValueError("Sudoku puzzle/candidate must both have shape [B, N*N]")
    if puzzle.shape != candidate.shape or puzzle.shape[1] != n * n:
        raise ValueError(
            f"Sudoku puzzle/candidate must have identical [B, {n*n}] shape; "
            f"got {tuple(puzzle.shape)} and {tuple(candidate.shape)}"
        )
    if puzzle.device != candidate.device:
        raise ValueError("Sudoku puzzle/candidate must be on the same device")
    return n


def _rows_cols_boxes(candidate: torch.Tensor, box: int) -> tuple[torch.Tensor, ...]:
    n = box * box
    grid = candidate.reshape(-1, n, n)
    rows = grid
    cols = grid.transpose(1, 2)
    boxes = (
        grid.reshape(-1, box, box, box, box)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, n, n)
    )
    return rows, cols, boxes


def _group_valid_fraction(groups: torch.Tensor, n: int) -> torch.Tensor:
    target = torch.arange(1, n + 1, device=groups.device, dtype=groups.dtype)
    ok = (groups.sort(dim=-1).values == target).all(dim=-1)
    return ok.float().mean(dim=-1)


def _partial_groups_valid(groups: torch.Tensor, n: int) -> torch.Tensor:
    """[B] true iff each group has only 0..n and no repeated non-zero digit."""
    domain = ((groups >= 0) & (groups <= n)).all(dim=(-1, -2))
    per_group = torch.ones(groups.shape[:2], dtype=torch.bool, device=groups.device)
    for value in range(1, n + 1):
        per_group &= (groups == value).sum(dim=-1) <= 1
    return domain & per_group.all(dim=-1)


def sudoku_puzzle_valid(puzzle: torch.Tensor, box: int) -> torch.Tensor:
    """Boolean [B] partial-grid validity matching ``data.sudoku.is_valid_grid``."""
    n = box * box
    if puzzle.ndim != 2 or puzzle.shape[1] != n * n:
        raise ValueError(f"Sudoku puzzle must have shape [B, {n*n}]")
    rows, cols, boxes = _rows_cols_boxes(puzzle, box)
    return (
        _partial_groups_valid(rows, n)
        & _partial_groups_valid(cols, n)
        & _partial_groups_valid(boxes, n)
    )


def sudoku_score(
    puzzle: torch.Tensor,
    candidate: torch.Tensor,
    box: int,
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> torch.Tensor:
    """Soft row/column/box/clue score; invalid puzzle domains receive score 0."""
    n = _require_sudoku_shape(puzzle, candidate, box)
    rows, cols, boxes = _rows_cols_boxes(candidate, box)
    v_row = _group_valid_fraction(rows, n)
    v_col = _group_valid_fraction(cols, n)
    v_box = _group_valid_fraction(boxes, n)
    given = puzzle != 0
    n_given = given.sum(dim=-1)
    preserved = ((candidate == puzzle) & given).sum(dim=-1).float()
    v_given = torch.where(
        n_given > 0,
        preserved / n_given.clamp_min(1).float(),
        torch.ones(candidate.shape[0], device=candidate.device),
    )
    wr, wc, wb, wg = weights
    score = wr * v_row + wc * v_col + wb * v_box + wg * v_given
    return score * sudoku_puzzle_valid(puzzle, box).float()


def sudoku_correct(puzzle: torch.Tensor, candidate: torch.Tensor, box: int) -> torch.Tensor:
    """Boolean [B]: solved permutation grid, valid puzzle, and every clue preserved."""
    n = _require_sudoku_shape(puzzle, candidate, box)
    rows, cols, boxes = _rows_cols_boxes(candidate, box)
    candidate_valid = (
        (_group_valid_fraction(rows, n) == 1.0)
        & (_group_valid_fraction(cols, n) == 1.0)
        & (_group_valid_fraction(boxes, n) == 1.0)
    )
    given = puzzle != 0
    clues_ok = ((candidate == puzzle) | ~given).all(dim=-1)
    return sudoku_puzzle_valid(puzzle, box) & candidate_valid & clues_ok


# --------------------------------------------------------------------------- #
# Maze
# --------------------------------------------------------------------------- #
def _require_maze_shape(input_grid: torch.Tensor, candidate: torch.Tensor, height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("maze height/width must be positive")
    if input_grid.ndim != 2 or candidate.ndim != 2:
        raise ValueError("maze input/candidate must have shape [B, H*W]")
    if input_grid.shape != candidate.shape or input_grid.shape[1] != height * width:
        raise ValueError(
            f"maze input/candidate must have identical [B, {height*width}] shape"
        )


def maze_correct(
    input_grid: torch.Tensor,
    candidate: torch.Tensor,
    height: int,
    width: int,
    *,
    require_optimal: bool = True,
) -> torch.Tensor:
    """Exact semantic maze success independent of the stored reference target."""
    _require_maze_shape(input_grid, candidate, height, width)
    inp = input_grid.detach().cpu().numpy().reshape(-1, height, width)
    cand = candidate.detach().cpu().numpy().reshape(-1, height, width)
    result = [
        maze_task.candidate_success(x, y, height, width, require_optimal=require_optimal)
        for x, y in zip(inp, cand)
    ]
    return torch.tensor(result, dtype=torch.bool, device=candidate.device)


def maze_score(input_grid: torch.Tensor, candidate: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Soft structural score plus the strict semantic-success component.

    Crucially, a candidate with no PATH cells receives zero for the path-overlay
    component.  Copying a non-trivial unsolved input therefore cannot score as a
    semantic success.
    """
    _require_maze_shape(input_grid, candidate, height, width)
    walls_kept = ((input_grid == maze_task.WALL) == (candidate == maze_task.WALL)).float().mean(dim=-1)
    endpoints_kept = (
        ((candidate == maze_task.START) == (input_grid == maze_task.START))
        & ((candidate == maze_task.GOAL) == (input_grid == maze_task.GOAL))
    ).float().mean(dim=-1)
    path_cells = candidate == maze_task.PATH
    n_path = path_cells.sum(dim=-1)
    path_on_open = torch.where(
        n_path > 0,
        (path_cells & (input_grid == maze_task.OPEN)).sum(dim=-1).float() / n_path.clamp_min(1).float(),
        torch.zeros(candidate.shape[0], device=candidate.device),
    )
    semantic = maze_correct(input_grid, candidate, height, width).float()
    return (walls_kept + endpoints_kept + path_on_open + semantic) / 4.0


# --------------------------------------------------------------------------- #
# Synthetic ARC-style soft verifier
# --------------------------------------------------------------------------- #
def arc_soft_score(
    candidate: torch.Tensor,
    target_palette: torch.Tensor | None = None,
    pad_token: int = 10,
) -> torch.Tensor:
    """Structural plausibility for SPECTRA's synthetic local ARC-style variant.

    This is not an official ARC/ARC-AGI benchmark verifier.
    """
    valid_colors = ((candidate >= 0) & (candidate <= pad_token)).float().mean(dim=-1)
    if target_palette is None:
        return valid_colors
    allowed = torch.zeros(candidate.shape[0], pad_token + 1, dtype=torch.bool, device=candidate.device)
    allowed.scatter_(1, target_palette.clamp(0, pad_token), True)
    in_palette = allowed.gather(1, candidate.clamp(0, pad_token)).float().mean(dim=-1)
    return 0.5 * valid_colors + 0.5 * in_palette
