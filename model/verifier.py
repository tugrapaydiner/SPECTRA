"""Symbolic verifiers (BLUEPRINT section 9.2-9.4).

Cheap, exact symbolic checks that score how valid a candidate answer is. They are
used as the first stage of verifier-guided inference (best-of-N, section 9.1) and
to mine hard negatives for the neural energy verifier (section 9.5).

Scores are in ``[0, 1]`` (1 = perfect) and computed batched over candidates of
shape ``[B, L]`` (flattened grids) as torch tensors.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
# Sudoku (section 9.2)
# --------------------------------------------------------------------------- #
def _group_valid_fraction(groups: torch.Tensor, n: int) -> torch.Tensor:
    """Fraction of groups (``[B, n, n]``) that are a permutation of ``1..n``."""
    target = torch.arange(1, n + 1, device=groups.device)
    ok = (groups.sort(dim=-1).values == target).all(dim=-1)  # [B, n]
    return ok.float().mean(dim=-1)  # [B]


def _rows_cols_boxes(candidate: torch.Tensor, box: int) -> tuple[torch.Tensor, ...]:
    """Reshape a flat candidate ``[B, n*n]`` into row/col/box groups ``[B, n, n]``."""
    n = box * box
    grid = candidate.view(-1, n, n)
    rows = grid
    cols = grid.transpose(1, 2)
    # Boxes: split each axis into (box, box) and regroup so each box's cells are
    # contiguous along the last dim.
    boxes = (
        grid.view(-1, box, box, box, box)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, n, n)
    )
    return rows, cols, boxes


def sudoku_score(
    puzzle: torch.Tensor,
    candidate: torch.Tensor,
    box: int,
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> torch.Tensor:
    """Weighted Sudoku validity score ``[B]`` (section 9.2).

    Combines row / column / box permutation validity and clue preservation.
    """
    n = box * box
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
    return wr * v_row + wc * v_col + wb * v_box + wg * v_given


def sudoku_correct(puzzle: torch.Tensor, candidate: torch.Tensor, box: int) -> torch.Tensor:
    """Boolean ``[B]``: candidate is a fully valid board respecting all clues."""
    n = box * box
    rows, cols, boxes = _rows_cols_boxes(candidate, box)
    valid = (
        (_group_valid_fraction(rows, n) == 1.0)
        & (_group_valid_fraction(cols, n) == 1.0)
        & (_group_valid_fraction(boxes, n) == 1.0)
    )
    given = puzzle != 0
    # Every given cell either matches the puzzle, or is not a given.
    clues_ok = ((candidate == puzzle) | ~given).all(dim=-1)
    return valid & clues_ok


# --------------------------------------------------------------------------- #
# Maze (section 9.3)
# --------------------------------------------------------------------------- #
def maze_score(input_grid: torch.Tensor, candidate: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Soft maze path score ``[B]``: rewards a contiguous wall-free path overlay.

    Checks (each contributes equally): the candidate keeps walls as walls, keeps
    the start/goal tokens, and the predicted PATH cells lie only on open cells.
    Tokens: 0=wall, 1=open, 2=start, 3=goal, 4=path.
    """
    walls_kept = ((input_grid == 0) == (candidate == 0)).float().mean(dim=-1)
    start_goal_kept = (
        ((candidate == 2) == (input_grid == 2)) & ((candidate == 3) == (input_grid == 3))
    ).float().mean(dim=-1)
    # PATH cells must sit on cells that were open in the input (not walls).
    path_cells = candidate == 4
    path_on_open = torch.where(
        path_cells.any(dim=-1),
        (path_cells & (input_grid == 1)).sum(dim=-1) / path_cells.sum(dim=-1).clamp_min(1),
        torch.ones(candidate.shape[0], device=candidate.device),
    )
    return (walls_kept + start_goal_kept + path_on_open) / 3.0


# --------------------------------------------------------------------------- #
# ARC-style soft verifier (section 9.4)
# --------------------------------------------------------------------------- #
def arc_soft_score(
    candidate: torch.Tensor,
    target_palette: torch.Tensor | None = None,
    pad_token: int = 10,
) -> torch.Tensor:
    """Soft ARC plausibility score ``[B]`` from cheap structural checks.

    Without the (hidden) target we can still check: the output uses a valid color
    set, and -- if a reference palette is given -- that the candidate's color set
    is a subset of it (transformation consistency). Scores in ``[0, 1]``.
    """
    valid_colors = ((candidate >= 0) & (candidate <= pad_token)).float().mean(dim=-1)
    if target_palette is None:
        return valid_colors
    # Fraction of candidate colors that appear in the allowed palette.
    allowed = torch.zeros(candidate.shape[0], pad_token + 1, dtype=torch.bool, device=candidate.device)
    allowed.scatter_(1, target_palette.clamp(0, pad_token), True)
    in_palette = allowed.gather(1, candidate.clamp(0, pad_token)).float().mean(dim=-1)
    return 0.5 * valid_colors + 0.5 * in_palette
