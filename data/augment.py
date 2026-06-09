"""Validity-preserving data augmentation (BLUEPRINT data/augment.py).

Augmentation multiplies a small generated dataset into many equivalent problems.
The non-negotiable property -- checked by the Phase 2 gate -- is that a
transformation applied to a ``(puzzle, solution)`` (or ``(input, target)``) pair
preserves *both* the solution's validity *and* the correspondence between the two
grids (clues stay consistent; the maze path stays valid).

  * Sudoku: the full Sudoku symmetry group -- digit relabelling, band/line
    permutations of rows and columns, and transposition.
  * Maze: the dihedral group D4 -- rotations and reflections (isometries of the
    grid, so path validity is preserved automatically).
"""

from __future__ import annotations

import numpy as np

from data.sudoku import band_preserving_order, grid_size


# --------------------------------------------------------------------------- #
# Sudoku
# --------------------------------------------------------------------------- #
def sample_sudoku_transform(box: int, rng: np.random.Generator) -> dict:
    """Sample one element of the Sudoku symmetry group as a parameter dict."""
    n = grid_size(box)
    return {
        "row_order": band_preserving_order(box, rng),
        "col_order": band_preserving_order(box, rng),
        "digit_perm": (rng.permutation(n) + 1).astype(np.int64),  # value v -> perm[v-1]
        "transpose": bool(rng.random() < 0.5),
    }


def apply_sudoku_transform(grid: np.ndarray, params: dict) -> np.ndarray:
    """Apply a sampled Sudoku transform to a single grid (0 stays blank)."""
    g = np.asarray(grid)
    g = g[params["row_order"], :][:, params["col_order"]]
    if params["transpose"]:
        g = g.T
    out = g.copy()
    nonzero = out != 0
    # Relabel only filled cells; blanks (0) are preserved.
    out[nonzero] = params["digit_perm"][out[nonzero] - 1]
    return out


def augment_sudoku_pair(
    puzzle: np.ndarray,
    solution: np.ndarray,
    box: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one identical random symmetry to a ``(puzzle, solution)`` pair."""
    params = sample_sudoku_transform(box, rng)
    return (
        apply_sudoku_transform(puzzle, params),
        apply_sudoku_transform(solution, params),
    )


# --------------------------------------------------------------------------- #
# Maze (dihedral group D4)
# --------------------------------------------------------------------------- #
# Each transform is (rot90_count, flip) applied with numpy; the same op is used
# on both input and target so start/goal/path tokens move together.
_DIHEDRAL = [
    (0, False), (1, False), (2, False), (3, False),  # rotations
    (0, True), (1, True), (2, True), (3, True),       # rotation + horizontal flip
]


def apply_dihedral(grid: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    """Apply a D4 element: optional horizontal flip then ``rot`` * 90deg rotation."""
    g = np.asarray(grid)
    if flip:
        g = np.fliplr(g)
    if rot:
        g = np.rot90(g, k=rot)
    return np.ascontiguousarray(g)


def augment_maze_pair(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one identical random dihedral transform to an ``(input, target)`` pair."""
    rot, flip = _DIHEDRAL[rng.integers(len(_DIHEDRAL))]
    return apply_dihedral(x, rot, flip), apply_dihedral(y, rot, flip)
