"""Sudoku generation, solving, and validation.

Parameterised by ``box`` size ``b`` so the same code produces 4x4 (b=2), 9x9
(b=3), and larger boards -- matching the curriculum in BLUEPRINT section 10.2
("4x4 -> 6x6 -> 9x9"). Boards are ``numpy`` int arrays of shape ``(N, N)`` with
``N = b * b``; values are ``1..N`` and ``0`` marks a blank.

Functions here are the symbolic ground truth used both for (a) data generation
quality gates (Phase 2) and (b) the Sudoku verifier in Phase 8 (section 9.2),
which composes these checks into a score rather than reimplementing them.
"""

from __future__ import annotations

import numpy as np


def grid_size(box: int) -> int:
    """Side length ``N = box * box`` of a Sudoku with sub-box size ``box``."""
    return box * box


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _groups_ok(grid: np.ndarray, box: int) -> bool:
    """Return True if no row/column/box has a duplicate among non-zero entries."""
    n = grid_size(box)
    for i in range(n):
        if _has_dup(grid[i, :]):  # row
            return False
        if _has_dup(grid[:, i]):  # column
            return False
    for br in range(box):
        for bc in range(box):
            block = grid[br * box : (br + 1) * box, bc * box : (bc + 1) * box]
            if _has_dup(block.reshape(-1)):
                return False
    return True


def _has_dup(values: np.ndarray) -> bool:
    """True if ``values`` contains a duplicated non-zero entry."""
    nonzero = values[values != 0]
    return nonzero.size != np.unique(nonzero).size


def is_valid_grid(grid: np.ndarray, box: int) -> bool:
    """True if the (possibly partial) grid violates no Sudoku constraint."""
    return _groups_ok(np.asarray(grid), box)


def is_solved(grid: np.ndarray, box: int) -> bool:
    """True if the grid is completely filled (no zeros) and valid."""
    grid = np.asarray(grid)
    return bool((grid != 0).all()) and is_valid_grid(grid, box)


def respects_clues(puzzle: np.ndarray, solution: np.ndarray) -> bool:
    """True if ``solution`` keeps every given clue from ``puzzle`` unchanged."""
    puzzle = np.asarray(puzzle)
    solution = np.asarray(solution)
    mask = puzzle != 0
    return bool((solution[mask] == puzzle[mask]).all())


# --------------------------------------------------------------------------- #
# Solving (MRV backtracking with bitmasks -- fast enough for 9x9)
# --------------------------------------------------------------------------- #
def count_solutions(puzzle: np.ndarray, box: int, limit: int = 2) -> int:
    """Count solutions of ``puzzle`` up to ``limit`` (early-stops at ``limit``).

    Used both to verify a generated puzzle is uniquely solvable and as a building
    block for the dig-holes puzzle maker.
    """
    n = grid_size(box)
    full = (1 << n) - 1

    rows = [0] * n
    cols = [0] * n
    boxes = [0] * n
    cells: list[int] = []

    grid = np.asarray(puzzle).astype(int).reshape(n, n)
    for r in range(n):
        for c in range(n):
            v = int(grid[r, c])  # Python int keeps bitmasks out of numpy types
            if v:
                bit = 1 << (v - 1)
                b = (r // box) * box + (c // box)
                rows[r] |= bit
                cols[c] |= bit
                boxes[b] |= bit
            else:
                cells.append(r * n + c)

    count = 0

    def backtrack() -> bool:
        """Recurse; return True once ``count`` reaches ``limit`` (cut search)."""
        nonlocal count
        if not cells:
            count += 1
            return count >= limit

        # Minimum-remaining-values: choose the empty cell with fewest candidates.
        best_i, best_mask, best_n = -1, 0, n + 1
        for idx, cell in enumerate(cells):
            r, c = divmod(cell, n)
            b = (r // box) * box + (c // box)
            avail = full & ~(rows[r] | cols[c] | boxes[b])
            popcount = bin(avail).count("1")
            if popcount < best_n:
                best_i, best_mask, best_n = idx, avail, popcount
                if popcount <= 1:
                    break

        if best_n == 0:  # dead end
            return False

        cell = cells.pop(best_i)
        r, c = divmod(cell, n)
        b = (r // box) * box + (c // box)
        mask = best_mask
        while mask:
            bit = mask & -mask
            mask ^= bit
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            if backtrack():
                rows[r] ^= bit
                cols[c] ^= bit
                boxes[b] ^= bit
                cells.insert(best_i, cell)
                return True
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
        cells.insert(best_i, cell)
        return False

    backtrack()
    return count


def solve(puzzle: np.ndarray, box: int) -> np.ndarray | None:
    """Return one completed solution of ``puzzle``, or ``None`` if unsolvable."""
    n = grid_size(box)
    full = (1 << n) - 1
    grid = np.asarray(puzzle).astype(int).reshape(n, n).copy()

    rows = [0] * n
    cols = [0] * n
    boxes = [0] * n
    empties: list[int] = []
    for r in range(n):
        for c in range(n):
            v = int(grid[r, c])  # Python int keeps bitmasks out of numpy types
            if v:
                bit = 1 << (v - 1)
                b = (r // box) * box + (c // box)
                rows[r] |= bit
                cols[c] |= bit
                boxes[b] |= bit
            else:
                empties.append(r * n + c)

    def backtrack() -> bool:
        if not empties:
            return True
        best_i, best_mask, best_n = -1, 0, n + 1
        for idx, cell in enumerate(empties):
            r, c = divmod(cell, n)
            b = (r // box) * box + (c // box)
            avail = full & ~(rows[r] | cols[c] | boxes[b])
            popcount = bin(avail).count("1")
            if popcount < best_n:
                best_i, best_mask, best_n = idx, avail, popcount
                if popcount <= 1:
                    break
        if best_n == 0:
            return False
        cell = empties.pop(best_i)
        r, c = divmod(cell, n)
        b = (r // box) * box + (c // box)
        mask = best_mask
        while mask:
            bit = mask & -mask
            mask ^= bit
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            grid[r, c] = (bit.bit_length())
            if backtrack():
                return True
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
            grid[r, c] = 0
        empties.insert(best_i, cell)
        return False

    return grid if backtrack() else None


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_solution(box: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a uniformly-shuffled valid completed Sudoku grid (values 1..N).

    Uses the canonical base pattern (guaranteed valid) then applies the Sudoku
    symmetry group -- digit relabelling, row/column band/stack shuffles, and an
    optional transpose -- which all preserve validity.
    """
    n = grid_size(box)

    # Base pattern: cell (r, c) -> distinct value in each row/col/box.
    def pattern(r: int, c: int) -> int:
        return (box * (r % box) + r // box + c) % n

    rows_order = band_preserving_order(box, rng)
    cols_order = band_preserving_order(box, rng)
    labels = rng.permutation(n) + 1  # map base value -> digit 1..N

    grid = np.empty((n, n), dtype=np.int64)
    for i, r in enumerate(rows_order):
        for j, c in enumerate(cols_order):
            grid[i, j] = labels[pattern(r, c)]

    if rng.random() < 0.5:
        grid = grid.T.copy()
    return grid


def band_preserving_order(box: int, rng: np.random.Generator) -> list[int]:
    """A random row/column ordering that preserves Sudoku validity.

    Shuffles the order of bands and the lines within each band. Used both for
    generation (``generate_solution``) and for augmentation (``data/augment.py``).
    """
    bands = list(rng.permutation(box))
    order: list[int] = []
    for band in bands:
        lines = list(rng.permutation(box))
        order.extend(band * box + line for line in lines)
    return order


def make_puzzle(
    solution: np.ndarray,
    box: int,
    num_clues: int,
    rng: np.random.Generator,
    require_unique: bool = True,
) -> np.ndarray:
    """Carve a puzzle from ``solution`` by removing cells ("digging holes").

    When ``require_unique`` is True, a cell is only removed if the puzzle remains
    uniquely solvable, guaranteeing the (puzzle, solution) pair is unambiguous.
    The result has at least ``num_clues`` givens (it may have more if uniqueness
    blocks further removal).

    Args:
        solution: Completed grid (values 1..N).
        box: Sub-box size.
        num_clues: Target number of remaining givens.
        rng: NumPy random generator.
        require_unique: Enforce unique solvability while digging.

    Returns:
        Puzzle grid with blanks as 0.
    """
    n = grid_size(box)
    puzzle = np.asarray(solution).astype(np.int64).copy()
    cells = list(rng.permutation(n * n))
    target_blanks = n * n - num_clues
    blanks = 0

    for cell in cells:
        if blanks >= target_blanks:
            break
        r, c = divmod(int(cell), n)
        saved = puzzle[r, c]
        puzzle[r, c] = 0
        if require_unique and count_solutions(puzzle, box, limit=2) != 1:
            puzzle[r, c] = saved  # removal created ambiguity -> put it back
        else:
            blanks += 1
    return puzzle


def generate_pair(
    box: int,
    num_clues: int,
    rng: np.random.Generator,
    require_unique: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a ``(puzzle, solution)`` pair for the given difficulty."""
    solution = generate_solution(box, rng)
    puzzle = make_puzzle(solution, box, num_clues, rng, require_unique=require_unique)
    return puzzle, solution
