"""Phase 2 gate (part 1): augmentation correctness.

Augmentation must preserve solution validity and puzzle<->solution correspondence
for Sudoku, and path validity for mazes.
"""

import numpy as np

from data import augment as aug
from data import maze as mz
from data import sudoku as sk


def test_sudoku_augmentation_preserves_validity_and_clues():
    box = 3
    rng = np.random.default_rng(0)
    puzzle, solution = sk.generate_pair(box, num_clues=40, rng=rng, require_unique=True)
    assert sk.count_solutions(puzzle, box) == 1
    n_clues = int((puzzle != 0).sum())

    for _ in range(25):
        p_aug, s_aug = aug.augment_sudoku_pair(puzzle, solution, box, rng)
        # Solution stays a valid completed board.
        assert sk.is_solved(s_aug, box)
        # Clues are preserved in count and remain consistent with the solution.
        assert int((p_aug != 0).sum()) == n_clues
        assert sk.respects_clues(p_aug, s_aug)
        # Uniqueness is preserved (symmetry is a bijection on solutions).
        assert sk.count_solutions(p_aug, box) == 1


def test_sudoku_transform_is_invertible_on_solution():
    """Two different transforms generally produce different but valid boards."""
    box = 2
    rng = np.random.default_rng(1)
    solution = sk.generate_solution(box, rng)
    seen = set()
    for _ in range(10):
        params = aug.sample_sudoku_transform(box, rng)
        t = aug.apply_sudoku_transform(solution, params)
        assert sk.is_solved(t, box)
        seen.add(t.tobytes())
    assert len(seen) > 1  # augmentation actually varies the board


def test_maze_augmentation_preserves_path():
    rng = np.random.default_rng(2)
    x, y = mz.generate_pair(15, 15, rng, min_path_len=12)
    orig_path_cells = int((y == mz.PATH).sum())

    for _ in range(16):
        x_aug, y_aug = aug.augment_maze_pair(x, y, rng)
        # Token vocabularies preserved.
        assert set(np.unique(x_aug)).issubset({0, 1, 2, 3})
        assert set(np.unique(y_aug)).issubset({0, 1, 2, 3, 4})
        # Path-cell count is invariant under isometry.
        assert int((y_aug == mz.PATH).sum()) == orig_path_cells
        # Input/target differ only on the path overlay.
        diff = x_aug != y_aug
        assert (y_aug[diff] == mz.PATH).all()
        # The transformed maze is still solvable with the same shortest length.
        grid = (x_aug != mz.WALL).astype(np.int64)
        start = tuple(int(v) for v in np.argwhere(x_aug == mz.START)[0])
        goal = tuple(int(v) for v in np.argwhere(x_aug == mz.GOAL)[0])
        path = mz.shortest_path(grid, start, goal)
        assert path is not None
        assert len(path) == orig_path_cells + 2  # +2 for start and goal
