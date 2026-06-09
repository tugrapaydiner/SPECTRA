"""Correctness tests for Sudoku generation, solving, and validation."""

import numpy as np
import pytest

from data import sudoku as sk


@pytest.mark.parametrize("box", [2, 3])
def test_generate_solution_is_valid_and_complete(box):
    rng = np.random.default_rng(0)
    n = sk.grid_size(box)
    for _ in range(20):
        grid = sk.generate_solution(box, rng)
        assert grid.shape == (n, n)
        assert grid.min() >= 1 and grid.max() <= n
        assert sk.is_solved(grid, box)


def test_validators_detect_violations():
    box = 2
    grid = sk.generate_solution(box, np.random.default_rng(1))
    assert sk.is_valid_grid(grid, box)
    # Introduce a duplicate in a row -> invalid.
    bad = grid.copy()
    bad[0, 0] = bad[0, 1]
    assert not sk.is_valid_grid(bad, box)
    # A partial grid (with zeros) can still be valid.
    partial = grid.copy()
    partial[0, 0] = 0
    assert sk.is_valid_grid(partial, box)
    assert not sk.is_solved(partial, box)


@pytest.mark.parametrize("box", [2, 3])
def test_solve_recovers_a_valid_completion(box):
    rng = np.random.default_rng(2)
    puzzle, solution = sk.generate_pair(
        box, num_clues=sk.grid_size(box) ** 2 // 2, rng=rng, require_unique=True
    )
    solved = sk.solve(puzzle, box)
    assert solved is not None
    assert sk.is_solved(solved, box)
    assert sk.respects_clues(puzzle, solved)
    # Unique puzzle -> solver must recover the exact generating solution.
    assert np.array_equal(solved, solution)


def test_make_puzzle_is_unique_and_consistent():
    box = 3
    rng = np.random.default_rng(3)
    solution = sk.generate_solution(box, rng)
    puzzle = sk.make_puzzle(solution, box, num_clues=40, rng=rng, require_unique=True)
    # Clues are a subset of the solution.
    assert sk.respects_clues(puzzle, solution)
    # Exactly one solution.
    assert sk.count_solutions(puzzle, box, limit=2) == 1
    # At least the requested number of clues remain.
    assert int((puzzle != 0).sum()) >= 40


def test_count_solutions_detects_ambiguity():
    box = 2
    # An empty 4x4 grid has many solutions; count is capped at the limit.
    empty = np.zeros((4, 4), dtype=np.int64)
    assert sk.count_solutions(empty, box, limit=2) == 2
