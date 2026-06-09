"""Correctness tests for maze generation, shortest path, and validation."""

import numpy as np
import pytest

from data import maze as mz


def test_generate_maze_is_solvable():
    rng = np.random.default_rng(0)
    for _ in range(20):
        grid, start, goal = mz.generate_maze(15, 15, rng)
        assert grid.shape == (15, 15)
        assert grid[start] == mz.OPEN and grid[goal] == mz.OPEN
        path = mz.shortest_path(grid, start, goal)
        assert path is not None
        assert mz.is_valid_path(grid, start, goal, path)


def test_generate_maze_rejects_even_dims():
    with pytest.raises(ValueError):
        mz.generate_maze(14, 15, np.random.default_rng(0))


def test_is_valid_path_rejects_bad_paths():
    rng = np.random.default_rng(1)
    grid, start, goal = mz.generate_maze(15, 15, rng)
    good = mz.shortest_path(grid, start, goal)
    assert mz.is_valid_path(grid, start, goal, good)

    # Wrong endpoints.
    assert not mz.is_valid_path(grid, start, goal, good[:-1])
    # A diagonal jump is illegal (manhattan distance 2).
    jump = [start, (start[0] + 1, start[1] + 1)]
    assert not mz.is_valid_path(grid, start, (start[0] + 1, start[1] + 1), jump)


def test_target_marks_path_tokens():
    rng = np.random.default_rng(2)
    grid, start, goal = mz.generate_maze(15, 15, rng)
    path = mz.shortest_path(grid, start, goal)
    target = mz.make_target(grid, start, goal, path)
    # Intermediate path cells become PATH; endpoints keep START/GOAL.
    assert target[start] == mz.START
    assert target[goal] == mz.GOAL
    assert (target == mz.PATH).sum() == len(path) - 2
    # Path cells recovered.
    assert len(mz.extract_path(target)) == len(path)


def test_generate_pair_token_ranges():
    rng = np.random.default_rng(3)
    x, y = mz.generate_pair(15, 15, rng, min_path_len=10)
    assert set(np.unique(x)).issubset({0, 1, 2, 3})
    assert set(np.unique(y)).issubset({0, 1, 2, 3, 4})
    # Input and target agree everywhere except the path overlay.
    diff = (x != y)
    assert (y[diff] == mz.PATH).all()
