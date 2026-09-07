"""Additional M03 cross-implementation parity checks."""
import numpy as np
import pytest
import torch

from data import sudoku as sk
from model.verifier import sudoku_correct, sudoku_puzzle_valid


def test_sudoku_numpy_tensor_both_reject_float_grid_representation():
    solved = sk.generate_solution(2, np.random.default_rng(2026))
    puzzle = solved.copy()
    puzzle[0, 0] = 0

    assert sk.is_valid_grid(puzzle, 2)
    assert not sk.is_valid_grid(puzzle.astype(np.float32), 2)

    puzzle_i = torch.from_numpy(puzzle.reshape(1, -1))
    solved_i = torch.from_numpy(solved.reshape(1, -1))
    assert sudoku_puzzle_valid(puzzle_i, 2).item()
    assert sudoku_correct(puzzle_i, solved_i, 2).item()

    with pytest.raises(TypeError, match="integer dtype"):
        sudoku_puzzle_valid(puzzle_i.float(), 2)
    with pytest.raises(TypeError, match="integer dtype"):
        sudoku_correct(puzzle_i.float(), solved_i.float(), 2)
