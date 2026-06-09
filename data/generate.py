"""Infinite synthetic task generator + self-play flywheel (BLUEPRINT section 10).

A finite dataset saturates; SPECTRA keeps producing training pressure with a
procedural task generator (section 10.2) and an asynchronous "night-shift"
flywheel (section 10.1): generate -> validate -> solve with System 2 -> keep only
verifier-confirmed solutions -> compile them into System 1 training data.

This module implements the Sudoku instance of the generator plus the generic
flywheel loop; other task families plug in the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from data import sudoku as sk
from model.verifier import sudoku_correct


@dataclass
class GeneratedTask:
    """A generated task plus the curriculum metadata of section 10.2."""

    puzzle: np.ndarray  # flattened input grid
    solution: np.ndarray  # flattened reference solution
    difficulty: float  # 0 (easy) .. 1 (hard)
    num_clues: int
    box: int
    failure_mode: str = ""  # tag set when the quality gate rejects a task


class SyntheticSudokuGenerator:
    """Procedural Sudoku generator with a difficulty knob and a solution verifier.

    Difficulty linearly interpolates the number of given clues between an easy
    (many clues) and hard (few clues) bound -- the section 10.2 difficulty
    parameter / curriculum metadata.
    """

    def __init__(self, box: int = 3, min_clue_frac: float = 0.30, max_clue_frac: float = 0.65):
        self.box = box
        self.n_cells = sk.grid_size(box) ** 2
        self.min_clues = max(1, int(min_clue_frac * self.n_cells))
        self.max_clues = int(max_clue_frac * self.n_cells)

    def clues_for_difficulty(self, difficulty: float) -> int:
        difficulty = float(min(1.0, max(0.0, difficulty)))
        # Hard (difficulty 1) -> few clues; easy (0) -> many clues.
        return int(round(self.max_clues - difficulty * (self.max_clues - self.min_clues)))

    def generate(
        self, difficulty: float, rng: np.random.Generator, require_unique: bool = True
    ) -> GeneratedTask:
        """Generate one validated task at the given difficulty."""
        num_clues = self.clues_for_difficulty(difficulty)
        puzzle, solution = sk.generate_pair(
            self.box, num_clues, rng, require_unique=require_unique
        )
        return GeneratedTask(
            puzzle=puzzle.reshape(-1),
            solution=solution.reshape(-1),
            difficulty=difficulty,
            num_clues=int((puzzle != 0).sum()),
            box=self.box,
        )


# System 2 solver: maps a batch of puzzles [B, L] to answers [B, L].
System2Solver = Callable[[torch.Tensor], torch.Tensor]


def self_play_flywheel(
    generator: SyntheticSudokuGenerator,
    system2_solve: System2Solver,
    n_tasks: int,
    rng: np.random.Generator,
    difficulty: float = 0.6,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the night-shift loop and return compiled System 1 training data.

    Generates ``n_tasks`` tasks, solves them with System 2, and admits only those
    whose System 2 answer the *symbolic verifier* confirms correct (the section
    10.3 data-quality gate). The admitted ``(puzzle, system2_answer)`` pairs are
    the distillation targets that compile System 2's discoveries into System 1.

    Returns:
        ``(inputs [M, L], targets [M, L], stats)`` where ``M <= n_tasks``.
    """
    tasks = [generator.generate(difficulty, rng) for _ in range(n_tasks)]
    puzzles = torch.from_numpy(np.stack([t.puzzle for t in tasks])).to(device)

    answers = system2_solve(puzzles)  # [n_tasks, L]
    verified = sudoku_correct(puzzles, answers, generator.box)  # [n_tasks] bool

    inputs = puzzles[verified].cpu().numpy()
    targets = answers[verified].cpu().numpy()
    stats = {
        "generated": n_tasks,
        "admitted": int(verified.sum()),
        "admit_rate": float(verified.float().mean()),
        "difficulty": difficulty,
    }
    return inputs, targets, stats
