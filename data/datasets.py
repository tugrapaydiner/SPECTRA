"""Torch dataset wrappers and builders for the symbolic tasks.

``GridDataset`` holds flattened integer grids and yields ``LongTensor`` pairs
``(input_ids, target_ids)`` of shape ``[L]``. Task builders generate the raw
grids (optionally augmented) and record the ``height``/``width`` the recursive
core needs for positional encoding.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from data import arc as arc_task
from data import augment as aug
from data import babyai as babyai_task
from data import maze as mz
from data import smarthome as sh_task
from data import sudoku as sk


class GridDataset(Dataset):
    """In-memory dataset of flattened grid problems.

    Args:
        inputs: Integer array ``[M, L]`` of input token ids.
        targets: Integer array ``[M, L]`` of target token ids.
        height: Grid height (for the model's row positional embedding).
        width: Grid width (for the model's column positional embedding).
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray, height: int, width: int):
        assert inputs.shape == targets.shape, "inputs/targets shape mismatch"
        self.inputs = np.ascontiguousarray(inputs, dtype=np.int64)
        self.targets = np.ascontiguousarray(targets, dtype=np.int64)
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.inputs[idx]),
            torch.from_numpy(self.targets[idx]),
        )


def build_sudoku_arrays(
    box: int,
    n: int,
    num_clues: int,
    rng: np.random.Generator,
    require_unique: bool = True,
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Generate ``n`` flattened ``(puzzle, solution)`` Sudoku pairs."""
    side = sk.grid_size(box)
    inputs = np.empty((n, side * side), dtype=np.int64)
    targets = np.empty((n, side * side), dtype=np.int64)
    for i in range(n):
        puzzle, solution = sk.generate_pair(box, num_clues, rng, require_unique=require_unique)
        if augment:
            puzzle, solution = aug.augment_sudoku_pair(puzzle, solution, box, rng)
        inputs[i] = puzzle.reshape(-1)
        targets[i] = solution.reshape(-1)
    return inputs, targets, side, side


def build_maze_arrays(
    h: int,
    w: int,
    n: int,
    rng: np.random.Generator,
    min_path_len: int = 0,
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Generate ``n`` flattened ``(input, target)`` maze pairs."""
    inputs = np.empty((n, h * w), dtype=np.int64)
    targets = np.empty((n, h * w), dtype=np.int64)
    for i in range(n):
        x, y = mz.generate_pair(h, w, rng, min_path_len=min_path_len)
        if augment:
            x, y = aug.augment_maze_pair(x, y, rng)
        inputs[i] = x.reshape(-1)
        targets[i] = y.reshape(-1)
    return inputs, targets, h, w


def build_smarthome_arrays(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Generate ``n`` smart-home ``(sensors, action)`` pairs."""
    length = sh_task.SEQ_LEN
    inputs = np.empty((n, length), dtype=np.int64)
    targets = np.empty((n, length), dtype=np.int64)
    for i in range(n):
        inputs[i], targets[i] = sh_task.generate_pair(rng)
    return inputs, targets, 1, length


def build_arc_arrays(
    n: int, rng: np.random.Generator, transform: str = "flip_h",
    canvas_h: int = 10, canvas_w: int = 10, n_colors: int = 5, pad_token: int = 10,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Generate ``n`` ARC-style ``(input, output)`` pairs for one transform rule."""
    inputs = np.empty((n, canvas_h * canvas_w), dtype=np.int64)
    targets = np.empty((n, canvas_h * canvas_w), dtype=np.int64)
    for i in range(n):
        inputs[i], targets[i] = arc_task.generate_pair(
            transform, rng, canvas_h=canvas_h, canvas_w=canvas_w,
            n_colors=n_colors, pad_token=pad_token,
        )
    return inputs, targets, canvas_h, canvas_w


def build_babyai_arrays(
    n: int, rng: np.random.Generator, height: int = 8, width: int = 8, wall_prob: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Generate ``n`` BabyAI-style ``(state, next_state)`` pairs."""
    inputs = np.empty((n, height * width), dtype=np.int64)
    targets = np.empty((n, height * width), dtype=np.int64)
    for i in range(n):
        inputs[i], targets[i] = babyai_task.generate_pair(height, width, rng, wall_prob)
    return inputs, targets, height, width


def build_dataset(task: str, n: int, rng: np.random.Generator, **kwargs) -> GridDataset:
    """Build a :class:`GridDataset` for a task (sudoku/maze/arc/babyai/smarthome)."""
    if task == "sudoku":
        inputs, targets, h, w = build_sudoku_arrays(
            kwargs.get("box", 3), n, kwargs.get("num_clues", 40), rng,
            require_unique=kwargs.get("require_unique", True),
            augment=kwargs.get("augment", False),
        )
    elif task == "maze":
        inputs, targets, h, w = build_maze_arrays(
            kwargs.get("height", 15), kwargs.get("width", 15), n, rng,
            min_path_len=kwargs.get("min_path_len", 0),
            augment=kwargs.get("augment", False),
        )
    elif task == "smarthome":
        inputs, targets, h, w = build_smarthome_arrays(n, rng)
    elif task == "arc":
        inputs, targets, h, w = build_arc_arrays(
            n, rng, transform=kwargs.get("transform", "flip_h"),
            canvas_h=kwargs.get("canvas_h", 10), canvas_w=kwargs.get("canvas_w", 10),
            n_colors=kwargs.get("n_colors", 5), pad_token=kwargs.get("pad_token", 10),
        )
    elif task == "babyai":
        inputs, targets, h, w = build_babyai_arrays(
            n, rng, height=kwargs.get("height", 8), width=kwargs.get("width", 8),
            wall_prob=kwargs.get("wall_prob", 0.2),
        )
    else:
        raise ValueError(f"Unknown task: {task!r}")
    return GridDataset(inputs, targets, h, w)
