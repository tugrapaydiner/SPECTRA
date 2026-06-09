"""Synthetic ARC-AGI few-shot primitive generator (Blank-Check #1).

Sudoku is computationally reducible; ARC-AGI is not -- it tests *abstraction*: infer
a transformation rule from a handful of (input, output) demonstration pairs and
apply it to a held-out test input. The single grid->grid transforms in ``data/arc``
do NOT capture this; this module does.

A task is the real ARC format: ``n_demos`` demonstration pairs + one test input,
all sharing ONE rule whose parameters are fixed per task (so the only way to solve
the test is to abstract the rule, not memorize a mapping). Tasks are serialized
into a single token canvas the recursive core / Latent MCTS consumes: the
demonstrations are the in-context "prompt", and the model must fill the final
(test-output) cell. Generalization is forced because the test grid is novel.

Primitives: horizontal/vertical reflection, 180 rotation, color permutation,
gravity, and 4-connected flood fill -- compositional ARC building blocks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

Grid = np.ndarray  # int [g, g], colors 0..n_colors-1 (0 = background)


# --------------------------------------------------------------------------- #
# Primitives: (grid, params) -> grid   (all size-preserving)
# --------------------------------------------------------------------------- #
def reflect_h(grid: Grid, params: dict) -> Grid:
    return grid[:, ::-1].copy()


def reflect_v(grid: Grid, params: dict) -> Grid:
    return grid[::-1, :].copy()


def rotate_180(grid: Grid, params: dict) -> Grid:
    return grid[::-1, ::-1].copy()


def recolor(grid: Grid, params: dict) -> Grid:
    """Apply a fixed color permutation (the rule = the permutation)."""
    perm = params["perm"]
    return perm[grid]


def gravity_down(grid: Grid, params: dict) -> Grid:
    """Non-zero cells fall to the bottom of each column (stable order)."""
    out = np.zeros_like(grid)
    for c in range(grid.shape[1]):
        col = grid[:, c]
        nz = col[col != 0]
        out[grid.shape[0] - len(nz):, c] = nz
    return out


def flood_fill(grid: Grid, params: dict) -> Grid:
    """Flood-fill the background region 4-connected to (0,0) with ``fill``."""
    fill = params["fill"]
    out = grid.copy()
    if out[0, 0] != 0:
        return out
    h, w = out.shape
    seen = np.zeros_like(out, dtype=bool)
    q: deque = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        out[r, c] = fill
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == 0:
                seen[nr, nc] = True
                q.append((nr, nc))
    return out


def _perm_sampler(n_colors: int) -> Callable[[np.random.Generator], dict]:
    def sample(rng):
        # Permute non-background colors only (keep 0 as background).
        perm = np.arange(n_colors)
        nb = perm[1:].copy()
        rng.shuffle(nb)
        perm[1:] = nb
        return {"perm": perm}
    return sample


def _fill_sampler(n_colors: int) -> Callable[[np.random.Generator], dict]:
    return lambda rng: {"fill": int(rng.integers(1, n_colors))}


# rule name -> (transform, param_sampler)
def rule_registry(n_colors: int) -> dict[str, tuple[Callable, Callable]]:
    none = lambda rng: {}
    return {
        "reflect_h": (reflect_h, none),
        "reflect_v": (reflect_v, none),
        "rotate_180": (rotate_180, none),
        "recolor": (recolor, _perm_sampler(n_colors)),
        "gravity": (gravity_down, none),
        "flood_fill": (flood_fill, _fill_sampler(n_colors)),
    }


ALL_RULES = list(rule_registry(10).keys())


# --------------------------------------------------------------------------- #
# Task = n_demos demonstration pairs + a test pair (one shared rule)
# --------------------------------------------------------------------------- #
@dataclass
class ArcTask:
    rule: str
    demos: list[tuple[Grid, Grid]]
    test_input: Grid
    test_output: Grid


def _random_grid(g: int, n_colors: int, rng: np.random.Generator, density: float = 0.5) -> Grid:
    grid = rng.integers(1, n_colors, size=(g, g))
    mask = rng.random((g, g)) > density
    grid[mask] = 0  # sparse foreground over background
    return grid.astype(np.int64)


def generate_task(
    rule: str, rng: np.random.Generator, n_demos: int = 3, g: int = 5, n_colors: int = 6
) -> ArcTask:
    """Generate one few-shot ARC task: shared rule, ``n_demos`` demos + 1 test."""
    transform, sampler = rule_registry(n_colors)[rule]
    params = sampler(rng)  # rule parameters fixed for the whole task

    demos = []
    for _ in range(n_demos):
        gin = _random_grid(g, n_colors, rng)
        demos.append((gin, transform(gin, params)))
    test_in = _random_grid(g, n_colors, rng)
    return ArcTask(rule, demos, test_in, transform(test_in, params))


# --------------------------------------------------------------------------- #
# Serialization: few-shot context -> one token canvas + answer mask
# --------------------------------------------------------------------------- #
def serialize_task(task: ArcTask) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Stack [d1_in, d1_out, ..., test_in] as the input canvas and the same with the
    final cell replaced by test_out as the target. Returns flat ``(input, target,
    answer_mask, height, width)``; ``answer_mask`` marks the test-output cell.
    """
    g = task.test_input.shape[0]
    cells_in = []
    cells_out = []
    for gin, gout in task.demos:
        cells_in += [gin, gout]
        cells_out += [gin, gout]
    cells_in.append(task.test_input)   # last cell: the test input (model sees it)
    cells_out.append(task.test_output)  # target: the test output (model must produce it)

    canvas_in = np.vstack(cells_in)
    canvas_out = np.vstack(cells_out)
    height, width = canvas_in.shape

    answer_mask = np.zeros((height, width), dtype=bool)
    answer_mask[height - g:, :] = True  # only the final (test) cell is supervised/scored

    return (
        canvas_in.reshape(-1), canvas_out.reshape(-1), answer_mask.reshape(-1), height, width,
    )


def build_arc_fewshot_arrays(
    n: int,
    rng: np.random.Generator,
    rules: list[str] | None = None,
    n_demos: int = 3,
    g: int = 5,
    n_colors: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Generate ``n`` serialized few-shot ARC tasks.

    Returns ``(inputs [n, L], targets [n, L], answer_mask [L], height, width)``.
    """
    rules = rules or ALL_RULES
    probe = serialize_task(generate_task(rules[0], rng, n_demos, g, n_colors))
    length, answer_mask, height, width = probe[0].shape[0], probe[2], probe[3], probe[4]

    inputs = np.empty((n, length), dtype=np.int64)
    targets = np.empty((n, length), dtype=np.int64)
    for i in range(n):
        rule = rules[int(rng.integers(len(rules)))]
        x, y, _, _, _ = serialize_task(generate_task(rule, rng, n_demos, g, n_colors))
        inputs[i], targets[i] = x, y
    return inputs, targets, answer_mask, height, width
