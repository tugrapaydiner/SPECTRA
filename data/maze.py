"""Maze generation, shortest-path ground truth, and path validation.

A maze is an ``H x W`` grid of cells (``H``, ``W`` odd) generated as a *perfect
maze* (spanning tree) by randomized backtracking, so every passage cell is
reachable from every other -- guaranteeing a start->goal path exists.

Token scheme (BLUEPRINT maze config / section 24.2):
  * input grid:  0=wall, 1=open, 2=start, 3=goal
  * target grid: same, plus 4=path for the shortest-path cells strictly between
    start and goal. The model's job is to overlay the correct path.
"""

from __future__ import annotations

from collections import deque

import numpy as np

WALL, OPEN, START, GOAL, PATH = 0, 1, 2, 3, 4
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def generate_maze(
    h: int,
    w: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Generate a perfect maze and pick start/goal at opposite passage corners.

    Args:
        h: Maze height (odd).
        w: Maze width (odd).
        rng: NumPy random generator.

    Returns:
        ``(grid, start, goal)`` where ``grid`` is ``H x W`` with 0=wall, 1=open.
    """
    if h % 2 == 0 or w % 2 == 0:
        raise ValueError(f"Maze dims must be odd, got {h}x{w}")

    grid = np.zeros((h, w), dtype=np.int64)  # all walls initially
    start_cell = (1, 1)
    grid[start_cell] = OPEN

    # Iterative recursive backtracker: carve passages between odd cells.
    stack = [start_cell]
    while stack:
        r, c = stack[-1]
        # Candidate neighbours two cells away (the wall between is carved out).
        candidates = []
        for dr, dc in _NEIGHBORS:
            nr, nc = r + 2 * dr, c + 2 * dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == WALL:
                candidates.append((nr, nc, dr, dc))
        if not candidates:
            stack.pop()
            continue
        nr, nc, dr, dc = candidates[rng.integers(len(candidates))]
        grid[r + dr, c + dc] = OPEN  # knock down the wall between
        grid[nr, nc] = OPEN
        stack.append((nr, nc))

    start = (1, 1)
    goal = (h - 2, w - 2)
    return grid, start, goal


def shortest_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """BFS shortest path over open cells (4-connectivity). ``None`` if none."""
    h, w = grid.shape
    if grid[start] == WALL or grid[goal] == WALL:
        return None
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            # Reconstruct path from goal back to start.
            path = [cell]
            while prev[cell] is not None:
                cell = prev[cell]  # type: ignore[assignment]
                path.append(cell)
            path.reverse()
            return path
        r, c = cell
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != WALL:
                if (nr, nc) not in prev:
                    prev[(nr, nc)] = cell
                    queue.append((nr, nc))
    return None


def make_input(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> np.ndarray:
    """Input grid with start/goal tokens overlaid on the open/wall maze."""
    out = grid.copy()
    out[start] = START
    out[goal] = GOAL
    return out


def make_target(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    path: list[tuple[int, int]],
) -> np.ndarray:
    """Target grid: maze + start/goal + PATH tokens on intermediate path cells."""
    out = grid.copy()
    for cell in path[1:-1]:  # exclude start/goal endpoints
        out[cell] = PATH
    out[start] = START
    out[goal] = GOAL
    return out


def is_valid_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    path: list[tuple[int, int]],
) -> bool:
    """Check a path is a legal start->goal walk (section 9.3).

    Validates: starts at start, ends at goal, every step moves to a 4-adjacent
    cell, never enters a wall, and never revisits a cell (no illegal jumps/loops).
    """
    if not path or path[0] != start or path[-1] != goal:
        return False
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    for i, (r, c) in enumerate(path):
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] == WALL:
            return False
        if (r, c) in seen:  # no revisits
            return False
        seen.add((r, c))
        if i > 0:
            pr, pc = path[i - 1]
            if abs(pr - r) + abs(pc - c) != 1:  # must be a single 4-adjacent step
                return False
    return True


def extract_path(target: np.ndarray) -> list[tuple[int, int]]:
    """Return cells marked PATH/START/GOAL in a target grid (unordered)."""
    cells = np.argwhere(np.isin(target, [START, GOAL, PATH]))
    return [tuple(int(v) for v in cell) for cell in cells]


def generate_pair(
    h: int,
    w: int,
    rng: np.random.Generator,
    min_path_len: int = 0,
    max_attempts: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an ``(input_grid, target_grid)`` pair with a solvable shortest path.

    Retries generation until the shortest path is at least ``min_path_len`` long
    (or ``max_attempts`` is exhausted, in which case the last maze is returned).
    """
    grid = path = start = goal = None
    for _ in range(max_attempts):
        grid, start, goal = generate_maze(h, w, rng)
        path = shortest_path(grid, start, goal)
        if path is not None and len(path) >= min_path_len:
            break
    assert grid is not None and path is not None
    return make_input(grid, start, goal), make_target(grid, start, goal, path)
