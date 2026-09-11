"""Maze generation, shortest-path ground truth, and semantic validation.

Token scheme:
  input:  0=wall, 1=open, 2=start, 3=goal
  target: same, plus 4=path for intermediate route cells.

Milestone 03 adds a strict candidate-success checker.  A successful candidate
must preserve the declared maze, connect the unique start to the unique goal by
one simple wall-free 4-neighbour path, and satisfy the configured optimality
requirement.  Merely copying an unsolved input is not success.
"""
from __future__ import annotations

from collections import deque
import numpy as np

WALL, OPEN, START, GOAL, PATH = 0, 1, 2, 3, 4
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _checked_grid(grid: np.ndarray, h: int, w: int, allowed: set[int]) -> np.ndarray | None:
    arr = np.asarray(grid)
    if arr.shape != (h, w) or arr.dtype.kind not in "iu":
        return None
    values = set(int(v) for v in np.unique(arr))
    if not values.issubset(allowed):
        return None
    return arr.astype(np.int64, copy=False)


def generate_maze(
    h: int, w: int, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
        raise ValueError(f"Maze dims must be odd and >=3, got {h}x{w}")
    if h == 3 and w == 3:
        raise ValueError("Maze needs distinct start and goal cells; 3x3 has only one interior room")
    grid = np.zeros((h, w), dtype=np.int64)
    start_cell = (1, 1)
    grid[start_cell] = OPEN
    stack = [start_cell]
    while stack:
        r, c = stack[-1]
        candidates = []
        for dr, dc in _NEIGHBORS:
            nr, nc = r + 2 * dr, c + 2 * dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == WALL:
                candidates.append((nr, nc, dr, dc))
        if not candidates:
            stack.pop()
            continue
        nr, nc, dr, dc = candidates[int(rng.integers(len(candidates)))]
        grid[r + dr, c + dc] = OPEN
        grid[nr, nc] = OPEN
        stack.append((nr, nc))
    return grid, (1, 1), (h - 2, w - 2)


def shortest_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    arr = np.asarray(grid)
    if arr.ndim != 2:
        return None
    h, w = arr.shape
    for r, c in (start, goal):
        if not (0 <= r < h and 0 <= c < w):
            return None
    if arr[start] == WALL or arr[goal] == WALL:
        return None
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = [cell]
            while prev[cell] is not None:
                cell = prev[cell]  # type: ignore[assignment]
                path.append(cell)
            path.reverse()
            return path
        r, c = cell
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w and arr[nxt] != WALL and nxt not in prev:
                prev[nxt] = cell
                queue.append(nxt)
    return None


def make_input(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> np.ndarray:
    out = np.asarray(grid, dtype=np.int64).copy()
    out[start] = START
    out[goal] = GOAL
    return out


def make_target(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    path: list[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=np.int64).copy()
    for cell in path[1:-1]:
        out[cell] = PATH
    out[start] = START
    out[goal] = GOAL
    return out


def is_valid_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    path: list[tuple[int, int]] | None,
) -> bool:
    if not path or path[0] != start or path[-1] != goal:
        return False
    arr = np.asarray(grid)
    if arr.ndim != 2:
        return False
    h, w = arr.shape
    seen: set[tuple[int, int]] = set()
    for i, (r, c) in enumerate(path):
        if not (0 <= r < h and 0 <= c < w) or arr[r, c] == WALL or (r, c) in seen:
            return False
        seen.add((r, c))
        if i:
            pr, pc = path[i - 1]
            if abs(pr - r) + abs(pc - c) != 1:
                return False
    return True


def extract_path(target: np.ndarray) -> list[tuple[int, int]]:
    cells = np.argwhere(np.isin(target, [START, GOAL, PATH]))
    return [tuple(int(v) for v in cell) for cell in cells]


def candidate_success(
    input_grid: np.ndarray,
    candidate: np.ndarray,
    height: int | None = None,
    width: int | None = None,
    *,
    require_optimal: bool = True,
) -> bool:
    """Strict semantic maze success independent of a reference target.

    Requirements:
      * input/candidate shapes and token domains are valid;
      * exactly one start and one goal are preserved at their declared cells;
      * walls are preserved exactly and PATH is only placed on input OPEN cells;
      * START + PATH + GOAL form one connected simple path with no branches;
      * when ``require_optimal`` is true, route length equals BFS shortest length.
    """
    inp0 = np.asarray(input_grid)
    if inp0.ndim != 2:
        return False
    h = int(height if height is not None else inp0.shape[0])
    w = int(width if width is not None else inp0.shape[1])
    inp = _checked_grid(inp0, h, w, {WALL, OPEN, START, GOAL})
    cand = _checked_grid(candidate, h, w, {WALL, OPEN, START, GOAL, PATH})
    if inp is None or cand is None:
        return False

    starts = np.argwhere(inp == START)
    goals = np.argwhere(inp == GOAL)
    if len(starts) != 1 or len(goals) != 1:
        return False
    start = tuple(int(v) for v in starts[0])
    goal = tuple(int(v) for v in goals[0])
    if start == goal:
        return False
    if not np.array_equal(cand == START, inp == START) or not np.array_equal(cand == GOAL, inp == GOAL):
        return False
    if not np.array_equal(cand == WALL, inp == WALL):
        return False
    if np.any((cand == PATH) & (inp != OPEN)):
        return False
    # Open cells may remain OPEN or be promoted to PATH, but nothing else.
    if np.any((inp == OPEN) & ~np.isin(cand, [OPEN, PATH])):
        return False

    route_mask = (cand == PATH) | (cand == START) | (cand == GOAL)
    route_cells = {tuple(int(v) for v in rc) for rc in np.argwhere(route_mask)}
    if start not in route_cells or goal not in route_cells or len(route_cells) < 2:
        return False

    def degree(cell: tuple[int, int]) -> int:
        r, c = cell
        return sum((r + dr, c + dc) in route_cells for dr, dc in _NEIGHBORS)

    for cell in route_cells:
        deg = degree(cell)
        if cell in {start, goal}:
            if deg != 1:
                return False
        elif deg != 2:
            return False

    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _NEIGHBORS:
            nxt = (r + dr, c + dc)
            if nxt in route_cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    if seen != route_cells or goal not in seen:
        return False

    base = inp.copy()
    base[base == START] = OPEN
    base[base == GOAL] = OPEN
    optimum = shortest_path(base, start, goal)
    if optimum is None:
        return False
    if require_optimal and len(route_cells) != len(optimum):
        return False
    return True


def generate_pair(
    h: int,
    w: int,
    rng: np.random.Generator,
    min_path_len: int = 0,
    max_path_len: int | None = None,
    max_attempts: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a pair satisfying the declared path-length bounds or raise."""
    if min_path_len < 0:
        raise ValueError("min_path_len must be non-negative")
    if max_path_len is not None and max_path_len < min_path_len:
        raise ValueError("max_path_len must be >= min_path_len")
    for _ in range(max_attempts):
        grid, start, goal = generate_maze(h, w, rng)
        path = shortest_path(grid, start, goal)
        if path is None:
            continue
        if len(path) < min_path_len:
            continue
        if max_path_len is not None and len(path) > max_path_len:
            continue
        x = make_input(grid, start, goal)
        y = make_target(grid, start, goal, path)
        if not candidate_success(x, y, h, w, require_optimal=True):
            raise RuntimeError("maze generator produced a target that fails its own semantic contract")
        return x, y
    raise RuntimeError(
        f"failed to generate maze satisfying path length [{min_path_len}, {max_path_len}] "
        f"within {max_attempts} attempts"
    )
