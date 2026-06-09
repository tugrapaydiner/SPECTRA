"""BabyAI-style local planning task (BLUEPRINT section 24.4).

Grounded sequential reasoning on a small gridworld. We implement the "Predict"
variant (predict the next state after an action); the action is encoded in the
agent's token (a directional agent), so input and target are same-shape grids and
the existing cell/board metrics apply. The agent moves one cell in its direction
unless blocked by a wall or the boundary. Plan/Decompose are natural extensions.
"""

from __future__ import annotations

import numpy as np

EMPTY, WALL, GOAL, AGENT = 0, 1, 2, 3
AGENT_UP, AGENT_DOWN, AGENT_LEFT, AGENT_RIGHT = 4, 5, 6, 7
NUM_TOKENS = 8

_DELTAS = {AGENT_UP: (-1, 0), AGENT_DOWN: (1, 0), AGENT_LEFT: (0, -1), AGENT_RIGHT: (0, 1)}


def generate_world(
    h: int, w: int, rng: np.random.Generator, wall_prob: float = 0.2
) -> tuple[np.ndarray, tuple[int, int], int]:
    """Generate a gridworld with walls, a goal, and a directional agent.

    Returns ``(grid, agent_pos, agent_dir_token)`` where ``grid`` already holds the
    directional agent token at ``agent_pos``.
    """
    grid = (rng.random((h, w)) < wall_prob).astype(np.int64)  # 1 = wall
    free = np.argwhere(grid == EMPTY)
    # Place goal and agent on distinct free cells.
    gi, ai = rng.choice(len(free), size=2, replace=False)
    goal = tuple(int(v) for v in free[gi])
    agent = tuple(int(v) for v in free[ai])
    grid[goal] = GOAL
    direction = int(rng.choice([AGENT_UP, AGENT_DOWN, AGENT_LEFT, AGENT_RIGHT]))
    grid[agent] = direction
    return grid, agent, direction


def step(grid: np.ndarray, agent: tuple[int, int], direction: int) -> np.ndarray:
    """Return the next-state grid after the agent moves (or stays if blocked)."""
    h, w = grid.shape
    dr, dc = _DELTAS[direction]
    nr, nc = agent[0] + dr, agent[1] + dc

    out = grid.copy()
    out[agent] = EMPTY  # agent leaves its cell
    blocked = not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc] == WALL
    dest = agent if blocked else (nr, nc)
    # Preserve a goal underneath only if the agent did not land on it.
    out[dest] = AGENT
    return out


def generate_pair(
    h: int, w: int, rng: np.random.Generator, wall_prob: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Return a flattened ``(input_grid, next_state_grid)`` pair."""
    grid, agent, direction = generate_world(h, w, rng, wall_prob)
    nxt = step(grid, agent, direction)
    return grid.reshape(-1), nxt.reshape(-1)
