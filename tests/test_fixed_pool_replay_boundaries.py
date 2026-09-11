"""Boundary counterexamples for interpreting replay diagnostics."""
import torch

from data import maze
from eval.checkable_tasks import MAZE11
from eval.fixed_pool_replay import decode_metrics


def test_maze_improvement_identity_does_not_extend_to_adjacent_endpoints():
    # An empty valid route and a nonempty invalid route can both score 0.75.
    # The solve-transition simplification applies to the retained nonadjacent
    # endpoint distribution, not every syntactically valid 11x11 maze.
    grid = torch.full((11, 11), maze.WALL, dtype=torch.int64)
    grid[1, 1], grid[1, 2], grid[1, 3] = maze.START, maze.GOAL, maze.OPEN
    x = grid.reshape(1, -1).repeat(2, 1)
    logits = torch.zeros(2, 121, 5)
    logits[:, :, maze.OPEN] = 1.
    logits[0, 14, maze.PATH] = 2.
    _, valid, quality = decode_metrics(x, logits, MAZE11)
    assert valid.tolist() == [False, True]
    assert quality.tolist() == [.75, .75]
    assert not bool(quality[1] > quality[0] + 1e-6)
    assert bool((~valid[0]) & valid[1])
