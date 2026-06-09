"""Correctness tests for the ARC / BabyAI / smart-home task generators."""

import numpy as np

from data import arc, babyai
from data import smarthome as sh
from data.datasets import build_dataset


# --------------------------------------------------------------------------- #
# Smart-home (section 24.5)
# --------------------------------------------------------------------------- #
def test_smarthome_policy():
    away_open_locked = {"door": 1, "lock": 1, "motion": 0, "user": 1, "time": 0, "battery": 0}
    assert sh.safe_action(away_open_locked) == sh.ALERT
    away_open_unlocked = {**away_open_locked, "lock": 0}
    assert sh.safe_action(away_open_unlocked) == sh.LOCK_ALERT
    home_low_batt = {"door": 0, "lock": 0, "motion": 0, "user": 0, "time": 1, "battery": 1}
    assert sh.safe_action(home_low_batt) == sh.CONSERVE
    all_safe = {f: 0 for f in ["door", "lock", "motion", "user", "time", "battery"]}
    assert sh.safe_action(all_safe) == sh.IGNORE


def test_smarthome_safety_false_negative():
    assert sh.safety_false_negative(sh.ALERT, sh.IGNORE) is True
    assert sh.safety_false_negative(sh.LOCK_ALERT, sh.ALERT) is False  # still critical
    assert sh.safety_false_negative(sh.IGNORE, sh.IGNORE) is False


def test_smarthome_pair_encoding():
    x, y = sh.generate_pair(np.random.default_rng(0))
    assert x.shape == (sh.SEQ_LEN,) and y.shape == (sh.SEQ_LEN,)
    assert (y == y[0]).all()  # action broadcast across the sequence
    assert int(x.max()) < sh.NUM_TOKENS


# --------------------------------------------------------------------------- #
# ARC-style (section 24.3)
# --------------------------------------------------------------------------- #
def test_arc_transforms():
    g = np.array([[1, 2], [3, 4]])
    assert np.array_equal(arc.TRANSFORMS["flip_h"](g), np.array([[2, 1], [4, 3]]))
    assert np.array_equal(arc.TRANSFORMS["rotate180"](g), np.array([[4, 3], [2, 1]]))
    assert np.array_equal(arc.TRANSFORMS["transpose"](g), np.array([[1, 3], [2, 4]]))


def test_arc_recolor_preserves_structure():
    rng = np.random.default_rng(0)
    g = np.array([[1, 1, 2], [2, 3, 3]])
    r = arc.recolor(g, rng, n_colors=5)
    # Cells that shared a color still share a color (partition preserved).
    for a in range(g.size):
        for b in range(g.size):
            assert (g.flat[a] == g.flat[b]) == (r.flat[a] == r.flat[b])


def test_arc_pair_shapes_and_palette():
    rng = np.random.default_rng(1)
    x, y = arc.generate_pair("flip_h", rng, canvas_h=10, canvas_w=10, n_colors=5, pad_token=10)
    assert x.shape == (100,) and y.shape == (100,)
    assert set(np.unique(y)).issubset(set(range(11)))


# --------------------------------------------------------------------------- #
# BabyAI-style (section 24.4)
# --------------------------------------------------------------------------- #
def test_babyai_step_moves_agent():
    grid = np.zeros((3, 3), dtype=np.int64)
    grid[1, 1] = babyai.AGENT_RIGHT
    nxt = babyai.step(grid, (1, 1), babyai.AGENT_RIGHT)
    assert nxt[1, 2] == babyai.AGENT and nxt[1, 1] == babyai.EMPTY


def test_babyai_step_blocked_by_wall_and_boundary():
    grid = np.zeros((3, 3), dtype=np.int64)
    grid[1, 1] = babyai.AGENT_RIGHT
    grid[1, 2] = babyai.WALL
    assert babyai.step(grid, (1, 1), babyai.AGENT_RIGHT)[1, 1] == babyai.AGENT  # blocked, stays

    edge = np.zeros((3, 3), dtype=np.int64)
    edge[1, 2] = babyai.AGENT_RIGHT
    assert babyai.step(edge, (1, 2), babyai.AGENT_RIGHT)[1, 2] == babyai.AGENT  # boundary, stays


# --------------------------------------------------------------------------- #
# build_dataset integration
# --------------------------------------------------------------------------- #
def test_build_dataset_for_new_tasks():
    rng = np.random.default_rng(0)
    for task, kw in [("smarthome", {}), ("arc", {"transform": "flip_h"}), ("babyai", {})]:
        ds = build_dataset(task, 16, rng, **kw)
        assert len(ds) == 16
        x, y = ds[0]
        assert x.shape == y.shape
        assert x.dtype.is_floating_point is False  # LongTensor token ids
