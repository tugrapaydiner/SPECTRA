"""Blank-Check #1: few-shot ARC-AGI primitive generator.

Proves tasks are genuine few-shot abstraction problems -- one consistent rule
across all demonstrations + the held-out test -- and serialize to a token canvas
whose only supervised region is the test output.
"""

import numpy as np

from data import arc_primitives as arc


def test_param_free_rules_are_exact_across_demos_and_test():
    rng = np.random.default_rng(0)
    for rule in ["reflect_h", "reflect_v", "rotate_180", "gravity"]:
        task = arc.generate_task(rule, rng, n_demos=3, g=5, n_colors=6)
        fn = arc.rule_registry(6)[rule][0]
        for gin, gout in task.demos:
            assert np.array_equal(fn(gin, {}), gout)
        assert np.array_equal(fn(task.test_input, {}), task.test_output)


def test_gravity_pulls_nonzero_down():
    out = arc.gravity_down(np.array([[1], [0], [2], [0]]), {})
    assert np.array_equal(out, np.array([[0], [0], [1], [2]]))


def test_recolor_is_one_globally_consistent_permutation():
    rng = np.random.default_rng(1)
    task = arc.generate_task("recolor", rng, n_demos=3, g=6, n_colors=6)
    mapping: dict[int, int] = {}
    for gin, gout in task.demos + [(task.test_input, task.test_output)]:
        for a, b in zip(gin.flatten().tolist(), gout.flatten().tolist()):
            if a in mapping:
                assert mapping[a] == b  # same rule explains every pair (abstraction)
            else:
                mapping[a] = b


def test_serialization_is_few_shot_with_masked_answer_cell():
    rng = np.random.default_rng(0)
    g = 5
    task = arc.generate_task("reflect_h", rng, n_demos=3, g=g, n_colors=6)
    x, y, mask, h, w = arc.serialize_task(task)
    assert h == (2 * 3 + 1) * g and w == g           # 3 demo pairs + test input, stacked
    assert (x[~mask] == y[~mask]).all()              # context identical; only answer differs
    assert np.array_equal(x[mask].reshape(g, g), task.test_input)   # model sees the test input
    assert np.array_equal(y[mask].reshape(g, g), task.test_output)  # must produce the test output


def test_build_fewshot_arrays():
    rng = np.random.default_rng(0)
    X, Y, mask, h, w = arc.build_arc_fewshot_arrays(8, rng, n_demos=3, g=5, n_colors=6)
    assert X.shape == (8, h * w) and Y.shape == (8, h * w)
    assert mask.sum() == 25                          # the 5x5 test-output cell
