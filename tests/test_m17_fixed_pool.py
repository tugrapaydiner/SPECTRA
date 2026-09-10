from copy import deepcopy
import pytest

from eval.fixed_pool import GATE, TARGET_NAMES, score_pool, summarize_pools
from eval.verified_search import ValueTarget


def pool(seed=1, example="a", covered=True):
    # Perfect improvement predicts an incorrect improving answer, not the correct
    # fixed point. This is a wrong-objective counterexample, not predictor error.
    return {"core_seed": seed, "example_id": example, "validity": [False, covered],
            "depths": [1, 4], "actions": [0, 0], "improvement_labels": [1, 0],
            "scores": {ValueTarget.IMPROVEMENT.value: [1.0, 0.0],
                       ValueTarget.TERMINAL.value: [0.0, 1.0],
                       ValueTarget.QUALITY.value: [0.25, 1.0]}}


def test_perfect_improvement_counterexample_and_coverage_identity():
    p = score_pool(pool())
    assert p["successes"]["oracle_improvement"] == 0
    assert p["successes"][ValueTarget.IMPROVEMENT.value] == 0
    assert p["successes"][ValueTarget.QUALITY.value] == 1
    assert p["successes"]["uniform_random_expectation"] == .5
    s = summarize_pools([pool(a, b) for a in (1, 2) for b in ("a", "b")], replicates=50)
    assert s["quality_minus_improvement"] == 1.0
    assert s["ci95"] == [1., 1.]
    assert s["gate_pass"]
    for r in s["selectors"].values():
        assert r["success"] == r["coverage"] * r["conditional_selection_reliability"]


def test_no_coverage_does_not_turn_into_selection_success():
    s = summarize_pools([pool(covered=False)], replicates=10)
    assert not s["gate_pass"]
    assert s["coverage"] == 0
    assert s["ci95"] == [0., 0.]
    assert all(r["conditional_selection_reliability"] is None for r in s["selectors"].values())


def test_stored_selections_are_recomputed_not_trusted():
    p = pool()
    p.update(successes={name: 0 for name in TARGET_NAMES}, selected_indices={"fake": 10})
    assert summarize_pools([p], replicates=5)["quality_minus_improvement"] == 1.0


def test_paired_bootstrap_requires_common_unique_model_examples():
    with pytest.raises(ValueError, match="duplicate"):
        summarize_pools([pool(), pool()])
    with pytest.raises(ValueError, match="common"):
        summarize_pools([pool(1, "a"), pool(2, "b")])


@pytest.mark.parametrize("mutation", ["bad_flag", "nan", "outside", "wrong_shape", "duplicate", "missing_depth", "bad_event"])
def test_malformed_pool_rejected(mutation):
    p = pool()
    if mutation == "bad_flag": p["validity"] = [0, 1]
    elif mutation == "nan": p["scores"][TARGET_NAMES[0]][0] = float("nan")
    elif mutation == "outside": p["scores"][TARGET_NAMES[0]][0] = 1.1
    elif mutation == "wrong_shape": p["scores"][TARGET_NAMES[0]] = [0.]
    elif mutation == "duplicate": p["depths"] = [1, 1]
    elif mutation == "missing_depth": p["depths"] = [1, 3]
    else: p["improvement_labels"] = [.2, 0]
    with pytest.raises(ValueError): score_pool(p)


def test_ties_are_first_candidate_and_protocol_thresholds_are_fixed():
    p = pool()
    for name in TARGET_NAMES:
        p["scores"][name] = [.5, .5]
    out = score_pool(p)
    assert all(out["selected_indices"][name] == 0 for name in TARGET_NAMES)
    assert GATE == {"min_coverage": .20, "min_quality_minus_improvement": .05,
                    "min_ci_lower_strict": 0., "bootstrap_replicates": 2000, "bootstrap_seed": 17091}
