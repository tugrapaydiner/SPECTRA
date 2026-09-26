"""A favorable speed number cannot bypass completeness or equivalence checks."""
import copy

import pytest

from scripts.bench_integrated_runtime import ARMS, analyze


def complete_rows():
    cases = {"fixture:b1:i0": {"fixture": {"name": "fixture", "depth": 4, "trained": True}}}
    rows = []
    for arm in ARMS:
        for round_id in range(3):
            rows.append(dict(case="fixture:b1:i0", arm=arm, round=round_id,
                api_ns=100 + round_id, complete_ns=120 + round_id, outputs={"logits": "a", "answer": "b", "halt": "c"},
                valid=[False], native_calls=52 if arm.endswith("trace") else 49,
                halt_calls=4 if arm.endswith("trace") else 1, returned_tensor_bytes=100))
    return rows, cases


def test_complete_matrix_has_exact_denominator_and_no_quality_claim():
    rows, cases = complete_rows()
    report = analyze(rows, cases, 3)
    assert report["calls"] == 18 and report["artifacts"] == 1
    assert report["ratio_of_summed_case_medians"]["blocked_final_over_fastest_existing_per_case"] == 1
    assert report["quality_improvement_claimed"] is False


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_round", "bool_round", "bits", "validity", "negative_time", "bool_time", "missing_check", "work", "heads", "storage"])
def test_corrupt_or_incomplete_matrix_is_rejected(kind):
    rows, cases = complete_rows()
    if kind == "missing": rows.pop()
    elif kind == "duplicate": rows.append(copy.deepcopy(rows[0]))
    elif kind == "wrong_round": rows[0]["round"] = 77
    elif kind == "bool_round": rows[0]["round"] = False
    elif kind == "bits": rows[0]["outputs"]["logits"] = "changed"
    elif kind == "validity": rows[0]["valid"] = [True]
    elif kind == "negative_time": rows[0]["api_ns"] = -1
    elif kind == "bool_time": rows[0]["api_ns"] = True
    elif kind == "missing_check": rows[0]["complete_ns"] = 1
    elif kind == "work": rows[0]["native_calls"] = 3
    elif kind == "heads": rows[0]["halt_calls"] = 0
    elif kind == "storage": rows[0]["returned_tensor_bytes"] = 99
    with pytest.raises(ValueError): analyze(rows, cases, 3)
