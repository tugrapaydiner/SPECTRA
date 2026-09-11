from copy import deepcopy

import pytest

from scripts.m16_cpu_experiment import effect
from scripts.m17_cross_task import closed_summary


def timing_rows():
    return [dict(seed=seed, core_seed=seed, example_index=example, example_id=str(example),
                 arm=arm, round=repeat, latency_ms=2. if arm == "reference" else 1.,
                 valid=True, answer=[1], work={"transitions": 1, "value_calls": 0})
            for seed in (1, 2) for example in (0, 1) for repeat in range(3)
            for arm in ("reference", "native")]


def summarize(rows, version):
    rows = deepcopy(rows)
    if version == 16:
        return effect(rows)
    for row in rows:
        row["arm"] = {"reference": "reference_k4", "native": "native_k4"}[row["arm"]]
    return closed_summary(rows)


@pytest.mark.parametrize("version", [16, 17])
@pytest.mark.parametrize("latency", [float("nan"), float("inf"), -1., 0., True, "1"])
def test_invalid_latencies_cannot_produce_a_result(version, latency):
    rows = timing_rows()
    rows[0]["latency_ms"] = latency
    with pytest.raises(ValueError, match="latency"):
        summarize(rows, version)


@pytest.mark.parametrize("version", [16, 17])
@pytest.mark.parametrize("mutation", ["duplicate", "missing", "bool_round", "soft_valid", "changed_valid", "empty", "missing_arm"])
def test_malformed_timing_inventories_fail_closed(version, mutation):
    rows = timing_rows()
    if mutation == "duplicate": rows.append(deepcopy(rows[0]))
    elif mutation == "missing": rows.pop()
    elif mutation == "bool_round": rows[0]["round"] = False
    elif mutation == "soft_valid": rows[0]["valid"] = .99
    elif mutation == "changed_valid": rows[0]["valid"] = False
    elif mutation == "empty": rows = []
    else: rows = [r for r in rows if r["arm"] == "reference"]
    with pytest.raises(ValueError):
        summarize(rows, version)


def test_correct_matched_measurements_keep_the_original_effect():
    result = summarize(timing_rows(), 16)
    assert result["paired_mean_ratio_after_round_medians"] == .5
    assert result["ci95"] == [.5, .5]
    assert result["gate_pass"]
    result = summarize(timing_rows(), 17)
    assert result["arms"]["native_k4"]["valid_answers"] == 4
    assert result["arms"]["native_k4"]["timing_rows"] == 12
