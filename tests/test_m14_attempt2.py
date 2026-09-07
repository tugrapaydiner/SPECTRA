"""Contracts for M14 Attempt 2 before any Attempt 2 result."""
from __future__ import annotations

import torch

from scripts.m14_attempt2_early_exit import (
    ATTEMPT1_DATA_SEED,
    CANDIDATE_BUDGETS,
    CONFIRM_DATA_SEED,
    DATA_SEED,
    EARLY_WEIGHTS,
    MODEL_SEEDS,
    RESERVE_CONFIRM_DATA_SEED,
)
from scripts.m14_primary_experiment import practical_gate


def test_attempt2_fresh_hierarchy_and_training_seeds():
    assert DATA_SEED not in {ATTEMPT1_DATA_SEED, CONFIRM_DATA_SEED, RESERVE_CONFIRM_DATA_SEED}
    assert len(MODEL_SEEDS) == 5 and len(set(MODEL_SEEDS)) == 5
    assert set(MODEL_SEEDS).isdisjoint({1401, 2402, 3403, 4404, 5405})


def test_intervention_is_one_frozen_early_heavy_weight_vector():
    assert EARLY_WEIGHTS == [0.70, 0.10, 0.10, 0.10]
    assert abs(sum(EARLY_WEIGHTS) - 1.0) < 1e-12
    assert EARLY_WEIGHTS[0] == max(EARLY_WEIGHTS)
    assert CANDIDATE_BUDGETS == [1, 2, 3, 4]


def test_original_m14_quality_superiority_threshold_is_unchanged():
    fail = practical_gate({
        "quality_difference": 0.029,
        "quality_ci95": [0.01, 0.05],
        "latency_ratio_candidate_over_baseline": 1.0,
        "latency_ratio_ci95": [0.98, 1.02],
    })
    assert fail["pass"] is False
    passing = practical_gate({
        "quality_difference": 0.031,
        "quality_ci95": [0.001, 0.05],
        "latency_ratio_candidate_over_baseline": 1.14,
        "latency_ratio_ci95": [1.10, 1.19],
    })
    assert passing["pass"] is True and passing["path"] == "quality_superiority"


def test_original_tradeoff_threshold_is_unchanged():
    fail = practical_gate({
        "quality_difference": -0.01,
        "quality_ci95": [-0.02, 0.01],
        "latency_ratio_candidate_over_baseline": 0.66,
        "latency_ratio_ci95": [0.60, 0.70],
    })
    assert fail["pass"] is False
    passing = practical_gate({
        "quality_difference": -0.01,
        "quality_ci95": [-0.02, 0.01],
        "latency_ratio_candidate_over_baseline": 0.64,
        "latency_ratio_ci95": [0.60, 0.74],
    })
    assert passing["pass"] is True and passing["path"] == "quality_cost_tradeoff"


def test_confirmation_seed_remains_original_first_confirmation():
    assert CONFIRM_DATA_SEED == 2026091402
    assert RESERVE_CONFIRM_DATA_SEED == 2026091403
