from __future__ import annotations

import pytest

from experiments.general_competition.adaptive_backend import (
    POLICY,
    choose_frontend,
    compare_adaptive_backend,
)
from experiments.general_competition.matched_backend import prepare_matched_formula


def test_frontend_policy_uses_only_unit_and_empty_clause_geometry():
    assert POLICY == "general-frontend-v1"
    assert choose_frontend(prepare_matched_formula(1, ((),))) == (
        "reduce",
        "empty_clause",
    )
    assert choose_frontend(prepare_matched_formula(3, ((1,), (1, 2, 3)))) == (
        "reduce",
        "unit_clause",
    )
    assert choose_frontend(prepare_matched_formula(3, ((1, 2), (1, 2, 3)))) == (
        "plain",
        "no_unit_clause",
    )
    with pytest.raises(ValueError, match="prepared"):
        choose_frontend(object())


def test_direct_route_preserves_the_matched_contract():
    pytest.importorskip("pysat")
    prepared = prepare_matched_formula(
        4,
        ((1, 2, 3), (-1, 2, 4), (1, -3, 4)),
    )
    result = compare_adaptive_backend(prepared, order="hybrid-first")
    record = result.record()
    assert result.accepted_for_timing
    assert record["adaptive_policy"] == POLICY
    assert record["adaptive_decision"] == "plain"
    assert record["adaptive_reason"] == "no_unit_clause"
    assert record["frontend"] == "plain-direct-v1"
    assert record["core_elapsed_ns"] == 0
    assert record["residual_variables"] == record["original_variables"]
    assert record["residual_clauses"] == record["original_clauses"]
    assert record["plain"]["status"] == "SAT_CHECKED"
    assert record["hybrid"]["status"] == "SAT_CHECKED"


def test_unit_route_keeps_the_replayable_reducer():
    pytest.importorskip("pysat")
    prepared = prepare_matched_formula(
        4,
        ((1,), (-1, 2), (-2, 3, 4)),
    )
    result = compare_adaptive_backend(prepared)
    record = result.record()
    assert result.accepted_for_timing
    assert record["adaptive_decision"] == "reduce"
    assert record["adaptive_reason"] == "unit_clause"
    assert record["frontend"] == "python-replayable-binary-core"
