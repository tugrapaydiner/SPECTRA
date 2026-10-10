from __future__ import annotations

import pytest

from experiments.real_traffic.fair_benchmark import (
    assignment_to_labels,
    complete_assignment,
    minfill_order,
    model_to_labels,
)


def test_conditioned_values_replace_arbitrary_partial_values() -> None:
    names = ("v0", "v1", "v2")
    observed = complete_assignment(
        {"v0": False, "v1": True},
        {"v0": True, "v2": False},
        names,
    )
    assert observed == {"v0": True, "v1": True, "v2": False}


def test_missing_bdd_variable_is_refused() -> None:
    with pytest.raises(AssertionError, match="materialize every variable"):
        complete_assignment({"v0": True}, {}, ("v0", "v1"))


def test_cudd_conditioning_keeps_the_original_fixed_value() -> None:
    cudd = pytest.importorskip("dd.cudd")
    bdd = cudd.BDD()
    bdd.declare("v0", "v1")
    relation = bdd.var("v0") & bdd.var("v1")
    fixed = {"v0": True}
    restricted = bdd.let(fixed, relation)
    partial = bdd.pick(restricted, care_vars=("v1",))
    assert partial is not None
    completed = complete_assignment(partial, fixed, ("v0", "v1"))
    assert completed == {"v0": True, "v1": True}


def test_model_and_assignment_decoders_agree() -> None:
    masks = (0b0011, 0b0110, 0b1000)
    from_sat = model_to_labels((1, -2, 3), masks)
    from_bdd = assignment_to_labels(
        {"v0": True, "v1": False, "v2": True},
        masks,
        ("v0", "v1", "v2"),
    )
    assert from_sat == from_bdd == (1, 1, 3)


def test_minfill_is_deterministic_and_complete() -> None:
    edges = ((0, 1), (1, 2), (2, 3), (0, 3))
    first = minfill_order(4, edges)
    second = minfill_order(4, edges)
    assert first == second
    assert sorted(first) == [0, 1, 2, 3]
