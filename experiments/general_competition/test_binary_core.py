from __future__ import annotations

from dataclasses import replace
import builtins
from itertools import combinations, product

import pytest

from experiments.general_competition.binary_core import (
    BinaryCore,
    UnitStep,
    build_binary_core,
    canonical_cnf,
    model_satisfies,
    verify_binary_core,
)


def row_satisfied(row, model):
    return any(model[abs(literal) - 1] == (literal > 0) for literal in row)


def models(nvars, clauses):
    return {
        values
        for values in product((False, True), repeat=nvars)
        if all(row_satisfied(row, values) for row in clauses)
    }


def lifted_models(core: BinaryCore):
    return {
        core.lift(values)
        for values in product((False, True), repeat=core.reduced_nvars)
        if all(row_satisfied(row, values) for row in core.residual_clauses)
    }


def test_signed_equivalence_is_substituted_but_long_clause_is_retained():
    clauses = ((1, -2), (-1, 2), (2, 3, 4))
    core = build_binary_core(4, clauses)
    assert not core.contradiction
    assert core.class_count == 3
    assert core.class_literals == (1, 1, 2, 3)
    assert core.mapping == (1, 1, 2, 3)
    assert core.residual_clauses == ((1, 2, 3),)
    assert core.equivalent_variables_removed == 1
    assert core.long_input_clauses == 1
    assert core.component_proofs
    assert verify_binary_core(4, clauses, core)
    assert lifted_models(core) == models(4, canonical_cnf(4, clauses))


def test_negative_equivalence_and_assumption_projection():
    clauses = ((1, 2), (-1, -2), (2, 3, 4))
    core = build_binary_core(4, clauses)
    assert core.class_literals[:2] == (1, -1)
    assert core.mapping[:2] == (1, -1)
    assert core.project_assumptions((1, -2)) == (1,)
    assert core.project_assumptions((1, 2)) is None
    assert verify_binary_core(4, clauses, core)
    assert lifted_models(core) == models(4, canonical_cnf(4, clauses))


def test_binary_contradiction_has_mutual_reachability_evidence():
    clauses = ((1,), (-1,), (2, 3, 4))
    core = build_binary_core(4, clauses)
    assert core.contradiction
    assert core.conflict_kind == "binary"
    assert core.conflict_variable == 1
    assert core.residual_clauses == ((),)
    assert verify_binary_core(4, clauses, core)
    assert not models(4, canonical_cnf(4, clauses))
    with pytest.raises(ValueError, match="no model"):
        core.lift(())


def test_long_clause_conflict_is_replayed_as_unit_propagation():
    clauses = ((1,), (-1, 2), (-2, -3, -4), (3,), (4,))
    core = build_binary_core(4, clauses)
    assert core.contradiction
    assert core.conflict_kind == "unit"
    assert core.conflict_clause == (-2, -3, -4)
    assert tuple(step.literal for step in core.unit_trace) == (1, 3, 4, 2)
    assert verify_binary_core(4, clauses, core)
    assert not models(4, canonical_cnf(4, clauses))


def test_unit_propagation_lifts_fixed_original_values():
    clauses = ((1,), (-1, 2), (-2, 3, 4))
    core = build_binary_core(4, clauses)
    assert not core.contradiction
    assert core.mapping[:2] == (0, 0)
    assert core.fixed[:2] == (True, True)
    assert core.project_assumptions((1, 2)) == ()
    assert core.project_assumptions((-1,)) is None
    assert verify_binary_core(4, clauses, core)
    for reduced in product((False, True), repeat=core.reduced_nvars):
        if all(row_satisfied(row, reduced) for row in core.residual_clauses):
            original = core.lift(reduced)
            assert original[:2] == (True, True)
            assert model_satisfies(4, clauses, original)


def test_canonicalization_validates_before_removing_tautologies():
    assert canonical_cnf(
        3,
        ((1, 1, -2), (-1, 1, 3), (3, -2, 1), (1, -2)),
    ) == ((1, -2), (1, -2, 3))
    with pytest.raises(ValueError, match="nonzero"):
        canonical_cnf(1, ((1, -1, 0),))
    with pytest.raises(ValueError, match="exact integer"):
        canonical_cnf(1, ((True,),))
    with pytest.raises(ValueError, match="iterable"):
        canonical_cnf(1, 7)
    with pytest.raises(ValueError, match="each clause"):
        canonical_cnf(1, (7,))


def test_zero_variable_true_and_false_formulas():
    true_core = build_binary_core(0, ())
    assert not true_core.contradiction
    assert true_core.reduced_nvars == 0
    assert true_core.residual_clauses == ()
    assert true_core.lift(()) == ()
    assert verify_binary_core(0, (), true_core)

    false_core = build_binary_core(0, ((),))
    assert false_core.contradiction
    assert false_core.conflict_kind == "unit"
    assert verify_binary_core(0, ((),), false_core)


def test_replay_rejects_changed_component_and_unit_evidence():
    clauses = ((1, -2), (-1, 2), (2, 3, 4))
    core = build_binary_core(4, clauses)
    proof = core.component_proofs[0]
    assert proof.outward
    source, target, _reason = proof.outward[0]
    changed_proof = replace(
        proof,
        outward=((source, target, (4,)),) + proof.outward[1:],
    )
    changed_core = replace(
        core,
        component_proofs=(changed_proof,) + core.component_proofs[1:],
    )
    assert not verify_binary_core(4, clauses, changed_core)

    unit_clauses = ((1,), (-1, 2), (2, 3, 4))
    unit_core = build_binary_core(4, unit_clauses)
    first = unit_core.unit_trace[0]
    changed_step = UnitStep(-first.literal, first.reason)
    assert not verify_binary_core(
        4,
        unit_clauses,
        replace(unit_core, unit_trace=(changed_step,) + unit_core.unit_trace[1:]),
    )


def test_replay_rejects_unproved_variable_merge():
    clauses = ((1, 2, 3),)
    core = build_binary_core(3, clauses)
    malicious = replace(
        core,
        class_count=2,
        class_literals=(1, 1, 2),
        reduced_nvars=2,
        residual_clauses=((1, 2),),
        mapping=(1, 1, 2),
        equivalent_variables_removed=1,
    )
    assert not verify_binary_core(3, clauses, malicious)


def test_singleton_class_replay_work_is_linear(monkeypatch):
    from experiments.general_competition import binary_core

    # Count literal inspections, not elapsed time: unused declared variables
    # must not cause a complete mapping scan for every singleton class.
    nvars = 128
    core = build_binary_core(nvars, ())
    inspections = 0

    def counted_abs(value):
        nonlocal inspections
        inspections += 1
        return builtins.abs(value)

    monkeypatch.setattr(binary_core, "abs", counted_abs, raising=False)
    assert verify_binary_core(nvars, (), core)
    assert inspections <= 16 * nvars


def test_class_grouping_preserves_signed_and_noncontiguous_members():
    clauses = ((1, 3), (-1, -3), (2, -5), (-2, 5), (3, 4, 6))
    core = build_binary_core(6, clauses)
    assert core.class_literals == (1, 2, -1, 3, 2, 4)
    assert verify_binary_core(6, clauses, core)
    assert lifted_models(core) == models(6, canonical_cnf(6, clauses))
    # Negating a representative or changing the sign of a proved member must
    # still fail even when its absolute class index is unchanged.
    for changed in ((-1, 2, -1, 3, 2, 4), (1, 2, 1, 3, 2, 4)):
        assert not verify_binary_core(
            6, clauses, replace(core, class_literals=changed)
        )


@pytest.mark.parametrize(("field", "value"), [
    ("component_proofs", None), ("component_proofs", [None]),
    ("component_proofs", (None,)),
    ("class_literals", None), ("class_literals", [1]),
    ("class_literals", ("1",)), ("class_literals", (True,)),
    ("class_literals", (0,)), ("class_literals", (2,)),
    ("class_count", True), ("class_count", 1.0),
    ("unit_trace", None), ("unit_trace", [None]), ("unit_trace", (None,)),
    ("mapping", None), ("mapping", (True,)),
    ("fixed", None), ("fixed", (0,)),
    ("residual_clauses", None), ("residual_clauses", ([1],)),
    ("conflict_clause", [1]), ("conflict_kind", []),
    ("original_nvars", True), ("normalized_clause_count", False),
    ("reduced_nvars", True), ("propagated_classes", -1),
])
def test_replay_rejects_malformed_core_receipt(field, value):
    core = build_binary_core(1, ())
    assert not verify_binary_core(1, (), replace(core, **{field: value}))


def test_replay_bounds_class_count_before_allocating(monkeypatch):
    from experiments.general_competition import binary_core

    core = build_binary_core(1, ())

    def bounded_range(*args):
        values = builtins.range(*args)
        assert len(values) <= 2, "receipt count reached an allocation loop"
        return values

    monkeypatch.setattr(binary_core, "range", bounded_range, raising=False)
    assert not verify_binary_core(1, (), replace(core, class_count=10**12))


@pytest.mark.parametrize(("field", "value"), [
    ("nodes", None), ("nodes", ([1],)), ("nodes", (True, -2)),
    ("root", []), ("root", True),
    ("outward", None), ("outward", ((1, 2),)),
    ("outward", (([], 2, (1, 2)),)),
    ("outward", ((1, 2, [1, 2]),)),
    ("outward", ((True, 2, (1, 2)),)),
    ("inward", (None,)),
])
def test_replay_rejects_malformed_component_receipt(field, value):
    clauses = ((1, -2), (-1, 2))
    core = build_binary_core(2, clauses)
    changed = replace(core.component_proofs[0], **{field: value})
    assert not verify_binary_core(
        2, clauses,
        replace(core, component_proofs=(changed,) + core.component_proofs[1:]),
    )


@pytest.mark.parametrize(("literal", "reason"), [
    (True, (1,)), (1.0, (1,)), ([], (1,)),
    (1, [1]), (1, ([1],)), (1, (True,)),
])
def test_replay_rejects_malformed_unit_receipt(literal, reason):
    clauses = ((1,),)
    core = build_binary_core(1, clauses)
    assert not verify_binary_core(
        1, clauses, replace(core, unit_trace=(UnitStep(literal, reason),))
    )


def test_assumptions_are_all_validated_even_after_contradiction():
    core = build_binary_core(1, ((1,), (-1,)))
    assert core.project_assumptions((1,)) is None
    with pytest.raises(ValueError, match="assumption"):
        core.project_assumptions((1, 2))
    with pytest.raises(ValueError, match="assumption"):
        core.project_assumptions((True,))


def test_exhaustive_small_formula_model_equivalence():
    literals = (-3, -2, -1, 1, 2, 3)
    clause_pool = {()}
    for width in (1, 2, 3):
        for values in combinations(literals, width):
            if any(-literal in values for literal in values):
                continue
            clause_pool.add(tuple(sorted(values, key=lambda value: (abs(value), value))))
    ordered = tuple(sorted(clause_pool, key=lambda row: (len(row), row)))

    checked = 0
    for count in range(4):
        for formula in combinations(ordered, count):
            core = build_binary_core(3, formula)
            assert verify_binary_core(3, formula, core)
            original = models(3, formula)
            if core.contradiction:
                assert not original
            else:
                assert lifted_models(core) == original
            checked += 1
    assert checked > 3000


@pytest.mark.parametrize(
    ("clauses", "prefix"),
    [(((1,),), (-1,)), ((), (1, -1))],
    ids=["fixed-value", "projected-pair"],
)
@pytest.mark.parametrize("invalid", [0, 2, True, 1.0, "1"])
@pytest.mark.parametrize("one_shot", [False, True])
def test_assumption_conflict_does_not_hide_invalid_suffix(
    clauses, prefix, invalid, one_shot,
):
    core = build_binary_core(1, clauses)
    values = (*prefix, invalid)
    assumptions = (value for value in values) if one_shot else values
    with pytest.raises(ValueError, match="assumption"):
        core.project_assumptions(assumptions)


@pytest.mark.parametrize(
    ("clauses", "assumptions"),
    [(((1,),), (-1, 1)), ((), (1, -1, 1))],
    ids=["fixed-value", "projected-pair"],
)
def test_valid_contradictory_assumptions_consume_the_complete_iterator(
    clauses, assumptions,
):
    core = build_binary_core(1, clauses)
    seen = []

    def values():
        for literal in assumptions:
            seen.append(literal)
            yield literal

    assert core.project_assumptions(values()) is None
    assert seen == list(assumptions)
