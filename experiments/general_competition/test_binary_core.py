from __future__ import annotations

from dataclasses import replace
from itertools import combinations, product

import pytest

from experiments.general_competition.binary_core import (
    BinaryCore,
    ComponentProof,
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
    clauses = ((1,), (-1, 2), (-2, 3, 4), (-3,), (-4,))
    core = build_binary_core(4, clauses)
    assert core.contradiction
    assert core.conflict_kind == "unit"
    assert core.conflict_clause == (-2, 3, 4)
    assert tuple(step.literal for step in core.unit_trace) == (-3, -4, 1, 2)
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
