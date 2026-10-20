from __future__ import annotations

from dataclasses import replace
from itertools import product

import pytest

from experiments.general_competition.binary_core import model_satisfies
from experiments.general_competition.residual_solver import (
    GeneralResult,
    ResidualProof,
    solve_general,
    verify_general_result,
    verify_residual_proof,
)


def triple_clauses():
    return tuple(
        tuple(variable if positive else -variable for variable, positive in enumerate(signs, 1))
        for signs in product((False, True), repeat=3)
    )


def has_model(nvars, clauses):
    return any(
        model_satisfies(nvars, clauses, assignment)
        for assignment in product((False, True), repeat=nvars)
    )


def test_sat_model_is_lifted_through_the_binary_core():
    clauses = ((1, -2), (-1, 2), (2, 3, 4), (-3, 4))
    result = solve_general(4, clauses)
    assert result.status == "SAT_VERIFIED"
    assert result.binary_core.equivalent_variables_removed == 1
    assert result.residual_model
    assert model_satisfies(4, clauses, result.model)
    assert verify_general_result(4, clauses, result)


def test_residual_unsat_has_a_replayable_complete_search_tree():
    clauses = triple_clauses()
    result = solve_general(3, clauses)
    assert result.status == "UNSAT_VERIFIED"
    assert not result.binary_core.contradiction
    assert result.residual_proof is not None
    assert result.proof_nodes == len(result.residual_proof.nodes)
    assert result.propagations > 0
    assert verify_residual_proof(
        result.binary_core.reduced_nvars,
        result.binary_core.residual_clauses,
        result.residual_proof,
    )
    assert verify_general_result(3, clauses, result)


def test_binary_core_unsat_does_not_forge_a_residual_proof():
    clauses = ((1,), (-1,), (1, 2, 3))
    result = solve_general(3, clauses)
    assert result.status == "UNSAT_VERIFIED"
    assert result.binary_core.contradiction
    assert result.residual_proof is None
    assert result.proof_nodes == 0
    assert verify_general_result(3, clauses, result)


def test_all_three_variable_long_clause_subsets_match_truth_tables():
    pool = triple_clauses()
    for mask in range(1 << len(pool)):
        clauses = tuple(
            row for index, row in enumerate(pool) if mask & (1 << index)
        )
        result = solve_general(3, clauses)
        expected_sat = has_model(3, clauses)
        assert result.status == (
            "SAT_VERIFIED" if expected_sat else "UNSAT_VERIFIED"
        )
        assert verify_general_result(3, clauses, result)


def test_changed_model_and_proof_are_rejected():
    sat_clauses = ((1, 2, 3),)
    sat = solve_general(3, sat_clauses)
    changed_model = (not sat.model[0],) + sat.model[1:]
    if model_satisfies(3, sat_clauses, changed_model):
        changed_model = tuple(not value for value in sat.model)
    assert not verify_general_result(
        3, sat_clauses, replace(sat, model=changed_model)
    )

    unsat_clauses = triple_clauses()
    unsat = solve_general(3, unsat_clauses)
    assert unsat.residual_proof is not None
    nodes = list(unsat.residual_proof.nodes)
    leaf = next(index for index, node in enumerate(nodes) if node.kind == "conflict")
    nodes[leaf] = replace(nodes[leaf], conflict_clause=(1,))
    changed = replace(unsat.residual_proof, nodes=tuple(nodes))
    assert not verify_residual_proof(
        unsat.binary_core.reduced_nvars,
        unsat.binary_core.residual_clauses,
        changed,
    )
    assert not verify_general_result(
        3, unsat_clauses, replace(unsat, residual_proof=changed)
    )


def test_resource_limits_return_unknown_without_a_fake_certificate():
    result = solve_general(3, triple_clauses(), max_decisions=0)
    assert result.status == "UNKNOWN"
    assert result.reason == "decision limit"
    assert result.residual_proof is None
    assert result.model == ()
    assert verify_general_result(3, triple_clauses(), result)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("max_decisions", -1, "non-negative"),
        ("max_propagations", -1, "non-negative"),
        ("max_proof_nodes", 0, "positive"),
        ("max_depth", -1, "non-negative"),
    ],
)
def test_invalid_limits_fail_closed(keyword, value, message):
    with pytest.raises(ValueError, match=message):
        solve_general(1, ((1,),), **{keyword: value})


def test_verifiers_reject_non_contract_objects():
    result = solve_general(1, ((1,),))
    assert not verify_general_result(1, ((1,),), object())
    assert not verify_residual_proof(1, ((1,),), object())
    assert isinstance(result, GeneralResult)
    assert result.residual_proof is None or isinstance(result.residual_proof, ResidualProof)
