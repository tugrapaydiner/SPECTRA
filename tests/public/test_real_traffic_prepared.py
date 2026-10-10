from __future__ import annotations

import copy

import pytest

from experiments.real_traffic import (
    InvalidContradiction,
    PreparedContradictionChecker,
    solve_with_proof,
    verify_contradiction,
)


def contradiction_fixture():
    edges = ((0, 1), (1, 2))
    masks = (0b11, 0b11, 0b11)
    restrictions = ((0, 0b01), (1, 0b01))
    outcome = solve_with_proof(edges, masks, restrictions)
    assert outcome.status == "UNSAT" and outcome.contradiction is not None
    return edges, masks, restrictions, outcome.contradiction


def test_prepared_checker_matches_one_off_verifier() -> None:
    edges, masks, restrictions, certificate = contradiction_fixture()
    expected = verify_contradiction(edges, masks, restrictions, certificate)
    prepared = PreparedContradictionChecker(edges, masks)
    observed = prepared.verify(restrictions, certificate)
    assert observed["valid"] is True
    assert observed["variable"] == expected["variable"]
    assert observed["path_edges"] == expected["path_edges"]
    assert observed["base_implications"] == len(prepared.base_implications)
    assert observed["query_implications"] == len(restrictions)


def test_repeated_proof_checks_do_not_mutate_the_prepared_relation() -> None:
    edges, masks, restrictions, certificate = contradiction_fixture()
    prepared = PreparedContradictionChecker(edges, masks)
    identity = prepared.base_implications
    first = prepared.verify(restrictions, certificate)
    second = prepared.verify(restrictions, certificate)
    assert first == second
    assert prepared.base_implications is identity


def test_prepared_checker_rejects_a_query_edge_from_another_request() -> None:
    edges, masks, restrictions, certificate = contradiction_fixture()
    prepared = PreparedContradictionChecker(edges, masks)
    other = ((0, 0b10), (1, 0b01))
    with pytest.raises(InvalidContradiction):
        prepared.verify(other, certificate)


def test_prepared_checker_rejects_corrupted_or_noncanonical_inputs() -> None:
    edges, masks, restrictions, certificate = contradiction_fixture()
    prepared = PreparedContradictionChecker(edges, masks)
    altered = copy.deepcopy(certificate)
    altered["negative_to_positive"][1] *= -1
    with pytest.raises(InvalidContradiction):
        prepared.verify(restrictions, altered)
    with pytest.raises(InvalidContradiction, match="noncanonical"):
        prepared.verify(tuple(reversed(restrictions)), certificate)
    with pytest.raises(InvalidContradiction, match="duplicate"):
        prepared.verify((restrictions[0], restrictions[0]), certificate)


def test_prepared_checker_refuses_noncanonical_original_edges() -> None:
    with pytest.raises(InvalidContradiction, match="canonical"):
        PreparedContradictionChecker(((1, 0),), (0b11, 0b11))
    with pytest.raises(InvalidContradiction, match="canonical"):
        PreparedContradictionChecker(((0, 1), (0, 1)), (0b11, 0b11))
