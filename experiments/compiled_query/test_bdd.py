"""Small exhaustive independent oracles, not an oracle used by the benchmark.

Run this directory directly so the historical neural-test conftest is not loaded.
Set SPECTRA_REQUIRE_CUDD=1 in native validation: missing CUDD is then a FAILURE,
not a successful run with a skipped comparator.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import itertools
import os
import random

import pytest

from experiments.compiled_query.bdd import (
    BackendUnavailable, CompilationLimit, CompiledCNF, freeze_cnf, variable_order,
)
from experiments.compiled_query.binary_list import BinaryListProblem, CompiledListSession


@pytest.fixture(params=['cudd', 'autoref'])
def backend(request):
    try:
        importlib.import_module('dd.' + request.param)
    except ImportError:
        if request.param == 'cudd' and os.environ.get('SPECTRA_REQUIRE_CUDD') == '1':
            pytest.fail('Required actual native dd.cudd backend is unavailable')
        pytest.skip('Explicit optional backend dd.' + request.param + ' not installed')
    return request.param


def satisfies(clauses, model):
    return all(any(model[abs(literal) - 1] == (literal > 0) for literal in row)
               for row in clauses)


def all_models(nvars, clauses, assumptions=()):
    return [model for model in itertools.product((False, True), repeat=nvars)
            if satisfies(clauses, model)
            and all(model[abs(literal) - 1] == (literal > 0) for literal in assumptions)]


def labels_oracle(problem, query):
    """Direct enumeration from original colours; no CNF encoder or BDD."""
    choices = [tuple(c for c in range(problem.palette) if (mask >> c) & 1)
               for mask in problem.masks]
    return [bytes(row) for row in itertools.product(*choices)
            if all(row[left] != row[right] for left, right in problem.edges)
            and all((allowed >> row[vertex]) & 1 for vertex, allowed in query)]


@pytest.mark.parametrize('mode', ['natural', 'occurrence', 'bfs'])
def test_query_blind_order_contains_every_variable(mode):
    clauses = ((1, -3), (3, 5), (5, -1), (2,))
    order = variable_order(8, clauses, mode)
    assert sorted(order) == list(range(1, 9))
    assert order == variable_order(8, clauses, mode)


def test_long_clause_bfs_is_incidence_based_and_keeps_isolates():
    clause = tuple(range(1, 10001))
    result = variable_order(10002, (clause,), 'bfs')
    assert result == tuple(range(1, 10003))


@pytest.mark.parametrize('nvars,clauses', [(-1, ()), (True, ()), (1.0, ()),
    (2, ((0,),)), (2, ((True,),)), (2, ((3,),)), (0, ((1,),)),
    (2, ((1, -1, 999),)), (2, ((1.0,),))])
def test_cnf_validation_cannot_hide_invalid_literal_in_tautology(nvars, clauses):
    with pytest.raises(ValueError):
        freeze_cnf(nvars, clauses)


def test_empty_formula_clause_and_generic_deduplication():
    assert freeze_cnf(0, ()) == ()
    assert freeze_cnf(0, ((),)) == ((),)
    assert freeze_cnf(3, ((1, -1), (3, 2, 2), (2, 3))) == ((2, 3),)


def test_native_backend_never_silently_falls_back(monkeypatch):
    imports = []
    def unavailable(name):
        imports.append(name)
        raise ImportError('deliberately absent')
    monkeypatch.setattr('experiments.compiled_query.bdd.importlib.import_module', unavailable)
    with pytest.raises(BackendUnavailable, match='no fallback'):
        CompiledCNF(1, ((1,),), backend='cudd')
    assert imports == ['dd.cudd']


@pytest.mark.parametrize('kwargs', [dict(backend='auto'), dict(order='oracle'),
    dict(reordering=1), dict(max_nodes=True), dict(max_nodes=0),
    dict(memory_bytes=True), dict(memory_bytes=1024)])
def test_invalid_configuration_is_rejected_before_import(kwargs):
    with pytest.raises(ValueError):
        CompiledCNF(1, ((1,),), **kwargs)


@pytest.mark.parametrize('order', ['natural', 'occurrence', 'bfs'])
@pytest.mark.parametrize('reordering', [False, True])
def test_every_three_variable_boolean_function_and_partial_request(backend, order, reordering):
    # 256 Boolean functions x 27 partial assignments = 6,912 requests per
    # backend/order/reordering configuration.  No training or timing data.
    nvars = 3
    requests = [tuple((v + 1) * sign for v, sign in enumerate(states) if sign)
                for states in itertools.product((-1, 0, 1), repeat=nvars)]
    for truth_table in range(256):
        clauses = []
        for assignment in range(8):
            if not ((truth_table >> assignment) & 1):
                clauses.append(tuple(-(v + 1) if (assignment >> v) & 1 else v + 1
                                     for v in range(nvars)))
        with CompiledCNF(nvars, clauses, backend=backend, order=order,
                         reordering=reordering) as compiled:
            for request in requests:
                result = compiled.model(request)
                expected = all_models(nvars, clauses, request)
                assert (result is None) == (not expected)
                if result is not None:
                    assert len(result) == nvars
                    assert result in expected


def test_constants_unconstrained_and_conditioned_away_variables(backend):
    with CompiledCNF(0, (), backend=backend) as compiled:
        assert compiled.model() == ()  # an empty dictionary is a model of TRUE
    with CompiledCNF(0, ((),), backend=backend) as compiled:
        assert compiled.model() is None
    with CompiledCNF(5, ((1, 2),), backend=backend) as compiled:
        answer = compiled.model((1, 4, -5))
        assert answer[0] is True and answer[3:] == (True, False)
        assert len(answer) == 5
        assert compiled.model((-1, -2)) is None
        assert compiled.model((1,)) is not None  # previous query did not persist


@pytest.mark.parametrize('bad', [(0,), (True,), (4,), (-4,), (1.0,), (1, -1, 9)])
def test_assumptions_are_validated_even_after_a_contradiction(backend, bad):
    with CompiledCNF(3, (), backend=backend) as compiled:
        with pytest.raises(ValueError):
            compiled.model(bad)
        assert compiled.model((1, -1)) is None
        assert compiled.model((1, 1))[0] is True


def test_input_ownership_and_lifecycle(backend):
    clauses = [[1, 2]]
    compiled = CompiledCNF(2, clauses, backend=backend)
    clauses[0][:] = [-1]
    assert compiled.model((1,)) is not None
    info = compiled.info
    assert info['backend'] == 'dd.' + backend
    assert info['uses_spectra_quotient'] is False
    assert info['query_answer_cache'] is False
    compiled.close()
    compiled.close()
    for action in (lambda: compiled.model(), lambda: compiled.info, compiled.__enter__):
        with pytest.raises(RuntimeError, match='closed'):
            action()


def test_failed_compilation_is_a_limit_not_unsat(backend):
    with pytest.raises(CompilationLimit):
        CompiledCNF(8, ((1, 2),), backend=backend, max_nodes=1)
    # A failed manager must not poison a subsequent independent manager.
    with CompiledCNF(1, ((1,),), backend=backend) as compiled:
        assert compiled.model() == (True,)


def test_concurrent_queries_never_share_assumptions(backend):
    with CompiledCNF(6, ((1, 2), (-2, 3)), backend=backend) as compiled:
        requests = [(1, 4), (-1, 2, 3), (-2, -3), (1, -1)] * 32
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(compiled.model, requests))
        for query, result in zip(requests, results):
            expected = all_models(6, ((1, 2), (-2, 3)), query)
            assert (result is None) == (not expected)
            if result is not None:
                assert result in expected


def test_binary_encoding_all_three_vertex_graphs_and_singleton_binary_lists():
    edge_bank = ((0, 1), (0, 2), (1, 2))
    masks = (1, 2, 3, 4, 5, 6)
    for edge_mask in range(8):
        edges = tuple(edge for bit, edge in enumerate(edge_bank) if (edge_mask >> bit) & 1)
        for domains in itertools.product(masks, repeat=3):
            problem = BinaryListProblem.create(3, 3, edges, domains)
            clauses = problem.cnf()
            choices = problem.colors()
            actual = {bytes(choices[v][int(value)] for v, value in enumerate(model))
                      for model in all_models(3, clauses)}
            expected = set(labels_oracle(problem, ()))
            assert actual == expected


def test_native_binary_lists_all_three_vertex_graphs_and_restrictions(backend):
    # 8 graphs x 27 binary-list banks x 64 original-address requests = 13,824.
    edge_bank = ((0, 1), (0, 2), (1, 2))
    for edge_mask in range(8):
        edges = tuple(edge for bit, edge in enumerate(edge_bank) if (edge_mask >> bit) & 1)
        for domains in itertools.product((3, 5, 6), repeat=3):
            problem = BinaryListProblem.create(3, 3, edges, domains)
            options = [(None, 0, 1 << low, 1 << high) for low, high in problem.colors()]
            with CompiledListSession(problem, backend=backend) as session:
                for limits in itertools.product(*options):
                    query = tuple((v, mask) for v, mask in enumerate(limits) if mask is not None)
                    expected = labels_oracle(problem, query)
                    result = session.solve(query)
                    assert (result.status == 'SAT_VERIFIED') == bool(expected)
                    if expected:
                        assert result.labels in expected
                    else:
                        assert result.status == 'UNSAT'
                        assert result.labels == b'' and result.unsat_certificate is None


@pytest.mark.parametrize('kwargs', [dict(n=True), dict(n=-1), dict(palette=0),
    dict(palette=65), dict(masks=(3,)), dict(masks=(3, 0)),
    dict(masks=(3, True)), dict(masks=(3, 7)), dict(masks=(3, 8)),
    dict(edges=((0, 0),)), dict(edges=((0, 2),)), dict(edges=((0, True),))])
def test_binary_problem_validation(kwargs):
    values = dict(n=2, palette=3, edges=((0, 1),), masks=(3, 6))
    values.update(kwargs)
    with pytest.raises(ValueError):
        BinaryListProblem.create(**values)


@pytest.mark.parametrize('query', [((2, 1),), ((True, 1),), ((0, True),),
    ((0, -1),), ((0, 1 << 64),), ((0, 1, 2),)])
def test_restriction_validation(query):
    problem = BinaryListProblem.create(2, 3, ((0, 1),), (3, 6))
    with pytest.raises(ValueError):
        problem.freeze_query(query)


def test_duplicates_are_intersections_and_bit63_is_not_signed(backend):
    problem = BinaryListProblem.create(2, 64, ((0, 1),), ((1 << 63) | 1, 1 << 63))
    with CompiledListSession(problem, backend=backend) as session:
        assert session.solve(((0, 1), (1, 1 << 63))).labels == bytes((0, 63))
        assert session.solve(((0, 1), (0, 1 << 63))).status == 'UNSAT'
        assert session.solve(((0, 2),)).status == 'UNSAT'
        assert session.solve(((0, (1 << 64) - 1),)).labels == bytes((0, 63))
        assert session.solve().labels == bytes((0, 63))


def test_separately_owned_checker_is_not_bypassed(backend):
    problem = BinaryListProblem.create(2, 3, ((0, 1),), (3, 6))
    with CompiledListSession(problem, backend=backend, checker=lambda labels, query: False) as session:
        with pytest.raises(AssertionError, match='original-input observer'):
            session.solve()
    with CompiledListSession(problem, backend=backend) as session:
        # Deliberately corrupt decoding, but do not alter the original observer.
        session._choices = ((63, 63), (63, 63))
        with pytest.raises(AssertionError, match='original-input observer'):
            session.solve()


def test_list_session_ownership_validation_and_close(backend):
    edges, masks = [[0, 1]], [3, 6]
    problem = BinaryListProblem.create(2, 3, edges, masks)
    with CompiledListSession(problem, backend=backend) as session:
        edges[0][:] = [0, 0]
        masks[:] = [0, 0]
        query = [[0, 1]]
        answer = session.solve(iter(query))
        query[0][1] = 0
        assert answer.status == 'SAT_VERIFIED' and answer.labels[0] == 0
    session.close()
    for action in (lambda: session.solve(), lambda: session.info, session.__enter__):
        with pytest.raises(RuntimeError, match='closed'):
            action()
    bypassed = BinaryListProblem(1, 2, (), (0,))
    with pytest.raises(ValueError):
        CompiledListSession(bypassed, backend=backend)


def test_random_four_to_seven_vertex_domains_against_original_oracle(backend):
    rng = random.Random(0xC0DD2026)
    for _ in range(80):
        n = rng.randint(4, 7)
        edges = tuple((u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < .35)
        masks = tuple(rng.choice((1, 2, 3, 4, 5, 6, 8, 9, 10, 12)) for _ in range(n))
        problem = BinaryListProblem.create(n, 4, edges, masks)
        with CompiledListSession(problem, backend=backend) as session:
            for _query in range(24):
                query = tuple((rng.randrange(n), rng.randrange(16)) for _ in range(rng.randrange(5)))
                expected = labels_oracle(problem, query)
                actual = session.solve(query)
                assert (actual.status == 'SAT_VERIFIED') == bool(expected)
                if expected:
                    assert actual.labels in expected
