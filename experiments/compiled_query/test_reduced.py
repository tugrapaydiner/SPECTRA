"""Independent exhaustive original-model checks for generic CNF elimination."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import itertools
import os
import random

import pytest

from .bdd import BackendUnavailable
from .binary_list import BinaryListProblem, CompiledListSession
from .reduced import ReducedCNF, reduce_cnf


def models(n, clauses, request=()):
    return {row for row in itertools.product((False, True), repeat=n)
        if all(any(row[abs(lit)-1] == (lit > 0) for lit in clause) for clause in clauses)
        and all(row[abs(lit)-1] == (lit > 0) for lit in request)}


@pytest.fixture(params=['cudd', 'autoref'])
def backend(request):
    try:
        importlib.import_module('dd.' + request.param)
    except ImportError:
        if request.param == 'cudd' and os.environ.get('SPECTRA_REQUIRE_CUDD') == '1':
            pytest.fail('required native CUDD unavailable')
        pytest.skip('explicit optional backend unavailable')
    return request.param


def assert_reduction(n, clauses):
    reduced = reduce_cnf(n, clauses)
    lifted = {reduced.lift(row) for row in models(reduced.nvars, reduced.clauses)}
    assert lifted == models(n, clauses)
    assert len(reduced.mapping) == len(reduced.fixed) == n
    assert set(abs(x) for x in reduced.mapping if x) == set(range(1, reduced.nvars+1))
    return reduced


def test_every_binary_two_variable_formula_retains_exactly_all_models():
    bank = ((), (1,), (-1,), (2,), (-2,), (1, 2), (1, -2), (-1, 2), (-1, -2))
    for mask in range(512):
        assert_reduction(4, tuple(c for i, c in enumerate(bank) if mask >> i & 1))


def test_random_mixed_long_clauses_never_lose_or_invent_models():
    rng = random.Random(619026)
    for _ in range(1200):
        n = rng.randrange(1, 7)
        bank = [i*s for i in range(1, n+1) for s in (-1, 1)]
        clauses = tuple(tuple(rng.choices(bank, k=rng.randrange(1, 8)))
                        for _ in range(rng.randrange(1, 24)))
        assert_reduction(n, clauses)
    assert_reduction(0, ())
    assert_reduction(0, ((),))


def test_signed_classes_unit_chains_and_isolates():
    reduced = assert_reduction(6, ((1, 2), (-1, -2), (-2, 3), (2, -3),
                                  (1,), (-1, 4), (-4, 5)))
    assert reduced.nvars == 1
    assert reduced.equivalent_variables_removed == 2
    assert reduced.fixed_classes == 3
    assert reduced.assumptions((1, 2)) is None
    assert reduced.assumptions((6,)) == (1,)
    with pytest.raises(ValueError, match='assumption'):
        reduced.assumptions((2, 0))
    with pytest.raises(ValueError, match='partial'):
        reduced.lift(())


@pytest.mark.parametrize('order', ['natural', 'bfs', 'occurrence'])
@pytest.mark.parametrize('reordering', [False, True])
def test_all_boolean_functions_all_partial_requests(backend, order, reordering):
    requests = [tuple((i+1)*s for i, s in enumerate(row) if s)
                for row in itertools.product((-1, 0, 1), repeat=3)]
    for function in range(256):
        clauses = tuple(tuple(-(v+1) if row >> v & 1 else v+1 for v in range(3))
                        for row in range(8) if not (function >> row & 1))
        truth = models(3, clauses)
        with ReducedCNF(3, clauses, backend=backend, order=order, reordering=reordering) as engine:
            for query in requests:
                expected = {r for r in truth if all(r[abs(lit)-1] == (lit > 0) for lit in query)}
                actual = engine.model(query)
                assert (actual is not None) == bool(expected)
                if expected:
                    assert actual in expected


def test_repeated_queries_signed_equivalences_and_lifecycle(backend):
    clauses = ((-1, 2), (1, -2), (2, 3), (-2, -3), (4,))
    requests = [(), (1,), (-1,), (2, -3), (1, -2), (-4,), (6,), (6, -6)]
    engine = ReducedCNF(6, clauses, backend=backend)
    info = engine.info
    assert info['original_variables'] == 6 and info['compiled_variables'] == 3
    assert info['equivalent_variables_removed'] == 2 and info['unit_fixed_classes'] == 1
    assert not info['uses_spectra_quotient'] and not info['query_answer_cache']
    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(engine.model, requests * 8))
    for query, row in zip(requests * 8, rows):
        truth = models(6, clauses, query)
        assert (row is not None) == bool(truth)
        if truth:
            assert row in truth
    engine.close(); engine.close()
    for action in (lambda: engine.model(), lambda: engine.info, lambda: engine.__enter__()):
        with pytest.raises(RuntimeError, match='closed'):
            action()


def test_backend_required_even_for_preprocessing_contradiction(monkeypatch):
    imports = []
    def missing(name):
        imports.append(name)
        raise ImportError('deliberately unavailable')
    monkeypatch.setattr('experiments.compiled_query.bdd.importlib.import_module', missing)
    with pytest.raises(BackendUnavailable, match='no fallback'):
        ReducedCNF(1, ((1,), (-1,)))
    assert imports == ['dd.cudd']


def test_all_invalid_literals_checked_before_constant_shortcut(backend):
    for bad in [0, True, 2, 1.0]:
        with pytest.raises(ValueError, match='literal'):
            ReducedCNF(1, ((), (bad,)), backend=backend)
    with ReducedCNF(1, ((),), backend=backend) as engine:
        for bad in [0, True, 2, 1.0]:
            with pytest.raises(ValueError, match='assumption'):
                engine.model((1, -1, bad))
    for clauses in [(), ((),)]:
        with ReducedCNF(0, clauses, backend=backend) as engine:
            assert engine.model() == (None if clauses else ())


def test_original_colour_checker_cannot_be_bypassed(backend):
    p = BinaryListProblem.create(4, 3, ((0, 1), (1, 2)), (3, 3, 3, 4))
    with CompiledListSession(p, construction='reduced', backend=backend) as engine:
        for query in [(), ((0, 1),), ((2, 2),), ((0, 1), (2, 2))]:
            answer = engine.solve(query)
            if answer.status == 'SAT_VERIFIED':
                assert p.check(answer.labels, query) and len(answer.labels) == p.n
    with CompiledListSession(p, construction='reduced', backend=backend,
                             checker=lambda *_: False) as engine:
        with pytest.raises(AssertionError, match='observer'):
            engine.solve()


def test_event_driven_long_unit_chain_has_no_recursive_stack():
    n = 10000
    result = reduce_cnf(n, [(1,)] + [(-v, v+1) for v in range(1, n)])
    assert result.nvars == 0 and result.fixed_classes == n
    assert result.lift(()) == (True,) * n


def test_random_current_requests_after_elimination_and_propagation(backend):
    rng = random.Random(202610131)
    for _ in range(180):
        n = rng.randint(1, 6)
        literals = [i*s for i in range(1, n+1) for s in (-1, 1)]
        clauses = tuple(tuple(rng.choices(literals, k=rng.randint(1, 4)))
                        for _ in range(rng.randint(0, 18)))
        truth = models(n, clauses)
        with ReducedCNF(n, clauses, backend=backend) as engine:
            for _ in range(16):
                query = tuple(rng.choices(literals, k=rng.randint(0, 5)))
                expected = {row for row in truth
                            if all(row[abs(x)-1] == (x > 0) for x in query)}
                actual = engine.model(query)
                assert (actual is not None) == bool(expected)
                if expected:
                    assert actual in expected


@pytest.mark.parametrize('options', [dict(backend='auto'), dict(order='oracle'),
    dict(reordering=1), dict(max_nodes=True), dict(memory_bytes=1), dict(construction='frontier')])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        ReducedCNF(1, (), **options)


def test_lift_refuses_non_boolean_and_partial_models():
    reduced = reduce_cnf(1, ())
    for value in [(), (0,), (1,), [True], (True, False)]:
        with pytest.raises(ValueError, match='model'):
            reduced.lift(value)
