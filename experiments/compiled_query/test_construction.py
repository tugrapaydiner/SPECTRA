"""Adversarial semantic contracts for separately named stronger CUDD controls."""
from __future__ import annotations

import importlib
import itertools
import os
import random

import pytest

from .bdd import CompiledCNF, freeze_cnf
from .binary_list import BinaryListProblem, CompiledListSession
from .preprocess import binary_equivalences


@pytest.fixture(params=['cudd', 'autoref'])
def backend(request):
    try:
        importlib.import_module('dd.' + request.param)
    except ImportError:
        if request.param == 'cudd' and os.environ.get('SPECTRA_REQUIRE_CUDD') == '1':
            pytest.fail('Required native CUDD is missing')
        pytest.skip('Explicit optional backend not installed')
    return request.param


def models(n, clauses, assumptions=()):
    return [row for row in itertools.product((False, True), repeat=n)
            if all(any(row[abs(x)-1] == (x > 0) for x in c) for c in clauses)
            and all(row[abs(x)-1] == (x > 0) for x in assumptions)]


@pytest.mark.parametrize('construction', ['frontier', 'seeded'])
@pytest.mark.parametrize('order', ['natural', 'bfs', 'occurrence'])
@pytest.mark.parametrize('reordering', [False, True])
def test_all_three_variable_functions_and_requests(backend, construction, order, reordering):
    requests = [tuple((i+1)*sign for i, sign in enumerate(row) if sign)
                for row in itertools.product((-1, 0, 1), repeat=3)]
    for function in range(256):
        clauses = tuple(tuple(-(i+1) if a >> i & 1 else i+1 for i in range(3))
                        for a in range(8) if not (function >> a & 1))
        truth = models(3, clauses)
        with CompiledCNF(3, clauses, backend=backend, order=order,
                         reordering=reordering, construction=construction) as engine:
            for request in requests:
                expected = [x for x in truth if all(x[abs(v)-1] == (v > 0) for v in request)]
                actual = engine.model(request)
                assert (actual is not None) == bool(expected)
                if expected:
                    assert actual in expected


def test_all_binary_formulas_preprocessor_entails_every_seed():
    bank = ((), (1,), (-1,), (2,), (-2,), (1, 2), (1, -2), (-1, 2), (-1, -2))
    for mask in range(1 << len(bank)):
        clauses = freeze_cnf(2, (c for i, c in enumerate(bank) if mask >> i & 1))
        truth = models(2, clauses)
        contradiction, pairs = binary_equivalences(2, clauses)
        assert contradiction == (not truth)
        for a, b in pairs:
            assert all(row[a-1] == (row[abs(b)-1] if b > 0 else not row[abs(b)-1]) for row in truth)
        if not contradiction:
            # The binary preprocessor must not infer one-way implication as equality.
            for a in (1, 2):
                for b in (1, -1, 2, -2):
                    if a < abs(b):
                        equivalent = all(row[a-1] == (row[abs(b)-1] if b > 0 else not row[abs(b)-1]) for row in truth)
                        # Equal constants need not be connected by the implication graph.
                        if (a, b) in pairs:
                            assert equivalent


def test_nonbinary_constraints_not_treated_as_binary_and_cycles_are_signed():
    assert binary_equivalences(3, freeze_cnf(3, ((1, 2, 3), (-1, -2, -3)))) == (False, ())
    assert binary_equivalences(3, freeze_cnf(3, ((-1, 2), (1, -2), (2, 3), (-2, -3)))) == (False, ((1, 2), (1, -3)))
    assert binary_equivalences(2, freeze_cnf(2, ((-1, 2),))) == (False, ())


def test_seeded_constants_isolated_variables_and_every_request(backend):
    bank = ((), (1,), (-1,), (2,), (-2,), (1, 2), (1, -2), (-1, 2), (-1, -2))
    requests = [(), (1,), (-1,), (2,), (-2,), (1, 2), (1, -2), (-1, 2), (-1, -2), (1, -1), (4,)]
    for mask in range(1 << len(bank)):
        clauses = tuple(c for i, c in enumerate(bank) if mask >> i & 1)
        with CompiledCNF(4, clauses, backend=backend, construction='seeded') as engine:
            for request in requests:
                expected = models(4, clauses, request)
                actual = engine.model(request)
                assert (actual is not None) == bool(expected)
                if expected:
                    assert actual in expected
    for clauses in ((), ((),)):
        with CompiledCNF(0, clauses, backend=backend, construction='seeded') as engine:
            assert engine.model() == (None if clauses else ())


def test_long_signed_cycle_is_iterative_and_does_not_read_a_solver():
    n = 5000
    clauses = [(-i, i+1) for i in range(1, n)] + [(-n, 1)]
    contradiction, seeds = binary_equivalences(n, freeze_cnf(n, clauses))
    assert not contradiction and len(seeds) == n - 1
    assert seeds[0] == (1, 2) and seeds[-1] == (1, n)


@pytest.mark.parametrize('construction', ['frontier', 'seeded'])
def test_random_original_labels_and_independent_checker(backend, construction):
    rng = random.Random(91941)
    for _ in range(100):
        n = rng.randint(1, 7)
        p = BinaryListProblem.create(n, 4,
            ((u, v) for u in range(n) for v in range(u+1, n) if rng.random() < .3),
            (rng.choice((1, 2, 3, 4, 5, 6, 8, 9, 10, 12)) for _ in range(n)))
        truth = [bytes(row) for row in itertools.product(*(tuple(c for c in range(4) if m >> c & 1) for m in p.masks))
                 if p.check(bytes(row), ())]
        with CompiledListSession(p, backend=backend, construction=construction) as session:
            for _ in range(15):
                query = tuple((rng.randrange(n), rng.randrange(16)) for _ in range(rng.randrange(5)))
                expected = [row for row in truth if p.check(row, query)]
                actual = session.solve(query)
                assert (actual.status == 'SAT_VERIFIED') == bool(expected)
                if expected:
                    assert actual.labels in expected
    p = BinaryListProblem.create(2, 2, ((0, 1),), (3, 3))
    with CompiledListSession(p, backend=backend, construction=construction,
                             checker=lambda *args: False) as session:
        with pytest.raises(AssertionError, match='observer'):
            session.solve()


def test_construction_is_explicit_and_invalid_input_is_never_hidden():
    with pytest.raises(ValueError, match='construction'):
        CompiledCNF(1, (), construction='auto')
    with pytest.raises(ValueError, match='literal'):
        CompiledCNF(1, ((), (1, -1, 99)), construction='seeded')


def test_equivalence_order_groups_without_dropping_or_relabeling():
    from .preprocess import equivalence_order
    order = (3, 1, 5, 2, 4)
    assert equivalence_order(order, ((1, -4), (2, 3))) == (3, 2, 1, 4, 5)
    assert equivalence_order(order, ()) == order
    assert equivalence_order((), ()) == ()


@pytest.mark.parametrize('construction', ['frontier', 'seeded'])
def test_amortized_guard_still_checks_final_build_and_queries(backend, construction):
    from .bdd import CompilationLimit
    clauses = ((1, 2), (2, 3), (3, 4))
    with pytest.raises(CompilationLimit):
        CompiledCNF(4, clauses, backend=backend, construction=construction, max_nodes=1)
    with CompiledCNF(4, clauses, backend=backend, construction=construction) as engine:
        assert engine._guard_interval == 64
        assert engine._guard_scans >= 2  # Before construction and after final root.
        engine._max_nodes = 0  # Deliberate runtime resource-injection test.
        for _ in range(64):
            try:
                engine.model()
            except CompilationLimit:
                break
        else:
            pytest.fail('soft runtime guard was silently disabled')
    with CompiledCNF(4, clauses, backend=backend) as engine:
        assert engine._guard_interval == 1
