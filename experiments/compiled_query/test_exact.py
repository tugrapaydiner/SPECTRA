"""Exact service, proof forgery, ownership, and resource contracts.

SPECTRA_EXACT_LIBRARY optionally selects a prebuilt sanitizer library. No tests
are skipped when a native backend is missing: build/import must succeed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import itertools
import os
from pathlib import Path
import random
import struct

import pytest

from .binary_list import BinaryListProblem
from .exact import ExactAnswer, ExactRuntime, build_exact_runtime
from .proof import decode_proof, verify_answer


@pytest.fixture(scope='module')
def exact_runtime(tmp_path_factory):
    explicit = os.environ.get('SPECTRA_EXACT_LIBRARY')
    library = Path(explicit) if explicit else build_exact_runtime(tmp_path_factory.mktemp('exact-native'))
    return ExactRuntime(library)


def brute(problem, query):
    rows = [tuple(c for c in range(problem.palette) if (mask >> c) & 1) for mask in problem.masks]
    return {bytes(row) for row in itertools.product(*rows)
            if all(row[u] != row[v] for u, v in problem.edges)
            and all((mask >> row[v]) & 1 for v, mask in query)}


def check_answer(problem, query, answer, expected):
    assert answer.status == ('SAT_VERIFIED' if expected else 'UNSAT_VERIFIED')
    assert verify_answer(problem, query, answer.status, answer.labels, answer.proof)
    if expected:
        assert answer.labels in expected and not answer.proof
    else:
        assert not answer.labels and len(answer.proof) >= 20
    assert answer.visited >= 0 and answer.arcs_scanned >= 0


@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
def test_exhaustive_three_vertex_original_graphs_domains_and_requests(exact_runtime, mode):
    # Every graph on 3 labelled vertices; every one/two-colour domain from 3
    # colours; every per-vertex subset of that original domain, without filtering.
    bank = ((0, 1), (0, 2), (1, 2))
    for graph in range(8):
        edges = tuple(bank[i] for i in range(3) if (graph >> i) & 1)
        for masks in itertools.product((1, 2, 3, 4, 5, 6), repeat=3):
            p = BinaryListProblem.create(3, 3, edges, masks)
            original_models = brute(p, ())
            subsets = [tuple(x for x in range(8) if x & ~d == 0) for d in masks]
            with exact_runtime.prepare(p, mode=mode) as session:
                for request in itertools.product(*subsets):
                    q = tuple(enumerate(request))
                    expected = {row for row in original_models if all(request[v] >> row[v] & 1 for v in range(3))}
                    check_answer(p, q, session.solve(q), expected)


@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
def test_random_lists_against_native_minicard(exact_runtime, mode):
    from pysat.solvers import Solver
    rng = random.Random(14972)
    for _ in range(90):
        n = rng.randint(1, 60)
        p = BinaryListProblem.create(n, 6,
            ((u, v) for u in range(n) for v in range(u+1, n) if rng.random() < .12),
            (sum(1 << c for c in rng.sample(range(6), rng.choice((1, 2, 2, 2)))) for _ in range(n)))
        with exact_runtime.prepare(p, mode=mode) as session, Solver(name='minicard', bootstrap_with=p.cnf()) as sat:
            for _ in range(35):
                q = tuple((rng.randrange(n), rng.randrange(64)) for _ in range(rng.randrange(9)))
                literals = p.assumptions(q)
                expected = literals is not None and sat.solve(assumptions=list(literals))
                a = session.solve(q)
                assert (a.status == 'SAT_VERIFIED') == expected
                assert verify_answer(p, q, a.status, a.labels, a.proof)


@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
@pytest.mark.parametrize('kind', (1, 2, 3))
def test_every_proof_kind_and_single_byte_mutations(exact_runtime, mode, kind):
    if kind == 1:
        p = BinaryListProblem.create(2, 2, (), (3, 3)); q = ((0, 1), (0, 2))
    elif kind == 2:
        p = BinaryListProblem.create(4, 2, ((0, 1), (1, 2), (2, 3)), (3,)*4); q = ((0, 1), (3, 1))
    else:
        p = BinaryListProblem.create(3, 2, ((0, 1), (1, 2), (0, 2)), (3,)*3); q = ()
    with exact_runtime.prepare(p, mode=mode) as session:
        a = session.solve(q)
        assert decode_proof(a.proof, p.n)[0] == kind
        for index in range(len(a.proof)):
            for value in (0, 1, 127, 255):
                payload = bytearray(a.proof); payload[index] = value
                forged = replace(a, proof=bytes(payload))
                # A mutation may form a different valid proof. Check semantics,
                # not the unsupported premise that every change must be invalid.
                assert session.check(q, forged) == verify_answer(p, q, forged.status, forged.labels, forged.proof)
        for bad in (b'', a.proof+b'\0', a.proof[:-1], b'SBP1'+b'\xff'*12,
                    bytearray(a.proof), memoryview(a.proof)):
            forged = replace(a, proof=bad)
            assert not session.check(q, forged)
            assert not verify_answer(p, q, forged.status, forged.labels, forged.proof)


@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
def test_current_request_isolation_and_stale_proofs(exact_runtime, mode):
    p = BinaryListProblem.create(2, 2, ((0, 1),), (3, 3))
    with exact_runtime.prepare(p, mode=mode) as session:
        bad = ((0, 1), (1, 1)); a = session.solve(bad)
        assert not session.check((), a)
        assert not verify_answer(p, (), a.status, a.labels, a.proof)
        for q in ((), ((0, 1),), ((0, 2),), bad, ((1, 0),), (), ((0, 1 << 63),), ()):
            check_answer(p, q, session.solve(q), brute(p, q))
        # A sound proof is permitted to remain valid for the same contradiction
        # with an extra unrelated pin; validity is rechecked rather than cached.
        assert session.check(bad + ((0, 3),), a)


@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
def test_empty_graph_singletons_and_bit63(exact_runtime, mode):
    for p, queries in [
        (BinaryListProblem.create(0, 64, (), ()), ((),)),
        (BinaryListProblem.create(3, 64, ((0, 1),), (1 << 63, 1+(1 << 63), 1)),
         ((), ((1, 1 << 63),), ((0, 0),), ((2, (1 << 64)-1),))),
        (BinaryListProblem.create(2, 1, ((0, 1),), (1, 1)), ((),)),
    ]:
        with exact_runtime.prepare(p, mode=mode) as session:
            for q in queries:
                check_answer(p, q, session.solve(q), brute(p, q))


@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
def test_long_chain_and_odd_cycle_no_recursion(exact_runtime, mode):
    n = 10001
    p = BinaryListProblem.create(n, 2, tuple((v, v+1) for v in range(n-1)), (3,)*n)
    with exact_runtime.prepare(p, mode=mode) as session:
        q = ((0, 1), (n-1, 2))
        answer = session.solve(q)
        assert answer.status == 'UNSAT_VERIFIED'
        assert verify_answer(p, q, answer.status, answer.labels, answer.proof)
        assert session.solve(((0, 1),)).status == 'SAT_VERIFIED'
    p = BinaryListProblem.create(n, 2, p.edges+((0, n-1),), (3,)*n)
    with exact_runtime.prepare(p, mode=mode) as session:
        answer = session.solve(())
        assert decode_proof(answer.proof, n)[0] == 3
        assert verify_answer(p, (), answer.status, answer.labels, answer.proof)


@pytest.mark.parametrize('query', [((0, 0), (2, 1)), ((0, 0), (True, 1)),
    ((0, 0), (0, True)), ((0, 0), (0, 1 << 64)), ((0, 0), (0, -1)), ((0, 0), (0,)),
    ((0, 0), (0., 1)), ((0, 0), (-1, 1))])
def test_no_early_contradiction_hides_malformed_later_pins(exact_runtime, query):
    p = BinaryListProblem.create(2, 2, (), (3, 3))
    with exact_runtime.prepare(p) as session:
        with pytest.raises(ValueError):
            session.solve(query)
        with pytest.raises(ValueError):
            exact_runtime._module.solve(session._solver, query)
        assert session.solve(()).status == 'SAT_VERIFIED'


@pytest.mark.parametrize('change', [dict(n=True), dict(n=-1), dict(palette=0),
    dict(palette=65), dict(masks=(0, 3)), dict(masks=(7, 3)), dict(masks=(3,)),
    dict(edges=((0, 0),)), dict(edges=((0, 2),)), dict(masks=(True, 3))])
def test_original_input_validation_native_and_python(exact_runtime, change):
    p = replace(BinaryListProblem.create(2, 2, (), (3, 3)), **change)
    with pytest.raises((ValueError, TypeError)):
        exact_runtime.prepare(p)
    with pytest.raises(ValueError):
        exact_runtime._module.create(p.n, p.palette, p.edges, p.masks, 1, 512 << 20)


@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
def test_memory_lifecycle_concurrency_and_fail_closed(exact_runtime, mode, monkeypatch):
    p = BinaryListProblem.create(8, 2, tuple((v, v+1) for v in range(7)), (3,)*8)
    for cap in (True, -1, 0, 100):
        with pytest.raises(ValueError):
            exact_runtime.prepare(p, max_bytes=cap)
    with pytest.raises(MemoryError):
        exact_runtime._module.create(p.n, p.palette, p.edges, p.masks, 1, 100)
    with exact_runtime.prepare(p, mode=mode) as session:
        queries = [(), ((0, 1),), ((0, 1), (7, 1)), ((2, 0),)] * 30
        with ThreadPoolExecutor(4) as pool:
            answers = list(pool.map(session.solve, queries))
        for q, a in zip(queries, answers):
            check_answer(p, q, a, brute(p, q))
        old = session._module.verify
        monkeypatch.setattr(session._module, 'verify', lambda *a: False)
        for q in ((), ((0, 0),)):
            with pytest.raises(AssertionError, match='observer'):
                session.solve(q)
        monkeypatch.setattr(session._module, 'verify', old)
    session.close()
    with pytest.raises(RuntimeError): session.solve(())
    with pytest.raises(RuntimeError): _ = session.info
    with pytest.raises(RuntimeError): session.__enter__()


def test_observer_rejects_status_and_sat_witness_forgery(exact_runtime):
    p = BinaryListProblem.create(2, 2, ((0, 1),), (3, 3))
    with exact_runtime.prepare(p) as session:
        answer = session.solve(())
        for change in (dict(labels=b'\0\0'), dict(labels=b'\xff\0'), dict(labels=b'\0'),
                       dict(labels=bytearray(answer.labels)), dict(proof=b'junk'), dict(status='UNKNOWN')):
            bad = replace(answer, **change)
            assert not session.check((), bad)
            assert not verify_answer(p, (), bad.status, bad.labels, bad.proof)


def test_sparse_graph_preserves_proofs_and_drops_dense_arcs(exact_runtime):
    n = 120
    p = BinaryListProblem.create(n, 2, tuple((u, v) for u in range(n//2) for v in range(n//2, n)), (3,)*n)
    with exact_runtime.prepare(p, mode='sparse') as sparse, exact_runtime.prepare(p, mode='raw') as raw:
        info = sparse.info
        assert info['retained_proof_arcs'] <= 2*(2*n-info['components'])+info['condensation_arcs']
        assert info['retained_proof_arcs'] < info['original_arcs'] // 10
        assert info['retained_array_bytes'] < raw.info['retained_array_bytes']
        for q in ((), ((0, 1),), ((0, 1), (n-1, 1)), ((0, 2), (n-1, 1))):
            a = sparse.solve(q); b = raw.solve(q)
            assert a.status == b.status
            assert verify_answer(p, q, a.status, a.labels, a.proof)
            assert sparse.check(q, b)  # observer never trusts the sparse graph


def test_routed_proofs_have_no_original_search_adjacency(exact_runtime):
    n=300
    p=BinaryListProblem.create(n,2,tuple((u,v) for u in range(n//2) for v in range(n//2,n)),(3,)*n)
    with exact_runtime.prepare(p,mode='route') as routed, exact_runtime.prepare(p,mode='raw') as raw:
        assert routed.info['original_adjacency_retained'] is False
        assert routed.info['retained_proof_arcs']==0
        assert routed.info['routing_array_bytes']>0
        for q in ((),((0,1),(n-1,1)),((0,1),),((n-1,2),(1,2))):
            a=routed.solve(q);b=raw.solve(q)
            assert a.status==b.status
            assert verify_answer(p,q,a.status,a.labels,a.proof)
        assert routed.info['retained_array_bytes'] < raw.info['retained_array_bytes']//5


def test_old_abi_capsules_are_rejected_before_native_dereference(exact_runtime):
    import ctypes
    constructor = ctypes.pythonapi.PyCapsule_New
    constructor.restype = ctypes.py_object
    constructor.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    # Invalid address is safe only if the old ABI name is rejected BEFORE use.
    # Keep the name bytes alive for the lifetime of their capsules.
    solver_name = b'spectra.exact_binary.solver.v1'
    observer_name = b'spectra.exact_binary.observer.v1'
    old_solver = constructor(1, solver_name, None)
    old_observer = constructor(1, observer_name, None)
    with pytest.raises(ValueError, match='handle'):
        exact_runtime._module.solve(old_solver, ())
    with pytest.raises(ValueError, match='handle'):
        exact_runtime._module.verify(old_observer, (), 1, b'', b'')
    assert exact_runtime._module.abi() == 2
