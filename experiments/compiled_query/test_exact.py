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
from .exact import (ExactAnswer, ExactRuntime, build_exact_runtime,
                    pack_query_bank, pack_session_packet,
                    unpack_query_bank, unpack_session_packet)
from .proof import decode_proof, verify_answer
from .sessions import decode_certified_output


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


@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
def test_batched_bank_matches_individually_checked_answers(exact_runtime, mode):
    p = BinaryListProblem.create(8, 3,
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)),
        (3, 6, 5, 3, 6, 5, 3, 6))
    queries = ((), ((0, 1),), ((0, 1), (7, 2)), ((3, 0),),
               ((2, 1), (2, 4)), ((4, 2), (5, 4)), ((7, 1 << 63),))
    packet = pack_query_bank(p, queries)
    assert unpack_query_bank(packet, p.n) == queries
    with exact_runtime.prepare(p, mode=mode) as session:
        bank = session.solve_bank(packet)
        rows = decode_certified_output(bank.output, p.n, len(queries))
        assert bank.query_count == len(queries)
        assert bank.sat_count + bank.unsat_count == len(queries)
        assert bank.proof_bytes == sum(len(proof) for _, _, proof in rows)
        assert session.check_bank(packet, bank.output)
        for query, (decision, labels, proof) in zip(queries, rows):
            single = session.solve(query)
            assert decision == (single.status == 'SAT_VERIFIED')
            assert labels == (single.labels if decision else bytes(p.n))
            assert verify_answer(p, query, single.status, labels if decision else b'', proof)


def test_batched_packet_and_serialized_output_fail_closed(exact_runtime):
    p = BinaryListProblem.create(4, 2, ((0, 1), (1, 2), (2, 3)), (3,) * 4)
    queries = ((), ((0, 1),), ((0, 1), (3, 1)), ((1, 0),))
    packet = pack_query_bank(p, queries)
    with exact_runtime.prepare(p, mode='route') as session:
        answer = session.solve_bank(packet)
        with pytest.raises(ValueError, match='immutable'):
            session.solve_bank(bytearray(packet))
        corruptions = []
        for offset, value in ((0, ord('X')), (12, 1), (16, 1), (32, 255)):
            changed = bytearray(packet); changed[offset] = value; corruptions.append(bytes(changed))
        corruptions += [packet[:-1], packet + b'\0']
        for changed in corruptions:
            with pytest.raises(ValueError):
                session.solve_bank(changed)
        for offset in (0, 4, 16, len(answer.output) - 1):
            changed = bytearray(answer.output); changed[offset] ^= 1
            assert not session.check_bank(packet, bytes(changed))
        assert not session.check_bank(packet, bytearray(answer.output))


def test_batched_worst_case_output_is_admitted_before_work(exact_runtime):
    n = 1000
    p = BinaryListProblem.create(n, 2, tuple((v, v + 1) for v in range(n - 1)), (3,) * n)
    packet = pack_query_bank(p, ((),) * 100)
    with exact_runtime.prepare(p, mode='route', max_bytes=2 << 20) as session:
        with pytest.raises(MemoryError, match='admission'):
            session.solve_bank(packet)


@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
def test_prepared_session_packet_matches_persistent_exact_endpoint(exact_runtime, mode):
    p = BinaryListProblem.create(8, 3,
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)),
        (3, 6, 5, 3, 6, 5, 3, 6))
    queries = ((), ((0, 1),), ((0, 1), (7, 2)), ((3, 0),),
               ((2, 1), (2, 4)), ((4, 2), (5, 4)), ((7, 1 << 63),))
    packet = pack_session_packet(p, queries)
    decoded_problem, decoded_queries = unpack_session_packet(packet)
    assert decoded_problem == p and decoded_queries == queries
    prepared = exact_runtime.solve_session_packet(packet, mode=mode)
    assert prepared.info['prepared_input'] is True
    assert prepared.info['input_schema'] == 'SBC1'
    assert prepared.info['mode'] == mode
    query_packet = pack_query_bank(p, queries)
    with exact_runtime.prepare(p, mode=mode) as session:
        persistent = session.solve_bank(query_packet)
        assert prepared.output == persistent.output
        assert session.check_bank(query_packet, prepared.output)
    assert prepared.query_count == len(queries)
    assert prepared.sat_count + prepared.unsat_count == len(queries)
    rows = decode_certified_output(prepared.output, p.n, len(queries))
    for query, (decision, labels, proof) in zip(queries, rows):
        status = 'SAT_VERIFIED' if decision else 'UNSAT_VERIFIED'
        assert verify_answer(p, query, status, labels if decision else b'', proof)



@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
def test_prepared_session_packet_tile64_matches_individual_exact_answers(exact_runtime, mode):
    n = 24
    p = BinaryListProblem.create(n, 2, tuple((v, v + 1) for v in range(n - 1)), (3,) * n)
    rows = []
    for index in range(130):
        vertex = index % (n - 1)
        kind = index % 6
        if kind == 0:
            query = ()
        elif kind == 1:
            query = ((vertex, 1),)
        elif kind == 2:
            query = ((vertex, 1), (vertex + 1, 1))
        elif kind == 3:
            query = ((vertex, 2), (vertex + 1, 2))
        elif kind == 4:
            query = ((vertex, 1), (vertex, 2))
        else:
            query = ((vertex, 1 << 63),)
        rows.append(query)
    queries = tuple(rows)
    packet = pack_session_packet(p, queries)
    prepared = exact_runtime.solve_session_packet(packet, mode=mode)
    assert prepared.info['bank_kernel'] == ('serial' if mode == 'raw' else 'tile64')
    assert prepared.info['bank_tiles'] == (0 if mode == 'raw' else 3)
    decoded = decode_certified_output(prepared.output, p.n, len(queries))
    with exact_runtime.prepare(p, mode=mode) as session:
        for query, (decision, labels, proof) in zip(queries, decoded):
            single = session.solve(query)
            assert decision == (single.status == 'SAT_VERIFIED')
            assert labels == (single.labels if decision else bytes(p.n))
            assert proof == single.proof
            assert verify_answer(p, query, single.status,
                                 labels if decision else b'', proof)


def test_prepared_tile_threshold_is_explicit(exact_runtime):
    p = BinaryListProblem.create(8, 2, tuple((v, v + 1) for v in range(7)), (3,) * 8)
    for count, expected, tiles in ((63, 'serial', 0), (64, 'tile64', 1), (129, 'tile64', 3)):
        packet = pack_session_packet(p, ((),) * count)
        answer = exact_runtime.solve_session_packet(packet, mode='route')
        assert answer.info['bank_kernel'] == expected
        assert answer.info['bank_tiles'] == tiles


@pytest.mark.parametrize('mode', ('scc', 'sparse', 'route'))
def test_mixed_domain_tiles_keep_exact_scalar_proofs(exact_runtime, mode):
    # Adjacent domains share one colour, producing a nontrivial condensation
    # DAG. Include repeated/reordered pins, immediate and propagated conflicts,
    # unforced requests, and a partial last tile.
    p = BinaryListProblem.create(8, 4, tuple((v, v+1) for v in range(7)),
                                (3, 6, 12, 9, 3, 6, 12, 9))
    rng = random.Random(73051)
    rows = [(), ((0, 2), (1, 2)), ((0, 1), (0, 2)),
            ((1, 6), (0, 2), (1, 2)), ((7, 8), (6, 8)), ((0, 1 << 63),)]
    rows += [tuple((rng.randrange(p.n), rng.randrange(16))
                   for _ in range(rng.randrange(1, 9))) for _ in range(123)]
    rows = tuple(rows)
    expected = []
    with exact_runtime.prepare(p, mode=mode) as session:
        for query in rows:
            answer = session.solve(query)
            check_answer(p, query, answer, brute(p, query))
            expected.append((answer.status == 'SAT_VERIFIED',
                             answer.labels if answer.labels else bytes(p.n), answer.proof))
    for count in (63, 64, 65, 129):
        bank = exact_runtime.solve_session_packet(pack_session_packet(p, rows[:count]), mode=mode)
        decoded = decode_certified_output(bank.output, p.n, count)
        assert decoded == tuple(expected[:count])
    reverse = exact_runtime.solve_session_packet(pack_session_packet(p, rows[::-1]), mode=mode)
    assert decode_certified_output(reverse.output, p.n, len(rows)) == tuple(expected[::-1])


@pytest.mark.parametrize('mode', ('scc', 'sparse', 'route', 'auto'))
def test_tiled_witnesses_keep_high_colours_and_last_lane(exact_runtime, mode):
    n = 71
    masks = tuple(1 if v % 11 == 0 else 1 | (1 << 63) for v in range(n))
    p = BinaryListProblem.create(n, 64, tuple((v, v+1) for v in range(0, n-1, 2)), masks)
    rng = random.Random(84617)
    rows = []
    for index in range(129):
        labels = [0] * n
        for v in range(0, n-1, 2):
            first = 0 if masks[v] == 1 else 63 if masks[v+1] == 1 else rng.choice((0, 63))
            labels[v], labels[v+1] = first, 63-first
        rows.append(tuple((v, 1 << labels[v]) for v in range(n)
                          if index % 3 and rng.random() < .3))
    rows = tuple(rows)
    expected = []
    with exact_runtime.prepare(p, mode='scc' if mode == 'auto' else mode) as session:
        for query in rows:
            answer = session.solve(query)
            assert answer.status == 'SAT_VERIFIED'
            assert verify_answer(p, query, answer.status, answer.labels, answer.proof)
            expected.append((1, answer.labels, b''))
    for count in (64, 65, 129):
        answer = exact_runtime.solve_session_packet(pack_session_packet(p, rows[:count]), mode=mode)
        assert answer.sat_count == count and answer.proof_bytes == 0
        assert decode_certified_output(answer.output, n, count) == tuple(expected[:count])


@pytest.mark.parametrize('count', (0, 1, 63, 64, 65, 129))
@pytest.mark.parametrize('structure', ('empty', 'independent', 'chain', 'odd_cycle'))
def test_auto_packet_exposes_and_matches_its_selected_engine(exact_runtime, count, structure):
    n = 0 if structure == 'empty' else 9
    edges = () if structure in ('empty', 'independent') else tuple((v, v+1) for v in range(n-1))
    if structure == 'odd_cycle':
        edges += ((0, n-1),)
    p = BinaryListProblem.create(n, 2, edges, (3,) * n)
    requests = ((),) if not n else ((), ((0, 1),), ((0, 1), (1, 1)), ((0, 0),))
    rows = tuple(requests[i % len(requests)] for i in range(count))
    packet = pack_session_packet(p, rows)
    answer = exact_runtime.solve_session_packet(packet, mode='auto')
    selected = ('raw' if count < 64 or structure == 'odd_cycle' else
                'route' if structure in ('chain', 'empty') else 'scc')
    assert answer.info['mode'] == 'auto'
    assert answer.info['selected_mode'] == selected
    assert answer.output == exact_runtime.solve_session_packet(packet, mode=selected).output
    for query, (decision, labels, certificate) in zip(rows, decode_certified_output(answer.output, n, count)):
        assert bool(decision) == bool(brute(p, query))
        assert verify_answer(p, query, 'SAT_VERIFIED' if decision else 'UNSAT_VERIFIED',
                             labels if decision else b'', certificate)


def test_auto_does_not_hide_malformed_packets_or_resource_limits(exact_runtime):
    p = BinaryListProblem.create(2, 2, ((0, 1),), (3, 3))
    packet = pack_session_packet(p, (((0, 0),), ((1, 1),)))
    # Corrupt the final pin, after a valid immediate contradiction in row one.
    damaged = bytearray(packet)
    struct.pack_into('<I', damaged, 32 + 8*p.n + 8*len(p.edges) + 4*3 + 4, p.n)
    with pytest.raises(ValueError):
        exact_runtime.solve_session_packet(bytes(damaged), mode='auto')
    large = BinaryListProblem.create(1000, 2, (), (3,) * 1000)
    with pytest.raises(MemoryError):
        exact_runtime.solve_session_packet(pack_session_packet(large, ((),)*100),
                                           mode='auto', max_bytes=2 << 20)


@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
def test_impossible_graph_bank_keeps_current_request_proofs(exact_runtime, mode):
    # An odd cycle is impossible before requests arrive. Reusing its fixed proof
    # must not replace an earlier empty-domain proof or leak into another call.
    n = 101
    p = BinaryListProblem.create(n, 2,
        tuple((v, v + 1) for v in range(n - 1)) + ((0, n - 1),), (3,) * n)
    rows = (((0, 1), (0, 2)), (), ((7, 1),), ((9, 0),), ()) * 27
    answer = exact_runtime.solve_session_packet(pack_session_packet(p, rows), mode=mode)
    decoded = decode_certified_output(answer.output, n, len(rows))
    visits = arcs = 0
    with exact_runtime.prepare(p, mode=mode) as session:
        for query, (decision, labels, certificate) in zip(rows, decoded):
            single = session.solve(query)
            assert not decision and labels == bytes(n)
            assert certificate == single.proof
            assert verify_answer(p, query, single.status, b'', certificate)
            visits += single.visited
            arcs += single.arcs_scanned
    assert answer.visited <= visits and answer.arcs_scanned < arcs
    assert exact_runtime.solve_session_packet(pack_session_packet(p, rows[::-1]),
                                             mode=mode).unsat_count == len(rows)
    feasible = BinaryListProblem.create(2, 2, ((0, 1),), (3, 3))
    assert exact_runtime.solve_session_packet(pack_session_packet(feasible, ((),) * 65),
                                             mode=mode).sat_count == 65


def test_prepared_session_packet_rejects_every_corrupted_region(exact_runtime):
    p = BinaryListProblem.create(4, 2, ((0, 1), (1, 2), (2, 3)), (3,) * 4)
    queries = ((), ((0, 1),), ((0, 1), (3, 1)), ((1, 0),))
    packet = pack_session_packet(p, queries)
    with pytest.raises(ValueError, match='immutable'):
        exact_runtime.solve_session_packet(bytearray(packet))
    masks_at = 32
    edges_at = masks_at + 8 * p.n
    offsets_at = edges_at + 8 * len(p.edges)
    vertices_at = offsets_at + 4 * (len(queries) + 1)
    mutations = []
    for offset, value in ((0, ord('X')), (8, 0), (24, 1), (masks_at, 0),
                          (edges_at + 4, 0), (offsets_at, 1), (vertices_at, 255)):
        changed = bytearray(packet); changed[offset] = value; mutations.append(bytes(changed))
    mutations += [packet[:-1], packet + b'\0']
    for changed in mutations:
        with pytest.raises(ValueError):
            exact_runtime.solve_session_packet(changed)


def test_prepared_session_packet_admits_worst_case_output_before_construction(exact_runtime):
    n = 1000
    p = BinaryListProblem.create(n, 2, tuple((v, v + 1) for v in range(n - 1)), (3,) * n)
    packet = pack_session_packet(p, ((),) * 100)
    with pytest.raises(MemoryError, match='admission'):
        exact_runtime.solve_session_packet(packet, mode='route', max_bytes=2 << 20)


def test_old_abi_capsules_are_rejected_before_native_dereference(exact_runtime):
    import ctypes
    constructor = ctypes.pythonapi.PyCapsule_New
    constructor.restype = ctypes.py_object
    constructor.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    # Invalid address is safe only if the old ABI name is rejected BEFORE use.
    # Keep the name bytes alive for the lifetime of their capsules.
    solver_name = b'spectra.exact_binary.solver.v2'
    observer_name = b'spectra.exact_binary.observer.v2'
    old_solver = constructor(1, solver_name, None)
    old_observer = constructor(1, observer_name, None)
    with pytest.raises(ValueError, match='handle'):
        exact_runtime._module.solve(old_solver, ())
    with pytest.raises(ValueError, match='handle'):
        exact_runtime._module.verify(old_observer, (), 1, b'', b'')
    assert exact_runtime._module.abi() == 3
