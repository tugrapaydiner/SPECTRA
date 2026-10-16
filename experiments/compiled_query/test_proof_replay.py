"""The faster audit must agree with the retained scalar original-input reference."""
from dataclasses import replace
import itertools
from pathlib import Path
import os
import random
import struct
from types import SimpleNamespace

import pytest

from .audit import original_feasible
from .binary_list import BinaryListProblem
from .exact import ExactRuntime, build_exact_runtime
from .proof import verify_answer
from .proof_replay import OriginalReplay


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    path = os.environ.get('SPECTRA_EXACT_LIBRARY')
    return ExactRuntime(Path(path) if path else build_exact_runtime(tmp_path_factory.mktemp('replay-native')))


def test_every_three_vertex_domain_graph_and_complete_assignment():
    edges = ((0, 1), (0, 2), (1, 2))
    for graph in range(8):
        for masks in itertools.product((1, 2, 3, 4, 5, 6), repeat=3):
            p = BinaryListProblem.create(3, 3, [e for i, e in enumerate(edges) if graph >> i & 1], masks)
            observer = OriginalReplay(p)
            for values in itertools.product(range(4), repeat=3):
                row = bytes(values)
                assert observer.verify((), 'SAT_VERIFIED', row, b'') == verify_answer(p, (), 'SAT_VERIFIED', row, b'')


def test_replayed_current_feasibility_and_proofs_match_independent_references(runtime):
    rng = random.Random(184371)
    for _ in range(120):
        n = rng.randrange(1, 15)
        p = BinaryListProblem.create(n, 5,
            ((u, v) for u in range(n) for v in range(u+1, n) if rng.random() < .22),
            (sum(1 << c for c in rng.sample(range(5), rng.choice((1, 2, 2, 2)))) for _ in range(n)))
        observer = OriginalReplay(p)
        with runtime.prepare(p, mode='route') as session:
            for _ in range(15):
                query = tuple((rng.randrange(n), rng.randrange(32)) for _ in range(rng.randrange(6)))
                expected = original_feasible(p, query)
                assert observer.feasible(query) == expected
                answer = session.solve(query)
                assert (answer.status == 'SAT_VERIFIED') == expected
                assert observer.verify(query, answer.status, answer.labels, answer.proof)
                assert verify_answer(p, query, answer.status, answer.labels, answer.proof)
                # Hashes are not evidence of logical validity. Challenge both
                # verifiers with altered bytes, including changed path endpoints.
                if answer.proof:
                    for index in rng.sample(range(len(answer.proof)), min(8, len(answer.proof))):
                        bad = bytearray(answer.proof); bad[index] ^= rng.randrange(1, 256)
                        assert observer.verify(query, answer.status, b'', bytes(bad)) == verify_answer(
                            p, query, answer.status, b'', bytes(bad))
            assert observer.feasible(()) == original_feasible(p, ())


@pytest.mark.parametrize('n', (4096, 4097))
def test_bitset_threshold_and_original_edge_fallback(runtime, n):
    p = BinaryListProblem.create(n, 64, ((v, v+1) for v in range(n-1)), (1+(1 << 63),)*n)
    observer = OriginalReplay(p)
    assert (observer.adjacency is not None) == (n <= 4096)
    with runtime.prepare(p, mode='route') as session:
        for query in ((), ((0, 1),), ((0, 1), (1, 1)), ((0, 0),)):
            answer = session.solve(query)
            assert observer.verify(query, answer.status, answer.labels, answer.proof)
            assert verify_answer(p, query, answer.status, answer.labels, answer.proof)
            assert observer.feasible(query) == (answer.status == 'SAT_VERIFIED')


def test_prepared_original_snapshot_is_owned():
    edges, masks = [[0, 1]], [3, 3]
    p = SimpleNamespace(n=2, palette=2, edges=edges, masks=masks)
    observer = OriginalReplay(p)
    masks[:] = [1, 1]; edges.clear()
    assert observer.verify((), 'SAT_VERIFIED', b'\0\1', b'')
    assert not observer.verify((), 'SAT_VERIFIED', b'\0\0', b'')
    assert observer.feasible(())
    assert not observer.feasible(((0, 1), (1, 1)))
    assert observer.feasible(())


@pytest.mark.parametrize('query', [((0, 0), (True, 1)), ((0, 0), (2, 1)),
    ((0, 0), (0, -1)), ((0, 0), (0, 1 << 64)), ((0, 0), (0, True)), ((0, 0), (0,))])
def test_full_current_input_validation_and_malformed_proofs(query):
    p = BinaryListProblem.create(2, 2, (), (3, 3))
    observer = OriginalReplay(p)
    proof = struct.pack('<4sIIII', b'SBP1', 1, 1, 0, 0)
    assert not observer.verify(query, 'UNSAT_VERIFIED', b'', proof)
    with pytest.raises((ValueError, TypeError)):
        observer.feasible(query)
    for bad in (proof+b'!', proof[:-1], b'', bytearray(proof), memoryview(proof)):
        assert not observer.verify(((0, 0),), 'UNSAT_VERIFIED', b'', bad)
    assert not observer.verify((), 'UNKNOWN', b'', b'')


@pytest.mark.parametrize('change', [dict(n=True), dict(palette=65), dict(masks=(0, 3)),
    dict(masks=(7, 3)), dict(masks=(True, 3)), dict(edges=((0, 0),)), dict(edges=((0, 3),))])
def test_original_admission_never_uses_compiled_assumptions(change):
    p = replace(BinaryListProblem.create(2, 2, (), (3, 3)), **change)
    with pytest.raises((ValueError, TypeError)):
        OriginalReplay(p)


def test_common_mask_projection_preserves_all_effective_domains(runtime):
    from .common_masks import project_queries
    from .mixed_cases import unfiltered_requests
    p = BinaryListProblem.create(5, 3, ((0, 1), (1, 2), (2, 3), (3, 4)), (3, 6, 5, 3, 6))
    old = unfiltered_requests(p, 128)
    new = project_queries(p, old)
    assert len(new) == len(old) and new != old
    with runtime.prepare(p, mode='route') as session:
        for a, b in zip(old, new):
            assert len(a) == len(b)
            assert all(u == v and (x & p.masks[u]) == (y & p.masks[v])
                       for (u, x), (v, y) in zip(a, b))
            answer = session.solve(a)
            assert session.solve(b).status == answer.status
            assert verify_answer(p, b, answer.status, answer.labels, answer.proof)
    assert project_queries(p, new) == new
