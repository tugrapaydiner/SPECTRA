"""Check sort elision against the previous native array construction.

The diagnostic build is never used in the timing panel. No external compiler or
solver is downloaded here; the same configured native builder is used.
"""
from __future__ import annotations

import itertools
import random

import pytest

from . import exact
from .binary_list import BinaryListProblem
from .proof import verify_answer
from .sessions import decode_certified_output


@pytest.fixture(scope='module')
def diagnostic(tmp_path_factory):
    folder = tmp_path_factory.mktemp('linear-csr-oracle')
    source = folder / 'diagnostic.cpp'
    original = exact.SOURCE
    source.write_text('#define SPECTRA_EXACT_VALIDATE_LINEAR_CSR 1\n' +
                      original.read_text(encoding='utf-8'), encoding='utf-8')
    try:
        exact.SOURCE = source
        library = exact.build_exact_runtime(folder / 'build')
    finally:
        exact.SOURCE = original
    return exact.ExactRuntime(library)


def replay(runtime, problem, rows):
    packet = exact.pack_session_packet(problem, rows)
    for mode in ('raw', 'scc', 'sparse', 'route', 'auto'):
        result = runtime.solve_session_packet(packet, mode=mode)
        decoded = decode_certified_output(result.output, problem.n, len(rows))
        for query, (code, labels, proof) in zip(rows, decoded):
            assert verify_answer(problem, query,
                                 'SAT_VERIFIED' if code else 'UNSAT_VERIFIED',
                                 labels if code else b'', proof)


@pytest.mark.parametrize('graph', range(8))
def test_every_three_vertex_graph_and_original_domain(diagnostic, graph):
    # 8 * 6**3 original problems; both full native CSR arrays are compared inside
    # each constructor, before SCC processing. This covers singleton unit rows.
    bank = ((0, 1), (0, 2), (1, 2))
    edges = tuple(edge for i, edge in enumerate(bank) if graph >> i & 1)
    for masks in itertools.product((1, 2, 3, 4, 5, 6), repeat=3):
        p = BinaryListProblem.create(3, 3, edges, masks)
        replay(diagnostic, p, ((), ((0, 1),), ((1, 2),), ((2, 0),)))


@pytest.mark.parametrize('n', (0, 1, 2, 31, 64, 129))
def test_large_labels_relabelling_and_tile_boundaries(diagnostic, n):
    rng = random.Random(58013 + n)
    labels = (0, 1, 17, 62, 63)
    masks = tuple(sum(1 << c for c in rng.sample(labels, rng.choice((1, 2))))
                  for _ in range(n))
    edges = tuple((u, v) for u in range(n) for v in range(u + 1, n)
                  if rng.random() < .19)
    p = BinaryListProblem.create(n, 64, edges, masks)
    rows = tuple(tuple((rng.randrange(n), rng.getrandbits(64))
                       for _ in range(rng.randrange(5))) if n else ()
                 for _ in range(129))
    for count in (1, 63, 64, 65, 129):
        replay(diagnostic, p, rows[:count])


def test_tuple_normalization_precedes_linear_native_construction(diagnostic):
    # Bypass BinaryListProblem normalization: the native tuple path must still
    # supply canonical edges to the new constructor before either CSR is built.
    module = diagnostic._module
    canonical = ((0, 1), (0, 3), (1, 2), (2, 3))
    noisy = ((3, 2), (3, 0), (0, 1), (2, 1), (1, 0), (0, 3))
    masks = (3, 1, 3, 2)
    for mode in range(4):
        a = module.create(4, 2, canonical, masks, mode, 16 << 20)
        b = module.create(4, 2, noisy, masks, mode, 16 << 20)
        assert module.info(a) == module.info(b)
        for query in ((), ((0, 1),), ((0, 2),), ((3, 0),)):
            assert module.solve(a, query) == module.solve(b, query)
