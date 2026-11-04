"""Tile64 bank preparation contracts for the installed exact-query API.

These tests protect answer bytes and parser semantics.  Performance is measured by
the separate development benchmark, never by a wall-clock unit-test assertion.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

import pytest

from spectra.exact import (
    ExactProblem,
    ExactRuntime,
    build_exact_runtime,
    decode_certified_output,
    pack_query_bank,
    pack_session_packet,
    verify_certified_output,
    verify_session_output,
)


def _compiler_available() -> bool:
    return bool(shutil.which('cl' if sys.platform == 'win32' else 'g++'))


@pytest.fixture(scope='module')
def tile_runtime(tmp_path_factory):
    supplied = os.environ.get('SPECTRA_EXACT_LIBRARY')
    if supplied is not None:
        return ExactRuntime(Path(supplied).resolve())
    if sys.platform not in ('linux', 'win32') or not _compiler_available():
        pytest.skip('supported native compiler is unavailable')
    library = build_exact_runtime(tmp_path_factory.mktemp('tile64-exact-runtime'))
    return ExactRuntime(library)


def _problem_and_queries():
    problem = ExactProblem.create(
        8, 2, tuple((vertex, vertex + 1) for vertex in range(7)), (0b11,) * 8)
    queries = []
    for index in range(65):
        if index % 5 == 0:
            # Two duplicate-pin empty domains in deliberately noncanonical order.
            # Scalar preparation sorts by original vertex, so vertex 2 is the
            # canonical kind-1 proof target rather than vertex 5.
            query = ((5, 0b01), (2, 0b01), (5, 0b10), (2, 0b10))
        elif index % 5 == 1:
            query = ((2, 0b01), (1, 0b01))
        elif index % 5 == 2:
            query = ((7, 0b10), (0, 0b01))
        elif index % 5 == 3:
            query = ((4, 0b11), (4, 0b01), (6, 0b10))
        else:
            query = ()
        queries.append(query)
    return problem, tuple(queries)


def test_tile64_bank_matches_every_scalar_scc_answer(tile_runtime):
    problem, queries = _problem_and_queries()
    packet = pack_query_bank(problem, queries)
    with tile_runtime.prepare(problem, mode='scc') as session:
        bank = session.solve_bank(packet)
        rows = decode_certified_output(bank.output, problem.n, len(queries))
        assert session.check_bank(packet, bank.output)
        for query, row in zip(queries, rows):
            scalar = session.solve(query)
            code, labels, proof = row
            expected_labels = scalar.labels if scalar.status == 'SAT_VERIFIED' else bytes(problem.n)
            assert (code, labels, proof) == (
                int(scalar.status == 'SAT_VERIFIED'), expected_labels, scalar.proof)
    assert verify_certified_output(problem, queries, bank.output)


def test_sbq1_and_sbc1_tile64_paths_are_byte_identical(tile_runtime):
    problem, queries = _problem_and_queries()
    bank_packet = pack_query_bank(problem, queries)
    with tile_runtime.prepare(problem, mode='scc') as session:
        prepared = session.solve_bank(bank_packet)
    session_packet = pack_session_packet(problem, queries)
    one_call = tile_runtime.solve_session_packet(session_packet, mode='scc')
    assert one_call.info['bank_kernel'] == 'tile64'
    assert one_call.info['bank_tiles'] == 2
    assert one_call.output == prepared.output
    assert verify_session_output(session_packet, one_call.output)
