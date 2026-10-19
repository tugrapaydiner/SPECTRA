"""Contracts for the independent direct 2-SAT challenger."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from experiments.compiled_query.twosat_baseline import (
    TwoSatBaseline,
    build_twosat_baseline,
)
from spectra.exact import (
    ExactProblem,
    ExactRuntime,
    build_exact_runtime,
    decode_certified_output,
    pack_session_packet,
    verify_session_output,
)
from spectra.exact_proof import decode_proof


def _compiler_available() -> bool:
    return bool(shutil.which("cl" if sys.platform == "win32" else "g++"))


@pytest.fixture(scope="session")
def twosat_library(tmp_path_factory):
    if sys.platform not in ("linux", "win32") or not _compiler_available():
        pytest.skip("supported native compiler is unavailable")
    sanitizer = os.environ.get("SPECTRA_TWOSAT_SANITIZER", "none")
    return build_twosat_baseline(
        tmp_path_factory.mktemp("direct-twosat"), sanitizer=sanitizer
    )


@pytest.fixture(scope="session")
def twosat(twosat_library):
    return TwoSatBaseline(twosat_library)


@pytest.fixture(scope="session")
def exact_runtime(tmp_path_factory):
    supplied = os.environ.get("SPECTRA_EXACT_LIBRARY")
    library = (
        Path(supplied).resolve()
        if supplied
        else build_exact_runtime(tmp_path_factory.mktemp("exact-for-twosat"))
    )
    return ExactRuntime(library)


def _codes(output: bytes, n: int, count: int) -> tuple[int, ...]:
    return tuple(code for code, _labels, _proof in decode_certified_output(output, n, count))


def _brute(problem: ExactProblem, query) -> bool:
    allowed = list(problem.masks)
    for vertex, mask in query:
        allowed[vertex] &= mask
    if any(mask == 0 for mask in allowed):
        return False
    choices = [
        tuple(colour for colour in range(problem.palette) if mask >> colour & 1)
        for mask in allowed
    ]
    return any(
        all(labels[left] != labels[right] for left, right in problem.edges)
        for labels in itertools.product(*choices)
    )


def test_direct_baseline_matches_exhaustive_three_vertex_oracle(twosat):
    edge_bank = tuple((left, right) for left in range(3) for right in range(left + 1, 3))
    queries = [()]
    queries.extend(((vertex, mask),) for vertex in range(3) for mask in range(4))
    queries.extend(
        ((left, left_mask), (right, right_mask))
        for left in range(3)
        for right in range(left + 1, 3)
        for left_mask in range(4)
        for right_mask in range(4)
    )
    rows = tuple(queries)

    for masks in itertools.product((1, 2, 3), repeat=3):
        for edge_bits in range(1 << len(edge_bank)):
            edges = tuple(
                edge for index, edge in enumerate(edge_bank) if edge_bits >> index & 1
            )
            problem = ExactProblem.create(3, 2, edges, masks)
            packet = pack_session_packet(problem, rows)
            answer = twosat.solve_session_packet(packet)
            assert answer.query_count == len(rows)
            assert answer.assignment_fallbacks == 0
            assert verify_session_output(packet, answer.output)
            expected = tuple(1 if _brute(problem, row) else 0 for row in rows)
            assert _codes(answer.output, problem.n, len(rows)) == expected


def test_direct_baseline_matches_exact_decisions_and_all_proof_kinds(
    twosat, exact_runtime
):
    mixed = ExactProblem.create(4, 2, ((0, 1), (1, 2), (2, 3)), (3, 3, 3, 3))
    rows = (
        (),
        ((0, 1),),
        ((0, 1), (1, 1)),
        ((0, 0),),
        ((0, 1), (3, 2)),
    )
    packet = pack_session_packet(mixed, rows)
    direct = twosat.solve_session_packet(packet)
    exact = exact_runtime.solve_session_packet(packet, mode="auto")
    assert verify_session_output(packet, direct.output)
    assert _codes(direct.output, mixed.n, len(rows)) == _codes(
        exact.output, mixed.n, len(rows)
    )
    kinds = {
        decode_proof(proof, mixed.n)[0]
        for code, _labels, proof in decode_certified_output(
            direct.output, mixed.n, len(rows)
        )
        if code == 0
    }
    assert {1, 2}.issubset(kinds)

    static = ExactProblem.create(2, 1, ((0, 1),), (1, 1))
    static_packet = pack_session_packet(static, ((),))
    static_answer = twosat.solve_session_packet(static_packet)
    assert verify_session_output(static_packet, static_answer.output)
    [(code, _labels, proof)] = decode_certified_output(
        static_answer.output, static.n, 1
    )
    assert code == 0 and decode_proof(proof, static.n)[0] == 3


def test_direct_baseline_has_a_sparse_fallback(twosat):
    n = 4200
    problem = ExactProblem.create(n, 2, (), (3,) * n)
    packet = pack_session_packet(problem, ((),))
    answer = twosat.solve_session_packet(packet, max_bytes=64 << 20)
    assert answer.components == 2 * n
    assert answer.dense_closure is False
    assert answer.closure_bytes == 0
    assert answer.assignment_fallbacks == 0
    assert verify_session_output(packet, answer.output)


def test_direct_baseline_rejects_malformed_packet(twosat):
    problem = ExactProblem.create(2, 2, ((0, 1),), (3, 3))
    packet = bytearray(pack_session_packet(problem, ((),)))
    packet[0] ^= 1
    with pytest.raises(ValueError, match="SBC1"):
        twosat.solve_session_packet(bytes(packet))


def test_direct_baseline_receipt_rejects_tampering(tmp_path, twosat_library):
    copied = tmp_path / "copied"
    copied.mkdir()
    library = copied / twosat_library.name
    shutil.copy2(twosat_library, library)
    receipt = json.loads(twosat_library.with_name("build.json").read_text())
    receipt["source_sha256"] = "0" * 64
    library.with_name("build.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        TwoSatBaseline(library)
