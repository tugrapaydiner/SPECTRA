"""Contracts for the independent native 2-SAT challenger.

The challenger is deliberately separate from SPECTRA's native exact solver.  The
public SBC1/SCQ2 formats and the standard-library verifier are the only shared
semantic boundary.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import random
import shutil
import sys

import pytest

from experiments.compiled_query.twosat_baseline import (
    TwoSatRuntime,
    build_twosat_runtime,
)
from spectra.exact import (
    ExactProblem,
    decode_certified_output,
    pack_session_packet,
    verify_session_output,
)
from spectra.exact_proof import decode_proof


def _compiler_available() -> bool:
    return bool(shutil.which("cl" if sys.platform == "win32" else "g++"))


@pytest.fixture(scope="module")
def twosat_runtime(tmp_path_factory):
    supplied = os.environ.get("SPECTRA_TWOSAT_LIBRARY")
    if supplied is not None:
        return TwoSatRuntime(Path(supplied))
    if sys.platform not in ("linux", "win32") or not _compiler_available():
        pytest.skip("supported native compiler is unavailable")
    library = build_twosat_runtime(tmp_path_factory.mktemp("native-twosat"))
    return TwoSatRuntime(library)


def _brute(problem: ExactProblem, query):
    effective = list(problem.masks)
    for vertex, allowed in query:
        effective[vertex] &= allowed
    choices = [
        tuple(colour for colour in range(problem.palette) if mask & (1 << colour))
        for mask in effective
    ]
    if any(not row for row in choices):
        return None
    for labels in itertools.product(*choices):
        if all(labels[left] != labels[right] for left, right in problem.edges):
            return bytes(labels)
    return None


def _decision_codes(output: bytes, vertices: int, rows: int) -> tuple[int, ...]:
    return tuple(
        code for code, _labels, _proof
        in decode_certified_output(output, vertices, rows)
    )


def test_all_modes_match_exhaustive_small_oracle(twosat_runtime):
    for vertices in range(1, 4):
        possible_edges = tuple(
            (left, right)
            for left in range(vertices)
            for right in range(left + 1, vertices)
        )
        for edge_bits in range(1 << len(possible_edges)):
            edges = tuple(
                edge for index, edge in enumerate(possible_edges)
                if edge_bits & (1 << index)
            )
            for masks in itertools.product((0b01, 0b10, 0b11), repeat=vertices):
                problem = ExactProblem.create(vertices, 2, edges, masks)
                rows = [()]
                for vertex in range(vertices):
                    rows.extend(
                        [
                            ((vertex, 0),),
                            ((vertex, 0b01),),
                            ((vertex, 0b10),),
                            ((vertex, 0b11),),
                        ]
                    )
                if vertices >= 2:
                    rows.extend(
                        [
                            ((0, 0b01), (1, 0b01)),
                            ((0, 0b10), (1, 0b10)),
                            ((0, 0b01), (1, 0b10)),
                            ((0, 0b10), (1, 0b01)),
                        ]
                    )
                frozen_rows = tuple(rows)
                packet = pack_session_packet(problem, frozen_rows)
                expected = tuple(
                    int(_brute(problem, row) is not None) for row in frozen_rows
                )
                for mode in ("raw", "scc", "auto"):
                    answer = twosat_runtime.solve_session_packet(
                        packet, mode=mode, max_bytes=64 << 20
                    )
                    assert answer.query_count == len(frozen_rows)
                    assert answer.sat_count + answer.unsat_count == len(frozen_rows)
                    assert verify_session_output(packet, answer.output)
                    assert _decision_codes(
                        answer.output, problem.n, len(frozen_rows)
                    ) == expected
                    assert answer.info["independent_implementation"] is True
                    assert answer.info["algorithm"] == (
                        "independent-static-scc-unit-closure-v1"
                    )


def test_seeded_four_vertex_panel(twosat_runtime):
    rng = random.Random(0x2A5A7)
    possible_edges = tuple(
        (left, right) for left in range(4) for right in range(left + 1, 4)
    )
    for _ in range(128):
        edges = tuple(edge for edge in possible_edges if rng.getrandbits(1))
        masks = tuple(rng.choice((0b01, 0b10, 0b11)) for _ in range(4))
        problem = ExactProblem.create(4, 2, edges, masks)
        rows = tuple(
            tuple((rng.randrange(4), rng.randrange(4)) for _ in range(rng.randrange(5)))
            for _ in range(17)
        )
        packet = pack_session_packet(problem, rows)
        expected = tuple(int(_brute(problem, row) is not None) for row in rows)
        for mode in ("raw", "scc", "auto"):
            answer = twosat_runtime.solve_session_packet(packet, mode=mode)
            assert verify_session_output(packet, answer.output)
            assert _decision_codes(answer.output, problem.n, len(rows)) == expected


def test_all_proof_kinds_and_tile64_are_exercised(twosat_runtime):
    mixed = ExactProblem.create(2, 2, ((0, 1),), (0b11, 0b11))
    mixed_rows = (
        (),
        ((0, 0b01),),
        ((0, 0b01), (1, 0b01)),
        ((0, 0),),
    )
    mixed_packet = pack_session_packet(mixed, mixed_rows)
    mixed_answer = twosat_runtime.solve_session_packet(mixed_packet, mode="scc")
    assert verify_session_output(mixed_packet, mixed_answer.output)
    decoded = decode_certified_output(mixed_answer.output, mixed.n, len(mixed_rows))
    assert tuple(code for code, _labels, _proof in decoded) == (1, 1, 0, 0)
    assert decode_proof(decoded[2][2], mixed.n)[0] == 2
    assert decode_proof(decoded[3][2], mixed.n)[0] == 1

    static = ExactProblem.create(2, 1, ((0, 1),), (0b1, 0b1))
    static_packet = pack_session_packet(static, ((),))
    static_answer = twosat_runtime.solve_session_packet(static_packet, mode="raw")
    assert verify_session_output(static_packet, static_answer.output)
    static_row = decode_certified_output(static_answer.output, static.n, 1)[0]
    assert static_row[0] == 0
    assert decode_proof(static_row[2], static.n)[0] == 3

    chain = ExactProblem.create(
        8,
        2,
        tuple((vertex, vertex + 1) for vertex in range(7)),
        (0b11,) * 8,
    )
    chain_rows = ((),) * 64
    chain_packet = pack_session_packet(chain, chain_rows)
    scc = twosat_runtime.solve_session_packet(chain_packet, mode="scc")
    raw = twosat_runtime.solve_session_packet(chain_packet, mode="raw")
    assert verify_session_output(chain_packet, scc.output)
    assert verify_session_output(chain_packet, raw.output)
    assert _decision_codes(scc.output, chain.n, len(chain_rows)) == (1,) * 64
    assert _decision_codes(raw.output, chain.n, len(chain_rows)) == (1,) * 64
    assert scc.info["bank_kernel"] == "tile64"
    assert scc.info["selected_mode"] == "scc"
    assert raw.info["bank_kernel"] == "serial"
    assert raw.info["selected_mode"] == "raw"


def test_receipt_and_packet_fail_closed(tmp_path, twosat_runtime):
    source_library = Path(twosat_runtime._module.__file__).resolve()
    copied = tmp_path / source_library.name
    copied.write_bytes(source_library.read_bytes())
    receipt = dict(twosat_runtime.build_receipt)
    receipt["source_sha256"] = "0" * 64
    (tmp_path / "twosat-build.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source hash"):
        TwoSatRuntime(copied)

    problem = ExactProblem.create(2, 2, ((0, 1),), (0b11, 0b11))
    packet = pack_session_packet(problem, ((),))
    with pytest.raises(ValueError, match="immutable"):
        twosat_runtime.solve_session_packet(bytearray(packet))
    with pytest.raises(ValueError, match="mode"):
        twosat_runtime.solve_session_packet(packet, mode="unknown")
    with pytest.raises(ValueError, match="at least 1 MiB"):
        twosat_runtime.solve_session_packet(packet, max_bytes=(1 << 20) - 1)
    damaged = b"FAIL" + packet[4:]
    with pytest.raises(ValueError, match="SBC1"):
        twosat_runtime.solve_session_packet(damaged)
