"""Independent direct 2-SAT challenge contracts.

These tests deliberately compare through public SBC1/SCQ2 bytes and the
standard-library replay verifier. The challenger does not import SPECTRA's native
solver implementation.
"""
from __future__ import annotations

import itertools
import json
import os
import random

import pytest

from spectra.exact import (
    ExactProblem,
    ExactRuntime,
    build_exact_runtime,
    decode_certified_output,
    pack_session_packet,
    verify_session_output,
)
from .direct_twosat import Direct2SATRuntime, build_direct_twosat_runtime


@pytest.fixture(scope="module")
def direct_runtime(tmp_path_factory):
    sanitizer = os.environ.get("DIRECT_2SAT_SANITIZER", "none")
    library = build_direct_twosat_runtime(
        tmp_path_factory.mktemp("direct-twosat"), sanitizer=sanitizer
    )
    return Direct2SATRuntime(library)


@pytest.fixture(scope="module")
def spectra_runtime(tmp_path_factory):
    library = build_exact_runtime(tmp_path_factory.mktemp("spectra-exact"))
    return ExactRuntime(library)


def _models(problem: ExactProblem, query):
    domains = [
        tuple(colour for colour in range(problem.palette) if (mask >> colour) & 1)
        for mask in problem.masks
    ]
    effective = list(problem.masks)
    for vertex, allowed in query:
        effective[vertex] &= allowed
    return {
        bytes(candidate)
        for candidate in itertools.product(*domains)
        if all((effective[v] >> candidate[v]) & 1 for v in range(problem.n))
        and all(candidate[u] != candidate[v] for u, v in problem.edges)
    }


def _statuses(output: bytes, n: int, count: int) -> tuple[int, ...]:
    return tuple(
        code for code, _labels, _proof
        in decode_certified_output(output, n, count)
    )


@pytest.mark.parametrize("engine", ("auto", "dag", "tile"))
def test_mixed_bank_covers_sat_and_all_three_proof_kinds(
    direct_runtime, engine
):
    mixed = ExactProblem.create(2, 2, ((0, 1),), (0b11, 0b11))
    rows = ((), ((0, 0b01),), ((0, 0b01), (1, 0b01)), ((0, 0),))
    packet = pack_session_packet(mixed, rows)
    result = direct_runtime.solve_session_packet(packet, engine=engine)
    expected_engine = {"auto": "closure", "dag": "dag", "tile": "tile"}[engine]
    assert result.info["engine"] == expected_engine
    assert (result.info["sat"], result.info["unsat"]) == (2, 2)
    assert verify_session_output(packet, result.output)
    decoded = decode_certified_output(result.output, mixed.n, len(rows))
    assert tuple(code for code, _labels, _proof in decoded) == (1, 1, 0, 0)

    static = ExactProblem.create(2, 1, ((0, 1),), (0b1, 0b1))
    static_packet = pack_session_packet(static, ((),))
    static_result = direct_runtime.solve_session_packet(static_packet, engine=engine)
    assert verify_session_output(static_packet, static_result.output)
    proof_kinds = []
    for output, n, count in (
        (result.output, mixed.n, len(rows)),
        (static_result.output, static.n, 1),
    ):
        for code, _labels, proof in decode_certified_output(output, n, count):
            if not code:
                proof_kinds.append(int.from_bytes(proof[4:8], "little"))
    assert sorted(set(proof_kinds)) == [1, 2, 3]


@pytest.mark.parametrize("engine", ("auto", "dag", "tile"))
def test_exhaustive_three_vertex_formulas_and_requests(direct_runtime, engine):
    edge_bank = ((0, 1), (0, 2), (1, 2))
    domains = (1, 2, 3, 4, 5, 6)
    for graph_bits in range(8):
        edges = tuple(
            edge_bank[index] for index in range(3) if graph_bits >> index & 1
        )
        for masks in itertools.product(domains, repeat=3):
            problem = ExactProblem.create(3, 3, edges, masks)
            subsets = [
                tuple(value for value in range(8) if value & ~mask == 0)
                for mask in masks
            ]
            rows = tuple(
                tuple(enumerate(request))
                for request in itertools.product(*subsets)
            )
            packet = pack_session_packet(problem, rows)
            result = direct_runtime.solve_session_packet(packet, engine=engine)
            assert verify_session_output(packet, result.output)
            expected = tuple(1 if _models(problem, row) else 0 for row in rows)
            assert _statuses(result.output, problem.n, len(rows)) == expected


@pytest.mark.parametrize("engine", ("auto", "dag", "tile"))
def test_random_banks_match_bruteforce_and_spectra(
    direct_runtime, spectra_runtime, engine
):
    rng = random.Random(20261019)
    for _ in range(160):
        n = rng.randint(0, 8)
        palette = rng.randint(1, 5)
        masks = tuple(
            sum(
                1 << colour
                for colour in rng.sample(
                    range(palette), rng.randint(1, min(2, palette))
                )
            )
            for _ in range(n)
        )
        edges = tuple(
            (u, v)
            for u in range(n)
            for v in range(u + 1, n)
            if rng.random() < 0.28
        )
        problem = ExactProblem.create(n, palette, edges, masks)
        rows = []
        for _query in range(rng.randint(1, 18)):
            pins = []
            for _pin in range(rng.randint(0, 5)):
                if not n:
                    break
                pins.append((rng.randrange(n), rng.randrange(1 << palette)))
            rows.append(tuple(pins))
        rows = tuple(rows)
        packet = pack_session_packet(problem, rows)
        direct = direct_runtime.solve_session_packet(packet, engine=engine)
        spectra = spectra_runtime.solve_session_packet(packet, mode="raw")
        assert verify_session_output(packet, direct.output)
        assert verify_session_output(packet, spectra.output)
        expected = tuple(1 if _models(problem, row) else 0 for row in rows)
        assert _statuses(direct.output, n, len(rows)) == expected
        assert _statuses(spectra.output, n, len(rows)) == expected


def test_explicit_engines_and_memory_fallback(direct_runtime):
    problem = ExactProblem.create(256, 2, (), (0b11,) * 256)
    packet = pack_session_packet(problem, ((),))

    for engine in ("dag", "tile"):
        explicit = direct_runtime.solve_session_packet(packet, engine=engine)
        assert explicit.info["requested_engine"] == engine
        assert explicit.info["engine"] == engine
        assert explicit.info["closure_bytes"] == 0
        assert verify_session_output(packet, explicit.output)

    fallback = direct_runtime.solve_session_packet(packet, max_bytes=64 << 10)
    assert fallback.info["requested_engine"] == "auto"
    assert fallback.info["engine"] == "dag"
    assert fallback.info["closure_bytes"] == 0
    assert verify_session_output(packet, fallback.output)


def test_receipt_and_packet_tampering_fail_closed(tmp_path, direct_runtime):
    copied = tmp_path / direct_runtime.path.name
    copied.write_bytes(direct_runtime.path.read_bytes())
    receipt = json.loads(direct_runtime.path.with_name("build.json").read_text())
    receipt["source_sha256"] = "0" * 64
    copied.with_name("build.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        Direct2SATRuntime(copied)

    problem = ExactProblem.create(2, 2, ((0, 1),), (3, 3))
    packet = bytearray(pack_session_packet(problem, ((),)))
    packet[0] ^= 1
    with pytest.raises(ValueError, match="SBC1"):
        direct_runtime.solve_session_packet(bytes(packet))
