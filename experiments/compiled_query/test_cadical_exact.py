"""Correctness contracts for the official CaDiCaL proof-carrying adapter."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
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
from spectra.exact_proof import decode_proof

from .cadical_exact import (
    CADICAL_COMMIT,
    CadicalExactRuntime,
    build_cadical_exact_runtime,
)


@pytest.fixture(scope="module")
def cadical_runtime(tmp_path_factory):
    source = os.environ.get("CADICAL_SOURCE")
    if not source:
        raise RuntimeError("CADICAL_SOURCE must point to the pinned official checkout")
    sanitizer = os.environ.get("CADICAL_ADAPTER_SANITIZER", "none")
    library = build_cadical_exact_runtime(
        tmp_path_factory.mktemp("cadical-exact"),
        Path(source),
        sanitizer=sanitizer,
    )
    return CadicalExactRuntime(library)


@pytest.fixture(scope="module")
def spectra_runtime(tmp_path_factory):
    return ExactRuntime(build_exact_runtime(tmp_path_factory.mktemp("spectra-exact")))


def _models(problem: ExactProblem, query):
    domains = [
        tuple(colour for colour in range(problem.palette) if mask >> colour & 1)
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
        code
        for code, _labels, _proof
        in decode_certified_output(output, n, count)
    )


def test_official_source_identity_and_signature(cadical_runtime):
    assert cadical_runtime.receipt["cadical_commit"] == CADICAL_COMMIT
    assert cadical_runtime.receipt["cadical_repository"] == "arminbiere/cadical"
    assert "cadical" in cadical_runtime.signature.lower()


def test_mixed_bank_covers_sat_and_all_three_proof_kinds(cadical_runtime):
    mixed = ExactProblem.create(2, 2, ((0, 1),), (0b11, 0b11))
    rows = ((), ((0, 0b01),), ((0, 0b01), (1, 0b01)), ((0, 0),))
    packet = pack_session_packet(mixed, rows)
    result = cadical_runtime.solve_session_packet(packet)
    assert result.info["cadical_commit"] == CADICAL_COMMIT
    assert (result.info["sat"], result.info["unsat"]) == (2, 2)
    assert result.info["solves"] == 3
    assert verify_session_output(packet, result.output)
    decoded = decode_certified_output(result.output, mixed.n, len(rows))
    assert tuple(code for code, _labels, _proof in decoded) == (1, 1, 0, 0)

    static = ExactProblem.create(2, 1, ((0, 1),), (0b1, 0b1))
    static_packet = pack_session_packet(static, ((), ((0, 0),)))
    static_result = cadical_runtime.solve_session_packet(static_packet)
    assert verify_session_output(static_packet, static_result.output)
    assert static_result.info["solves"] == 1

    kinds = []
    for output, n, count in (
        (result.output, mixed.n, len(rows)),
        (static_result.output, static.n, 2),
    ):
        for code, _labels, proof in decode_certified_output(output, n, count):
            if not code:
                kinds.append(decode_proof(proof, n)[0])
    assert sorted(set(kinds)) == [1, 2, 3]


@pytest.mark.exhaustive
def test_every_three_vertex_formula_and_restriction(cadical_runtime):
    edge_bank = ((0, 1), (0, 2), (1, 2))
    domains = (1, 2, 3, 4, 5, 6)
    for graph_bits in range(8):
        edges = tuple(
            edge_bank[index]
            for index in range(3)
            if graph_bits >> index & 1
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
            answer = cadical_runtime.solve_session_packet(packet)
            assert verify_session_output(packet, answer.output)
            expected = tuple(1 if _models(problem, row) else 0 for row in rows)
            assert _statuses(answer.output, problem.n, len(rows)) == expected


def test_random_banks_match_bruteforce_and_spectra(
    cadical_runtime, spectra_runtime
):
    rng = random.Random(20261020)
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
            (left, right)
            for left in range(n)
            for right in range(left + 1, n)
            if rng.random() < 0.28
        )
        problem = ExactProblem.create(n, palette, edges, masks)
        rows = []
        for _query in range(rng.randint(1, 18)):
            pins = []
            for _pin in range(rng.randint(0, 6)):
                if not n:
                    break
                pins.append((rng.randrange(n), rng.randrange(1 << palette)))
            rows.append(tuple(pins))
        rows = tuple(rows)
        packet = pack_session_packet(problem, rows)
        cadical = cadical_runtime.solve_session_packet(packet)
        spectra = spectra_runtime.solve_session_packet(packet, mode="auto")
        assert verify_session_output(packet, cadical.output)
        assert verify_session_output(packet, spectra.output)
        expected = tuple(1 if _models(problem, row) else 0 for row in rows)
        assert _statuses(cadical.output, n, len(rows)) == expected
        assert _statuses(spectra.output, n, len(rows)) == expected


def test_declared_but_unused_variables_return_complete_colouring(cadical_runtime):
    problem = ExactProblem.create(64, 3, (), (0b101,) * 64)
    rows = ((), ((0, 0b001),), ((63, 0b100),))
    packet = pack_session_packet(problem, rows)
    result = cadical_runtime.solve_session_packet(packet)
    assert verify_session_output(packet, result.output)
    decoded = decode_certified_output(result.output, problem.n, len(rows))
    assert all(code == 1 and len(labels) == problem.n for code, labels, _ in decoded)


def test_receipt_packet_and_output_tampering_fail_closed(
    tmp_path, cadical_runtime
):
    copied = tmp_path / cadical_runtime.path.name
    copied.write_bytes(cadical_runtime.path.read_bytes())
    receipt = json.loads(
        cadical_runtime.path.with_name("build.json").read_text(encoding="utf-8")
    )
    receipt["cadical_commit"] = "0" * 40
    copied.with_name("build.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="external source identity"):
        CadicalExactRuntime(copied)

    problem = ExactProblem.create(2, 2, ((0, 1),), (3, 3))
    packet = bytearray(pack_session_packet(problem, ((), ((0, 1), (1, 1)))))
    packet[0] ^= 1
    with pytest.raises(ValueError, match="SBC1"):
        cadical_runtime.solve_session_packet(bytes(packet))

    clean = pack_session_packet(problem, ((), ((0, 1), (1, 1))))
    answer = cadical_runtime.solve_session_packet(clean)
    damaged = bytearray(answer.output)
    damaged[-1] ^= 1
    assert not verify_session_output(clean, bytes(damaged))
