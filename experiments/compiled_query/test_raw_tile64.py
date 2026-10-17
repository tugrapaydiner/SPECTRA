"""Direct contracts for the 64-lane raw implication kernel.

The existing serial ``raw`` mode remains the control. ``raw-tile`` must preserve
its complete SCQ2 bytes while sharing closure work across request lanes.
"""
from __future__ import annotations

import os
from pathlib import Path
import random

import pytest

from .binary_list import BinaryListProblem
from .exact import ExactRuntime, build_exact_runtime, pack_query_bank, pack_session_packet
from .proof import verify_answer
from .sessions import decode_certified_output


@pytest.fixture(scope="module")
def exact_runtime(tmp_path_factory):
    explicit = os.environ.get("SPECTRA_EXACT_LIBRARY")
    library = Path(explicit) if explicit else build_exact_runtime(
        tmp_path_factory.mktemp("raw-tile64-native")
    )
    return ExactRuntime(library)


def _rows(problem, count: int, seed: int, width: int = 10):
    rng = random.Random(seed)
    choices = problem.colors()
    rows = [()]
    for index in range(1, count):
        vertices = rng.sample(range(problem.n), min(width, problem.n))
        pins = [(vertex, 1 << rng.choice(choices[vertex])) for vertex in vertices]
        if index % 11 == 0:
            vertex, allowed = pins[0]
            pins.append((vertex, problem.masks[vertex] ^ allowed))
        if index % 13 == 0:
            pins.reverse()
        rows.append(tuple(pins))
    return tuple(rows)


def _problems():
    n = 96
    yield BinaryListProblem.create(
        n, 2, tuple((v, v + 1) for v in range(n - 1)), (3,) * n
    )
    yield BinaryListProblem.create(
        n, 3, tuple((v, v + 1) for v in range(n - 1)),
        tuple((3, 6, 5)[v % 3] for v in range(n)),
    )
    yield BinaryListProblem.create(
        n, 2,
        tuple((left, right) for left in range(n // 2)
              for right in range(n // 2, n)),
        (3,) * n,
    )


@pytest.mark.parametrize("count", (64, 65, 129))
def test_raw_tile64_is_byte_identical_to_serial_raw(exact_runtime, count):
    for case_index, problem in enumerate(_problems()):
        rows = _rows(problem, count, 9107 + case_index)
        packet = pack_session_packet(problem, rows)
        serial = exact_runtime.solve_session_packet(packet, mode="raw")
        tiled = exact_runtime.solve_session_packet(packet, mode="raw-tile")
        assert serial.info["bank_kernel"] == "serial"
        assert serial.info["selected_mode"] == "raw"
        assert tiled.info["bank_kernel"] == "tile64"
        assert tiled.info["selected_mode"] == "raw-tile"
        assert tiled.info["bank_tiles"] == (count + 63) // 64
        assert tiled.output == serial.output
        assert (tiled.sat_count, tiled.unsat_count, tiled.proof_bytes) == (
            serial.sat_count, serial.unsat_count, serial.proof_bytes
        )
        decoded = decode_certified_output(tiled.output, problem.n, count)
        for query, (decision, labels, proof) in zip(rows, decoded):
            assert verify_answer(
                problem,
                query,
                "SAT_VERIFIED" if decision else "UNSAT_VERIFIED",
                labels if decision else b"",
                proof,
            )


def test_raw_tile64_persistent_bank_matches_serial_control(exact_runtime):
    problem = next(iter(_problems()))
    rows = _rows(problem, 129, 17117)
    packet = pack_query_bank(problem, rows)
    with exact_runtime.prepare(problem, mode="raw") as serial, \
            exact_runtime.prepare(problem, mode="raw-tile") as tiled:
        serial_answer = serial.solve_bank(packet)
        tiled_answer = tiled.solve_bank(packet)
        assert tiled_answer.output == serial_answer.output
        assert tiled.check_bank(packet, tiled_answer.output)
        assert serial.check_bank(packet, tiled_answer.output)


def test_raw_tile64_threshold_and_base_contradiction_fall_back(exact_runtime):
    path = BinaryListProblem.create(
        12, 2, tuple((v, v + 1) for v in range(11)), (3,) * 12
    )
    for count, kernel, tiles in ((0, "serial", 0), (63, "serial", 0),
                                 (64, "tile64", 1), (129, "tile64", 3)):
        result = exact_runtime.solve_session_packet(
            pack_session_packet(path, ((),) * count), mode="raw-tile"
        )
        assert result.info["bank_kernel"] == kernel
        assert result.info["bank_tiles"] == tiles

    odd = BinaryListProblem.create(
        9, 2, tuple((v, v + 1) for v in range(8)) + ((0, 8),), (3,) * 9
    )
    packet = pack_session_packet(odd, ((),) * 129)
    serial = exact_runtime.solve_session_packet(packet, mode="raw")
    tiled = exact_runtime.solve_session_packet(packet, mode="raw-tile")
    assert tiled.info["bank_kernel"] == "serial"
    assert tiled.output == serial.output


def test_auto_high_pin_bank_uses_raw_tile64_and_keeps_control_bytes(exact_runtime):
    problem = BinaryListProblem.create(
        128, 2, tuple((v, v + 1) for v in range(127)), (3,) * 128
    )
    rows = _rows(problem, 129, 92117, width=12)
    packet = pack_session_packet(problem, rows)
    auto = exact_runtime.solve_session_packet(packet, mode="auto")
    tiled = exact_runtime.solve_session_packet(packet, mode="raw-tile")
    serial = exact_runtime.solve_session_packet(packet, mode="raw")
    assert auto.info["selected_mode"] == "raw-tile"
    assert auto.info["auto_policy_reason"] == "high_pin_density_raw_tile64"
    assert auto.info["bank_kernel"] == "tile64"
    assert auto.output == tiled.output == serial.output
