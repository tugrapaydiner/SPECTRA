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
    pack_session_packet,
    verify_session_output,
)


@pytest.fixture(scope="session")
def exact_runtime(tmp_path_factory):
    supplied = os.environ.get("SPECTRA_EXACT_LIBRARY")
    if supplied is not None:
        return ExactRuntime(Path(supplied).resolve())
    compiler = "cl" if sys.platform == "win32" else "g++"
    if sys.platform not in ("linux", "win32") or shutil.which(compiler) is None:
        pytest.skip("supported native compiler is unavailable")
    return ExactRuntime(build_exact_runtime(tmp_path_factory.mktemp("selector-v4-runtime")))


def solve(runtime, problem, rows):
    packet = pack_session_packet(problem, rows)
    answer = runtime.solve_session_packet(packet, mode="auto")
    assert verify_session_output(packet, answer.output)
    assert answer.info["auto_policy"] == "structural-v4"
    return answer


def path_problem(n: int = 8) -> ExactProblem:
    return ExactProblem.create(
        n, 2,
        tuple((vertex, vertex + 1) for vertex in range(n - 1)),
        (3,) * n,
    )


def independent_problem(n: int) -> ExactProblem:
    return ExactProblem.create(n, 2, (), (3,) * n)


def test_auto_keeps_a_tiny_bank_on_raw(exact_runtime):
    answer = solve(exact_runtime, path_problem(), ((),) * 7)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "short_weak_compression"


def test_auto_uses_scc_early_when_compression_is_strong(exact_runtime):
    answer = solve(exact_runtime, path_problem(), ((),) * 8)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_strong_compression"


def test_auto_keeps_a_short_weakly_compressed_bank_on_raw(exact_runtime):
    answer = solve(exact_runtime, independent_problem(128), ((),) * 32)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "short_weak_compression"


def test_auto_uses_tile64_scc_for_a_long_feasible_bank(exact_runtime):
    answer = solve(exact_runtime, independent_problem(128), ((),) * 64)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_tile64"


def test_high_absolute_but_low_density_pins_use_scc(exact_runtime):
    problem = independent_problem(1024)
    row = tuple((vertex, 3) for vertex in range(16))
    answer = solve(exact_runtime, problem, (row,) * 64)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_tile64"


def test_high_graph_relative_pin_density_uses_raw(exact_runtime):
    problem = independent_problem(128)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 64)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "dense_pin_fraction"


def test_auto_keeps_a_base_contradiction_on_raw(exact_runtime):
    problem = ExactProblem.create(3, 2, ((0, 1), (0, 2), (1, 2)), (3, 3, 3))
    answer = solve(exact_runtime, problem, ((),) * 64)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "base_infeasible"
    assert answer.unsat_count == 64
