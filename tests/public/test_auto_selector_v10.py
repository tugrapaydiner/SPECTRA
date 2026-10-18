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
    return ExactRuntime(build_exact_runtime(tmp_path_factory.mktemp("selector-v10-runtime")))


def solve(runtime, problem, rows):
    packet = pack_session_packet(problem, rows)
    answer = runtime.solve_session_packet(packet, mode="auto")
    assert verify_session_output(packet, answer.output)
    assert answer.info["auto_policy"] == "structural-v10"
    assert answer.info["mode"] == "auto"
    return answer


def path_problem(n: int = 8) -> ExactProblem:
    return ExactProblem.create(
        n,
        2,
        tuple((vertex, vertex + 1) for vertex in range(n - 1)),
        (3,) * n,
    )


def independent_problem(n: int) -> ExactProblem:
    return ExactProblem.create(n, 2, (), (3,) * n)


def dense_uncompressed_problem(n: int = 64) -> ExactProblem:
    if n % 2:
        raise ValueError("even vertex count required")
    half = n // 2
    edges = {
        (left, half + ((left * 7 + offset) % half))
        for left in range(half)
        for offset in range(8)
    }
    masks = tuple(5 if vertex < half else 10 for vertex in range(n))
    return ExactProblem.create(n, 4, tuple(sorted(edges)), masks)


def test_large_tiny_bank_uses_the_native_raw_path(exact_runtime):
    problem = independent_problem(1024)
    rows = ((),) * 8
    packet = pack_session_packet(problem, rows)
    automatic = exact_runtime.solve_session_packet(packet, mode="auto")
    explicit = exact_runtime.solve_session_packet(packet, mode="raw")
    assert automatic.output == explicit.output
    assert automatic.info["mode"] == "auto"
    assert automatic.info["auto_policy"] == "structural-v10"
    assert automatic.info["selected_mode"] == "raw"
    assert automatic.info["auto_policy_reason"] == "large_tiny_bank"
    assert verify_session_output(packet, automatic.output)


def test_large_graph_can_still_use_scc_after_eight_requests(exact_runtime):
    answer = solve(exact_runtime, independent_problem(1024), ((),) * 9)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "short_weak_compression"


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
    assert answer.info["auto_policy_reason"] == "scc_long_bank"


def test_high_absolute_but_low_density_pins_use_scc(exact_runtime):
    problem = independent_problem(1024)
    row = tuple((vertex, 3) for vertex in range(16))
    answer = solve(exact_runtime, problem, (row,) * 64)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"


def test_pin_density_below_one_in_24_uses_scc(exact_runtime):
    problem = independent_problem(200)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 64)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"


def test_dense_long_uncompressed_bank_below_96_uses_scc(exact_runtime):
    problem = independent_problem(95)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 1024)
    assert answer.info["components"] == 2 * problem.n
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"


def test_dense_small_compressed_bank_stays_raw(exact_runtime):
    problem = path_problem(78)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 1024)
    assert answer.info["components"] < 2 * problem.n
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "dense_small_compressed"


def test_dense_long_bank_switches_to_scc_at_96_vertices(exact_runtime):
    problem = independent_problem(96)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 64)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"


def test_dense_long_bank_at_120_vertices_uses_scc(exact_runtime):
    problem = independent_problem(120)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 1024)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"


def test_dense_medium_long_bank_uses_scc(exact_runtime):
    problem = independent_problem(192)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 1024)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"

def test_large_high_pin_bank_uses_scc_at_tile64(exact_runtime):
    problem = independent_problem(448)
    row = tuple((vertex, 3) for vertex in range(19))
    answer = solve(exact_runtime, problem, (row,) * 64)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"


def test_dense_uncompressed_graph_uses_raw_at_tile64(exact_runtime):
    problem = dense_uncompressed_problem()
    answer = solve(exact_runtime, problem, ((),) * 64)
    assert answer.info["components"] == 2 * problem.n
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "dense_uncompressed_tile64"


def test_dense_uncompressed_graph_uses_scc_after_tile64(exact_runtime):
    problem = dense_uncompressed_problem()
    answer = solve(exact_runtime, problem, ((),) * 128)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"


def test_auto_keeps_a_base_contradiction_on_raw(exact_runtime):
    problem = ExactProblem.create(3, 2, ((0, 1), (0, 2), (1, 2)), (3, 3, 3))
    answer = solve(exact_runtime, problem, ((),) * 64)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "base_infeasible"
    assert answer.unsat_count == 64
