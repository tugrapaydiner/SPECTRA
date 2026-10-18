from __future__ import annotations

from spectra.exact import ExactProblem, pack_session_packet, verify_session_output


def solve(runtime, problem, rows):
    packet = pack_session_packet(problem, rows)
    answer = runtime.solve_session_packet(packet, mode="auto")
    assert verify_session_output(packet, answer.output)
    assert answer.info["auto_policy"] == "structural-v3"
    return answer


def path_problem(n: int = 8) -> ExactProblem:
    return ExactProblem.create(
        n,
        2,
        tuple((vertex, vertex + 1) for vertex in range(n - 1)),
        (3,) * n,
    )


def independent_problem(n: int = 8) -> ExactProblem:
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
    answer = solve(exact_runtime, independent_problem(), ((),) * 32)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "short_weak_compression"


def test_auto_uses_tile64_scc_for_a_long_feasible_bank(exact_runtime):
    answer = solve(exact_runtime, independent_problem(), ((),) * 64)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_tile64"


def test_auto_keeps_a_high_pin_bank_on_raw(exact_runtime):
    problem = independent_problem()
    row = tuple((vertex, 3) for vertex in range(problem.n))
    answer = solve(exact_runtime, problem, (row,) * 64)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "high_pin_density"


def test_auto_keeps_a_base_contradiction_on_raw(exact_runtime):
    problem = ExactProblem.create(3, 2, ((0, 1), (0, 2), (1, 2)), (3, 3, 3))
    answer = solve(exact_runtime, problem, ((),) * 64)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "base_infeasible"
    assert answer.unsat_count == 64
