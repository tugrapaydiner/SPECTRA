"""Actual deductive-solving capability, checked against independent enumeration."""
import itertools
import json
import random

import pytest

from spectra.cnf import CNF, solve_indexed
from spectra.cnf.deductive import solve_deductive


def feasible(problem):
    return any(problem.satisfied(w) for w in itertools.product((False, True), repeat=problem.nvars))


def test_every_two_variable_binary_formula():
    possible = [(), (1,), (-1,), (2,), (-2,), (1, 2), (1, -2), (-1, 2), (-1, -2)]
    for mask in range(1 << len(possible)):
        problem = CNF(2, tuple(c for i, c in enumerate(possible) if mask & (1 << i)))
        result = solve_deductive(problem, max_flips=0)
        assert (result.status == "SAT_VERIFIED") == feasible(problem)
        assert result.unsatisfied == problem.violated(result.witness)
        assert result.flips == 0


def test_binary_graph_beyond_recursion_limit():
    n = 3000
    # A large equivalence ring has no initial units and forces one SCC per sign.
    problem = CNF(n, tuple(c for i in range(1, n+1) for c in
                          ((-i, i % n+1), (i, -(i % n+1)))))
    result = solve_deductive(problem, max_flips=0)
    assert result.strategy == "2sat" and problem.satisfied(result.witness)


def test_forced_mixed_formula_solved_without_random_flips():
    problem = CNF(5, ((1, 1), (-1, 2), (-1, -2, 3), (-3, 4), (-4, 5), (1, -1)))
    result = solve_deductive(problem, max_flips=0)
    assert result.status == "SAT_VERIFIED" and result.witness == (True,) * 5
    assert result.propagated_variables == 5 and result.flips == 0


def test_propagated_assignment_survives_residual_search():
    problem = CNF(5, ((1,), (-1, -2), (3, 4, 5), (-3, -4, -5)))
    result = solve_deductive(problem, seed=7, max_flips=256)
    assert result.strategy == "propagation+indexed"
    assert result.witness[:2] == (True, False) and problem.satisfied(result.witness)


def test_general_formulas_preserve_indexed_search_outcomes():
    from data.cnf import random_3sat
    for seed in range(12):
        problem, _ = random_3sat(32, 134, 202610030000+seed)
        for cap in (0, 128):
            a = solve_indexed(problem, seed=seed, max_flips=cap)
            b = solve_deductive(problem, seed=seed, max_flips=cap)
            for field in ("status", "witness", "unsatisfied", "flips", "queries", "path_sha256"):
                assert getattr(a, field) == getattr(b, field)
            assert b.strategy == "indexed"


def test_random_small_formulas_never_return_false_sat():
    rng = random.Random(202610030111)
    for _ in range(300):
        problem = CNF(4, tuple(tuple(rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
                                    for _ in range(rng.randrange(5)))
                               for _ in range(rng.randrange(12))))
        result = solve_deductive(problem, max_flips=256)
        assert (not result.unsatisfied) == (result.status == "SAT_VERIFIED")
        assert result.unsatisfied == problem.violated(result.witness)
        if result.status == "SAT_VERIFIED":
            assert feasible(problem)


@pytest.mark.parametrize("problem", [CNF(0, ()), CNF(0, ((),)), CNF(3, ((1, -1),)),
                                    CNF(2, ((1,), (-1,))), CNF(2, ((2, 2, 2),))])
def test_empty_tautological_duplicate_and_conflicting_cases(problem):
    result = solve_deductive(problem, max_flips=0)
    assert (result.status == "SAT_VERIFIED") == feasible(problem)
    assert result.record()["algorithm"] == "deduction_then_indexed_search"
    assert result.record()["learned"] is False


@pytest.mark.parametrize("settings", [{"seed": True}, {"seed": -1}, {"max_flips": True}, {"max_flips": -1}])
def test_invalid_search_settings(settings):
    with pytest.raises(ValueError):
        solve_deductive(CNF(1, ((1,),)), **settings)


def test_public_cli_deductive_backend(tmp_path, capsys):
    from spectra.cli import main
    source, output = tmp_path / "forced.cnf", tmp_path / "answer.json"
    source.write_text("p cnf 3 3\n1 0\n-1 2 0\n-2 3 0\n")
    assert main(["cnf", "solve", str(source), "--backend", "deductive", "--max-flips", "0", "--out", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "SAT_VERIFIED"
    assert main(["cnf", "check", str(source), str(output)]) == 0
