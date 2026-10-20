from __future__ import annotations

import random

from pysat.solvers import Solver

from experiments.general_competition.binary_core import model_satisfies
from experiments.general_competition.residual_solver import (
    solve_general,
    verify_general_result,
)


ORACLES = ("cadical195", "glucose42")


def random_formula(seed: int):
    rng = random.Random(seed)
    nvars = rng.randint(1, 8)
    clauses = []
    for _ in range(rng.randint(0, 28)):
        row = []
        for _ in range(rng.randint(0, 6)):
            variable = rng.randint(1, nvars)
            row.append(variable if rng.getrandbits(1) else -variable)
        clauses.append(tuple(row))
    return nvars, tuple(clauses)


def oracle_status(name: str, clauses) -> bool:
    with Solver(name=name, bootstrap_with=[list(row) for row in clauses]) as solver:
        return bool(solver.solve())


def test_random_general_cnf_matches_two_independent_solver_families():
    for seed in range(256):
        nvars, clauses = random_formula(seed)
        expected = tuple(oracle_status(name, clauses) for name in ORACLES)
        assert expected[0] == expected[1], (seed, nvars, clauses, expected)

        result = solve_general(
            nvars,
            clauses,
            max_decisions=100_000,
            max_propagations=1_000_000,
            max_proof_nodes=200_000,
            max_depth=64,
        )
        assert result.status != "UNKNOWN", (seed, result.reason)
        assert (result.status == "SAT_VERIFIED") == expected[0]
        assert verify_general_result(nvars, clauses, result)
        if result.status == "SAT_VERIFIED":
            assert model_satisfies(nvars, clauses, result.model)
