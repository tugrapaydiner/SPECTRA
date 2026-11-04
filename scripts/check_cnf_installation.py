"""Exercise the installed CNF solvers from outside the source checkout.

Run with the clean installed wheel's Python and ``-I`` so no checkout module can
satisfy an import, for example after ``check_current_installation.py``::

    install/venv/bin/python -I scripts/check_cnf_installation.py

Every two-variable formula over nine candidate clauses (512 formulas) is solved
by the focused local search and by DPLL and compared with a truth table; the
focused backend is also driven through the installed ``spectra`` CLI.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

CANDIDATE_CLAUSES = ((), (1,), (-1,), (2,), (-2,), (1, 2), (1, -2), (-1, 2), (-1, -2))
OPTIONAL_DEPENDENCIES = ("torch", "numpy", "pytest", "pysat")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def satisfies(clauses, witness) -> bool:
    return all(any(witness[abs(literal) - 1] == (literal > 0) for literal in clause)
               for clause in clauses)


def all_formulas():
    from spectra.cnf import CNF
    for mask in range(1 << len(CANDIDATE_CLAUSES)):
        yield CNF(2, tuple(c for i, c in enumerate(CANDIDATE_CLAUSES) if mask & (1 << i)))


def check_installed_origin() -> None:
    import spectra.cnf.dpll
    import spectra.cnf.focused
    prefix = Path(sys.prefix).resolve()
    for module in (spectra.cnf.dpll, spectra.cnf.focused):
        require(Path(module.__file__).resolve().is_relative_to(prefix),
                f"{module.__name__} must come from the installed environment")
    for dependency in OPTIONAL_DEPENDENCIES:
        require(importlib.util.find_spec(dependency) is None,
                f"unexpected dependency: {dependency}")


def check_focused() -> int:
    from spectra.cnf import solve_focused
    checked = 0
    for problem in all_formulas():
        result = solve_focused(problem, seed=42, max_flips=64, restart_interval=7)
        bad = tuple(i for i, clause in enumerate(problem.clauses)
                    if not satisfies((clause,), result.witness))
        require(len(result.witness) == 2 and all(type(v) is bool for v in result.witness),
                "malformed focused witness")
        require(result.unsatisfied == bad, "focused original-clause mismatch")
        require((result.status == "SAT_VERIFIED") == (not bad), "incorrect focused status")
        require(0 <= result.flips <= 64 and 0 <= result.restarts <= 9, "focused budget exceeded")
        require(result.policy == "novelty_break", "unexpected focused default policy")
        checked += 1
    return checked


def check_dpll() -> int:
    from spectra.cnf.dpll import solve_dpll
    checked = 0
    for problem in all_formulas():
        expected = any(satisfies(problem.clauses, witness)
                       for witness in itertools.product((False, True), repeat=2))
        result = solve_dpll(problem, max_decisions=32)
        require((result.status == "SAT_VERIFIED") == expected, "DPLL truth-table mismatch")
        require(satisfies(problem.clauses, result.witness) == expected, "DPLL witness mismatch")
        require(result.decisions <= 32, "DPLL decision budget exceeded")
        require(json.loads(json.dumps(result.record())) == result.record(),
                "DPLL record is not JSON-stable")
        checked += 1
    return checked


def check_cli() -> int:
    Path("input.cnf").write_text("p cnf 2 2\n1 2 0\n-1 2 0\n", encoding="ascii")
    commands = (
        ("cnf", "solve", "input.cnf", "--backend", "focused", "--seed", "7", "--out", "answer.json"),
        ("cnf", "check", "input.cnf", "answer.json"),
    )
    for arguments in commands:
        call = subprocess.run([sys.executable, "-I", "-m", "spectra", *arguments],
                              text=True, capture_output=True, timeout=120)
        require(call.returncode == 0, f"installed command failed: {call.stderr}")
    record = json.loads(Path("answer.json").read_text(encoding="utf-8"))
    require(record["algorithm"] == "dense_focused_search" and record["learned"] is False,
            "unexpected CLI solver")
    return len(commands)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="spectra-cnf-installed-") as temporary:
        os.chdir(temporary)
        check_installed_origin()
        report = {
            "status": "PASS",
            "focused_formulas": check_focused(),
            "dpll_formulas": check_dpll(),
            "cli_checks": check_cli(),
            "optional_dependencies": False,
            "scope": "actual installed wheel; isolated Python; outside checkout",
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
