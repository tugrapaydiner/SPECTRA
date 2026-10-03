"""Run with the clean installed wheel's Python and -I; no checkout imports.

Example after check_current_installation.py:
  install/venv/bin/python -I scripts/check_focused_installation.py
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    with tempfile.TemporaryDirectory(prefix="spectra-focused-installed-") as temporary:
        os.chdir(temporary)
        import spectra.cnf.focused
        from spectra.cnf import CNF, solve_focused

        require(Path(spectra.cnf.focused.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),
                "focused solver must come from the installed environment")
        for dependency in ("torch", "numpy", "pytest", "pysat"):
            require(importlib.util.find_spec(dependency) is None, f"unexpected dependency: {dependency}")

        clauses = [(), (1,), (-1,), (2,), (-2,), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        for mask in range(1 << len(clauses)):
            problem = CNF(2, tuple(c for i, c in enumerate(clauses) if mask & (1 << i)))
            result = solve_focused(problem, seed=42, max_flips=64, restart_interval=7)
            bad = tuple(i for i, c in enumerate(problem.clauses)
                        if not any(result.witness[abs(lit)-1] == (lit > 0) for lit in c))
            require(len(result.witness) == 2 and all(type(v) is bool for v in result.witness),
                    "malformed witness")
            require(result.unsatisfied == bad, "original-clause mismatch")
            require((result.status == "SAT_VERIFIED") == (not bad), "incorrect status")
            require(0 <= result.flips <= 64 and 0 <= result.restarts <= 9, "budget exceeded")
            require(result.policy == "novelty_break", "unexpected default policy")

        Path("input.cnf").write_text("p cnf 2 2\n1 2 0\n-1 2 0\n", encoding="ascii")
        commands = [
            ["cnf", "solve", "input.cnf", "--backend", "focused", "--seed", "7", "--out", "answer.json"],
            ["cnf", "check", "input.cnf", "answer.json"],
        ]
        for arguments in commands:
            call = subprocess.run([sys.executable, "-I", "-m", "spectra", *arguments],
                                  text=True, capture_output=True, timeout=120)
            require(call.returncode == 0, f"installed command failed: {call.stderr}")
        record = json.loads(Path("answer.json").read_text())
        require(record["algorithm"] == "dense_focused_search" and record["learned"] is False,
                "unexpected CLI solver")
        print(json.dumps({"status": "PASS", "exhaustive_formula_checks": 512,
                          "cli_checks": 2, "optional_dependencies": False,
                          "scope": "actual installed wheel; isolated Python; outside checkout"}, indent=2))


if __name__ == "__main__":
    main()
