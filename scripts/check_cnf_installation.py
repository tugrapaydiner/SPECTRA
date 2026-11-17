"""Exercise the installed CNF solvers from outside the source checkout.

Run with the clean installed wheel's Python and ``-I`` so no checkout module can
satisfy an import, for example after ``check_current_installation.py``::

    install/venv/bin/python -I scripts/check_cnf_installation.py

Every two-variable formula over nine candidate clauses (512 formulas) is solved
by the focused local search and by DPLL and compared with a truth table; the
focused backend is also driven through the installed ``spectra`` CLI.
The installed native CDCL source is compiled in a child process and both public
solver paths are checked for SAT, UNSAT, and UNKNOWN. A separate child replays
the retained original-CNF proof after the native build has been removed.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
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
    import spectra.sat.binary_core
    import spectra.sat.native_binary_core
    import spectra.sat.native_cdcl
    prefix = Path(sys.prefix).resolve()
    for module in (spectra.cnf.dpll, spectra.cnf.focused,
                   spectra.sat.binary_core, spectra.sat.native_binary_core,
                   spectra.sat.native_cdcl):
        require(Path(module.__file__).resolve().is_relative_to(prefix),
                f"{module.__name__} must come from the installed environment")
    for module in (spectra.sat.native_binary_core, spectra.sat.native_cdcl):
        source = module.SOURCE.resolve()
        require(source.is_relative_to(prefix) and source.is_file(),
                f"installed wheel must include {module.__name__} source")
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


def check_native() -> dict:
    def isolated(code: str) -> dict:
        call = subprocess.run([sys.executable, "-I", "-c", code],
                              text=True, capture_output=True, timeout=360)
        require(call.returncode == 0,
                f"installed native check failed:\n{call.stdout}\n{call.stderr}")
        return json.loads(call.stdout)

    # The child owns the DLL lifetime, so cleanup also works on Windows.
    report = isolated('''
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
from spectra.sat.binary_core import (
    build_binary_core, canonical_cnf, model_satisfies, verify_binary_core,
)
from spectra.sat.native_binary_core import (
    NativeBinaryCore, assert_matches_python, build_native_binary_core,
)
from spectra.sat.native_cdcl import (
    NativeCDCLRuntime, build_native_cdcl, solve_cnf_native, solve_general_native,
    verify_direct_native_result, verify_native_result,
)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

runtime = NativeCDCLRuntime(build_native_cdcl("native-cdcl"))
sat_clauses = ((1, -2), (-1, 2), (2, 3, 4), (-3, 4), (-4, 5, 6))
unsat_clauses = tuple(tuple(sign * variable for variable, sign in enumerate(signs, 1))
                      for signs in product((-1, 1), repeat=3))
report = {}
core = NativeBinaryCore(build_native_binary_core("native-binary-core"))
cases = ((0, ()), (0, ((),)), (1, ((1,), (-1,))),
         (6, sat_clauses), (3, unsat_clauses))
for nvars, clauses in cases:
    reference = build_binary_core(nvars, clauses)
    require(verify_binary_core(nvars, clauses, reference), "binary-core reference replay failed")
    prepared = core.prepare_canonical(nvars, canonical_cnf(nvars, clauses))
    for reduced in (core.reduce(nvars, clauses), core.reduce_prepared(prepared)):
        assert_matches_python(nvars, clauses, reduced, reference)
        if not reduced.contradiction:
            for witness in product((False, True), repeat=reduced.reduced_nvars):
                if model_satisfies(reduced.reduced_nvars, reduced.residual_clauses, witness):
                    require(model_satisfies(nvars, clauses, reduced.lift(witness)),
                            "native binary-core model lift failed")
report["native_binary_core"] = {"formulas": len(cases), "paths": 2}
for name, solve, verify in (
    ("direct", solve_cnf_native, verify_direct_native_result),
    ("binary_core", solve_general_native, verify_native_result),
):
    sat = solve(runtime, 6, sat_clauses, max_conflicts=64, max_learned_literals=1024)
    require(sat.status == "SAT_VERIFIED" and sat.reason == "model", name + " SAT failed")
    require(model_satisfies(6, sat_clauses, sat.model) and verify(6, sat_clauses, sat),
            name + " original-CNF model replay failed")
    unsat = solve(runtime, 3, unsat_clauses, max_conflicts=64, max_learned_literals=1024)
    require(unsat.status == "UNSAT_VERIFIED" and unsat.reason == "drup",
            name + " UNSAT failed")
    require(unsat.proof is not None and len(unsat.proof.additions) > 1
            and unsat.proof.additions[-1] == () and verify(3, unsat_clauses, unsat),
            name + " nontrivial proof replay failed")
    unknown = solve(runtime, 3, unsat_clauses, max_conflicts=0, max_learned_literals=1024)
    require(unknown.status == "UNKNOWN" and unknown.reason == "conflict_budget"
            and unknown.model == () and unknown.proof is None
            and verify(3, unsat_clauses, unknown), name + " UNKNOWN budget check failed")
    report[name] = {"sat": 1, "unsat": 1, "unknown": 1,
                    "proof_clauses": unsat.proof_clauses}
    if name == "direct":
        Path("native-proof.json").write_text(json.dumps({
            "nvars": 3, "clauses": unsat_clauses, "proof": asdict(unsat.proof),
        }), encoding="utf-8")
print(json.dumps(report))
''')
    shutil.rmtree("native-cdcl")
    shutil.rmtree("native-binary-core")
    report["offline_replay"] = isolated('''
import ctypes
import json
from pathlib import Path
import sys

def reject_native_load(event, arguments):
    if event == "ctypes.dlopen":
        raise RuntimeError("offline proof replay attempted to load native code")

sys.addaudithook(reject_native_load)
from spectra.sat.native_cdcl import DRUPProof, verify_drup_proof

record = json.loads(Path("native-proof.json").read_text(encoding="utf-8"))
proof = DRUPProof(record["proof"]["schema"],
                  tuple(tuple(clause) for clause in record["proof"]["additions"]))
if not verify_drup_proof(record["nvars"], record["clauses"], proof):
    raise RuntimeError("standalone original-CNF proof replay failed")
if verify_drup_proof(record["nvars"], record["clauses"], DRUPProof(proof.schema, ((),))):
    raise RuntimeError("standalone replay accepted an invalid proof")
print(json.dumps({"verified": True, "rejected_invalid_proof": True,
                  "native_execution": False}))
''')
    return report


def main() -> None:
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="spectra-cnf-installed-") as temporary:
        try:
            os.chdir(temporary)
            check_installed_origin()
            report = {
                "status": "PASS",
                "focused_formulas": check_focused(),
                "dpll_formulas": check_dpll(),
                "cli_checks": check_cli(),
                "native_cdcl": check_native(),
                "optional_dependencies": False,
                "scope": "actual installed wheel; isolated Python; outside checkout",
            }
        finally:
            os.chdir(previous)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
