from __future__ import annotations

from itertools import product
import importlib.util
import os
from pathlib import Path

import pytest

requires_pysat = pytest.mark.skipif(
    importlib.util.find_spec("pysat") is None, reason="PySAT is not installed",
)

from experiments.general_competition.binary_core import model_satisfies
from experiments.general_competition.proof_backend import (
    check_drat,
    solve_original_with_drat,
    solve_safe_hybrid,
    write_dimacs,
    write_drat,
)


def unsat_cube():
    return tuple(
        tuple(
            variable if signs[variable - 1] else -variable
            for variable in range(1, 4)
        )
        for signs in product((False, True), repeat=3)
    )


@requires_pysat
def test_safe_hybrid_lifts_and_checks_sat_models():
    clauses = ((1, -2), (-1, 2), (2, 3, 4), (-3, 4))
    result = solve_safe_hybrid(4, clauses)
    assert result.status == "SAT_CHECKED"
    assert result.residual_status == "SAT_CHECKED"
    assert result.model_verified
    assert result.proof == ()
    assert result.proof_scope == "not-required-for-sat"
    assert model_satisfies(4, clauses, result.model)


@requires_pysat
def test_residual_unsat_is_confirmed_on_the_exact_original_formula():
    clauses = unsat_cube()
    result = solve_safe_hybrid(3, clauses)
    assert result.status == "UNSAT_WITH_DRAT"
    assert result.residual_status == "UNSAT_REPORTED"
    assert result.original_confirmation_ns > 0
    assert result.proof
    assert "0" in result.proof
    assert result.proof_scope == "exact-original-cnf"


@requires_pysat
def test_binary_core_unsat_still_gets_an_original_formula_proof():
    clauses = ((1,), (-1,))
    result = solve_safe_hybrid(1, clauses)
    assert result.status == "UNSAT_WITH_DRAT"
    assert result.residual_status == "SKIPPED_CORE_CONTRADICTION"
    assert result.proof_scope == "exact-original-cnf"
    assert "0" in result.proof


@requires_pysat
def test_original_backend_handles_the_valid_empty_clause():
    result = solve_original_with_drat(1, ((),))
    assert result.status == "UNSAT_WITH_DRAT"
    assert "0" in result.proof


@requires_pysat
def test_nonstandard_drat_inputs_fail_closed():
    with pytest.raises(ValueError, match="duplicate literals"):
        solve_original_with_drat(1, ((1, 1),))
    with pytest.raises(ValueError, match="tautological"):
        solve_original_with_drat(1, ((1, -1),))


@pytest.mark.skipif(
    not os.environ.get("DRAT_TRIM"),
    reason="pinned external DRAT-trim checker is not configured",
)
@requires_pysat
def test_retained_original_formula_proof_passes_drat_trim(tmp_path):
    clauses = unsat_cube()
    result = solve_safe_hybrid(3, clauses)
    formula = write_dimacs(tmp_path / "formula.cnf", 3, clauses)
    proof = write_drat(tmp_path / "proof.drat", result.proof)
    receipt = check_drat(os.environ["DRAT_TRIM"], formula, proof)
    assert receipt.verified, receipt.record()
    assert receipt.returncode == 0
    assert "VERIFIED" in (receipt.stdout + receipt.stderr).upper()


@pytest.mark.skipif(
    not os.environ.get("DRAT_TRIM"),
    reason="pinned external DRAT-trim checker is not configured",
)
@requires_pysat
def test_root_conflict_proof_passes_drat_trim(tmp_path):
    clauses = ((1,), (-1,))
    result = solve_safe_hybrid(1, clauses)
    formula = write_dimacs(tmp_path / "root-conflict.cnf", 1, clauses)
    proof = write_drat(tmp_path / "root-conflict.drat", result.proof)
    receipt = check_drat(os.environ["DRAT_TRIM"], formula, proof)
    assert receipt.verified, receipt.record()


def test_writers_refuse_to_replace_retained_evidence(tmp_path):
    formula = write_dimacs(tmp_path / "formula.cnf", 1, ((1,),))
    with pytest.raises(FileExistsError):
        write_dimacs(formula, 1, ((1,),))

    proof = write_drat(tmp_path / "proof.drat", ("0",))
    with pytest.raises(FileExistsError):
        write_drat(proof, ("0",))


def test_checker_timeout_must_be_positive(tmp_path):
    checker = Path(os.environ.get("DRAT_TRIM", "/bin/true"))
    with pytest.raises(ValueError, match="positive"):
        check_drat(checker, tmp_path / "missing.cnf", tmp_path / "missing.drat", timeout_seconds=0)


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "expected"),
    [
        (0, "s NOT VERIFIED\n", "", False),
        (0, "c proof not verified\n", "", False),
        (0, "c VERIFIED derivation: all lemmas preserve satisfiability\n"
            "s DERIVATION\n", "", False),
        (0, "c s VERIFIED\n", "", False),
        (0, "s VERIFIED extra\n", "", False),
        (0, "", "s VERIFIED\n", False),
        (0, "s verified\n", "", False),
        (1, "s VERIFIED\n", "", False),
        (-9, "s VERIFIED\n", "", False),
        (0, "", "", False),
        (0, "s VERIFIED\n", "", True),
        (0, "  s VERIFIED  \r\n", "", True),
    ],
    ids=["negative-status", "comment-only", "derivation-only", "commented-token",
         "extra-token", "stderr-only", "wrong-case", "nonzero-exit", "killed",
         "empty-output", "positive-status", "surrounding-whitespace"],
)
def test_checker_requires_exact_stdout_refutation_status(
    tmp_path, monkeypatch, returncode, stdout, stderr, expected,
):
    import subprocess
    from experiments.general_competition import proof_backend

    completed = subprocess.CompletedProcess([], returncode, stdout, stderr)
    monkeypatch.setattr(proof_backend.subprocess, "run", lambda *a, **kw: completed)
    receipt = check_drat(tmp_path / "checker", tmp_path / "input.cnf",
                         tmp_path / "proof.drat")
    assert receipt.verified is expected
    assert (receipt.stdout, receipt.stderr) == (stdout, stderr)


def test_checker_accepts_refutation_status_among_diagnostics(tmp_path, monkeypatch):
    import subprocess
    from experiments.general_competition import proof_backend

    stdout = ("c finished parsing\n\nc detected empty clause\n"
              "\ns VERIFIED\n\nc verification complete\n")
    completed = subprocess.CompletedProcess([], 0, stdout, "")
    monkeypatch.setattr(proof_backend.subprocess, "run", lambda *a, **kw: completed)
    receipt = check_drat(tmp_path / "checker", tmp_path / "input.cnf",
                         tmp_path / "proof.drat")
    assert receipt.verified
    assert receipt.stdout == stdout
    assert receipt.stderr == ""
