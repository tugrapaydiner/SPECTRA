"""DIMACS/DRAT writers and the external DRAT-trim result parser."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from spectra.sat.drat import check_drat, write_dimacs, write_drat


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
    from spectra.sat import drat

    completed = subprocess.CompletedProcess([], returncode, stdout, stderr)
    monkeypatch.setattr(drat.subprocess, "run", lambda *a, **kw: completed)
    receipt = check_drat(tmp_path / "checker", tmp_path / "input.cnf",
                         tmp_path / "proof.drat")
    assert receipt.verified is expected
    assert (receipt.stdout, receipt.stderr) == (stdout, stderr)


def test_checker_accepts_refutation_status_among_diagnostics(tmp_path, monkeypatch):
    import subprocess
    from spectra.sat import drat

    stdout = ("c finished parsing\n\nc detected empty clause\n"
              "\ns VERIFIED\n\nc verification complete\n")
    completed = subprocess.CompletedProcess([], 0, stdout, "")
    monkeypatch.setattr(drat.subprocess, "run", lambda *a, **kw: completed)
    receipt = check_drat(tmp_path / "checker", tmp_path / "input.cnf",
                         tmp_path / "proof.drat")
    assert receipt.verified
    assert receipt.stdout == stdout
    assert receipt.stderr == ""
