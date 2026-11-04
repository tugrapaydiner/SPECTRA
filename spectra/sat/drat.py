"""DIMACS/DRAT file writers and an adapter for an external DRAT-trim checker.

These helpers exist so SAT models and UNSAT proofs produced by
:mod:`spectra.sat.native_cdcl` can be replayed by an independently maintained
checker. Writers refuse to replace an existing file.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
from typing import Iterable

Clause = tuple[int, ...]


@dataclass(frozen=True)
class DratCheck:
    """Outcome of one external DRAT-trim run."""

    schema: str
    verified: bool
    returncode: int
    stdout: str
    stderr: str

    def record(self) -> dict:
        return asdict(self)


def _exact_rows(nvars: int, clauses: Iterable[Iterable[int]]) -> tuple[Clause, ...]:
    if type(nvars) is not int or not 0 <= nvars <= 10_000_000:
        raise ValueError("nvars must be an exact integer in [0, 10000000]")
    try:
        iterator = iter(clauses)
    except TypeError as error:
        raise ValueError("clauses must be iterable") from error
    rows: list[Clause] = []
    for clause in iterator:
        try:
            row = tuple(clause)
        except TypeError as error:
            raise ValueError("each clause must be iterable") from error
        seen: set[int] = set()
        for literal in row:
            if type(literal) is not int or literal == 0 or abs(literal) > nvars:
                raise ValueError("CNF literal must be a nonzero in-range exact integer")
            if literal in seen:
                raise ValueError("standard DRAT input cannot contain duplicate literals")
            if -literal in seen:
                raise ValueError("standard DRAT input cannot contain tautological clauses")
            seen.add(literal)
        rows.append(row)
    return tuple(rows)


def _proof_lines(raw_proof: Iterable[str | bytes]) -> tuple[str, ...]:
    """Normalize line endings and blank lines without changing any proof clause."""
    retained: list[str] = []
    for raw_line in raw_proof:
        if isinstance(raw_line, bytes):
            try:
                line = raw_line.decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError("DRAT proof line is not ASCII") from error
        elif isinstance(raw_line, str):
            line = raw_line
        else:
            raise ValueError("DRAT proof line must be text")
        line = line.strip()
        if line:
            retained.append(line)
    return tuple(retained)


def write_dimacs(path: str | Path, nvars: int, clauses: Iterable[Iterable[int]]) -> Path:
    """Write a standard DIMACS CNF file; duplicate and tautological clauses are rejected."""
    rows = _exact_rows(nvars, clauses)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"p cnf {nvars} {len(rows)}\n")
        for row in rows:
            if row:
                stream.write(" ".join(str(literal) for literal in row) + " ")
            stream.write("0\n")
    return destination


def write_drat(path: str | Path, proof: Iterable[str | bytes]) -> Path:
    """Write a textual DRAT proof that must derive the empty clause."""
    lines = _proof_lines(proof)
    if not any(line == "0" for line in lines):
        raise ValueError("DRAT proof must contain a derived empty clause")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="ascii", newline="\n") as stream:
        for line in lines:
            stream.write(line + "\n")
    return destination


def check_drat(
    checker: str | Path,
    cnf: str | Path,
    proof: str | Path,
    *,
    timeout_seconds: int = 300,
) -> DratCheck:
    """Run DRAT-trim; a proof is verified only by an exact ``s VERIFIED`` stdout line."""
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive exact integer")
    command = [str(Path(checker).resolve()), str(Path(cnf).resolve()), str(Path(proof).resolve())]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout_seconds,
    )
    verified = completed.returncode == 0 and any(
        line.strip() == "s VERIFIED" for line in completed.stdout.splitlines()
    )
    return DratCheck(
        schema="spectra.general_drat_check.v1",
        verified=verified,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
