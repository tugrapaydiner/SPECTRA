"""Standard-proof boundary for the general-CNF competition track.

SAT answers from a reduced residual are lifted and checked against the exact
original clause sequence. An UNSAT answer from the residual is never promoted on
its own: the exact original formula is solved again with a dedicated proof-logging
backend, and the retained DRAT proof is checked by the pinned DRAT-trim contract.

The performance backend and proof backend are deliberately separate. CaDiCaL is
used for residual solving, while Lingeling is the default proof producer because
PySAT 1.9.dev15's CaDiCaL wrapper does not reliably finalize the DRAT stream in
this environment. Until proof-producing preprocessing composition is implemented,
UNSAT pays for original-formula confirmation rather than turning an internal
reduction receipt into a Main-track certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
from time import perf_counter_ns
from typing import Iterable

from experiments.general_competition.binary_core import (
    BinaryCore,
    Clause,
    build_binary_core,
    model_satisfies,
    verify_binary_core,
)

DEFAULT_RESIDUAL_SOLVER = "cadical195"
DEFAULT_PROOF_SOLVER = "lingeling"


@dataclass(frozen=True)
class OriginalProofRun:
    schema: str
    solver: str
    status: str
    variables: int
    clauses: int
    elapsed_ns: int
    model: tuple[bool, ...]
    model_verified: bool
    proof: tuple[str, ...]


@dataclass(frozen=True)
class SafeHybridRun:
    schema: str
    solver: str
    proof_solver: str
    status: str
    original_variables: int
    original_clauses: int
    residual_variables: int
    residual_clauses: int
    core_elapsed_ns: int
    residual_elapsed_ns: int
    original_confirmation_ns: int
    total_elapsed_ns: int
    equivalent_variables_removed: int
    propagated_classes: int
    residual_status: str
    model: tuple[bool, ...]
    model_verified: bool
    proof: tuple[str, ...]
    proof_scope: str

    def record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DratCheck:
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


def _complete_model(raw: list[int] | None, nvars: int) -> tuple[bool, ...]:
    if raw is None:
        raise RuntimeError("SAT backend returned no model")
    values: dict[int, bool] = {}
    for literal in raw:
        if type(literal) is not int or literal == 0:
            raise RuntimeError("SAT backend returned a malformed model literal")
        variable = abs(literal)
        if variable > nvars:
            continue
        value = literal > 0
        previous = values.get(variable)
        if previous is not None and previous != value:
            raise RuntimeError("SAT backend returned contradictory model values")
        values[variable] = value
    return tuple(values.get(variable, False) for variable in range(1, nvars + 1))


def _new_solver(
    name: str,
    *,
    with_proof: bool,
    bootstrap_with: Iterable[Iterable[int]] = (),
):
    if type(name) is not str or not name:
        raise ValueError("solver name must be a non-empty string")
    try:
        from pysat.solvers import Solver
    except ImportError as error:
        raise RuntimeError("python-sat is required for the native proof backend") from error
    return Solver(
        name=name,
        bootstrap_with=[list(row) for row in bootstrap_with],
        with_proof=with_proof,
        use_timer=False,
    )


def _proof_lines(raw_proof: Iterable[str | bytes]) -> tuple[str, ...]:
    """Normalize the proof stream without changing any retained DRAT clause."""
    retained: list[str] = []
    for raw_line in raw_proof:
        if isinstance(raw_line, bytes):
            try:
                line = raw_line.decode("ascii")
            except UnicodeDecodeError as error:
                raise RuntimeError("UNSAT backend returned a non-ASCII proof line") from error
        elif isinstance(raw_line, str):
            line = raw_line
        else:
            raise RuntimeError("UNSAT backend returned a non-text proof line")
        line = line.strip()
        if line:
            retained.append(line)
    return tuple(retained)


def _has_empty_clause(proof: Iterable[str]) -> bool:
    return any(line.strip() == "0" for line in proof)


def solve_original_with_drat(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    *,
    solver_name: str = DEFAULT_PROOF_SOLVER,
) -> OriginalProofRun:
    """Solve the exact original clause sequence and retain a DRAT proof for UNSAT."""
    rows = _exact_rows(nvars, clauses)
    started = perf_counter_ns()

    # The original formula already proves UNSAT when it contains an empty clause.
    # PySAT 1.9.dev15 indexes clause[0] while bootstrapping and cannot accept this
    # valid DIMACS edge case, so retain the canonical one-line proof directly.
    if () in rows:
        return OriginalProofRun(
            schema="spectra.general_original_proof.v2",
            solver="input-empty-clause",
            status="UNSAT_WITH_DRAT",
            variables=nvars,
            clauses=len(rows),
            elapsed_ns=perf_counter_ns() - started,
            model=(),
            model_verified=False,
            proof=("0",),
        )

    with _new_solver(
        solver_name,
        with_proof=True,
        bootstrap_with=rows,
    ) as solver:
        sat = solver.solve()
        if sat:
            model = _complete_model(solver.get_model(), nvars)
            proof: tuple[str, ...] = ()
        else:
            model = ()
            raw_proof = solver.get_proof()
            if raw_proof is None:
                raise RuntimeError("UNSAT backend returned no proof")
            proof = _proof_lines(raw_proof)
    elapsed = perf_counter_ns() - started

    if sat:
        verified = model_satisfies(nvars, rows, model)
        if not verified:
            raise RuntimeError("original-clause checker rejected backend SAT model")
        status = "SAT_CHECKED"
    else:
        verified = False
        if not proof or not _has_empty_clause(proof):
            raise RuntimeError("UNSAT proof is empty or lacks the empty clause")
        status = "UNSAT_WITH_DRAT"
    return OriginalProofRun(
        schema="spectra.general_original_proof.v2",
        solver=solver_name,
        status=status,
        variables=nvars,
        clauses=len(rows),
        elapsed_ns=elapsed,
        model=model,
        model_verified=verified,
        proof=proof,
    )


def _solve_residual(
    core: BinaryCore,
    *,
    solver_name: str,
) -> tuple[str, tuple[bool, ...], int]:
    started = perf_counter_ns()
    with _new_solver(
        solver_name,
        with_proof=False,
        bootstrap_with=core.residual_clauses,
    ) as solver:
        sat = solver.solve()
        model = _complete_model(solver.get_model(), core.reduced_nvars) if sat else ()
    elapsed = perf_counter_ns() - started
    if sat and not model_satisfies(
        core.reduced_nvars, core.residual_clauses, model
    ):
        raise RuntimeError("residual model failed the residual-clause checker")
    return ("SAT_CHECKED" if sat else "UNSAT_REPORTED", model, elapsed)


def solve_safe_hybrid(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    *,
    solver_name: str = DEFAULT_RESIDUAL_SOLVER,
    proof_solver_name: str = DEFAULT_PROOF_SOLVER,
) -> SafeHybridRun:
    """Use the binary frontend for SAT but require an original-formula DRAT for UNSAT."""
    rows = _exact_rows(nvars, clauses)
    total_started = perf_counter_ns()
    core_started = perf_counter_ns()
    core = build_binary_core(nvars, rows)
    core_elapsed = perf_counter_ns() - core_started
    if not verify_binary_core(nvars, rows, core):
        raise AssertionError("binary core failed independent replay")

    residual_elapsed = 0
    confirmation_elapsed = 0
    if core.contradiction:
        residual_status = "SKIPPED_CORE_CONTRADICTION"
        original = solve_original_with_drat(
            nvars, rows, solver_name=proof_solver_name
        )
        confirmation_elapsed = original.elapsed_ns
        if original.status != "UNSAT_WITH_DRAT":
            raise RuntimeError("binary-core contradiction disagrees with original solver")
        status = original.status
        model = ()
        model_verified = False
        proof = original.proof
        proof_scope = "exact-original-cnf"
    else:
        residual_status, residual_model, residual_elapsed = _solve_residual(
            core, solver_name=solver_name
        )
        if residual_status == "SAT_CHECKED":
            model = core.lift(residual_model)
            model_verified = model_satisfies(nvars, rows, model)
            if not model_verified:
                raise RuntimeError("lifted residual model fails the exact original CNF")
            status = "SAT_CHECKED"
            proof = ()
            proof_scope = "not-required-for-sat"
        else:
            original = solve_original_with_drat(
                nvars, rows, solver_name=proof_solver_name
            )
            confirmation_elapsed = original.elapsed_ns
            if original.status != "UNSAT_WITH_DRAT":
                raise RuntimeError("residual UNSAT disagrees with original solver")
            status = original.status
            model = ()
            model_verified = False
            proof = original.proof
            proof_scope = "exact-original-cnf"

    total = perf_counter_ns() - total_started
    return SafeHybridRun(
        schema="spectra.general_safe_hybrid.v2",
        solver=solver_name,
        proof_solver=proof_solver_name,
        status=status,
        original_variables=nvars,
        original_clauses=len(rows),
        residual_variables=core.reduced_nvars,
        residual_clauses=len(core.residual_clauses),
        core_elapsed_ns=core_elapsed,
        residual_elapsed_ns=residual_elapsed,
        original_confirmation_ns=confirmation_elapsed,
        total_elapsed_ns=total,
        equivalent_variables_removed=core.equivalent_variables_removed,
        propagated_classes=core.propagated_classes,
        residual_status=residual_status,
        model=model,
        model_verified=model_verified,
        proof=proof,
        proof_scope=proof_scope,
    )


def write_dimacs(
    path: str | Path,
    nvars: int,
    clauses: Iterable[Iterable[int]],
) -> Path:
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
    lines = _proof_lines(proof)
    if not lines or not _has_empty_clause(lines):
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
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive exact integer")
    command = [str(Path(checker).resolve()), str(Path(cnf).resolve()), str(Path(proof).resolve())]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    combined = completed.stdout + "\n" + completed.stderr
    verified = completed.returncode == 0 and "VERIFIED" in combined.upper()
    return DratCheck(
        schema="spectra.general_drat_check.v1",
        verified=verified,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )