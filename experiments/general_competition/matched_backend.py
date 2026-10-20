"""Matched development comparison for the binary frontend and one SAT backend.

Both arms use the same native PySAT backend. The control receives the canonical
original CNF; the SPECTRA arm receives the independently replayed binary-core
residual and lifts SAT models back to the original variables. This isolates the
frontend contribution. Residual UNSAT answers remain explicitly unverified until
a standard proof-composition path is connected.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Iterable

from experiments.general_competition.binary_core import (
    Clause,
    build_binary_core,
    canonical_cnf,
    model_satisfies,
    verify_binary_core,
)


@dataclass(frozen=True)
class PreparedFormula:
    """One canonical formula prepared outside the measured solver endpoint."""

    nvars: int
    clauses: tuple[Clause, ...]
    empty_clauses: int
    unit_clauses: int
    binary_clauses: int
    long_clauses: int


@dataclass(frozen=True)
class BackendRun:
    solver: str
    status: str
    variables: int
    clauses: int
    elapsed_ns: int
    conflicts: int
    decisions: int
    propagations: int
    restarts: int
    model: tuple[bool, ...]
    model_verified: bool


@dataclass(frozen=True)
class MatchedComparison:
    schema: str
    solver: str
    frontend: str
    original_variables: int
    original_clauses: int
    empty_input_clauses: int
    unit_input_clauses: int
    binary_input_clauses: int
    long_input_clauses: int
    residual_variables: int
    residual_clauses: int
    core_elapsed_ns: int
    equivalent_variables_removed: int
    propagated_classes: int
    core_contradiction: bool
    plain: BackendRun
    hybrid: BackendRun
    hybrid_total_ns: int
    hybrid_over_plain: float
    answers_agree: bool
    accepted_for_timing: bool
    claim_boundary: str

    def record(self) -> dict:
        return asdict(self)


def prepare_matched_formula(
    nvars: int,
    clauses: Iterable[Iterable[int]],
) -> PreparedFormula:
    """Canonicalize and classify once outside a repeated matched timing loop."""
    canonical = canonical_cnf(nvars, clauses)
    empties = sum(len(row) == 0 for row in canonical)
    units = sum(len(row) == 1 for row in canonical)
    binaries = sum(len(row) == 2 for row in canonical)
    longs = sum(len(row) >= 3 for row in canonical)
    if empties + units + binaries + longs != len(canonical):
        raise AssertionError("canonical clause classification differs")
    return PreparedFormula(
        nvars=nvars,
        clauses=canonical,
        empty_clauses=empties,
        unit_clauses=units,
        binary_clauses=binaries,
        long_clauses=longs,
    )


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


def _stat(stats: dict, name: str) -> int:
    value = stats.get(name, 0)
    if type(value) is not int or value < 0:
        raise RuntimeError(f"SAT backend returned an invalid {name} statistic")
    return value


def run_backend(
    nvars: int,
    clauses: tuple[Clause, ...],
    *,
    solver_name: str = "cadical195",
) -> BackendRun:
    """Construct, solve and dispose one native backend instance."""
    try:
        from pysat.solvers import Solver
    except ImportError as error:
        raise RuntimeError("python-sat is required for matched backend development") from error

    started = perf_counter_ns()
    with Solver(
        name=solver_name,
        bootstrap_with=[list(row) for row in clauses],
        use_timer=False,
    ) as solver:
        sat = solver.solve()
        raw_model = solver.get_model() if sat else None
        stats = solver.accum_stats()
    elapsed = perf_counter_ns() - started

    if sat:
        model = _complete_model(raw_model, nvars)
        verified = model_satisfies(nvars, clauses, model)
        if not verified:
            raise RuntimeError("original-clause checker rejected backend SAT model")
        status = "SAT_CHECKED"
    else:
        model = ()
        verified = False
        status = "UNSAT_REPORTED"
    return BackendRun(
        solver=solver_name,
        status=status,
        variables=nvars,
        clauses=len(clauses),
        elapsed_ns=elapsed,
        conflicts=_stat(stats, "conflicts"),
        decisions=_stat(stats, "decisions"),
        propagations=_stat(stats, "propagations"),
        restarts=_stat(stats, "restarts"),
        model=model,
        model_verified=verified,
    )


def compare_prepared_backend(
    prepared: PreparedFormula,
    *,
    solver_name: str = "cadical195",
    order: str = "plain-first",
    native_core=None,
    native_prepared=None,
) -> MatchedComparison:
    """Compare one already-canonical formula without repeating untimed setup."""
    if type(prepared) is not PreparedFormula:
        raise ValueError("prepared must come from prepare_matched_formula")
    if order not in ("plain-first", "hybrid-first"):
        raise ValueError("order must be plain-first or hybrid-first")
    if native_prepared is not None and native_core is None:
        raise ValueError("native_prepared requires native_core")
    nvars = prepared.nvars
    canonical = prepared.clauses

    core_started = perf_counter_ns()
    if native_core is None:
        core = build_binary_core(nvars, canonical)
        if not verify_binary_core(nvars, canonical, core):
            raise AssertionError("binary core failed independent replay")
        frontend = "python-replayable-binary-core"
    else:
        if native_prepared is None:
            core = native_core.reduce(nvars, canonical)
        else:
            if (
                native_prepared.nvars != nvars
                or native_prepared.clause_count != len(canonical)
                or native_prepared.unit_clauses != prepared.unit_clauses
                or native_prepared.binary_clauses != prepared.binary_clauses
                or native_prepared.long_clauses != prepared.long_clauses
            ):
                raise ValueError("native prepared formula differs from prepared CNF")
            core = native_core.reduce_prepared(native_prepared)
        if core.original_nvars != nvars:
            raise AssertionError("native binary core belongs to another formula")
        frontend = "native-binary-core-prepared-v1"
    core_elapsed = perf_counter_ns() - core_started

    inferred_empty = (
        core.normalized_clause_count
        - core.unit_input_clauses
        - core.binary_input_clauses
        - core.long_input_clauses
    )
    if (
        core.normalized_clause_count != len(canonical)
        or inferred_empty != prepared.empty_clauses
        or core.unit_input_clauses != prepared.unit_clauses
        or core.binary_input_clauses != prepared.binary_clauses
        or core.long_input_clauses != prepared.long_clauses
    ):
        raise AssertionError("binary-core input classification differs")

    def hybrid_run() -> BackendRun:
        if core.contradiction:
            return BackendRun(
                solver=solver_name,
                status="UNSAT_VERIFIED_CORE",
                variables=core.reduced_nvars,
                clauses=len(core.residual_clauses),
                elapsed_ns=0,
                conflicts=1,
                decisions=0,
                propagations=getattr(core, "propagated_classes", 0),
                restarts=0,
                model=(),
                model_verified=True,
            )
        residual = run_backend(
            core.reduced_nvars,
            core.residual_clauses,
            solver_name=solver_name,
        )
        if residual.status == "SAT_CHECKED":
            lifted = core.lift(residual.model)
            if not model_satisfies(nvars, canonical, lifted):
                raise RuntimeError("lifted model fails the canonical original CNF")
            return BackendRun(
                **{
                    **asdict(residual),
                    "model": lifted,
                    "model_verified": True,
                }
            )
        return residual

    if order == "plain-first":
        plain = run_backend(nvars, canonical, solver_name=solver_name)
        hybrid = hybrid_run()
    else:
        hybrid = hybrid_run()
        plain = run_backend(nvars, canonical, solver_name=solver_name)

    plain_sat = plain.status == "SAT_CHECKED"
    hybrid_sat = hybrid.status == "SAT_CHECKED"
    answers_agree = plain_sat == hybrid_sat
    if not answers_agree:
        raise RuntimeError("matched backend arms disagree on SAT status")

    hybrid_total = core_elapsed + hybrid.elapsed_ns
    ratio = hybrid_total / max(plain.elapsed_ns, 1)
    accepted = plain_sat and hybrid_sat and plain.model_verified and hybrid.model_verified
    return MatchedComparison(
        schema="spectra.general_matched_backend.v2",
        solver=solver_name,
        frontend=frontend,
        original_variables=nvars,
        original_clauses=len(canonical),
        empty_input_clauses=prepared.empty_clauses,
        unit_input_clauses=prepared.unit_clauses,
        binary_input_clauses=prepared.binary_clauses,
        long_input_clauses=prepared.long_clauses,
        residual_variables=core.reduced_nvars,
        residual_clauses=len(core.residual_clauses),
        core_elapsed_ns=core_elapsed,
        equivalent_variables_removed=core.equivalent_variables_removed,
        propagated_classes=core.propagated_classes,
        core_contradiction=core.contradiction,
        plain=plain,
        hybrid=hybrid,
        hybrid_total_ns=hybrid_total,
        hybrid_over_plain=ratio,
        answers_agree=answers_agree,
        accepted_for_timing=accepted,
        claim_boundary=(
            "exposed post-admission development diagnostic; same backend in both arms; "
            f"frontend={frontend}; SAT models checked on original CNF; "
            "residual UNSAT has no standard proof yet"
        ),
    )


def compare_same_backend(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    *,
    solver_name: str = "cadical195",
    order: str = "plain-first",
    native_core=None,
) -> MatchedComparison:
    """Canonicalize once, then compare both arms under the same backend."""
    prepared = prepare_matched_formula(nvars, clauses)
    native_prepared = (
        native_core.prepare_canonical(prepared.nvars, prepared.clauses)
        if native_core is not None
        else None
    )
    return compare_prepared_backend(
        prepared,
        solver_name=solver_name,
        order=order,
        native_core=native_core,
        native_prepared=native_prepared,
    )
