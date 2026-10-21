"""Outcome-blind frontend routing for matched general-CNF development.

The policy uses only admitted formula geometry.  Unit or empty clauses take the
replayable binary-core path because propagation can remove substantial work or
certify an immediate contradiction.  Formulas without units go directly to the
matched solver, avoiding an SCC/substitution pass that exposed development data
showed could cost more than it saved.

This is an exposed-development policy.  It must be frozen before any prospective
or hidden inventory is opened and it does not establish general SAT leadership.
"""
from __future__ import annotations

from dataclasses import dataclass

from experiments.general_competition.matched_backend import (
    MatchedComparison,
    PreparedFormula,
    compare_prepared_backend,
    run_backend,
)

POLICY = "general-frontend-v1"


@dataclass(frozen=True)
class AdaptiveComparison:
    policy: str
    decision: str
    reason: str
    comparison: MatchedComparison

    @property
    def accepted_for_timing(self) -> bool:
        return self.comparison.accepted_for_timing

    def record(self) -> dict:
        value = self.comparison.record()
        value.update({
            "adaptive_policy": self.policy,
            "adaptive_decision": self.decision,
            "adaptive_reason": self.reason,
        })
        return value


def choose_frontend(prepared: PreparedFormula) -> tuple[str, str]:
    """Choose ``reduce`` or ``plain`` from immutable formula geometry only."""
    if type(prepared) is not PreparedFormula:
        raise ValueError("prepared must come from prepare_matched_formula")
    if prepared.empty_clauses:
        return "reduce", "empty_clause"
    if prepared.unit_clauses:
        return "reduce", "unit_clause"
    return "plain", "no_unit_clause"


def _compare_plain(
    prepared: PreparedFormula,
    *,
    solver_name: str,
    order: str,
) -> MatchedComparison:
    if order not in ("plain-first", "hybrid-first"):
        raise ValueError("order must be plain-first or hybrid-first")

    def solve():
        return run_backend(
            prepared.nvars,
            prepared.clauses,
            solver_name=solver_name,
        )

    if order == "plain-first":
        plain = solve()
        hybrid = solve()
    else:
        hybrid = solve()
        plain = solve()

    plain_sat = plain.status == "SAT_CHECKED"
    hybrid_sat = hybrid.status == "SAT_CHECKED"
    answers_agree = plain_sat == hybrid_sat
    if not answers_agree:
        raise RuntimeError("matched direct arms disagree on SAT status")
    accepted = (
        plain_sat
        and hybrid_sat
        and plain.model_verified
        and hybrid.model_verified
    )
    return MatchedComparison(
        schema="spectra.general_matched_backend.v2",
        solver=solver_name,
        frontend="plain-direct-v1",
        original_variables=prepared.nvars,
        original_clauses=len(prepared.clauses),
        empty_input_clauses=prepared.empty_clauses,
        unit_input_clauses=prepared.unit_clauses,
        binary_input_clauses=prepared.binary_clauses,
        long_input_clauses=prepared.long_clauses,
        residual_variables=prepared.nvars,
        residual_clauses=len(prepared.clauses),
        core_elapsed_ns=0,
        equivalent_variables_removed=0,
        propagated_classes=0,
        core_contradiction=False,
        plain=plain,
        hybrid=hybrid,
        hybrid_total_ns=hybrid.elapsed_ns,
        hybrid_over_plain=hybrid.elapsed_ns / max(plain.elapsed_ns, 1),
        answers_agree=answers_agree,
        accepted_for_timing=accepted,
        claim_boundary=(
            "exposed post-admission development diagnostic; identical direct "
            "backend in both arms; SAT models checked on original CNF"
        ),
    )


def compare_adaptive_backend(
    prepared: PreparedFormula,
    *,
    solver_name: str = "cadical195",
    order: str = "plain-first",
    native_core=None,
    native_prepared=None,
) -> AdaptiveComparison:
    """Run the frozen geometry policy under the existing matched-backend contract."""
    decision, reason = choose_frontend(prepared)
    if decision == "reduce":
        comparison = compare_prepared_backend(
            prepared,
            solver_name=solver_name,
            order=order,
            native_core=native_core,
            native_prepared=native_prepared,
        )
    else:
        comparison = _compare_plain(
            prepared,
            solver_name=solver_name,
            order=order,
        )
    return AdaptiveComparison(POLICY, decision, reason, comparison)
