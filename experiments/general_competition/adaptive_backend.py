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

from dataclasses import asdict, dataclass
import math

from experiments.general_competition.matched_backend import (
    MatchedComparison,
    ModelValidationError,
    PreparedFormula,
    compare_prepared_backend,
    run_backend,
)

POLICY = "general-frontend-v1"
RESULT_CLASSIFICATIONS = (
    "SPECTRA_NATIVE_WIN",
    "SPECTRA_FRONTEND_WIN",
    "UPSTREAM_FALLBACK_PARITY",
    "LOSS",
    "TIMEOUT",
    "UNKNOWN",
    "INVALID_MODEL",
    "INVALID_PROOF",
    "RESOURCE_FAILURE",
)


def classify_comparison(record: dict) -> str:
    """Classify checked matched evidence, never awarding an upstream timing win.

    This adapter has no standalone SPECTRA-G entry: even its native binary
    reducer is a frontend to an upstream solver. Unverified UNSAT is UNKNOWN,
    whereas a proof explicitly rejected by its checker is INVALID_PROOF.
    """
    arms = (record["plain"], record["hybrid"])
    statuses = {arm["status"] for arm in arms}
    if "INVALID_MODEL" in statuses or any(
        arm["status"] == "SAT_CHECKED" and arm["model_verified"] is not True
        for arm in arms
    ):
        return "INVALID_MODEL"
    if "INVALID_PROOF" in statuses:
        return "INVALID_PROOF"
    if "TIMEOUT" in statuses:
        return "TIMEOUT"
    if "RESOURCE_FAILURE" in statuses:
        return "RESOURCE_FAILURE"
    if (
        record["answers_agree"] is not True
        and "SAT_CHECKED" in statuses
        and any(status.startswith("UNSAT") for status in statuses)
    ):
        return "INVALID_PROOF"
    if (
        record["accepted_for_timing"] is not True
        or record["answers_agree"] is not True
        or statuses != {"SAT_CHECKED"}
    ):
        return "UNKNOWN"
    ratio = record["hybrid_over_plain"]
    if (
        type(ratio) not in (int, float)
        or not math.isfinite(ratio)
        or ratio <= 0
        or any(type(arm["elapsed_ns"]) is not int or arm["elapsed_ns"] <= 0 for arm in arms)
        or type(record["hybrid_total_ns"]) is not int
        or record["hybrid_total_ns"] <= 0
    ):
        return "UNKNOWN"
    if ratio != record["hybrid_total_ns"] / arms[0]["elapsed_ns"]:
        return "UNKNOWN"
    if record["adaptive_policy"] != POLICY:
        return "UNKNOWN"
    if any(arm["solver"] != record["solver"] for arm in arms):
        return "UNKNOWN"
    if record["adaptive_decision"] == "plain":
        if (
            record["frontend"] != "plain-direct-v1"
            or record["core_elapsed_ns"] != 0
            or record["residual_variables"] != record["original_variables"]
            or record["residual_clauses"] != record["original_clauses"]
        ):
            return "UNKNOWN"
        return "UPSTREAM_FALLBACK_PARITY"
    if (
        record["adaptive_decision"] != "reduce"
        or record["frontend"] not in (
            "python-replayable-binary-core", "native-binary-core-prepared-v1"
        )
    ):
        return "UNKNOWN"
    return "SPECTRA_FRONTEND_WIN" if ratio < 1.0 else "LOSS"


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
        value["result_classification"] = classify_comparison(value)
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

    completed_arms = {}
    for arm in (("plain", "hybrid") if order == "plain-first" else ("hybrid", "plain")):
        try:
            completed_arms[arm] = solve()
        except ModelValidationError as error:
            error.evidence.update({
                "failed_arm": arm,
                "completed_arms": {key: asdict(value) for key, value in completed_arms.items()},
            })
            raise
    plain, hybrid = completed_arms["plain"], completed_arms["hybrid"]

    plain_sat = plain.status == "SAT_CHECKED"
    hybrid_sat = hybrid.status == "SAT_CHECKED"
    answers_agree = plain_sat == hybrid_sat
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
