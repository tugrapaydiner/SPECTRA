"""Edge efficiency metrics and compute-optimal frontier curves (sections 23, 27).

The headline numbers are *accuracy per resource*: per measured joule, per MB, per
millisecond. The mandatory deliverable (section 27.0) is the compute-optimal
frontier -- accuracy vs cost as we vary parameter count, recursion depth,
active-token density, or MCTS rollouts -- summarised here as the Pareto-optimal
set of (cost, accuracy) operating points.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Efficiency metrics (section 23.2)
# --------------------------------------------------------------------------- #
def accuracy_per_joule(accuracy: float, joules_per_problem: float) -> float:
    return accuracy / (joules_per_problem + _EPS)


def accuracy_per_mb(accuracy: float, peak_ram_mb: float) -> float:
    return accuracy / (peak_ram_mb + _EPS)


def accuracy_per_ms(accuracy: float, latency_ms: float) -> float:
    return accuracy / (latency_ms + _EPS)


def edge_reasoning_score(
    accuracy: float, joules_per_problem: float, latency_ms: float, peak_ram_mb: float
) -> float:
    """Combined efficiency score (section 23.2). Report raw metrics alongside it."""
    return accuracy / (
        (joules_per_problem + _EPS) * (latency_ms + _EPS) * (peak_ram_mb + _EPS)
    )


# --------------------------------------------------------------------------- #
# Compute-optimal frontier (section 27.0)
# --------------------------------------------------------------------------- #
def compute_optimal_frontier(
    points: Sequence[dict[str, Any]],
    cost_key: str = "joules_per_problem",
    acc_key: str = "accuracy",
) -> list[dict[str, Any]]:
    """Return the Pareto-optimal operating points (lower cost, higher accuracy).

    A point is kept if no other point has both lower-or-equal cost *and* higher
    accuracy. Result is sorted by ascending cost -- the compute-optimal frontier.
    """
    kept: list[dict[str, Any]] = []
    for p in points:
        dominated = any(
            q is not p
            and q[cost_key] <= p[cost_key]
            and q[acc_key] > p[acc_key]
            for q in points
        )
        if not dominated:
            kept.append(p)
    # Deduplicate equal-cost points, keeping the most accurate, then sort by cost.
    best_by_cost: dict[float, dict[str, Any]] = {}
    for p in kept:
        c = p[cost_key]
        if c not in best_by_cost or p[acc_key] > best_by_cost[c][acc_key]:
            best_by_cost[c] = p
    return sorted(best_by_cost.values(), key=lambda p: p[cost_key])


def edge_report(
    accuracy: float,
    latency_ms: float,
    peak_ram_mb: float,
    model_size_mb: float,
    joules_per_problem: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble the full edge metrics dict (section 23.1 + efficiency ratios)."""
    report: dict[str, Any] = {
        "accuracy": accuracy,
        "latency_ms": latency_ms,
        "peak_ram_mb": peak_ram_mb,
        "model_size_mb": model_size_mb,
        "accuracy_per_ms": accuracy_per_ms(accuracy, latency_ms),
        "accuracy_per_mb": accuracy_per_mb(accuracy, peak_ram_mb),
        **extra,
    }
    if joules_per_problem is not None:
        report["joules_per_problem"] = joules_per_problem
        report["accuracy_per_joule"] = accuracy_per_joule(accuracy, joules_per_problem)
        report["edge_reasoning_score"] = edge_reasoning_score(
            accuracy, joules_per_problem, latency_ms, peak_ram_mb
        )
    else:
        report["joules_per_problem"] = None
        report["energy_note"] = "RAPL unavailable on this host; joules not measured"
    return report


def save_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a report dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
