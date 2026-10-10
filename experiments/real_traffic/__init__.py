"""Trace-driven repeated support research."""

from experiments.real_traffic.case import TrafficCase, build_case, build_queries
from experiments.real_traffic.certificate import (
    ExactOutcome,
    InvalidContradiction,
    PreparedContradictionChecker,
    solve_with_proof,
    verify_contradiction,
)
from experiments.real_traffic.plan import align_target_plan, color_graph
from experiments.real_traffic.observer import (
    binary_clauses,
    masks_from_plans,
    solve_binary,
    verify_coloring_edges,
    verify_labels,
)
from experiments.real_traffic.sources import (
    FIELDS_PER_DEMAND,
    MATRIX_WIDTH,
    SNAPSHOTS_PER_WEEK,
    SOURCE_ROOT,
    SOURCE_URLS,
    ConflictGraph,
    Demand,
    Link,
    TrafficWeek,
    TraceFormatError,
    build_conflict_graph,
    hash_json,
    mean_traffic,
    parse_demands,
    parse_links,
    parse_routes,
    parse_week,
    traffic_url,
)

__all__ = [
    "FIELDS_PER_DEMAND", "MATRIX_WIDTH", "SNAPSHOTS_PER_WEEK", "SOURCE_ROOT",
    "SOURCE_URLS", "ConflictGraph", "Demand", "ExactOutcome",
    "InvalidContradiction", "Link", "PreparedContradictionChecker", "TrafficCase",
    "TrafficWeek", "TraceFormatError", "align_target_plan", "binary_clauses",
    "build_case", "build_conflict_graph", "build_queries", "color_graph",
    "hash_json", "masks_from_plans", "mean_traffic", "parse_demands",
    "parse_links", "parse_routes", "parse_week", "solve_binary",
    "solve_with_proof", "traffic_url", "verify_coloring_edges",
    "verify_contradiction", "verify_labels",
]
