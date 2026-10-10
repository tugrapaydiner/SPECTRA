"""Build one long-lived service case from several measured Abilene weeks."""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

from experiments.real_traffic.case import TrafficCase, build_queries
from experiments.real_traffic.plan import align_target_plan, color_graph
from experiments.real_traffic.observer import masks_from_plans, verify_coloring_edges
from experiments.real_traffic.sources import (
    MATRIX_WIDTH,
    SNAPSHOTS_PER_WEEK,
    TraceFormatError,
    build_conflict_graph,
    hash_json,
    mean_traffic,
    parse_demands,
    parse_links,
    parse_routes,
    parse_week,
)


def combine_distinct_weeks(weeks: Sequence[str], query_banks, timestamp_banks,
                           status_banks, witness_banks, proof_banks):
    """Retain the first measured occurrence of each distinct support request."""
    if not weeks or len(set(weeks)) != len(weeks):
        raise ValueError("weeks must be nonempty and distinct")
    if not (len(weeks) == len(query_banks) == len(timestamp_banks)
            == len(status_banks) == len(witness_banks) == len(proof_banks)):
        raise ValueError("week result inventories differ")
    seen = set()
    queries = []
    timestamps = []
    statuses = []
    witnesses = []
    proofs = []
    cross_week_duplicates = 0
    for week_index, (bank, times, states, receipts, contradictions) in enumerate(zip(
            query_banks, timestamp_banks, status_banks, witness_banks, proof_banks)):
        if not (len(bank) == len(times) == len(states) == len(receipts)
                == len(contradictions)):
            raise ValueError("one week has ragged query records")
        for query, timestamp, status, witness, proof in zip(
                bank, times, states, receipts, contradictions):
            if query in seen:
                cross_week_duplicates += 1
                continue
            seen.add(query)
            queries.append(query)
            timestamps.append(week_index * SNAPSHOTS_PER_WEEK + timestamp)
            statuses.append(status)
            witnesses.append(witness)
            proofs.append(proof)
    return (tuple(queries), tuple(timestamps), tuple(statuses), tuple(witnesses),
            tuple(proofs), cross_week_duplicates)


def build_series_case(*, weeks: Sequence[str], routing_bytes: bytes,
                      demands_bytes: bytes, links_bytes: bytes,
                      traffic_bytes: Mapping[str, bytes], protect_count: int = 4,
                      migrate_count: int = 4,
                      expected_demand_count: int = MATRIX_WIDTH,
                      expected_snapshots: int = SNAPSHOTS_PER_WEEK) -> TrafficCase:
    """Compile once, then ask every distinct trace-defined question across weeks.

    The first week fixes both colour plans.  Later weeks influence only which
    original-address support restrictions are requested; they cannot retune the
    relation, solver, colour names, or migration options.
    """
    weeks = tuple(weeks)
    if not weeks or len(set(weeks)) != len(weeks):
        raise ValueError("weeks must be a nonempty distinct sequence")
    if set(traffic_bytes) != set(weeks):
        raise ValueError("traffic byte inventory differs from requested weeks")
    demands = parse_demands(
        demands_bytes.decode("ascii"), expected_count=expected_demand_count)
    links = parse_links(links_bytes.decode("ascii"))
    routes = parse_routes(routing_bytes.decode("ascii"), demands, links)
    graph = build_conflict_graph(
        demands, routes, source_bytes=(routing_bytes, demands_bytes, links_bytes))
    parsed = {
        week: parse_week(traffic_bytes[week], demands,
                         expected_snapshots=expected_snapshots)
        for week in weeks
    }
    anchor = parsed[weeks[0]]
    weights = mean_traffic(anchor)
    plan_a = color_graph(graph.adjacency)
    raw_target = color_graph(graph.adjacency, weights)
    plan_b = align_target_plan(plan_a, raw_target, weights)
    if not verify_coloring_edges(graph.edges, plan_b):
        raise AssertionError("aligned series target plan is invalid")
    masks = masks_from_plans(plan_a, plan_b)
    palette = max(max(plan_a), max(plan_b)) + 1
    if palette > 64:
        raise TraceFormatError("series plans exceed the native 64-colour contract")

    query_banks = []
    timestamp_banks = []
    status_banks = []
    witness_banks = []
    proof_banks = []
    attempts = local_duplicates = 0
    for week in weeks:
        queries, timestamps, statuses, witnesses, proofs, stats = build_queries(
            graph, parsed[week], plan_a, plan_b, masks,
            protect_count=protect_count, migrate_count=migrate_count)
        query_banks.append(queries)
        timestamp_banks.append(timestamps)
        status_banks.append(statuses)
        witness_banks.append(witnesses)
        proof_banks.append(proofs)
        attempts += stats["proposal_attempts"]
        local_duplicates += stats["duplicate_queries"]
    (queries, timestamps, statuses, witnesses, proofs,
     cross_duplicates) = combine_distinct_weeks(
        weeks, query_banks, timestamp_banks, status_banks,
        witness_banks, proof_banks)

    source_sha256 = {
        "routing": hashlib.sha256(routing_bytes).hexdigest(),
        "demands": hashlib.sha256(demands_bytes).hexdigest(),
        "links": hashlib.sha256(links_bytes).hexdigest(),
        "route_graph": graph.source_sha256,
    }
    source_sha256.update({
        "traffic_" + week: parsed[week].source_sha256 for week in weeks
    })
    provisional = {
        "schema": "spectra.real_traffic.case.v2",
        "source_week": ",".join(weeks),
        "source_sha256": source_sha256,
        "vertices": graph.n,
        "edges": graph.edges,
        "demand_indices": tuple(d.index for d in graph.demands),
        "demand_names": tuple(f"{d.source},{d.target}" for d in graph.demands),
        "route_links": graph.routes,
        "palette": palette,
        "plan_a": plan_a,
        "plan_b": plan_b,
        "masks": masks,
        "queries": queries,
        "timestamps": timestamps,
        "statuses": statuses,
        "witness_sha256": witnesses,
        "contradictions": proofs,
        "proposal_attempts": attempts,
        "duplicate_queries": local_duplicates + cross_duplicates,
        "sat_queries": statuses.count("SAT"),
        "unsat_queries": statuses.count("UNSAT"),
        "unique_witnesses": len({value for value in witnesses if value is not None}),
        "protect_count": protect_count,
        "migrate_count": migrate_count,
    }
    case = TrafficCase(**provisional, case_sha256=hash_json(provisional))
    case.validate()
    return case
