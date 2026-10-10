from __future__ import annotations

from decimal import Decimal
import gzip
import itertools
import json
import random

import pytest

from experiments.real_traffic import (
    ConflictGraph,
    Demand,
    InvalidContradiction,
    TrafficCase,
    TrafficWeek,
    TraceFormatError,
    hash_json,
    align_target_plan,
    build_conflict_graph,
    build_queries,
    color_graph,
    masks_from_plans,
    parse_demands,
    parse_links,
    parse_routes,
    parse_week,
    solve_binary,
    solve_with_proof,
    traffic_url,
    verify_labels,
    verify_contradiction,
)


def small_inventory() -> tuple[str, str, str]:
    demands = """# dmd(s,d) dmd_index
A,A 1
A,B 2
B,A 3
B,B 4
"""
    links = """# link(x,y) link_index link_type
A,B 1 0
B,A 2 0
*,A 3 1
A,* 4 2
*,B 5 1
B,* 6 2
"""
    routing = """# link(x,y) dmd(s,d) link_index dmd_index frac
*,A A,A 3 1 1
A,* A,A 4 1 1
*,A A,B 3 2 1
A,B A,B 1 2 1
B,* A,B 6 2 1
*,B B,A 5 3 1
B,A B,A 2 3 1
A,* B,A 4 3 1
*,B B,B 5 4 1
B,* B,B 6 4 1
"""
    return demands, links, routing


def compressed_rows(rows: list[list[int]]) -> bytes:
    text = "\n".join(" ".join(map(str, row)) for row in rows) + "\n"
    return gzip.compress(text.encode("ascii"), mtime=0)


def test_fixed_source_url_contract() -> None:
    assert traffic_url("X01").endswith("/X01.gz")
    assert traffic_url("X24").endswith("/X24.gz")
    for bad in ("x01", "X00", "X25", "X1", 1):
        with pytest.raises(ValueError):
            traffic_url(bad)  # type: ignore[arg-type]


def test_public_inventory_and_routes_parse_strictly() -> None:
    demands_text, links_text, routing_text = small_inventory()
    demands = parse_demands(demands_text, expected_count=4)
    links = parse_links(links_text)
    routes = parse_routes(routing_text, demands, links)
    assert routes == ((), (1,), (2,), ())

    graph = build_conflict_graph(demands, routes, source_bytes=(b"A", b"B"))
    assert tuple(d.index for d in graph.demands) == (2, 3)
    assert graph.routes == ((1,), (2,))
    assert graph.edges == ()
    assert len(graph.source_sha256) == 64

    with pytest.raises(TraceFormatError, match="expected 5 demands"):
        parse_demands(demands_text, expected_count=5)
    with pytest.raises(TraceFormatError, match="duplicate routing"):
        parse_routes(routing_text + "A,B A,B 1 2 1\n", demands, links)


def test_conflicts_use_shared_directed_internal_links() -> None:
    demands = tuple(Demand(index + 1, f"s{index}", f"t{index}") for index in range(4))
    graph = build_conflict_graph(demands, ((1, 2), (2, 3), (4,), (1,)))
    assert graph.edges == ((0, 1), (0, 3))
    assert graph.adjacency[0] == frozenset({1, 3})


def test_gzip_week_parser_keeps_real_od_column_and_exact_units() -> None:
    demands_text, _links_text, _routing_text = small_inventory()
    demands = parse_demands(demands_text, expected_count=4)
    rows = []
    for time in range(3):
        row = []
        for demand in range(4):
            real = 3_000_000 * (100 * time + demand)
            row.extend((real, 11, 12, 13, 14))
        rows.append(row)
    payload = compressed_rows(rows)
    week = parse_week(payload, demands, expected_snapshots=3)
    # Loop demands 1 and 4 are removed; conversion is to exact Mbit/s.
    assert week.snapshots == (
        (Decimal(8), Decimal(16)),
        (Decimal(808), Decimal(816)),
        (Decimal(1608), Decimal(1616)),
    )
    assert week.source_sha256

    broken = compressed_rows([rows[0][:-1], *rows[1:]])
    with pytest.raises(TraceFormatError, match="has 19 fields"):
        parse_week(broken, demands, expected_snapshots=3)


def test_colour_alignment_minimizes_weighted_migration() -> None:
    reference = (0, 1, 0, 2, 1)
    target = (2, 0, 2, 1, 0)
    weights = tuple(map(Decimal, (100, 20, 80, 1, 10)))
    aligned = align_target_plan(reference, target, weights)
    assert aligned == reference

    adjacency = (
        frozenset({1, 3}),
        frozenset({0, 2}),
        frozenset({1, 3}),
        frozenset({0, 2}),
    )
    assert color_graph(adjacency) == (0, 1, 0, 1)


def test_binary_observer_matches_exhaustive_search() -> None:
    rng = random.Random(20261010)
    for vertices in range(1, 7):
        for _ in range(80):
            palette = rng.randrange(1, 5)
            masks = []
            for _vertex in range(vertices):
                colours = rng.sample(range(palette), rng.randint(1, min(2, palette)))
                masks.append(sum(1 << colour for colour in colours))
            edges = tuple(
                (left, right)
                for left in range(vertices)
                for right in range(left + 1, vertices)
                if rng.random() < 0.3
            )
            restrictions = []
            for vertex in range(vertices):
                if rng.random() < 0.25:
                    choices = [c for c in range(palette) if masks[vertex] >> c & 1]
                    restrictions.append((vertex, 1 << rng.choice(choices)))
            domains = [[c for c in range(palette) if masks[v] >> c & 1]
                       for v in range(vertices)]
            expected = next(
                (labels for labels in itertools.product(*domains)
                 if verify_labels(edges, masks, labels, restrictions)),
                None,
            )
            observed = solve_binary(edges, masks, restrictions)
            assert (observed is None) == (expected is None)
            assert observed is None or verify_labels(edges, masks, observed, restrictions)


def path_graph() -> ConflictGraph:
    demands = tuple(Demand(index + 1, f"s{index}", f"t{index}") for index in range(4))
    edges = ((0, 1), (1, 2), (2, 3))
    adjacency = (
        frozenset({1}),
        frozenset({0, 2}),
        frozenset({1, 3}),
        frozenset({2}),
    )
    return ConflictGraph(demands, ((1,), (2,), (3,), (4,)), edges, adjacency, "0" * 64)


def test_trace_queries_keep_both_sat_and_unsat_outcomes() -> None:
    graph = path_graph()
    plan_a = (0, 1, 0, 1)
    plan_b = (1, 0, 2, 0)
    masks = masks_from_plans(plan_a, plan_b)
    week = TrafficWeek(
        (
            tuple(map(Decimal, (10, 0, 0, 0))),
            tuple(map(Decimal, (100, 90, 0, 0))),
            tuple(map(Decimal, (100, 90, 80, 0))),
        ),
        "1" * 64,
    )
    queries, timestamps, statuses, receipts, contradictions, stats = build_queries(
        graph, week, plan_a, plan_b, masks, protect_count=1, migrate_count=1,
    )
    assert timestamps == (1, 2)
    assert statuses == ("UNSAT", "SAT")
    assert receipts[0] is None and isinstance(receipts[1], str)
    assert contradictions[0] is not None and contradictions[1] is None
    verify_contradiction(graph.edges, masks, queries[0], contradictions[0])
    assert stats["sat_queries"] == stats["unsat_queries"] == 1
    assert len(queries) == 2


def test_case_round_trip_recomputes_every_outcome(tmp_path) -> None:
    graph = path_graph()
    plan_a = (0, 1, 0, 1)
    plan_b = (1, 0, 2, 0)
    masks = masks_from_plans(plan_a, plan_b)
    week = TrafficWeek(
        (
            tuple(map(Decimal, (10, 0, 0, 0))),
            tuple(map(Decimal, (100, 90, 0, 0))),
            tuple(map(Decimal, (100, 90, 80, 0))),
        ),
        "1" * 64,
    )
    queries, timestamps, statuses, receipts, contradictions, stats = build_queries(
        graph, week, plan_a, plan_b, masks, protect_count=1, migrate_count=1,
    )
    provisional = {
        "schema": "spectra.real_traffic.case.v2",
        "source_week": "fixture",
        "source_sha256": {"traffic": "1" * 64},
        "vertices": 4,
        "edges": graph.edges,
        "demand_indices": (1, 2, 3, 4),
        "demand_names": ("a", "b", "c", "d"),
        "route_links": graph.routes,
        "palette": 3,
        "plan_a": plan_a,
        "plan_b": plan_b,
        "masks": masks,
        "queries": queries,
        "timestamps": timestamps,
        "statuses": statuses,
        "witness_sha256": receipts,
        "contradictions": contradictions,
        **stats,
        "protect_count": 1,
        "migrate_count": 1,
    }
    case = TrafficCase(**provisional, case_sha256=hash_json(provisional))
    case.validate()
    path = tmp_path / "case.json"
    case.write(path)
    assert TrafficCase.read(path) == case

    altered = json.loads(path.read_text())
    altered["statuses"][0] = "SAT"
    path.write_text(json.dumps(altered))
    with pytest.raises(TraceFormatError, match="query receipt differs"):
        TrafficCase.read(path)


def test_contradiction_paths_are_small_and_corruption_is_rejected() -> None:
    edges = ((0, 1),)
    masks = (0b11, 0b11)
    restrictions = ((0, 0b01), (1, 0b01))
    outcome = solve_with_proof(edges, masks, restrictions)
    assert outcome.status == "UNSAT"
    assert outcome.labels is None and outcome.contradiction is not None
    receipt = verify_contradiction(edges, masks, restrictions, outcome.contradiction)
    assert receipt["valid"] is True and receipt["path_edges"] >= 2

    altered = json.loads(json.dumps(outcome.contradiction))
    altered["positive_to_negative"][1] *= -1
    with pytest.raises(InvalidContradiction):
        verify_contradiction(edges, masks, restrictions, altered)
