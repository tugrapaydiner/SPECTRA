"""Deterministic witness-colouring and topology fallback contracts."""
from __future__ import annotations

from pathlib import Path

from experiments.critical_clique_coloring.core import (
    parse_dimacs_col,
    verify_coloring,
)

from .witness_colouring import (
    best_witness_colouring,
    colour_count,
    degree_capped_subgraph,
    prepare_witness_topology,
)


def _write_graph(path: Path, n: int, edges) -> None:
    rows = [f"p edge {n} {len(edges)}"]
    rows.extend(f"e {left + 1} {right + 1}" for left, right in edges)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_portfolio_is_deterministic_and_proper(tmp_path):
    edges = tuple(
        (vertex, (vertex + 1) % 17)
        if vertex < (vertex + 1) % 17
        else ((vertex + 1) % 17, vertex)
        for vertex in range(17)
    )
    edges = tuple(sorted(set(edges)))
    path = tmp_path / "cycle.col"
    _write_graph(path, 17, edges)
    graph = parse_dimacs_col(path)

    first, first_receipt = best_witness_colouring(graph, restarts=16)
    second, second_receipt = best_witness_colouring(graph, restarts=16)
    assert first == second
    assert first_receipt == second_receipt
    assert colour_count(first) == 3
    verify_coloring(graph, first, 3)


def test_degree_capped_subgraph_is_stable_and_bounded(tmp_path):
    edges = tuple(
        (left, right)
        for left in range(70)
        for right in range(left + 1, 70)
    )
    path = tmp_path / "clique.col"
    _write_graph(path, 70, edges)
    graph = parse_dimacs_col(path)

    first = degree_capped_subgraph(graph, degree_cap=62)
    second = degree_capped_subgraph(graph, degree_cap=62)
    assert first.edges == second.edges
    assert first.sha256 == second.sha256
    assert set(first.edges).issubset(edges)
    assert max(map(len, first.adjacency)) <= 62
    assert len(first.edges) >= 512


def test_preparation_keeps_admissible_graph_and_caps_inadmissible_graph(tmp_path):
    cycle_path = tmp_path / "cycle.col"
    cycle_edges = tuple((vertex, vertex + 1) for vertex in range(20))
    _write_graph(cycle_path, 21, cycle_edges)
    cycle = parse_dimacs_col(cycle_path)
    retained = prepare_witness_topology(cycle, restarts=8)
    assert retained.receipt["topology_mode"] == "full"
    assert retained.graph.edges == cycle.edges
    assert retained.receipt["witness_colours"] <= 63

    clique_path = tmp_path / "clique.col"
    clique_edges = tuple(
        (left, right)
        for left in range(70)
        for right in range(left + 1, 70)
    )
    _write_graph(clique_path, 70, clique_edges)
    clique = parse_dimacs_col(clique_path)
    capped = prepare_witness_topology(clique, restarts=8)
    assert capped.receipt["topology_mode"] == "degree-capped"
    assert capped.receipt["full_graph_best_colours"] == 70
    assert capped.receipt["degree_cap"] == 62
    assert capped.receipt["maximum_retained_degree"] <= 62
    assert capped.receipt["witness_colours"] <= 63
    verify_coloring(
        capped.graph,
        capped.colouring,
        capped.receipt["witness_colours"],
    )
