from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "experiments" / "critical_clique_coloring" / "core.py"
SPEC = importlib.util.spec_from_file_location("spectra_critical_clique_core", MODULE)
assert SPEC and SPEC.loader
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def make_graph(n: int, edges: set[tuple[int, int]], name: str = "generated"):
    normalized = tuple(sorted((min(u, v), max(u, v)) for u, v in edges if u != v))
    adjacency = [set() for _ in range(n)]
    for u, v in normalized:
        adjacency[u].add(v)
        adjacency[v].add(u)
    return core.Graph(
        name=name,
        path=Path(name),
        n=n,
        edges=normalized,
        adjacency=tuple(frozenset(x) for x in adjacency),
        sha256=f"generated-{name}",
    )


def clone_true_twin(graph, vertex: int):
    new_vertex = graph.n
    edges = set(graph.edges)
    edges.add((vertex, new_vertex))
    for neighbor in graph.adjacency[vertex]:
        edges.add((min(neighbor, new_vertex), max(neighbor, new_vertex)))
    return make_graph(graph.n + 1, edges, graph.name + "-clone")


def test_true_twin_quotient_and_reconstruction():
    base = make_graph(4, {(0, 1), (1, 2), (2, 3), (3, 0)})
    graph = clone_true_twin(base, 0)
    quotient = core.build_critical_clique_quotient(graph)
    assert quotient.vertices_removed == 1
    original = core.brute_force_colorable(graph, 3)
    assert original is not None
    selected = core.selected_colors_from_coloring(original, quotient)
    lifted = core.reconstruct_coloring(graph, quotient, selected, 3)
    core.verify_coloring(graph, lifted, 3)


def test_weighted_quotient_exactness_on_random_small_graphs():
    rng = random.Random(0x53504543545241)
    checked = 0
    for n in range(1, 8):
        for sample in range(45):
            edges = {
                (u, v)
                for u in range(n)
                for v in range(u + 1, n)
                if rng.random() < 0.30
            }
            graph = make_graph(n, edges, f"r-{n}-{sample}")
            if n >= 2 and rng.random() < 0.65:
                graph = clone_true_twin(graph, rng.randrange(graph.n))
            quotient = core.build_critical_clique_quotient(graph)
            for color_count in range(1, 5):
                direct = core.brute_force_colorable(graph, color_count)
                weighted = core.brute_force_weighted_colorable(quotient, color_count)
                assert (direct is not None) == (weighted is not None)
                if weighted is not None:
                    lifted = core.reconstruct_coloring(graph, quotient, weighted, color_count)
                    core.verify_coloring(graph, lifted, color_count)
                checked += 1
    assert checked == 7 * 45 * 4


def test_pins_are_preserved_inside_one_critical_clique():
    graph = make_graph(3, {(0, 1), (0, 2), (1, 2)}, "triangle")
    quotient = core.build_critical_clique_quotient(graph)
    assert quotient.weights == (3,)
    selected = (frozenset({0, 1, 2}),)
    pins = {0: 2, 1: 0, 2: 1}
    lifted = core.reconstruct_coloring(graph, quotient, selected, 3, pinned_vertex_colors=pins)
    assert lifted == (2, 0, 1)


def test_invalid_weighted_overlap_is_rejected():
    graph = make_graph(3, {(0, 1), (1, 2)}, "path")
    quotient = core.build_critical_clique_quotient(graph)
    assert quotient.n == 3
    try:
        core.verify_weighted_quotient_coloring(quotient, ({0}, {0}, {1}), 2)
    except AssertionError as error:
        assert "shares colors" in str(error)
    else:
        raise AssertionError("overlapping quotient colors were accepted")
