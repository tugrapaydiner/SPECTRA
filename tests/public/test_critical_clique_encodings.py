from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

pysat = pytest.importorskip("pysat")
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Solver

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "critical_clique_coloring"
sys.path.insert(0, str(EXPERIMENT))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load("spectra_cc_core_encoding_tests", EXPERIMENT / "core.py")
bench = load("spectra_cc_benchmark_encoding_tests", EXPERIMENT / "benchmark.py")


def make_graph(n: int, edges: set[tuple[int, int]], name: str):
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


def solve_arm(graph, color_count: int, arm: str, pins=None):
    pins = dict(pins or {})
    quotient = core.build_critical_clique_quotient(graph) if arm == "critical_clique_assignment" else None
    base = (
        graph.n * color_count
        if arm == "direct_assignment"
        else graph.n * (color_count - 1)
        if arm == "direct_pop"
        else quotient.n * color_count
    )
    with Solver(name="cadical195") as solver:
        emitter = bench.FormulaEmitter(solver, top_variable=base, buffer_size=128)
        vpool = IDPool(start_from=base + 1)
        if arm == "direct_assignment":
            bench._add_direct_assignment(
                graph, color_count, pins, emitter,
                vpool=vpool, CardEnc=CardEnc, EncType=EncType,
            )
        elif arm == "direct_pop":
            bench._add_direct_pop(graph, color_count, pins, emitter)
        else:
            bench._add_critical_clique_assignment(
                quotient, color_count, pins, emitter,
                vpool=vpool, CardEnc=CardEnc, EncType=EncType,
            )
        emitter.flush()
        sat = solver.solve()
        if not sat:
            return None
        model = solver.get_model()
        assert model is not None
        if arm == "direct_assignment":
            coloring = bench._decode_direct_assignment(model, graph, color_count)
        elif arm == "direct_pop":
            coloring = bench._decode_direct_pop(model, graph, color_count)
        else:
            coloring = bench._decode_critical_clique_assignment(
                model, graph, quotient, color_count, pins,
            )
        core.verify_coloring(graph, coloring, color_count, pinned_vertex_colors=pins)
        return coloring


def test_all_three_encodings_match_bruteforce():
    rng = random.Random(0x434C49515545)
    checked = 0
    for n in range(1, 8):
        for sample in range(24):
            edges = {
                (u, v)
                for u in range(n)
                for v in range(u + 1, n)
                if rng.random() < 0.31
            }
            graph = make_graph(n, edges, f"enc-{n}-{sample}")
            if n >= 2 and rng.random() < 0.70:
                graph = clone_true_twin(graph, rng.randrange(graph.n))
            for color_count in range(1, 5):
                expected = core.brute_force_colorable(graph, color_count) is not None
                observed = {
                    arm: solve_arm(graph, color_count, arm) is not None
                    for arm in bench.ARMS
                }
                assert all(value == expected for value in observed.values()), (
                    graph.name, color_count, expected, observed
                )
                checked += 1
    assert checked == 7 * 24 * 4


def test_complete_clique_pins_are_identical_across_encodings():
    # Vertices 0 and 1 are one true-twin class; vertex 2 completes a triangle.
    graph = make_graph(3, {(0, 1), (0, 2), (1, 2)}, "pinned-triangle")
    pins = {0: 2, 1: 0, 2: 1}
    for arm in bench.ARMS:
        coloring = solve_arm(graph, 3, arm, pins)
        assert coloring is not None
        assert tuple(coloring[v] for v in range(3)) == (2, 0, 1)


def test_weight_larger_than_palette_is_clean_unsat():
    graph = make_graph(4, {(u, v) for u in range(4) for v in range(u + 1, 4)}, "k4")
    assert solve_arm(graph, 3, "critical_clique_assignment") is None
