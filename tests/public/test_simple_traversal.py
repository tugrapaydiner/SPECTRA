from __future__ import annotations

from itertools import combinations
import os
import random

import pytest

from experiments.real_traffic.observer import solve_binary, verify_labels
from experiments.real_traffic.simple_traversal import (
    SimpleTraversalRuntime,
    build_simple_traversal,
)


@pytest.fixture(scope="session")
def traversal(tmp_path_factory):
    override = os.environ.get("SPECTRA_SIMPLE_TRAVERSAL_LIBRARY")
    library = override or build_simple_traversal(
        tmp_path_factory.mktemp("simple-traversal")
    )
    return SimpleTraversalRuntime(library)


def test_every_five_vertex_graph_and_single_vertex_query(traversal):
    bank = tuple(combinations(range(5), 2))
    masks = (0b11,) * 5
    for bits in range(1 << len(bank)):
        edges = tuple(edge for index, edge in enumerate(bank) if bits >> index & 1)
        expected_base = solve_binary(edges, masks)
        if expected_base is None:
            with pytest.raises(ValueError):
                traversal.prepare(edges, masks)
            continue
        with traversal.prepare(edges, masks) as prepared:
            queries = [()]
            for vertex in range(5):
                queries.extend((((vertex, 1),), ((vertex, 2),)))
            for query in queries:
                expected = solve_binary(edges, masks, query)
                observed = prepared.solve(query)
                assert (observed.status == "SAT") == (expected is not None)
                if expected is not None:
                    assert verify_labels(edges, masks, observed.labels, query)


def test_random_binary_lists_match_independent_solver(traversal):
    rng = random.Random(771902)
    for _ in range(1000):
        vertices = rng.randrange(1, 25)
        palette = rng.randrange(1, 8)
        masks = []
        for _vertex in range(vertices):
            colours = rng.sample(
                range(palette), rng.randint(1, min(2, palette))
            )
            masks.append(sum(1 << colour for colour in colours))
        masks = tuple(masks)
        edges = tuple(
            (left, right)
            for left, right in combinations(range(vertices), 2)
            if rng.randrange(6) == 0
        )
        expected_base = solve_binary(edges, masks)
        if expected_base is None:
            with pytest.raises(ValueError):
                traversal.prepare(edges, masks)
            continue
        with traversal.prepare(edges, masks) as prepared:
            for _query in range(20):
                chosen = rng.sample(
                    range(vertices), rng.randrange(min(vertices, 6) + 1)
                )
                query = []
                for vertex in sorted(chosen):
                    colours = [
                        colour for colour in range(palette)
                        if masks[vertex] >> colour & 1
                    ]
                    query.append((vertex, 1 << rng.choice(colours)))
                query = tuple(query)
                expected = solve_binary(edges, masks, query)
                observed = prepared.solve(query)
                assert (observed.status == "SAT") == (expected is not None)
                if expected is not None:
                    assert verify_labels(edges, masks, observed.labels, query)


def test_payload_and_lifecycle(traversal):
    with pytest.raises(MemoryError):
        traversal.prepare((), (0b11,) * 100, max_bytes=1)
    prepared = traversal.prepare(((0, 1),), (0b11, 0b11))
    assert prepared.info["vertices"] == 2
    assert prepared.solve(((0, 1), (1, 1))).status == "UNSAT"
    prepared.close()
    prepared.close()
    with pytest.raises(RuntimeError):
        prepared.solve()


@pytest.mark.parametrize(
    "edges,masks",
    [
        ([], (0b11,)),
        ((), [0b11]),
        (((0, 0),), (0b11,)),
        (((0, 2),), (0b11, 0b11)),
        ((), (0,)),
        ((), (0b111,)),
        ((), (True,)),
    ],
)
def test_invalid_inputs(traversal, edges, masks):
    with pytest.raises((ValueError, TypeError)):
        traversal.prepare(edges, masks)
