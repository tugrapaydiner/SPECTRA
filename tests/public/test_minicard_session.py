from __future__ import annotations

from itertools import product
import os
import random

import pytest

from experiments.real_traffic.minicard_session import (
    MiniCardSessionRuntime,
    build_minicard_session,
)
from experiments.real_traffic.observer import solve_binary, verify_labels


@pytest.fixture(scope="session")
def runtime(tmp_path_factory):
    library = os.environ.get("SPECTRA_MINICARD_SESSION_LIBRARY")
    if library is None:
        source = os.environ.get("SPECTRA_MINICARD_SOURCE")
        if source is None:
            pytest.skip("pinned MiniCard source is not available in this environment")
        library = build_minicard_session(
            tmp_path_factory.mktemp("minicard-session"), source=source)
    return MiniCardSessionRuntime(library)


def all_queries(masks):
    banks = []
    for mask in masks:
        colors = [c for c in range(64) if mask >> c & 1]
        banks.append((None, *colors))
    return tuple(
        tuple((v, 1 << color) for v, color in enumerate(selection)
              if color is not None)
        for selection in product(*banks)
    )


def test_complete_query_relation_matches_independent_2sat(runtime) -> None:
    edges = ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3))
    masks = (0b0011, 0b0110, 0b1100, 0b1001)
    queries = all_queries(masks)
    with runtime.prepare(edges, masks) as prepared:
        result = prepared.solve_batch(queries)
        assert prepared.info["variables"] == 4
        assert prepared.info["clauses"] > 0
        for query, status, labels in zip(queries, result.statuses, result.labels):
            oracle = solve_binary(edges, masks, query)
            assert status == ("SAT" if oracle is not None else "UNSAT")
            assert labels is None or verify_labels(edges, masks, labels, query)


def test_random_sessions_match_independent_observer(runtime) -> None:
    rng = random.Random(608194)
    for _ in range(120):
        vertices = rng.randrange(1, 10)
        palette = rng.randrange(1, 6)
        masks = []
        for _vertex in range(vertices):
            colors = rng.sample(range(palette), rng.randint(1, min(2, palette)))
            masks.append(sum(1 << c for c in colors))
        masks = tuple(masks)
        edges = tuple((a, b) for a in range(vertices) for b in range(a + 1, vertices)
                      if rng.randrange(4) == 0)
        queries = []
        for __ in range(100):
            rows = []
            for vertex, mask in enumerate(masks):
                if rng.randrange(4) == 0:
                    colors = [c for c in range(palette) if mask >> c & 1]
                    rows.append((vertex, 1 << rng.choice(colors)))
            queries.append(tuple(rows))
        with runtime.prepare(edges, masks) as prepared:
            result = prepared.solve_batch(tuple(queries))
        for query, status, labels in zip(queries, result.statuses, result.labels):
            oracle = solve_binary(edges, masks, query)
            assert status == ("SAT" if oracle is not None else "UNSAT")
            assert labels is None or verify_labels(edges, masks, labels, query)


def test_persistent_learning_and_deterministic_statuses(runtime) -> None:
    vertices = 60
    edges = tuple((v, v + 1) for v in range(vertices - 1))
    masks = tuple(3 for _ in range(vertices))
    queries = tuple(((v, 1 << (v & 1)),) for v in range(vertices))
    with runtime.prepare(edges, masks) as prepared:
        first = prepared.solve_batch(queries)
        second = prepared.solve_batch(queries)
        assert first.statuses == second.statuses
        assert all(status == "SAT" for status in first.statuses)
        assert sum(second.conflicts) <= sum(first.conflicts)


def test_invalid_inputs_caps_and_lifecycle(runtime) -> None:
    with pytest.raises(ValueError):
        runtime.prepare(((1, 0),), (3, 3))
    with pytest.raises(ValueError):
        runtime.prepare((), (0,))
    with pytest.raises(ValueError):
        runtime.prepare((), (0b111,))
    with pytest.raises(MemoryError):
        runtime.prepare((), (3,) * 100, max_bytes=1)
    prepared = runtime.prepare(((0, 1),), (3, 3))
    with pytest.raises(MemoryError):
        prepared.solve_batch(((),) * 1000, max_bytes=1)
    prepared.close()
    prepared.close()
    with pytest.raises(RuntimeError):
        prepared.solve_batch(((),))
    with pytest.raises(RuntimeError):
        _ = prepared.info
