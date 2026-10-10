from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
import os
import random

import pytest

from experiments.real_traffic.compiled_support import (
    CompiledSupportRuntime,
    build_compiled_support,
)
from experiments.real_traffic.observer import solve_binary, verify_labels


@pytest.fixture(scope="session")
def compiled_runtime(tmp_path_factory):
    override = os.environ.get("SPECTRA_COMPILED_SUPPORT_LIBRARY")
    path = (override if override is not None else
            build_compiled_support(tmp_path_factory.mktemp("compiled-support")))
    return CompiledSupportRuntime(path)


def all_edges(vertices: int):
    return tuple(combinations(range(vertices), 2))


def test_every_four_vertex_graph_and_binary_assignment(compiled_runtime) -> None:
    bank = all_edges(4)
    masks = (0b11,) * 4
    for edge_mask in range(1 << len(bank)):
        edges = tuple(edge for index, edge in enumerate(bank) if edge_mask >> index & 1)
        with compiled_runtime.prepare(4, 2, edges, masks=masks) as prepared:
            for assignment in product((0, 1), repeat=4):
                restrictions = tuple(
                    (vertex, 1 << color)
                    for vertex, color in enumerate(assignment)
                )
                expected = solve_binary(edges, masks, restrictions)
                observed = prepared.solve(restrictions)
                assert (observed.status == "SAT_VERIFIED") == (expected is not None), (
                    edge_mask, assignment, observed)
                if expected is not None:
                    assert verify_labels(edges, masks, observed.labels, restrictions)


def test_3000_random_list_queries_match_independent_solver(compiled_runtime) -> None:
    rng = random.Random(490331)
    for index in range(3000):
        vertices = rng.randrange(1, 10)
        colors = rng.randrange(1, 7)
        masks = []
        for _ in range(vertices):
            chosen = rng.sample(range(colors), rng.randint(1, min(2, colors)))
            masks.append(sum(1 << color for color in chosen))
        masks = tuple(masks)
        edges = tuple(
            edge for edge in all_edges(vertices) if rng.randrange(4) == 0)
        restrictions = []
        for vertex, mask in enumerate(masks):
            if rng.randrange(3) == 0:
                choices = [color for color in range(colors) if mask >> color & 1]
                restrictions.append((vertex, 1 << rng.choice(choices)))
        restrictions = tuple(restrictions)
        expected = solve_binary(edges, masks, restrictions)
        with compiled_runtime.prepare(vertices, colors, edges, masks=masks) as prepared:
            observed = prepared.solve(restrictions)
            assert (observed.status == "SAT_VERIFIED") == (expected is not None), index
            if expected is not None:
                assert verify_labels(edges, masks, observed.labels, restrictions)


def test_base_contradiction_and_empty_query(compiled_runtime) -> None:
    with compiled_runtime.prepare(2, 1, ((0, 1),), masks=(1, 1)) as prepared:
        result = prepared.solve()
        assert result.status == "UNSAT_REPORTED" and result.labels == b""
    with compiled_runtime.prepare(3, 3, (), masks=(1, 2, 4)) as prepared:
        result = prepared.solve()
        assert result.status == "SAT_VERIFIED"
        assert result.labels == bytes((0, 1, 2))


def test_prepared_index_is_reusable_and_thread_safe(compiled_runtime) -> None:
    edges = ((0, 1), (1, 2), (2, 3))
    masks = (0b11, 0b11, 0b11, 0b11)
    queries = (
        (),
        ((0, 1),),
        ((0, 1), (1, 1)),
        ((0, 2), (3, 1)),
    )
    prepared = compiled_runtime.prepare(4, 2, edges, masks=masks)
    expected = [prepared.solve(query) for query in queries]
    with ThreadPoolExecutor(max_workers=8) as pool:
        observed = list(pool.map(prepared.solve, queries * 20))
    for index, result in enumerate(observed):
        reference = expected[index % len(queries)]
        assert result.status == reference.status
        assert result.labels == reference.labels
    prepared.close()
    prepared.close()
    with pytest.raises(RuntimeError):
        prepared.solve()
    with pytest.raises(RuntimeError):
        _ = prepared.info
    with pytest.raises(RuntimeError):
        prepared.__enter__()


@pytest.mark.parametrize("variables,colors,edges,masks", [
    (True, 2, (), ()),
    (-1, 2, (), ()),
    (100001, 2, (), ()),
    (1, True, (), (1,)),
    (1, 65, (), (1,)),
    (2, 2, [], (1, 2)),
    (2, 2, ((1, 0),), (1, 2)),
    (2, 2, ((0, 0),), (1, 2)),
    (2, 2, ((0, 2),), (1, 2)),
    (2, 2, ((0, 1), (0, 1)), (1, 2)),
    (2, 2, (), (1,)),
    (1, 2, (), [1]),
    (1, 2, (), (0,)),
    (1, 2, (), (0b111,)),
    (1, 2, (), (True,)),
    (1, 2, (), (4,)),
])
def test_invalid_inputs_are_rejected(compiled_runtime, variables, colors, edges, masks) -> None:
    with pytest.raises((ValueError, MemoryError)):
        compiled_runtime.prepare(variables, colors, edges, masks=masks)


@pytest.mark.parametrize("query", [
    [],
    ((0,),),
    ((0, 3),),
    ((-1, 1),),
    ((2, 1),),
    ((0, 1), (0, 2)),
    ((1, 1), (0, 1)),
    ((0, True),),
])
def test_invalid_queries_are_rejected(compiled_runtime, query) -> None:
    with compiled_runtime.prepare(2, 2, ((0, 1),), masks=(0b11, 0b11)) as prepared:
        with pytest.raises(ValueError):
            prepared.solve(query)  # type: ignore[arg-type]


def test_payload_cap_is_enforced(compiled_runtime) -> None:
    with pytest.raises(MemoryError):
        compiled_runtime.prepare(
            100, 2, tuple((index, index + 1) for index in range(99)),
            masks=(0b11,) * 100, max_bytes=100)


def test_explicit_build_does_not_overwrite(tmp_path) -> None:
    (tmp_path / "build.json").write_text("preserve")
    with pytest.raises(FileExistsError):
        build_compiled_support(tmp_path)
    assert (tmp_path / "build.json").read_text() == "preserve"


def test_missing_compiler_has_no_fallback(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        build_compiled_support(tmp_path, compiler="missing-compiled-support-compiler")
