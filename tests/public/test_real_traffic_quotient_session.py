from __future__ import annotations

from itertools import product
import os
import random

import pytest

from experiments.real_traffic.observer import solve_binary, verify_labels
from experiments.real_traffic.quotient_session import (
    QuotientSessionRuntime,
    build_quotient_session,
)
from spectra.cnf.quotient_query import QuotientRuntime, build_quotient_runtime


@pytest.fixture(scope="session")
def quotient_runtime(tmp_path_factory):
    path = os.environ.get("SPECTRA_QUOTIENT_LIBRARY")
    if path is None:
        path = build_quotient_runtime(tmp_path_factory.mktemp("compiled-quotient"))
    return QuotientRuntime(path)


@pytest.fixture(scope="session")
def quotient_session_runtime(tmp_path_factory):
    path = os.environ.get("SPECTRA_QUOTIENT_SESSION_LIBRARY")
    if path is None:
        path = build_quotient_session(tmp_path_factory.mktemp("quotient-session"))
    return QuotientSessionRuntime(path)


def restriction_bank(masks):
    choices = []
    for vertex, mask in enumerate(masks):
        colors = [c for c in range(64) if mask >> c & 1]
        choices.append((None, *colors))
    queries = []
    for selected in product(*choices):
        query = tuple((v, 1 << color) for v, color in enumerate(selected)
                      if color is not None)
        queries.append(query)
    return tuple(queries)


def test_all_path_restrictions_match_generic_search_and_original_observer(
        quotient_runtime, quotient_session_runtime) -> None:
    edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
    masks = (0b0011, 0b0110, 0b1100, 0b1001, 0b0011, 0b0110)
    queries = restriction_bank(masks)
    with quotient_runtime.prepare(6, 4, edges, masks=masks, mode="hybrid") as quotient:
        with quotient_session_runtime.prepare(quotient.certificate()) as compiled:
            scc = compiled.solve_batch(queries)
            closure = compiled.solve_compiled(queries)
            assert compiled.info["vertices"] == 6
            assert compiled.info["implication_edges"] > 0
            for index, query in enumerate(queries):
                generic = quotient.solve(query)
                expected = "SAT" if generic.status == "SAT_VERIFIED" else "UNSAT"
                assert scc.statuses[index] == closure.statuses[index] == expected
                for observed in (scc.labels[index], closure.labels[index]):
                    if expected == "SAT":
                        assert observed is not None
                        assert quotient.check(observed, query)
                        assert verify_labels(edges, masks, observed, query)
                    else:
                        assert observed is None


def test_random_binary_relations_match_independent_2sat(
        quotient_runtime, quotient_session_runtime) -> None:
    rng = random.Random(719381)
    checked = 0
    for _ in range(160):
        vertices = rng.randrange(1, 9)
        palette = rng.randrange(1, 6)
        masks = []
        for _vertex in range(vertices):
            colors = rng.sample(range(palette), rng.randint(1, min(2, palette)))
            masks.append(sum(1 << c for c in colors))
        masks = tuple(masks)
        edges = tuple((a, b) for a in range(vertices) for b in range(a + 1, vertices)
                      if rng.randrange(4) == 0)
        if solve_binary(edges, masks) is None:
            continue
        with quotient_runtime.prepare(vertices, palette, edges, masks=masks,
                                      mode="hybrid") as quotient:
            certificate = quotient.certificate()
            if certificate["impossible"] or any(certificate["wide"]):
                continue
            with quotient_session_runtime.prepare(certificate) as compiled:
                queries = []
                for __ in range(80):
                    rows = []
                    for vertex, mask in enumerate(masks):
                        if rng.randrange(4) == 0:
                            colors = [c for c in range(palette) if mask >> c & 1]
                            rows.append((vertex, 1 << rng.choice(colors)))
                    queries.append(tuple(rows))
                queries = tuple(queries)
                result = compiled.solve_compiled(queries)
                generic = compiled.solve_batch(queries)
                for query, status, labels, generic_status, generic_labels in zip(
                        queries, result.statuses, result.labels,
                        generic.statuses, generic.labels):
                    oracle = solve_binary(edges, masks, query)
                    expected = "SAT" if oracle is not None else "UNSAT"
                    assert status == generic_status == expected
                    if labels is not None:
                        assert quotient.check(labels, query)
                    if generic_labels is not None:
                        assert quotient.check(generic_labels, query)
                checked += 1
    assert checked >= 80


def test_closure_and_scc_paths_are_deterministic(quotient_runtime, quotient_session_runtime) -> None:
    edges = tuple((i, i + 1) for i in range(31))
    masks = tuple(3 for _ in range(32))
    queries = tuple(((v, 1 << (v & 1)),) for v in range(32))
    with quotient_runtime.prepare(32, 2, edges, masks=masks, mode="hybrid") as quotient:
        with quotient_session_runtime.prepare(quotient.certificate()) as compiled:
            first = compiled.solve_compiled(queries)
            second = compiled.solve_compiled(queries)
            assert first.statuses == second.statuses
            assert first.labels == second.labels
            assert first.traversed_edges == second.traversed_edges
            assert compiled.solve_batch(queries).statuses == first.statuses


def test_tampered_certificate_and_caps_are_rejected(quotient_runtime, quotient_session_runtime) -> None:
    with quotient_runtime.prepare(2, 2, ((0, 1),), masks=(3, 3), mode="hybrid") as q:
        certificate = q.certificate()
    broken = dict(certificate)
    broken["wide"] = list(broken["wide"])
    broken["wide"][0] = 1
    with pytest.raises(ValueError, match="binary original"):
        quotient_session_runtime.prepare(broken)
    with pytest.raises(MemoryError):
        quotient_session_runtime.prepare(certificate, max_bytes=1)
    prepared = quotient_session_runtime.prepare(certificate)
    with pytest.raises(MemoryError):
        prepared.solve_compiled(((),) * 1000, max_bytes=1)
    prepared.close()
    prepared.close()
    with pytest.raises(RuntimeError):
        prepared.solve_compiled(((),))
    with pytest.raises(RuntimeError):
        _ = prepared.info
