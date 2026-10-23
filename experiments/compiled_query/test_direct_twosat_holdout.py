"""Answer-free contracts for the frozen mixed direct 2-SAT holdout generator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.critical_clique_coloring.core import parse_dimacs_col
from experiments.wap_support.workload import deterministic_dsatur

from .cases import load_case
from .direct_twosat_holdout import (
    EXPECTED,
    GRAPH_INVENTORY,
    QUERY_COUNT,
    _git_blob_sha1,
    build_holdout_case,
)


def _write_graph(path: Path, n: int, edges) -> bytes:
    lines = [f"p edge {n} {len(edges)}"]
    lines.extend(f"e {left + 1} {right + 1}" for left, right in edges)
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return payload


def test_frozen_status_vector_is_exactly_balanced():
    assert len(EXPECTED) == QUERY_COUNT == 1024
    assert EXPECTED.count(1) == EXPECTED.count(0) == 512
    assert EXPECTED[:8] == (1, 0, 1, 0, 1, 0, 1, 0)


def test_generator_is_deterministic_and_answer_free(tmp_path, monkeypatch):
    name = "test-holdout.col"
    graph = tmp_path / name
    edges = tuple((vertex, vertex + 1) for vertex in range(519))
    payload = _write_graph(graph, 520, edges)
    record = {
        "family": "test-path",
        "bytes": len(payload),
        "git_blob_sha1": _git_blob_sha1(payload),
    }
    monkeypatch.setitem(GRAPH_INVENTORY, name, record)

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_receipt = build_holdout_case(graph, first)
    second_receipt = build_holdout_case(graph, second)
    assert first_receipt == second_receipt
    for filename in ("case.json", "expected.json", "generation.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    case, case_sha256 = load_case(first / "case.json")
    assert case_sha256 == first_receipt["case_sha256"]
    assert len(case.queries) == 1024
    assert len(set(case.queries)) == 1024
    expected = json.loads((first / "expected.json").read_text())
    assert expected["sat"] == expected["unsat"] == 512
    assert tuple(expected["statuses"]) == EXPECTED

    witness = deterministic_dsatur(parse_dimacs_col(graph))
    for row, query in enumerate(case.queries):
        if EXPECTED[row]:
            assert all((allowed >> witness[vertex]) & 1
                       for vertex, allowed in query)
        else:
            trap = 1 << first_receipt["trap_colour"]
            trapped = {vertex for vertex, allowed in query if allowed == trap}
            assert any(left in trapped and right in trapped for left, right in edges)


def test_graph_identity_and_edge_admission_fail_closed(tmp_path, monkeypatch):
    name = "tiny-holdout.col"
    graph = tmp_path / name
    edges = tuple((vertex, vertex + 1) for vertex in range(9))
    payload = _write_graph(graph, 10, edges)
    monkeypatch.setitem(GRAPH_INVENTORY, name, {
        "family": "test-tiny",
        "bytes": len(payload),
        "git_blob_sha1": _git_blob_sha1(payload),
    })
    with pytest.raises(ValueError, match="fewer than 512 edges"):
        build_holdout_case(graph, tmp_path / "too-small")

    monkeypatch.setitem(GRAPH_INVENTORY, name, {
        "family": "test-tiny",
        "bytes": len(payload),
        "git_blob_sha1": "0" * 40,
    })
    with pytest.raises(ValueError, match="Git-blob identity"):
        build_holdout_case(graph, tmp_path / "wrong-blob")
