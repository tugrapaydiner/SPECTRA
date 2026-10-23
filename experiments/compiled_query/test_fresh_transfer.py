"""Graph-free contracts for the prospective fresh transfer suite."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from .cases import load_case
from .fresh_transfer import (
    EXPECTED,
    GRAPH_INVENTORY,
    MASTER_SEED,
    QUERY_COUNT,
    SCHEMA,
    build_fresh_case,
)
from .fresh_transfer_benchmark import CADICAL_GATES, DIRECT_GATES, _gate


def _graph_payload(n: int, edges) -> bytes:
    rows = [f"p edge {n} {len(edges)}"]
    rows.extend(f"e {left + 1} {right + 1}" for left, right in edges)
    return ("\n".join(rows) + "\n").encode("utf-8")


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()


def test_frozen_inventory_and_status_schedule_are_exact():
    assert SCHEMA == "spectra.fresh_transfer.v1"
    assert MASTER_SEED == 0x6D4B1F29A7C3E850
    assert QUERY_COUNT == 1024
    assert EXPECTED == tuple(1 if row % 2 == 0 else 0 for row in range(1024))
    assert EXPECTED.count(1) == EXPECTED.count(0) == 512
    assert GRAPH_INVENTORY == {
        "DSJC500.5.col": {
            "family": "random-dimacs",
            "bytes": 599069,
            "git_blob_sha1": "5732c76326e835fa9dd61fb5c4251d93d9a34648",
        },
        "flat300_20_0.col": {
            "family": "flat-planted",
            "bytes": 198295,
            "git_blob_sha1": "3e37b08bfe90ed4dd57654ac959ef8c758729b1c",
        },
        "le450_25c.col": {
            "family": "leighton",
            "bytes": 165335,
            "git_blob_sha1": "615254414a3ef5784d0fc130573f2c14cbf7bba0",
        },
        "miles1500.col": {
            "family": "geometric",
            "bytes": 43473,
            "git_blob_sha1": "937ba541922a0e2ef4d1e9bf49cef49271a8e909",
        },
        "queen16_16.col": {
            "family": "queen",
            "bytes": 58010,
            "git_blob_sha1": "d1e1057d1e70fbea9a1318a0bf53c7321ec62b4c",
        },
        "school1.col": {
            "family": "school-scheduling",
            "bytes": 180775,
            "git_blob_sha1": "10c1165916a0edfb5941af2c5c914b1b6199ff28",
        },
    }


def test_generator_is_deterministic_unique_and_answer_free(tmp_path, monkeypatch):
    name = "synthetic-transfer.col"
    edges = tuple(
        (left, right)
        for left in range(33)
        for right in range(left + 1, 33)
    )
    payload = _graph_payload(33, edges)
    graph = tmp_path / name
    graph.write_bytes(payload)
    monkeypatch.setitem(GRAPH_INVENTORY, name, {
        "family": "synthetic-complete",
        "bytes": len(payload),
        "git_blob_sha1": _git_blob_sha1(payload),
    })

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_receipt = build_fresh_case(graph, first)
    second_receipt = build_fresh_case(graph, second)
    assert first_receipt == second_receipt
    for filename in ("case.json", "expected.json", "generation.json"):
        assert first.joinpath(filename).read_bytes() == second.joinpath(filename).read_bytes()

    case, observed_sha256 = load_case(first / "case.json")
    assert observed_sha256 == first_receipt["case_sha256"]
    assert len(case.queries) == QUERY_COUNT
    assert len(set(case.queries)) == QUERY_COUNT
    assert first_receipt["sat"] == first_receipt["unsat"] == 512
    assert first_receipt["answer_filtering"] is False
    expected = json.loads((first / "expected.json").read_text(encoding="utf-8"))
    assert tuple(expected["statuses"]) == EXPECTED


def test_identity_and_admission_fail_closed(tmp_path, monkeypatch):
    name = "identity-transfer.col"
    edges = tuple(
        (left, right)
        for left in range(33)
        for right in range(left + 1, 33)
    )
    payload = _graph_payload(33, edges)
    graph = tmp_path / name
    graph.write_bytes(payload)
    monkeypatch.setitem(GRAPH_INVENTORY, name, {
        "family": "identity-test",
        "bytes": len(payload),
        "git_blob_sha1": _git_blob_sha1(payload),
    })
    damaged = bytearray(payload)
    damaged[-2] = ord("1") if damaged[-2] != ord("1") else ord("2")
    graph.write_bytes(bytes(damaged))
    with pytest.raises(ValueError, match="Git-blob identity"):
        build_fresh_case(graph, tmp_path / "damaged")

    small_name = "small-transfer.col"
    small_edges = tuple((vertex, vertex + 1) for vertex in range(8))
    small_payload = _graph_payload(9, small_edges)
    small = tmp_path / small_name
    small.write_bytes(small_payload)
    monkeypatch.setitem(GRAPH_INVENTORY, small_name, {
        "family": "small-test",
        "bytes": len(small_payload),
        "git_blob_sha1": _git_blob_sha1(small_payload),
    })
    with pytest.raises(ValueError, match="fewer than 512 edges"):
        build_fresh_case(small, tmp_path / "too-small")


def test_generator_source_contains_no_solver_feedback():
    source = Path(__file__).with_name("fresh_transfer.py").read_text(encoding="utf-8")
    assert "pysat" not in source.lower()
    assert "Solver(" not in source
    assert ".solve(" not in source


def test_large_gap_gates_are_frozen_and_fail_independently():
    assert CADICAL_GATES == {
        1: {"geometric_mean": 0.30, "every_case": 0.50},
        8: {"geometric_mean": 0.25, "every_case": 0.40},
        64: {"geometric_mean": 0.15, "every_case": 0.25},
        1024: {"geometric_mean": 0.10, "every_case": 0.20},
    }
    assert DIRECT_GATES == {
        1: {"geometric_mean": 0.85, "every_case": 1.10},
        8: {"geometric_mean": 0.75, "every_case": 1.00},
        64: {"geometric_mean": 0.50, "every_case": 0.75},
        1024: {"geometric_mean": 0.35, "every_case": 0.60},
    }
    passing = _gate([0.10, 0.12, 0.14], CADICAL_GATES[64])
    assert passing["geometric_mean_pass"] and passing["every_case_pass"]
    mean_failure = _gate([0.31, 0.31, 0.31], CADICAL_GATES[64])
    assert not mean_failure["geometric_mean_pass"]
    case_failure = _gate([0.10, 0.10, 0.26], CADICAL_GATES[64])
    assert case_failure["geometric_mean_pass"]
    assert not case_failure["every_case_pass"]
