from __future__ import annotations

import os

import pytest

from experiments.real_traffic.certificate import solve_with_proof
from experiments.real_traffic.session_audit import (
    SessionAuditRuntime,
    build_session_audit,
)


@pytest.fixture(scope="session")
def audit_runtime(tmp_path_factory):
    override = os.environ.get("SPECTRA_SESSION_AUDIT_LIBRARY")
    path = (override if override is not None else
            build_session_audit(tmp_path_factory.mktemp("session-audit")))
    return SessionAuditRuntime(path)


def fixture_relation():
    edges = ((0, 1), (1, 2), (2, 3))
    masks = (0b0011, 0b0011, 0b0101, 0b0101)
    queries = (
        ((0, 0b0001),),
        ((0, 0b0001), (1, 0b0001)),
        ((2, 0b0100),),
    )
    outcomes = [solve_with_proof(edges, masks, query) for query in queries]
    statuses = tuple(outcome.status for outcome in outcomes)
    contradictions = tuple(outcome.contradiction for outcome in outcomes)
    status_bytes = bytes(0 if status == "SAT" else 1 for status in statuses)
    labels = b"".join(
        bytes(outcome.labels) if outcome.labels is not None else bytes(len(masks))
        for outcome in outcomes
    )
    return edges, masks, queries, statuses, contradictions, status_bytes, labels


def test_complete_session_is_checked_in_one_native_pass(audit_runtime) -> None:
    edges, masks, queries, statuses, proofs, observed, labels = fixture_relation()
    with audit_runtime.prepare(edges, masks, queries, statuses, proofs) as auditor:
        result = auditor.verify(observed, labels)
        assert auditor.info["queries"] == len(queries)
    assert result.sat == 2
    assert result.unsat == 1
    assert result.proof_edges > 0
    assert len(result.query_ns) == len(queries)
    assert result.elapsed_ns >= sum(result.query_ns)


def test_status_and_witness_corruption_are_rejected(audit_runtime) -> None:
    edges, masks, queries, statuses, proofs, observed, labels = fixture_relation()
    with audit_runtime.prepare(edges, masks, queries, statuses, proofs) as auditor:
        changed_status = bytes([observed[0] ^ 1, *observed[1:]])
        with pytest.raises(RuntimeError, match="status differs"):
            auditor.verify(changed_status, labels)
        changed_labels = bytearray(labels)
        changed_labels[0] = 63
        with pytest.raises(RuntimeError, match="original list"):
            auditor.verify(observed, bytes(changed_labels))
        with pytest.raises(ValueError, match="geometry differs"):
            auditor.verify(observed[:-1], labels)


def test_corrupted_retained_proof_is_rejected_during_preparation(audit_runtime) -> None:
    edges, masks, queries, statuses, proofs, _observed, _labels = fixture_relation()
    proofs = list(proofs)
    altered = dict(proofs[1])
    altered["positive_to_negative"] = list(altered["positive_to_negative"])
    altered["positive_to_negative"][0] *= -1
    proofs[1] = altered
    with pytest.raises(ValueError, match="endpoints differ"):
        audit_runtime.prepare(edges, masks, queries, statuses, tuple(proofs))


def test_sparse_fallback_and_dense_path_accept_the_same_relation(audit_runtime) -> None:
    masks = (0b11,) * 80
    sparse_edges = tuple((i, i + 1) for i in range(79))
    dense_edges = tuple((i, j) for i in range(20) for j in range(i + 1, 20))
    for edges in (sparse_edges, dense_edges):
        queries = (((0, 1),),)
        outcome = solve_with_proof(edges, masks, queries[0])
        statuses = (outcome.status,)
        proofs = (outcome.contradiction,)
        observed = bytes((0 if outcome.status == "SAT" else 1,))
        labels = (bytes(outcome.labels) if outcome.labels is not None
                  else bytes(len(masks)))
        with audit_runtime.prepare(edges, masks, queries, statuses, proofs) as auditor:
            result = auditor.verify(observed, labels)
        assert result.sat + result.unsat == 1


def test_payload_cap_and_closed_lifecycle(audit_runtime) -> None:
    edges, masks, queries, statuses, proofs, observed, labels = fixture_relation()
    with pytest.raises(MemoryError):
        audit_runtime.prepare(edges, masks, queries, statuses, proofs, max_bytes=1)
    prepared = audit_runtime.prepare(edges, masks, queries, statuses, proofs)
    assert prepared.verify(observed, labels).sat == 2
    prepared.close(); prepared.close()
    with pytest.raises(RuntimeError):
        prepared.verify(observed, labels)
    with pytest.raises(RuntimeError):
        _ = prepared.info


def test_builder_does_not_overwrite(tmp_path) -> None:
    (tmp_path / "build.json").write_text("preserve")
    with pytest.raises(FileExistsError):
        build_session_audit(tmp_path)
    assert (tmp_path / "build.json").read_text() == "preserve"
