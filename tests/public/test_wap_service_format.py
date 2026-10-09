"""Contracts for the hash-bound WAP deployment format and service endpoint."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import pickle
import struct

import pytest

from experiments.wap_service_format.format import (
    FLAGS, HEADER, MAGIC, VERSION, PackedCaseError, load_packed_case,
    pack_case, pack_directory, read_manifest, receipt_by_name,
)
from experiments.wap_support.controls import run_prevalidated_session, run_session
from experiments.wap_support.workload import WorkloadCase, _sha_json
from spectra.cnf.quotient_query import QuotientRuntime, build_quotient_runtime


def small_case() -> WorkloadCase:
    queries = tuple(
        tuple((vertex, 1 << ((query >> vertex) & 1)) for vertex in range(8))
        for query in range(8)
    )
    provisional = {
        "schema": "spectra.wap_support.case.v1", "graph_name": "small.col",
        "graph_sha256": "0" * 64, "graph_git_blob_sha1": "0" * 40,
        "n": 8, "edges": (), "dsatur_coloring": (0,) * 8,
        "dsatur_colors": 1, "palette": 3, "masks": (3,) * 8,
        "queries": queries, "generator_attempts": 8, "generator_rejections": 0,
        "generator_unique_model_hashes": 8, "generator_seed": 0,
        "query_width": 8, "query_count": 8, "clauses": 0,
    }
    result = WorkloadCase(**provisional, case_sha256=_sha_json(provisional))
    result.validate()
    return result


@pytest.fixture(scope="session")
def service_runtime(tmp_path_factory):
    path = build_quotient_runtime(tmp_path_factory.mktemp("packed-wap-native"))
    return QuotientRuntime(path)


def test_packed_round_trip_is_exact_and_manifest_bound(tmp_path):
    source = tmp_path / "canonical"
    packed = tmp_path / "packed"
    source.mkdir()
    case = small_case(); case.to_json(source / "small.json")
    manifest = pack_directory(source, packed)
    parsed = read_manifest(packed / "MANIFEST.json")
    assert parsed == manifest
    receipts = receipt_by_name(parsed)
    receipt = receipts["small.spwap"]
    loaded = load_packed_case(packed / receipt.packed_name,
                              expected_sha256=receipt.packed_file_sha256)
    assert loaded.case == case
    assert loaded.case.case_sha256 == receipt.case_sha256
    assert loaded.file_sha256 == hashlib.sha256((packed / receipt.packed_name).read_bytes()).hexdigest()
    assert receipt.packed_bytes == (packed / receipt.packed_name).stat().st_size
    assert receipt.source_file_sha256 == hashlib.sha256((source / "small.json").read_bytes()).hexdigest()


def test_loader_skips_canonical_revalidation_only_after_hash_check(tmp_path, monkeypatch):
    case = small_case(); source = tmp_path / "case.json"; case.to_json(source)
    destination = tmp_path / "case.spwap"
    receipt = pack_case(source, destination)
    monkeypatch.setattr(WorkloadCase, "validate", lambda _self: (_ for _ in ()).throw(AssertionError("revalidated")))
    loaded = load_packed_case(destination, expected_sha256=receipt.packed_file_sha256)
    assert loaded.case == case


def test_wrong_manifest_hash_and_payload_tampering_are_refused(tmp_path):
    case = small_case(); source = tmp_path / "case.json"; case.to_json(source)
    destination = tmp_path / "case.spwap"; receipt = pack_case(source, destination)
    with pytest.raises(PackedCaseError, match="manifest"):
        load_packed_case(destination, expected_sha256="0" * 64)
    raw = bytearray(destination.read_bytes()); raw[-1] ^= 1
    tampered = tmp_path / "tampered.spwap"; tampered.write_bytes(raw)
    with pytest.raises(PackedCaseError, match="payload"):
        load_packed_case(tampered, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_data_only_unpickler_refuses_global_objects(tmp_path):
    payload = pickle.dumps(__import__("os").system, protocol=5)
    header = HEADER.pack(MAGIC, VERSION, FLAGS, 1, 1, len(payload),
                         hashlib.sha256(payload).digest(), b"\0" * 32)
    artifact = header + payload
    path = tmp_path / "malicious.spwap"; path.write_bytes(artifact)
    with pytest.raises(PackedCaseError, match="decode failed"):
        load_packed_case(path, expected_sha256=hashlib.sha256(artifact).hexdigest())


def test_header_geometry_changes_are_refused_even_with_resealed_hash(tmp_path):
    case = small_case(); source = tmp_path / "case.json"; case.to_json(source)
    destination = tmp_path / "case.spwap"; pack_case(source, destination)
    raw = bytearray(destination.read_bytes())
    values = list(HEADER.unpack_from(raw)); values[3] += 1
    raw[:HEADER.size] = HEADER.pack(*values)
    altered = tmp_path / "geometry.spwap"; altered.write_bytes(raw)
    with pytest.raises(PackedCaseError, match="vertex count"):
        load_packed_case(altered, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_service_endpoint_matches_validating_endpoint(service_runtime, tmp_path):
    case = small_case(); source = tmp_path / "case.json"; case.to_json(source)
    destination = tmp_path / "case.spwap"; receipt = pack_case(source, destination)
    loaded = load_packed_case(destination, expected_sha256=receipt.packed_file_sha256)
    canonical = run_session(case, service_runtime, "scc", order_name="shuffle", order_seed=17)
    service = run_prevalidated_session(loaded.case, service_runtime, "scc",
                                       order_name="shuffle", order_seed=17)
    assert canonical.status == service.status == "COMPLETE"
    assert canonical.output_bytes == service.output_bytes
    assert canonical.output_sha256 == service.output_sha256
    assert canonical.unique_outputs == service.unique_outputs


def test_canonical_endpoint_still_rejects_an_invalid_direct_object(service_runtime):
    case = replace(small_case(), case_sha256="0" * 64)
    with pytest.raises(ValueError, match="digest"):
        run_session(case, service_runtime, "scc", order_name="forward", order_seed=0)


def test_packer_refuses_invalid_canonical_json(tmp_path):
    case = replace(small_case(), case_sha256="0" * 64)
    source = tmp_path / "bad.json"; case.to_json(source)
    with pytest.raises(ValueError, match="digest"):
        pack_case(source, tmp_path / "bad.spwap")


def test_manifest_rejects_unknown_receipt_fields(tmp_path):
    root = tmp_path / "packed"; root.mkdir()
    (root / "MANIFEST.json").write_text('{"schema":"spectra.wap_service_format.manifest.v1","format_magic_hex":"'
        + MAGIC.hex() + '","format_version":1,"cases":[{"extra":1}]}')
    with pytest.raises(PackedCaseError, match="fields"):
        read_manifest(root / "MANIFEST.json")
