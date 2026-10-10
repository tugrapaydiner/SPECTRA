"""Semantic evidence-tamper contracts for the WAP support-query study."""
from __future__ import annotations

from itertools import combinations
import hashlib
import json
from pathlib import Path
import zlib

import pytest

from experiments.wap_support import analyse
from experiments.wap_support.controls import ARMS, order_indices
from experiments.wap_support.run import ADDRESS_SPACE_BYTES, schedule, write_manifest
from experiments.wap_support.workload import QUERY_COUNT, QUERY_WIDTH, WorkloadCase


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _git_blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


class _Checker:
    def __init__(self, n):
        self.n = n
    def check(self, labels, restrictions=()):
        return (type(labels) is bytes and len(labels) == self.n
                and all(0 <= color < 3 for color in labels)
                and all((allowed >> labels[vertex]) & 1
                        for vertex, allowed in restrictions))
    def close(self):
        pass


class _Runtime:
    def __init__(self, _path):
        pass
    def checker(self, n, _k, _edges, _masks, _budget):
        return _Checker(n)


def _reseal(root):
    write_manifest(root)


def _fixture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(analyse, "QuotientRuntime", _Runtime)
    case_root = tmp_path / "cases-package"
    evidence = tmp_path / "evidence"
    runtime = tmp_path / "runtime.so"
    inventory_path = tmp_path / "inventory.json"
    runtime.write_bytes(b"synthetic runtime identity")
    for root in (case_root / "graphs", case_root / "cases", case_root / "generators",
                 evidence / "sessions"):
        root.mkdir(parents=True, exist_ok=True)

    graph_bytes = b"c synthetic audit graph\np edge 16 0\n"
    name = "synthetic.col"
    (case_root / "graphs" / name).write_bytes(graph_bytes)
    graph_sha = hashlib.sha256(graph_bytes).hexdigest()
    blob = _git_blob(graph_bytes)
    vertices = list(combinations(range(16), QUERY_WIDTH))[:QUERY_COUNT]
    queries = tuple(
        tuple((vertex, 1 << ((index + offset) & 1))
              for offset, vertex in enumerate(group))
        for index, group in enumerate(vertices)
    )
    provisional = {
        "schema": "spectra.wap_support.case.v1",
        "graph_name": name,
        "graph_sha256": graph_sha,
        "graph_git_blob_sha1": blob,
        "n": 16,
        "edges": (),
        "dsatur_coloring": (0,) * 16,
        "dsatur_colors": 1,
        "palette": 3,
        "masks": (3,) * 16,
        "queries": queries,
        "generator_attempts": QUERY_COUNT,
        "generator_rejections": 0,
        "generator_unique_model_hashes": QUERY_COUNT,
        "generator_seed": 1,
        "query_width": QUERY_WIDTH,
        "query_count": QUERY_COUNT,
        "clauses": 0,
    }
    case = WorkloadCase(**provisional, case_sha256=_sha_json(provisional))
    case.validate()
    case_path = case_root / "cases" / f"{name}.json"
    generator_path = case_root / "generators" / f"{name}.json"
    case.to_json(case_path)
    attempts = [
        {"query_sha256": _sha_json(query), "duplicate": False, "sat": True,
         "model_sha256": hashlib.sha256(str(index).encode()).hexdigest()}
        for index, query in enumerate(queries)
    ]
    generator = {
        "schema": "spectra.wap_support.generator_receipt.v1",
        "case_sha256": case.case_sha256,
        "attempts": attempts,
        "accepted": QUERY_COUNT,
        "rejected": 0,
        "unique_oracle_models": QUERY_COUNT,
        "construction_ns": 1,
        "graph_role": "prospective_holdout",
        "graph_bytes": len(graph_bytes),
        "graph_sha256": graph_sha,
    }
    generator_path.write_text(json.dumps(generator, sort_keys=True,
                                         separators=(",", ":")) + "\n")
    inventory = {
        "schema": "spectra.wap.upstream_inventory.v2",
        "repository": "synthetic/audit",
        "commit": "0" * 40,
        "graphs": {name: {"git_blob_sha1": blob, "bytes": len(graph_bytes),
                          "role": "prospective_holdout"}},
    }
    inventory_path.write_text(json.dumps(inventory, sort_keys=True) + "\n")
    receipt = {
        "name": name,
        "role": "prospective_holdout",
        "case_sha256": case.case_sha256,
        "case_file_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        "generator_file_sha256": hashlib.sha256(generator_path.read_bytes()).hexdigest(),
        "graph_sha256": graph_sha,
        "graph_git_blob_sha1": blob,
        "construction_ns": 1,
    }
    (case_root / "case_inventory.json").write_text(json.dumps({
        "schema": "spectra.wap_support.case_inventory.v1",
        "upstream_repository": inventory["repository"],
        "upstream_commit": inventory["commit"],
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "roles": ["prospective_holdout"],
        "cases": [receipt],
    }, sort_keys=True, separators=(",", ":")) + "\n")
    _reseal(case_root)

    jobs = schedule([case_path])
    (evidence / "schedule.json").write_text(json.dumps(jobs, sort_keys=True,
                                                        separators=(",", ":")) + "\n")
    (evidence / "environment.json").write_text(json.dumps({
        "address_space_limit_bytes": ADDRESS_SPACE_BYTES,
        "affinity": [0],
        "python_sat_version": "1.9.dev15",
    }, sort_keys=True, separators=(",", ":")) + "\n")
    rows = []
    for number, (case_file, arm, order, seed) in enumerate(jobs):
        session_id = f"{number:04d}-{name}-{arm}-{order}"
        folder = evidence / "sessions" / session_id
        folder.mkdir()
        outputs = []
        for query_index in order_indices(QUERY_COUNT, order, seed):
            labels = bytearray(16)
            for vertex, allowed in queries[query_index]:
                labels[vertex] = allowed.bit_length() - 1
            outputs.append(bytes(labels))
        output = b"".join(outputs)
        compressed = zlib.compress(output, 9)
        (folder / "outputs.bin.zlib").write_bytes(compressed)
        metadata = {
            "schema": "spectra.wap_support.session.v1",
            "arm": arm,
            "case_sha256": case.case_sha256,
            "order_name": order,
            "query_count": QUERY_COUNT,
            "status": "COMPLETE",
            "setup_ns": 100,
            "query_ns": 10 * QUERY_COUNT,
            "dispose_ns": 100,
            "session_ns": 10 * QUERY_COUNT + 300,
            "complete_ns": 10 * QUERY_COUNT + 400,
            "per_query_ns": [10] * QUERY_COUNT,
            "output_sha256": hashlib.sha256(output).hexdigest(),
            "output_file": "outputs.bin.zlib",
            "output_compressed_bytes": len(compressed),
            "output_compressed_sha256": hashlib.sha256(compressed).hexdigest(),
            "unique_outputs": len(set(outputs)),
            "cache_hits": 0,
            "solver_stats": {},
            "candidate_info": {},
            "machine": {"address_space_limit_bytes": ADDRESS_SPACE_BYTES,
                        "threads": 1},
        }
        metadata_path = folder / "session.json"
        metadata_path.write_text(json.dumps(metadata, sort_keys=True,
                                            separators=(",", ":")) + "\n")
        row = {
            "session_id": session_id,
            "case_file": case_file,
            "arm": arm,
            "order": order,
            "order_seed": seed,
            "cold_wall_ns": metadata["complete_ns"] + 1000,
            "complete_ns": metadata["complete_ns"],
            "session_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "output_compressed_sha256": metadata["output_compressed_sha256"],
        }
        (folder / "outer.json").write_text(json.dumps(row, sort_keys=True,
                                                       separators=(",", ":")) + "\n")
        rows.append(row)
    (evidence / "sessions.json").write_text(json.dumps(rows, sort_keys=True,
                                                        separators=(",", ":")) + "\n")
    (evidence / "sessions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows)
    )
    _reseal(evidence)
    return case_root, evidence, runtime, inventory_path


def _validate(paths):
    case_root, evidence, runtime, inventory = paths
    return analyse.validate(case_root, evidence, runtime, inventory,
                            reconstruct=False,
                            expected_roles={"prospective_holdout"})


def _rewrite_session(evidence, row_index, transform):
    rows_path = evidence / "sessions.json"
    rows = json.loads(rows_path.read_text())
    row = rows[row_index]
    folder = evidence / "sessions" / row["session_id"]
    metadata_path = folder / "session.json"
    metadata = json.loads(metadata_path.read_text())
    transform(row, metadata, folder)
    metadata_path.write_text(json.dumps(metadata, sort_keys=True,
                                        separators=(",", ":")) + "\n")
    row["session_metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    row["complete_ns"] = metadata["complete_ns"]
    row["output_compressed_sha256"] = metadata["output_compressed_sha256"]
    (folder / "outer.json").write_text(json.dumps(row, sort_keys=True,
                                                   separators=(",", ":")) + "\n")
    rows_path.write_text(json.dumps(rows, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    (evidence / "sessions.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                for item in rows)
    )
    _reseal(evidence)


def test_complete_semantic_fixture_passes(tmp_path, monkeypatch):
    cases, rows = _validate(_fixture(tmp_path, monkeypatch))
    assert len(cases) == 1 and len(rows) == len(ARMS) * 3


def test_invalid_full_witness_rejected_even_when_receipts_are_resealed(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    evidence = paths[1]
    def transform(_row, metadata, folder):
        output = bytearray(zlib.decompress((folder / "outputs.bin.zlib").read_bytes()))
        output[0] = 2
        compressed = zlib.compress(bytes(output), 9)
        (folder / "outputs.bin.zlib").write_bytes(compressed)
        metadata["output_sha256"] = hashlib.sha256(output).hexdigest()
        metadata["output_compressed_bytes"] = len(compressed)
        metadata["output_compressed_sha256"] = hashlib.sha256(compressed).hexdigest()
    _rewrite_session(evidence, 0, transform)
    with pytest.raises(ValueError, match="invalid output"):
        _validate(paths)


def test_trailing_compressed_payload_rejected(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    evidence = paths[1]
    def transform(_row, metadata, folder):
        path = folder / "outputs.bin.zlib"
        compressed = path.read_bytes() + b"trailing"
        path.write_bytes(compressed)
        metadata["output_compressed_bytes"] = len(compressed)
        metadata["output_compressed_sha256"] = hashlib.sha256(compressed).hexdigest()
    _rewrite_session(evidence, 0, transform)
    with pytest.raises(ValueError, match="trailing"):
        _validate(paths)


def test_boolean_timing_rejected(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    def transform(_row, metadata, _folder):
        metadata["setup_ns"] = True
    _rewrite_session(paths[1], 0, transform)
    with pytest.raises(ValueError, match="invalid timing setup_ns"):
        _validate(paths)


def test_missing_session_rejected_after_manifest_reseal(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    evidence = paths[1]
    rows = json.loads((evidence / "sessions.json").read_text())[:-1]
    (evidence / "sessions.json").write_text(json.dumps(rows, sort_keys=True,
                                                        separators=(",", ":")) + "\n")
    _reseal(evidence)
    with pytest.raises(ValueError, match="session inventory incomplete"):
        _validate(paths)


def test_generator_model_diversity_tamper_rejected(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    case_root = paths[0]
    path = next((case_root / "generators").glob("*.json"))
    data = json.loads(path.read_text())
    data["attempts"][1]["model_sha256"] = data["attempts"][0]["model_sha256"]
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n")
    case_inventory_path = case_root / "case_inventory.json"
    case_inventory = json.loads(case_inventory_path.read_text())
    case_inventory["cases"][0]["generator_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    case_inventory_path.write_text(json.dumps(case_inventory, sort_keys=True,
                                               separators=(",", ":")) + "\n")
    _reseal(case_root)
    with pytest.raises(ValueError, match="model diversity"):
        _validate(paths)


def test_wrong_case_role_rejected(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="roles differ"):
        analyse.validate(paths[0], paths[1], paths[2], paths[3],
                         reconstruct=False, expected_roles={"development"})
