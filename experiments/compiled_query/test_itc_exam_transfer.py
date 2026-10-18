"""End-to-end contracts for the ITC exam bounded-repair transfer harness."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.compiled_query import itc_exam_transfer as transfer
from spectra.itc2007_exam import parse_itc2007_exam


INSTANCE = """[Exams:5]
120, 1, 2
180, 2, 3
120, 9
120, 4
120, 4
[Periods:4]
01:01:2030, 09:00:00, 180, 0
01:01:2030, 14:00:00, 120, 5
02:01:2030, 09:00:00, 240, 0
02:01:2030, 14:00:00, 180, 0
[Rooms:1]
100, 0
[PeriodHardConstraints]
0, EXCLUSION, 2
1, AFTER, 0
[RoomHardConstraints]
2, ROOM_EXCLUSIVE
[InstitutionalWeightings]
TWOINAROW, 30
"""
SOLUTION = """0, 0
2, 0
1, 0
0, 0
3, 0
"""


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    instance = tmp_path / "fixture.exam"
    solution = tmp_path / "fixture.sol"
    instance.write_text(INSTANCE)
    solution.write_text(SOLUTION)
    license_path = tmp_path / "LICENSE"
    license_path.write_text("fixture redistribution license\n")
    return instance, solution, license_path


def test_solution_parser_and_full_hard_constraint_check():
    instance = parse_itc2007_exam(INSTANCE)
    periods, rooms = transfer.parse_solution(SOLUTION, instance)
    transfer.check_full_baseline(instance, periods, rooms)
    assert periods == (0, 2, 1, 0, 3)
    assert transfer.protected_exams(instance) == {0, 1, 2}
    with pytest.raises(ValueError, match="AFTER"):
        transfer.check_full_baseline(instance, (2, 0, 1, 0, 3), rooms)


def test_candidate_policy_retains_baseline_and_flexes_unprotected_exams():
    instance = parse_itc2007_exam(INSTANCE)
    periods, _ = transfer.parse_solution(SOLUTION, instance)
    candidates, flexible = transfer.derive_candidates(instance, periods, seed=7)
    assert candidates[0] == (0,) and candidates[1] == (2,) and candidates[2] == (1,)
    assert flexible == (3, 4)
    assert all(periods[index] in candidates[index] for index in range(5))


def test_freeze_run_and_independent_audit(tmp_path, monkeypatch):
    instance, solution, license_path = write_inputs(tmp_path)
    root = tmp_path / "evidence"
    monkeypatch.setattr(transfer, "WARMUPS", 0)
    monkeypatch.setattr(transfer, "REPEATS", 2)
    manifest = transfer.freeze(
        root, instance, solution, query_count=8, license_path=license_path,
        source_commit="fixture-source",
        dataset_instance_path="fixtures/custom.exam",
        dataset_solution_path="fixtures/custom.sol",
        evaluation_role="holdout",
    )
    assert manifest["query_count"] == 8
    assert manifest["model"]["exams"] == 5
    assert manifest["dataset_instance_path"] == "fixtures/custom.exam"
    assert manifest["dataset_solution_path"] == "fixtures/custom.sol"
    assert manifest["evaluation_role"] == "holdout"
    assert manifest["warmups"] == 0 and manifest["repeats"] == 2
    record = transfer.run(root)
    assert tuple(record["modes"]) == transfer.MODES
    assert record["warmups"] == 0
    assert record["execution_order"] == [
        ["auto", "raw", "scc", "route"],
        ["raw", "scc", "route", "auto"],
    ]
    result = transfer.audit(root)
    assert result["functional_complete"]
    assert result["baseline_valid_full_itc_hard_constraints"]
    assert result["flexible_exams"] == 2
    assert result["protected_exams"] == 3
    assert result["queries"] == 8
    assert result["evaluation_role"] == "holdout"
    assert result["dataset_instance_path"] == "fixtures/custom.exam"
    assert result["warmups"] == 0 and result["repeats"] == 2
    assert result["auto_vs_raw_median_ratio"] > 0
    assert all(item["sat"] + item["unsat"] == 8 for item in result["modes"].values())
    assert result["modes"]["auto"]["auto_policy"] == "structural-v9"
    assert result["modes"]["auto"]["bank_query_count"] == 8
    assert result["modes"]["raw"]["auto_policy"] == "explicit"
    assert json.loads((root / "audit.json").read_text()) == result



def test_manifest_rejects_invalid_role_and_schedule(tmp_path):
    instance, solution, license_path = write_inputs(tmp_path)
    with pytest.raises(ValueError, match="evaluation_role"):
        transfer.freeze(
            tmp_path / "bad-role", instance, solution, query_count=8,
            license_path=license_path, source_commit="fixture-source",
            evaluation_role="training",
        )

    root = tmp_path / "evidence"
    transfer.freeze(
        root, instance, solution, query_count=8, license_path=license_path,
        source_commit="fixture-source",
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["timing_schedule"] = "mode-by-mode"
    payload = transfer.canonical_json(manifest)
    manifest_path.write_bytes(payload)
    (root / "manifest.sha256").write_text(transfer.digest(payload) + "\n")
    with pytest.raises(ValueError, match="schedule differ"):
        transfer.load_manifest(root)

def test_frozen_artifact_hash_mutation_is_rejected(tmp_path):
    instance, solution, license_path = write_inputs(tmp_path)
    root = tmp_path / "evidence"
    transfer.freeze(root, instance, solution, query_count=8, license_path=license_path,
                    source_commit="fixture-source")
    packet = root / "session.sbc1"
    packet.write_bytes(packet.read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash differs"):
        transfer.load_manifest(root)


def test_audit_rederives_candidates_and_queries_instead_of_trusting_json(tmp_path):
    instance, solution, license_path = write_inputs(tmp_path)
    root = tmp_path / "evidence"
    transfer.freeze(root, instance, solution, query_count=8, license_path=license_path,
                    source_commit="fixture-source")

    state_path = root / "candidates.json"
    state = json.loads(state_path.read_text())
    state["seed"] += 1
    state_bytes = transfer.canonical_json(state)
    state_path.write_bytes(state_bytes)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["candidates.json"] = transfer.digest(state_bytes)
    manifest_bytes = transfer.canonical_json(manifest)
    manifest_path.write_bytes(manifest_bytes)
    (root / "manifest.sha256").write_text(transfer.digest(manifest_bytes) + "\n")

    loaded = transfer.load_manifest(root)
    with pytest.raises(ValueError, match="candidate policy differs"):
        transfer.reconstruct(root, loaded)


def test_failed_freeze_does_not_publish_a_partial_evidence_root(tmp_path):
    instance, solution, license_path = write_inputs(tmp_path)
    root = tmp_path / "evidence"
    solution.write_text("0, 0\n")
    with pytest.raises(ValueError, match="row count differs"):
        transfer.freeze(root, instance, solution, query_count=8, license_path=license_path,
                        source_commit="fixture-source")
    assert not root.exists()
    assert not root.with_name(root.name + ".partial").exists()
