"""Contracts for the prospective multi-instance ITC transfer suite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.compiled_query import itc_exam_suite as suite
from experiments.compiled_query import itc_exam_transfer as transfer
from experiments.compiled_query.test_itc_exam_transfer import INSTANCE, SOLUTION


def fixture_cases(upstream: Path) -> tuple[suite.SuiteCase, ...]:
    records = (
        suite.SuiteCase("fixture-development", "development", "instances/dev.exam", "solutions/dev.sol"),
        suite.SuiteCase("fixture-holdout-a", "holdout", "instances/a.exam", "solutions/a.sol"),
        suite.SuiteCase("fixture-holdout-b", "holdout", "instances/b.exam", "solutions/b.sol"),
    )
    (upstream / "instances").mkdir(parents=True)
    (upstream / "solutions").mkdir(parents=True)
    for case in records:
        (upstream / case.instance_path).write_text(INSTANCE)
        (upstream / case.solution_path).write_text(SOLUTION)
    (upstream / "LICENSE").write_text("fixture redistribution license\n")
    return records


def test_case_validation_rejects_escape_and_missing_holdout():
    with pytest.raises(ValueError, match="under upstream root"):
        suite.SuiteCase("bad", "holdout", "../bad.exam", "x.sol").validate()
    with pytest.raises(ValueError, match="holdout"):
        suite._cases((suite.SuiteCase("dev", "development", "x.exam", "x.sol"),))


def test_suite_freeze_run_and_audit(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    cases = fixture_cases(upstream)
    root = tmp_path / "suite"
    monkeypatch.setattr(transfer, "WARMUPS", 0)
    monkeypatch.setattr(transfer, "REPEATS", 1)
    manifest = suite.freeze(
        root, upstream, cases=cases, query_count=8,
        holdout_max_auto_raw_ratio=None,
        source_commit="fixture-source",
    )
    assert [item["role"] for item in manifest["cases"]] == [
        "development", "holdout", "holdout"
    ]
    assert manifest["holdout_max_auto_raw_ratio"] is None
    run = suite.run(root)
    assert len(run["cases"]) == 3
    result = suite.audit(root)
    assert result["functional_complete"]
    assert result["holdout_gate_pass"]
    assert result["selector_guard_pass"]
    assert result["selector_failures"] == []
    assert result["holdout_failures"] == []
    assert result["development_failures"] == []
    assert result["cases"][0]["selector_guard_pass"]
    assert result["cases"][0]["holdout_gate_pass"] is None
    assert all(
        case["holdout_gate_pass"] is True
        for case in result["cases"][1:]
    )
    assert all(case["queries"] == 8 for case in result["cases"])
    assert all(case["auto_vs_raw_paired_median_ratio"] > 0 for case in result["cases"])
    assert json.loads((root / "suite-audit.json").read_text()) == result


def test_suite_detects_child_manifest_substitution(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    cases = fixture_cases(upstream)
    root = tmp_path / "suite"
    monkeypatch.setattr(transfer, "WARMUPS", 0)
    monkeypatch.setattr(transfer, "REPEATS", 1)
    suite.freeze(
        root, upstream, cases=cases, query_count=8,
        holdout_max_auto_raw_ratio=None,
        source_commit="fixture-source",
    )
    child = root / "cases" / cases[1].name / "manifest.sha256"
    child.write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="manifest hash differs"):
        suite.load_manifest(root)
