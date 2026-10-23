from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from experiments.general_competition.adaptive_backend import (
    POLICY,
    choose_frontend,
    classify_comparison,
    compare_adaptive_backend,
)
from experiments.general_competition import exposed_panel
from experiments.general_competition import adaptive_backend
from experiments.general_competition.matched_backend import (
    BackendRun,
    ModelValidationError,
    _complete_model,
    prepare_matched_formula,
)


def test_frontend_policy_uses_only_unit_and_empty_clause_geometry():
    assert POLICY == "general-frontend-v1"
    assert choose_frontend(prepare_matched_formula(1, ((),))) == (
        "reduce",
        "empty_clause",
    )
    assert choose_frontend(prepare_matched_formula(3, ((1,), (1, 2, 3)))) == (
        "reduce",
        "unit_clause",
    )
    assert choose_frontend(prepare_matched_formula(3, ((1, 2), (1, 2, 3)))) == (
        "plain",
        "no_unit_clause",
    )
    with pytest.raises(ValueError, match="prepared"):
        choose_frontend(object())


def test_direct_route_preserves_the_matched_contract():
    pytest.importorskip("pysat")
    prepared = prepare_matched_formula(
        4,
        ((1, 2, 3), (-1, 2, 4), (1, -3, 4)),
    )
    result = compare_adaptive_backend(prepared, order="hybrid-first")
    record = result.record()
    assert result.accepted_for_timing
    assert record["adaptive_policy"] == POLICY
    assert record["adaptive_decision"] == "plain"
    assert record["adaptive_reason"] == "no_unit_clause"
    assert record["frontend"] == "plain-direct-v1"
    assert record["core_elapsed_ns"] == 0
    assert record["residual_variables"] == record["original_variables"]
    assert record["residual_clauses"] == record["original_clauses"]
    assert record["plain"]["status"] == "SAT_CHECKED"
    assert record["hybrid"]["status"] == "SAT_CHECKED"
    assert record["result_classification"] == "UPSTREAM_FALLBACK_PARITY"


def test_unit_route_keeps_the_replayable_reducer():
    pytest.importorskip("pysat")
    prepared = prepare_matched_formula(
        4,
        ((1,), (-1, 2), (-2, 3, 4)),
    )
    result = compare_adaptive_backend(prepared)
    record = result.record()
    assert result.accepted_for_timing
    assert record["adaptive_decision"] == "reduce"
    assert record["adaptive_reason"] == "unit_clause"
    assert record["frontend"] == "python-replayable-binary-core"
    assert record["result_classification"] in ("SPECTRA_FRONTEND_WIN", "LOSS")


def _sample(*, decision="plain", total_ns=900_000):
    core_ns = 100 if decision == "reduce" else 0
    record = {
        "solver": "cadical195",
        "adaptive_policy": POLICY,
        "adaptive_decision": decision,
        "frontend": "native-binary-core-prepared-v1" if decision == "reduce" else "plain-direct-v1",
        "original_variables": 2,
        "original_clauses": 1,
        "residual_variables": 2,
        "residual_clauses": 1,
        "core_elapsed_ns": core_ns,
        "hybrid_total_ns": total_ns,
        "hybrid_over_plain": total_ns / 1_000_000,
        "answers_agree": True,
        "accepted_for_timing": True,
        "plain": {
            "status": "SAT_CHECKED", "model_verified": True, "elapsed_ns": 1_000_000,
            "solver": "cadical195", "model": (True, True),
        },
        "hybrid": {
            "status": "SAT_CHECKED", "model_verified": True, "elapsed_ns": total_ns - core_ns,
            "solver": "cadical195", "model": (True, True),
        },
    }
    record["result_classification"] = classify_comparison(record)
    return record


@pytest.mark.parametrize("total_ns", [190_000, 999_999, 1_000_000, 5_000_000])
def test_same_backend_timing_variation_never_becomes_a_spectra_win(total_ns):
    record = _sample(total_ns=total_ns)
    assert record["result_classification"] == "UPSTREAM_FALLBACK_PARITY"
    assert record["hybrid_over_plain"] == total_ns / 1_000_000


@pytest.mark.parametrize(
    "total_ns,expected", [(190_000, "SPECTRA_FRONTEND_WIN"), (1_000_000, "LOSS"), (2_000_000, "LOSS")]
)
def test_native_reducer_is_frontend_work_and_ties_are_not_wins(total_ns, expected):
    record = _sample(decision="reduce", total_ns=total_ns)
    assert record["result_classification"] == expected
    assert record["result_classification"] != "SPECTRA_NATIVE_WIN"


@pytest.mark.parametrize("arm", ["plain", "hybrid"])
def test_rejected_sat_model_outranks_a_fast_timing_in_either_arm(arm):
    record = _sample(decision="reduce", total_ns=190_000)
    record[arm]["model_verified"] = False
    assert classify_comparison(record) == "INVALID_MODEL"


@pytest.mark.parametrize(
    "status,expected", [
        ("TIMEOUT", "TIMEOUT"), ("RESOURCE_FAILURE", "RESOURCE_FAILURE"),
        ("UNKNOWN", "UNKNOWN"), ("INVALID_PROOF", "INVALID_PROOF"),
        ("INVALID_MODEL", "INVALID_MODEL"), ("UNSAT_REPORTED", "UNKNOWN"),
    ]
)
def test_failure_and_unverified_statuses_never_become_wins(status, expected):
    record = _sample(decision="reduce", total_ns=190_000)
    record["hybrid"]["status"] = status
    record["hybrid"]["model_verified"] = False
    record["accepted_for_timing"] = False
    assert classify_comparison(record) == expected


def test_contradictory_unsat_against_a_checked_sat_model_is_a_correctness_failure():
    record = _sample()
    record["plain"]["status"] = "UNSAT_REPORTED"
    record["plain"]["model_verified"] = False
    record["answers_agree"] = False
    record["accepted_for_timing"] = False
    assert classify_comparison(record) == "INVALID_PROOF"


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), -1, 0, True, 0.01])
def test_unusable_or_inconsistent_timing_cannot_earn_a_win(ratio):
    record = _sample(decision="reduce")
    record["hybrid_over_plain"] = ratio
    assert classify_comparison(record) == "UNKNOWN"


def test_mismatched_route_or_backend_is_not_evidence_of_a_frontend_win():
    record = _sample()
    record["adaptive_decision"] = "reduce"
    assert classify_comparison(record) == "UNKNOWN"
    record = _sample(decision="reduce")
    record["hybrid"]["solver"] = "another-solver"
    assert classify_comparison(record) == "UNKNOWN"


def test_twelve_cell_accounting_keeps_three_frontend_wins_and_nine_fallback_cells():
    samples = [_sample(decision="reduce", total_ns=190_000) for _ in range(3)]
    samples += [_sample(total_ns=990_000) for _ in range(4)]
    samples += [_sample(total_ns=1_010_000) for _ in range(5)]
    original = deepcopy(samples)
    result = exposed_panel.summarize_samples(samples)
    assert sum(sample["hybrid_over_plain"] < 1 for sample in samples) == 7
    assert result["spectra_frontend_wins"] == result["frontend_cells"] == 3
    assert result["spectra_native_wins"] == 0
    assert result["upstream_fallback_parity"] == 9
    assert result["cells"] == result["timed_cells"] == 12
    assert result["non_timing_cells"] == 0
    assert sum(result["classification_counts"].values()) == 12
    assert result["frontend_median_hybrid_over_plain"] == 0.19
    assert result["fallback_median_hybrid_over_plain"] == 1.01
    assert "hybrid_wins" not in result
    assert samples == original


def test_incomplete_cells_remain_in_summary_without_fabricated_timing():
    samples = [exposed_panel._failure_sample(MemoryError("resource limit"), phase="preparation", repeat=0)]
    result = exposed_panel.summarize_samples(samples)
    assert result["classification_counts"]["RESOURCE_FAILURE"] == 1
    assert result["cells"] == result["non_timing_cells"] == 1
    assert result["timed_cells"] == 0
    assert result["median_hybrid_over_plain"] is None
    assert result["geometric_mean_hybrid_over_plain"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("forged_label", ["SPECTRA_NATIVE_WIN", "SPECTRA_FRONTEND_WIN"])
def test_summary_rejects_a_fallback_cell_relabelled_as_an_algorithm_win(forged_label):
    sample = _sample()
    sample["result_classification"] = forged_label
    with pytest.raises(ValueError, match="classification disagrees"):
        exposed_panel.summarize_samples([sample])


def _mock_panel_io(monkeypatch, tmp_path):
    monkeypatch.setattr(exposed_panel, "PANEL", exposed_panel.PANEL[:2])
    monkeypatch.setattr(exposed_panel, "build_native_binary_core", lambda directory: tmp_path / "fake.so")
    monkeypatch.setattr(
        exposed_panel, "NativeBinaryCore",
        lambda path: SimpleNamespace(prepare_canonical=lambda nvars, clauses: object()),
    )
    source = tmp_path / "formula.cnf"
    source.write_text("p cnf 2 1\n1 2 0\n", encoding="ascii")
    monkeypatch.setattr(exposed_panel, "_download", lambda case, directory: source)
    return source


def test_panel_keeps_acquisition_failure_and_the_rest_of_the_declared_inventory(monkeypatch, tmp_path):
    source = _mock_panel_io(monkeypatch, tmp_path)
    calls = []

    def download(case, directory):
        calls.append(case.instance_id)
        if len(calls) == 1:
            raise TimeoutError("download timed out")
        return source

    monkeypatch.setattr(exposed_panel, "_download", download)
    monkeypatch.setattr(exposed_panel, "compare_adaptive_backend", lambda *args, **kwargs: SimpleNamespace(record=_sample))
    result = exposed_panel.run_panel(tmp_path / "run", repeats=2)
    assert result["status"] == "INCOMPLETE"
    assert [case["instance_id"] for case in result["cases"]] == calls
    assert result["cells"] == 4
    assert result["classification_counts"]["TIMEOUT"] == 2
    assert result["upstream_fallback_parity"] == 2
    assert result["cases"][0]["samples"][0]["phase"] == "acquisition"
    assert not result["cases"][0]["samples"][0]["attempted"]
    assert result["cases"][1]["source_sha256"]


def test_panel_retains_bad_model_and_marks_remaining_cells_unrun(monkeypatch, tmp_path):
    _mock_panel_io(monkeypatch, tmp_path)
    calls = []

    def compare(*args, **kwargs):
        calls.append(kwargs["order"])
        if len(calls) == 2:
            raise ModelValidationError("rejected witness", evidence={"raw_model": [1, -1]})
        return SimpleNamespace(record=_sample)

    monkeypatch.setattr(exposed_panel, "compare_adaptive_backend", compare)
    result = exposed_panel.run_panel(tmp_path / "run", repeats=3)
    assert result["status"] == "FAILED_CORRECTNESS"
    assert calls == ["plain-first", "hybrid-first"]
    assert result["cells"] == 6
    assert result["upstream_fallback_parity"] == 1
    assert result["classification_counts"]["INVALID_MODEL"] == 1
    assert result["classification_counts"]["UNKNOWN"] == 4
    failed = result["cases"][0]["samples"][1]
    assert failed["rejected_evidence"]["raw_model"] == [1, -1]
    assert failed["attempted"]
    assert result["cases"][1]["samples"][0]["phase"] == "blocked_by_correctness_failure"
    assert not result["cases"][1]["samples"][0]["attempted"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("raw", [None, [0], [True], [1, -1]])
def test_model_validation_retains_rejected_backend_output(raw):
    with pytest.raises(ModelValidationError) as failure:
        _complete_model(raw, 2)
    assert failure.value.evidence["raw_model"] == raw


def _backend(status="SAT_CHECKED"):
    return BackendRun(
        solver="cadical195", status=status, variables=2, clauses=1, elapsed_ns=100,
        conflicts=0, decisions=1, propagations=1, restarts=0,
        model=(True, True) if status == "SAT_CHECKED" else (),
        model_verified=status == "SAT_CHECKED",
    )


@pytest.mark.parametrize("order", ["plain-first", "hybrid-first"])
def test_rejected_second_arm_keeps_the_first_arm_receipt(monkeypatch, order):
    calls = []

    def backend(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ModelValidationError("bad model", evidence={"raw_model": [1, -1]})
        return _backend()

    monkeypatch.setattr(adaptive_backend, "run_backend", backend)
    prepared = prepare_matched_formula(2, ((1, 2),))
    with pytest.raises(ModelValidationError) as failure:
        compare_adaptive_backend(prepared, order=order)
    completed_arm = "plain" if order == "plain-first" else "hybrid"
    failed_arm = "hybrid" if completed_arm == "plain" else "plain"
    evidence = failure.value.evidence
    assert evidence["failed_arm"] == failed_arm
    assert evidence["completed_arms"][completed_arm]["elapsed_ns"] == 100
    assert evidence["completed_arms"][completed_arm]["model"] == (True, True)
    assert evidence["raw_model"] == [1, -1]


def test_disagreeing_backend_receipts_are_preserved_for_correctness_review(monkeypatch):
    arms = iter((_backend(), _backend("UNSAT_REPORTED")))
    monkeypatch.setattr(adaptive_backend, "run_backend", lambda *args, **kwargs: next(arms))
    record = compare_adaptive_backend(prepare_matched_formula(2, ((1, 2),))).record()
    assert record["result_classification"] == "INVALID_PROOF"
    assert not record["answers_agree"]
    assert not record["accepted_for_timing"]
    assert record["plain"]["model"] == (True, True)
    assert record["hybrid"]["status"] == "UNSAT_REPORTED"


def test_failed_cli_report_is_written_before_nonzero_exit(monkeypatch, tmp_path, capsys):
    report = {"schema": exposed_panel.SCHEMA, "status": "INCOMPLETE", "cases": []}
    monkeypatch.setattr(exposed_panel, "run_panel", lambda *args, **kwargs: report)
    destination = tmp_path / "failed.json"
    with pytest.raises(SystemExit) as failure:
        exposed_panel.main([str(tmp_path), "--out", str(destination)])
    assert failure.value.code == 1
    assert json.loads(destination.read_text()) == report
    assert json.loads(capsys.readouterr().out) == report
