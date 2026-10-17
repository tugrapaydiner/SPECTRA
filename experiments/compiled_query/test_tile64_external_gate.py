from __future__ import annotations

from copy import deepcopy

from .tile64_external_gate import evaluate


def fixture() -> dict:
    comparisons = []
    for baseline, mean, clusters in (
        ("exact-bank-raw", .70, {"wap:05": .72, "wap:06": .68}),
        ("minicard", .40, {"wap:05": .39, "wap:06": .41}),
        ("cadical195", .42, {"wap:05": .40, "wap:06": .44}),
        ("cudd-reduced-natural", .45, {"wap:05": .43, "wap:06": .47}),
    ):
        comparisons.append({
            "candidate": "exact-bank-scc",
            "baseline": baseline,
            "count": 1024,
            "eligible": True,
            "excluded_or_incomplete": [],
            "matching_certificate_contract": baseline == "exact-bank-raw",
            "graph_clusters": 2,
            "cluster_ratios": clusters,
            "geomean_complete_ratio": mean,
        })
    return {
        "schema": "spectra.compiled_query.audit.v1",
        "scheduled_jobs": 10,
        "all_scheduled_decisions_complete": True,
        "audited_decisions": {"SAT_VERIFIED": 10 * 1024},
        "original_non_sat_requests_checked": 0,
        "outside_replication": False,
        "comparisons": comparisons,
    }


def test_external_gate_passes_only_clear_wins():
    result = evaluate(fixture(), label="test", expect_cudd=True)
    assert result["pass"]
    assert not result["failures"]


def test_external_gate_rejects_slight_external_win():
    value = fixture()
    row = next(row for row in value["comparisons"] if row["baseline"] == "minicard")
    row["geomean_complete_ratio"] = .51
    result = evaluate(value, label="test", expect_cudd=True)
    assert not result["pass"]
    assert any("minicard: geomean ratio" in failure for failure in result["failures"])


def test_external_gate_rejects_one_weak_graph():
    value = fixture()
    row = next(row for row in value["comparisons"] if row["baseline"] == "cadical195")
    row["cluster_ratios"]["wap:06"] = .68
    result = evaluate(value, label="test", expect_cudd=True)
    assert not result["pass"]
    assert any("cadical195: worst graph ratio" in failure for failure in result["failures"])


def test_external_gate_rejects_non_sat_or_missing_work():
    value = fixture()
    value["audited_decisions"] = {"SAT_VERIFIED": 10 * 1024 - 1, "UNSAT": 1}
    value["all_scheduled_decisions_complete"] = False
    result = evaluate(value, label="test", expect_cudd=False)
    assert not result["pass"]
    assert any("not COMPLETE" in failure for failure in result["failures"])
    assert any("non-SAT decisions" in failure for failure in result["failures"])
