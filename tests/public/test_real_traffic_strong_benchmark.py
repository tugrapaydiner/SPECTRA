from __future__ import annotations

from experiments.real_traffic.strong_benchmark import (
    ARMS,
    add_owned_costs,
    schedule,
    summarize,
)


def fake_cell(arm: str, repeat: int, session_ns: int) -> dict:
    return {
        "arm": arm,
        "repeat": repeat,
        "session_ns": session_ns,
        "process_wall_ns": session_ns + 10,
        "peak_rss_kib": 100,
        "sat_queries": 1,
        "unsat_queries": 1,
        "query_rows": [
            {"complete_ns": session_ns // 4},
            {"complete_ns": session_ns // 3},
        ],
    }


def test_schedule_is_balanced_and_complete() -> None:
    jobs = schedule(5)
    assert len(jobs) == 5 * len(ARMS)
    for repeat in range(5):
        assert sorted(arm for observed, arm in jobs if observed == repeat) == sorted(ARMS)


def test_owned_setup_and_disposal_are_charged() -> None:
    totals = {"setup_ns": 10, "dispose_ns": 20, "session_ns": 100}
    add_owned_costs(totals, setup_ns=3, dispose_ns=4, prefix="control")
    assert totals == {
        "setup_ns": 13,
        "dispose_ns": 24,
        "session_ns": 107,
        "control_setup_ns": 3,
        "control_dispose_ns": 4,
    }


def test_summary_uses_the_strongest_external_control() -> None:
    costs = {
        "spectra_scc": 20,
        "spectra_hybrid": 22,
        "spectra_parity": 60,
        "spectra_none": 100,
        "minicard_native": 30,
        "minicard_python": 50,
        "cudd_minfill": 80,
    }
    cells = [fake_cell(arm, repeat, cost)
             for repeat in range(3) for arm, cost in costs.items()]
    result = summarize(cells, [])
    assert result["best_external_baseline"] == "minicard_native"
    assert result["spectra_to_best_external_mean_ratio"] == 2 / 3
    assert result["candidate_mean_ratios"]["spectra_none"] == 0.2
    assert result["failures"] == []
