"""Phase 13 tests: latency/memory/energy profiling, edge metrics, frontier curves.

Physical joules require Linux RAPL, so on this host the energy path returns None
(verified) while latency/memory/metrics/frontier all run cross-platform.
"""

import pytest
import torch

from eval.edge_energy import (
    cgroup_command,
    cgroups_available,
    measure_energy_joules,
    rapl_available,
)
from eval.latency import measure_latency
from eval.memory import model_size_mb, process_rss_mb
from eval.reports import (
    accuracy_per_joule,
    accuracy_per_mb,
    accuracy_per_ms,
    compute_optimal_frontier,
    edge_report,
    edge_reasoning_score,
)
from model.trm import TRM


# --------------------------------------------------------------------------- #
# Energy / cgroups (graceful)
# --------------------------------------------------------------------------- #
def test_rapl_energy_graceful():
    assert isinstance(rapl_available(), bool)
    result = measure_energy_joules(lambda: sum(range(1000)), n_runs=2)
    if rapl_available():
        assert result["energy_joules"] >= 0
    else:
        assert result is None  # no RAPL on this host


def test_cgroup_command_builder():
    cmd = cgroup_command(1, 512, ["python", "eval_edge.py"])
    assert cmd[:2] == ["systemd-run", "--scope"]
    assert "CPUQuota=100%" in cmd
    assert "MemoryMax=512M" in cmd
    assert cmd[-2:] == ["python", "eval_edge.py"]
    # Unconstrained (0) omits the limits.
    assert "CPUQuota" not in " ".join(cgroup_command(0, 0, ["x"]))
    assert isinstance(cgroups_available(), bool)


# --------------------------------------------------------------------------- #
# Latency / memory
# --------------------------------------------------------------------------- #
def test_measure_latency_positive():
    stats = measure_latency(lambda: torch.randn(64, 64) @ torch.randn(64, 64), n_runs=10, warmup=2)
    assert stats["latency_ms_mean"] > 0
    assert stats["latency_ms_p95"] >= stats["latency_ms_p50"]


def test_memory_metrics():
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=2, max_grid_size=8)
    size = model_size_mb(model)
    assert size > 0
    rss = process_rss_mb()
    assert rss != rss or rss > 0  # NaN if psutil missing, else positive


# --------------------------------------------------------------------------- #
# Edge metrics + frontier
# --------------------------------------------------------------------------- #
def test_efficiency_metrics():
    assert accuracy_per_joule(0.9, 3.0) == pytest.approx(0.3, abs=1e-6)
    assert accuracy_per_mb(0.8, 4.0) == pytest.approx(0.2, abs=1e-6)
    assert accuracy_per_ms(0.5, 10.0) == pytest.approx(0.05, abs=1e-6)
    assert edge_reasoning_score(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0, abs=1e-3)


def test_compute_optimal_frontier():
    # (cost, acc): B dominates A (cheaper+better); frontier keeps the efficient ones.
    points = [
        {"joules_per_problem": 2.0, "accuracy": 0.70, "name": "A"},
        {"joules_per_problem": 1.0, "accuracy": 0.75, "name": "B"},  # dominates A
        {"joules_per_problem": 4.0, "accuracy": 0.90, "name": "C"},
        {"joules_per_problem": 3.0, "accuracy": 0.80, "name": "D"},
    ]
    frontier = compute_optimal_frontier(points)
    names = [p["name"] for p in frontier]
    assert "A" not in names  # dominated by B
    assert "B" in names and "C" in names
    # Sorted by ascending cost.
    costs = [p["joules_per_problem"] for p in frontier]
    assert costs == sorted(costs)


def test_edge_report_without_energy():
    report = edge_report(accuracy=0.9, latency_ms=12.0, peak_ram_mb=300.0, model_size_mb=1.4)
    assert report["joules_per_problem"] is None
    assert "energy_note" in report
    assert report["accuracy_per_ms"] == pytest.approx(0.9 / 12.0, abs=1e-6)


def test_edge_report_with_energy():
    report = edge_report(
        accuracy=0.9, latency_ms=12.0, peak_ram_mb=300.0, model_size_mb=1.4,
        joules_per_problem=2.5,
    )
    assert report["accuracy_per_joule"] == pytest.approx(0.36, abs=1e-3)
    assert report["edge_reasoning_score"] > 0
