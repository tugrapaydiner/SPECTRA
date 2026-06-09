"""Phase 13: measured edge benchmarking.

Physical joules need Linux RAPL (None here), but latency/memory savings ARE
measurable on any host. These tests show the benchmark harness runs end-to-end,
that recursion depth trades latency for accuracy (a real measured frontier), and
that System 1 is a measured latency win over recursive System 2 (BLUEPRINT
section 23, gate 13).
"""

import torch

from common.seed import resolve_device, set_seed
from eval.benchmarks import benchmark, sweep_recursion_depth
from eval.latency import measure_latency
from eval.reports import compute_optimal_frontier
from model.system1_student import System1Student
from model.trm import TRM


def test_benchmark_runs_end_to_end():
    set_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=2, max_grid_size=8)
    x = torch.randint(0, 5, (16, 16))
    y = torch.randint(1, 5, (16, 16))
    report = benchmark(model, x, y, height=4, width=4, n_latency_runs=8)
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["latency_ms"] > 0 and report["model_size_mb"] > 0
    assert report["joules_per_problem"] is None  # no RAPL on this host


def test_recursion_depth_latency_frontier():
    set_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=8, max_grid_size=8)
    x = torch.randint(0, 5, (8, 16))
    y = torch.randint(1, 5, (8, 16))
    points = sweep_recursion_depth(model, x, y, 4, 4, depths=[1, 2, 4, 8], n_latency_runs=6)
    lat = [p["latency_ms"] for p in points]
    assert lat[-1] > lat[0]  # deeper recursion measurably costs more latency
    assert model.N_sup == 8  # depth restored after the sweep
    frontier = compute_optimal_frontier(points, cost_key="latency_ms", acc_key="accuracy")
    assert len(frontier) >= 1


def test_system1_is_measured_latency_win():
    set_seed(0)
    device = resolve_device("cpu")  # CPU: block-count gap is clear, no launch noise
    trm = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=4, max_grid_size=8).to(device).eval()
    s1 = System1Student(dim=32, num_tokens=5, seq_len=16, n_layers=3, max_grid_size=8).to(device).eval()
    x = torch.randint(0, 5, (1, 16), device=device)

    with torch.no_grad():
        lat_trm = measure_latency(lambda: trm(x, height=4, width=4), n_runs=10, warmup=3)
        lat_s1 = measure_latency(lambda: s1(x, height=4, width=4), n_runs=10, warmup=3)
    # One feed-forward pass beats the deep recursive loop -- a measured saving.
    assert lat_s1["latency_ms_mean"] < lat_trm["latency_ms_mean"]
