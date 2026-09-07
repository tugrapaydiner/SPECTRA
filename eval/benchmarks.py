"""Benchmark harness: accuracy + representative B=1 latency/memory/energy.

M13 keeps physical package-energy availability explicit, times distinct examples,
and labels current RSS separately from the sampled window-local peak.
"""
from __future__ import annotations

from typing import Any
import torch

from eval.edge_energy import measure_energy_record
from eval.latency import measure_latency
from eval.memory import measure_peak_ram, model_size_mb, process_rss_mb
from eval.metrics import board_accuracy, cell_accuracy
from eval.reports import edge_report
from model.trm import TRM


@torch.inference_mode()
def evaluate_accuracy(model: TRM, x: torch.Tensor, y: torch.Tensor,
                      height: int, width: int) -> dict[str, float]:
    model.eval(); logits, _ = model(x, height=height, width=width); pred = logits.argmax(-1)
    return {"cell_acc": cell_accuracy(pred, y), "board_acc": board_accuracy(pred, y)}


def _representative_latency(model: TRM, x: torch.Tensor, height: int, width: int,
                            n_runs: int) -> dict[str, float]:
    n = min(max(1, x.shape[0]), max(16, min(32, x.shape[0])))
    rows: list[float] = []
    with torch.inference_mode():
        model.eval()
        for i in range(n):
            xi = x[i:i+1]
            stat = measure_latency(lambda xi=xi: model(xi, height=height, width=width),
                                   n_runs=max(1, n_runs // n), warmup=1)
            rows.append(float(stat["latency_ms_mean"]))
    s = sorted(rows); m = sum(s) / len(s)
    return {"latency_ms_mean": m, "latency_ms_p50": s[len(s)//2],
            "latency_ms_p95": s[min(len(s)-1, int(.95*len(s)))],
            "representative_examples": len(s)}


def benchmark(model: TRM, x: torch.Tensor, y: torch.Tensor, height: int, width: int,
              n_latency_runs: int = 30) -> dict[str, Any]:
    """Edge report over distinct B=1 examples; physical energy may be unavailable."""
    acc = evaluate_accuracy(model, x, y, height, width)
    latency = _representative_latency(model, x, height, width, n_latency_runs)
    n_energy = min(max(1, x.shape[0]), 16)

    def representative_pass():
        with torch.inference_mode():
            model.eval()
            for i in range(n_energy):
                model(x[i:i+1], height=height, width=width)

    energy = measure_energy_record(representative_pass, n_runs=1)
    joules = (energy["energy_joules"] / n_energy) if energy["available"] else None
    memory = measure_peak_ram(representative_pass, interval_s=0.001)
    current_rss = process_rss_mb()
    report = edge_report(accuracy=acc["board_acc"], latency_ms=latency["latency_ms_mean"],
        peak_ram_mb=memory["process_rss_peak_sampled_mb"], model_size_mb=model_size_mb(model),
        joules_per_problem=joules, cell_acc=acc["cell_acc"],
        latency_ms_p95=latency["latency_ms_p95"])
    report["energy_available"] = bool(energy["available"])
    report["energy_failure_reason"] = energy["failure_reason"]
    report["energy_scope"] = energy["scope"]
    report["representative_latency_examples"] = latency["representative_examples"]
    report["process_rss_current_mb"] = current_rss
    report["process_rss_peak_sampled_mb"] = memory["process_rss_peak_sampled_mb"]
    report["peak_ram_definition"] = memory["peak_ram_definition"]
    report["rss_sampling_interval_ms"] = memory["sampling_interval_ms"]
    report["rss_sampling_limitation"] = memory["sampling_limitation"]
    report["linux_vmhwm_mb"] = memory["linux_vmhwm_after_mb"]
    return report


def sweep_recursion_depth(model: TRM, x: torch.Tensor, y: torch.Tensor,
                          height: int, width: int, depths: list[int],
                          n_latency_runs: int = 20) -> list[dict[str, Any]]:
    original = model.N_sup; points: list[dict[str, Any]] = []
    try:
        for depth in depths:
            model.N_sup = depth; acc = evaluate_accuracy(model, x, y, height, width)
            latency = _representative_latency(model, x, height, width, n_latency_runs)
            points.append({"recursion_depth": depth, "accuracy": acc["board_acc"],
                           "cell_acc": acc["cell_acc"], "latency_ms": latency["latency_ms_mean"],
                           "representative_examples": latency["representative_examples"]})
    finally:
        model.N_sup = original
    return points
