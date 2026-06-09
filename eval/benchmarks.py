"""Benchmark harness: accuracy + latency + memory (+ energy) for a model.

Ties the metric modules together to produce one edge report per model/task, and
provides a recursion-depth sweep that yields compute-optimal frontier points
(accuracy vs cost). Edge inference is B=1 (section 26.2), so latency is measured
on single instances.
"""

from __future__ import annotations

from typing import Any

import torch

from eval.edge_energy import measure_energy_joules
from eval.latency import measure_latency
from eval.memory import model_size_mb, process_rss_mb
from eval.metrics import board_accuracy, cell_accuracy
from eval.reports import edge_report
from model.trm import TRM


@torch.no_grad()
def evaluate_accuracy(
    model: TRM, x: torch.Tensor, y: torch.Tensor, height: int, width: int
) -> dict[str, float]:
    """Cell and board accuracy of ``model`` on ``(x, y)``."""
    model.eval()
    logits, _ = model(x, height=height, width=width)
    pred = logits.argmax(dim=-1)
    return {"cell_acc": cell_accuracy(pred, y), "board_acc": board_accuracy(pred, y)}


def benchmark(
    model: TRM,
    x: torch.Tensor,
    y: torch.Tensor,
    height: int,
    width: int,
    n_latency_runs: int = 30,
) -> dict[str, Any]:
    """Full edge benchmark: accuracy, B=1 latency, peak RAM, size, energy if RAPL."""
    acc = evaluate_accuracy(model, x, y, height, width)

    x1 = x[:1]  # B=1 edge inference
    latency = measure_latency(
        lambda: model(x1, height=height, width=width), n_runs=n_latency_runs
    )
    energy = measure_energy_joules(
        lambda: model(x1, height=height, width=width), n_runs=n_latency_runs
    )
    joules = energy["joules_per_run"] if energy is not None else None

    report = edge_report(
        accuracy=acc["board_acc"],
        latency_ms=latency["latency_ms_mean"],
        peak_ram_mb=process_rss_mb(),
        model_size_mb=model_size_mb(model),
        joules_per_problem=joules,
        cell_acc=acc["cell_acc"],
        latency_ms_p95=latency["latency_ms_p95"],
    )
    return report


def sweep_recursion_depth(
    model: TRM,
    x: torch.Tensor,
    y: torch.Tensor,
    height: int,
    width: int,
    depths: list[int],
    n_latency_runs: int = 20,
) -> list[dict[str, Any]]:
    """Vary supervision depth ``N_sup`` and record accuracy vs latency.

    Produces compute-optimal frontier points (section 27.0): more recursion steps
    cost latency/energy and (ideally) buy accuracy. ``N_sup`` is restored after.
    """
    original = model.N_sup
    points: list[dict[str, Any]] = []
    try:
        for depth in depths:
            model.N_sup = depth
            acc = evaluate_accuracy(model, x, y, height, width)
            latency = measure_latency(
                lambda: model(x[:1], height=height, width=width), n_runs=n_latency_runs
            )
            points.append(
                {
                    "recursion_depth": depth,
                    "accuracy": acc["board_acc"],
                    "cell_acc": acc["cell_acc"],
                    "latency_ms": latency["latency_ms_mean"],
                }
            )
    finally:
        model.N_sup = original
    return points
