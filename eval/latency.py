"""Latency profiling (BLUEPRINT section 23).

Times a callable over repeated runs (with warmup and optional CUDA sync) and
reports mean / p50 / p95 in milliseconds -- the latency metrics the edge report
needs. Works on any platform.
"""

from __future__ import annotations

import time
from typing import Callable

import torch


def measure_latency(
    fn: Callable[[], object],
    n_runs: int = 30,
    warmup: int = 5,
    sync_cuda: bool = True,
) -> dict[str, float]:
    """Time ``fn`` and return latency statistics in milliseconds.

    Args:
        fn: Zero-arg callable to time (e.g. ``lambda: model(x))``.
        n_runs: Number of timed runs.
        warmup: Number of untimed warmup runs (JIT/caches/cuDNN autotune).
        sync_cuda: Synchronize CUDA before/after each timed run for honest timing.
    """
    use_cuda = sync_cuda and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000.0)

    times_ms.sort()
    n = len(times_ms)
    mean = sum(times_ms) / n
    return {
        "latency_ms_mean": mean,
        "latency_ms_p50": times_ms[n // 2],
        "latency_ms_p95": times_ms[min(n - 1, int(0.95 * n))],
        "latency_ms_min": times_ms[0],
        "latency_ms_std": (sum((t - mean) ** 2 for t in times_ms) / n) ** 0.5,
    }
