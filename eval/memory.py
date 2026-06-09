"""Memory profiling (BLUEPRINT section 23).

Reports process RSS (peak RAM) and model weight footprint. RSS uses psutil when
available; the model size is computed from parameter/buffer dtypes and is always
available.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

try:
    import psutil
except ImportError:  # pragma: no cover - environment dependent
    psutil = None


def process_rss_mb() -> float:
    """Current resident set size of this process in MB (NaN if psutil missing)."""
    if psutil is None:
        return float("nan")
    return psutil.Process().memory_info().rss / (1024 * 1024)


def model_size_mb(model: nn.Module) -> float:
    """Footprint of a model's parameters + buffers in MB (dense, current dtypes)."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 * 1024)


def measure_peak_ram(fn: Callable[[], object]) -> dict[str, float]:
    """Run ``fn`` and report RSS before/after and the delta (MB)."""
    before = process_rss_mb()
    fn()
    after = process_rss_mb()
    return {
        "rss_before_mb": before,
        "peak_ram_mb": after,
        "rss_delta_mb": after - before,
    }
