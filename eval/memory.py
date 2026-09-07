"""Memory accounting with explicit scopes and window-local peak sampling.

M13 distinguishes model-state bytes, search-tree tensor storage, current RSS,
window-local sampled RSS maximum, and Linux lifetime high-water. Sampling can miss
allocations shorter than the interval; that limit is always returned.
"""
from __future__ import annotations

import math
from pathlib import Path
import threading
import time
from typing import Any, Callable

import torch
import torch.nn as nn

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

MIB = 1024 * 1024


def process_rss_mb() -> float:
    """Current process RSS in MiB (NaN when psutil is unavailable)."""
    if psutil is None:
        return float("nan")
    return psutil.Process().memory_info().rss / MIB


def process_rss_bytes() -> int | None:
    if psutil is None:
        return None
    return int(psutil.Process().memory_info().rss)


def linux_process_hwm_mb(status_path: str | Path = "/proc/self/status") -> float | None:
    """Linux lifetime process high-water RSS (`VmHWM`), not window-local peak."""
    try:
        for line in Path(status_path).read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                fields = line.split()
                if len(fields) >= 2:
                    return int(fields[1]) / 1024.0  # kB -> MiB
    except (OSError, ValueError):
        return None
    return None


def model_state_bytes(model: nn.Module) -> int:
    """Parameters + persistent buffers in their actual in-memory dtypes."""
    return int(sum(p.numel() * p.element_size() for p in model.parameters())
               + sum(b.numel() * b.element_size() for b in model.buffers()))


def model_size_mb(model: nn.Module) -> float:
    return model_state_bytes(model) / MIB


def packed_weight_bytes(artifact_or_payload: Any) -> int | None:
    """Bytes of tensors explicitly stored under a deployment `packed_linears` map."""
    payload = getattr(artifact_or_payload, "payload", artifact_or_payload)
    if not isinstance(payload, dict) or not isinstance(payload.get("packed_linears"), dict):
        return None
    total = 0
    for entry in payload["packed_linears"].values():
        if not isinstance(entry, dict):
            continue
        packed = entry.get("packed")
        if torch.is_tensor(packed):
            total += packed.numel() * packed.element_size()
    return int(total)


def _storage_key_and_bytes(t: torch.Tensor) -> tuple[tuple[int, int, str], int]:
    storage = t.untyped_storage()
    nbytes = int(storage.nbytes())
    return (int(storage.data_ptr()), nbytes, str(t.device)), nbytes


def search_tree_tensor_bytes(root: Any) -> dict[str, Any]:
    """Count unique tensor storages reachable through an MCTS node tree.

    This intentionally excludes Python object/list allocator overhead; process RSS
    captures the whole process separately.
    """
    if root is None:
        return {
            "applicable": False,
            "node_count": 0,
            "tensor_storage_bytes": None,
            "scope": "not_applicable_no_search_tree",
        }
    stack = [root]; seen_nodes: set[int] = set(); seen_storage: set[tuple[int, int, str]] = set()
    total = 0; nodes = 0
    while stack:
        node = stack.pop()
        if id(node) in seen_nodes:
            continue
        seen_nodes.add(id(node)); nodes += 1
        for name in ("y", "z", "z_codes", "z_scale"):
            value = getattr(node, name, None)
            if torch.is_tensor(value):
                key, nbytes = _storage_key_and_bytes(value)
                if key not in seen_storage:
                    seen_storage.add(key); total += nbytes
        children = getattr(node, "children", None)
        if children:
            stack.extend(list(children))
    return {
        "applicable": True,
        "node_count": nodes,
        "tensor_storage_bytes": int(total),
        "scope": "unique_tensor_storages_only_excludes_python_object_allocator_overhead",
    }


class _RSSSampler(threading.Thread):
    def __init__(self, interval_s: float):
        super().__init__(daemon=True)
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.interval_s = float(interval_s)
        self._stop_event = threading.Event()
        self.samples: list[tuple[int, int]] = []

    def run(self) -> None:
        while not self._stop_event.is_set():
            rss = process_rss_bytes()
            if rss is not None:
                self.samples.append((time.perf_counter_ns(), rss))
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set(); self.join(timeout=2.0)


def measure_peak_ram(
    fn: Callable[[], object],
    *,
    interval_s: float = 0.001,
) -> dict[str, Any]:
    """Measure a window-local sampled RSS maximum plus Linux lifetime HWM.

    `process_rss_peak_sampled_mb` is a sampled maximum, not a perfect allocator
    high-water: allocations shorter than `interval_s` can be missed.
    """
    before = process_rss_bytes(); hwm_before = linux_process_hwm_mb()
    sampler = _RSSSampler(interval_s)
    t0 = time.perf_counter(); sampler.start()
    try:
        result = fn()
    finally:
        sampler.stop()
    elapsed = time.perf_counter() - t0
    after = process_rss_bytes(); hwm_after = linux_process_hwm_mb()
    candidates = [v for _, v in sampler.samples]
    if before is not None: candidates.append(before)
    if after is not None: candidates.append(after)
    peak = max(candidates) if candidates else None
    before_mb = before / MIB if before is not None else None
    after_mb = after / MIB if after is not None else None
    peak_mb = peak / MIB if peak is not None else None
    return {
        "rss_available": peak is not None,
        "rss_before_mb": before_mb,
        "rss_after_mb": after_mb,
        "process_rss_peak_sampled_mb": peak_mb,
        # Compatibility field now has an accurate definition rather than post-run RSS.
        "peak_ram_mb": peak_mb,
        "peak_ram_definition": "window_local_sampled_process_rss_maximum",
        "rss_delta_after_minus_before_mb": (
            after_mb - before_mb if before_mb is not None and after_mb is not None else None
        ),
        "linux_vmhwm_before_mb": hwm_before,
        "linux_vmhwm_after_mb": hwm_after,
        "linux_vmhwm_scope": "process_lifetime_high_water_not_window_local" if hwm_after is not None else None,
        "sampling_interval_ms": interval_s * 1000.0,
        "sample_count": len(sampler.samples),
        "sampling_limitation": "allocations shorter than the sampling interval can be missed",
        "elapsed_seconds": elapsed,
        "result_type": type(result).__name__,
    }
