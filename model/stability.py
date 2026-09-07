"""Stability Shield helpers for ternary training."""
from __future__ import annotations

from typing import Any, Iterator, Mapping

import torch.nn as nn

from model.bitlinear import FakeBitLinear


def iter_bitlinears(model: nn.Module) -> Iterator[tuple[str, FakeBitLinear]]:
    """Yield ``(name, module)`` for every FakeBitLinear in ``model``."""
    for name, module in model.named_modules():
        if isinstance(module, FakeBitLinear):
            yield name, module


def set_quant_strength(model: nn.Module, rho: float) -> None:
    """Set the soft-ternary interpolation rho on all FakeBitLinear layers."""
    rho = float(max(0.0, min(1.0, rho)))
    for _, module in iter_bitlinears(model):
        module.quant_strength.fill_(rho)


def quant_strength_state(model: nn.Module) -> dict[str, float]:
    """Return per-layer quantization strength for checkpoint provenance."""
    return {
        name: float(module.quant_strength.item())
        for name, module in iter_bitlinears(model)
    }


def load_quant_strength_state(
    model: nn.Module,
    state: Mapping[str, float],
) -> None:
    """Strictly restore per-layer quantization strength."""
    current = {name: module for name, module in iter_bitlinears(model)}
    if set(current) != set(state):
        missing = sorted(set(current) - set(state))
        extra = sorted(set(state) - set(current))
        raise ValueError(
            f"quantization-strength keys mismatch: missing={missing}, extra={extra}"
        )
    for name, module in current.items():
        rho = float(state[name])
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"invalid quantization strength for {name}: {rho}")
        module.quant_strength.fill_(rho)


class QuantWarmup:
    """Linear soft->hard ternarization schedule.

    ``rho = min(1, step / warmup_steps)``. The schedule has no hidden mutable
    counter; checkpoints record ``warmup_steps`` plus the actual per-layer rho.
    """

    def __init__(self, warmup_steps: int):
        self.warmup_steps = max(1, int(warmup_steps))

    def value(self, step: int) -> float:
        return min(1.0, step / self.warmup_steps)

    def apply(self, model: nn.Module, step: int) -> float:
        rho = self.value(step)
        set_quant_strength(model, rho)
        return rho


def ternary_report(model: nn.Module) -> dict[str, Any]:
    """Aggregate ternary weight statistics across all FakeBitLinear layers."""
    layers: list[tuple[str, dict[str, float]]] = []
    neg = zero = pos = total = 0.0
    for name, module in iter_bitlinears(model):
        stats = module.ternary_stats()
        count = module.weight.numel()
        layers.append((name, stats))
        neg += stats["neg"] * count
        zero += stats["zero"] * count
        pos += stats["pos"] * count
        total += count

    if total == 0:
        return {"num_ternary_layers": 0}
    return {
        "num_ternary_layers": len(layers),
        "neg": neg / total,
        "zero": zero / total,
        "pos": pos / total,
        "layers": layers,
    }


def is_ternary_saturated(
    report: dict[str, Any], max_zero: float = 0.90, min_zero: float = 0.005
) -> bool:
    """Detect dead-zero or effectively-binary ternary saturation."""
    if report.get("num_ternary_layers", 0) == 0:
        return False
    zero = report["zero"]
    return zero > max_zero or zero < min_zero
