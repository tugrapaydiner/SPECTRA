"""Stability Shield helpers for ternary training (BLUEPRINT section 11).

Progressive ternarization (section 11.5/11.6) is implemented as a smooth
*quant-strength warmup*: every ``FakeBitLinear`` carries a rho in [0, 1] that
interpolates its forward weight from full precision (rho=0) to hard ternary
(rho=1). Ramping rho up over training avoids the shock of an abrupt switch to
ternary. ``ternary_report`` exposes the weight distribution for the saturation
collapse detector (section 11.7).
"""

from __future__ import annotations

from typing import Any, Iterator

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


class QuantWarmup:
    """Linear soft->hard ternarization schedule (section 11.6).

    ``rho = min(1, step / warmup_steps)``. Call :meth:`apply` each training step
    to push the model from full precision toward hard ternary.
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
    """Detect ternary saturation collapse (section 11.7).

    Too many zeros (> ``max_zero``) means a dead model; almost no zeros
    (< ``min_zero``) means the ternary code collapsed to a noisy binary {-1,+1}.
    """
    if report.get("num_ternary_layers", 0) == 0:
        return False
    zero = report["zero"]
    return zero > max_zero or zero < min_zero
