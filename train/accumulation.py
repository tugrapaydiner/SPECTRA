"""Memory-bounded backward for a fixed-shape, per-example mean loss.

This preserves the optimizer batch and update count, not bitwise reduction order.
Use only deterministic, sample-separable models/losses: BatchNorm, stochastic
layers, cross-example losses and variable-denominator masked losses need their
own accumulation semantics. Call optimizer.zero_grad/step outside this function.
"""
from __future__ import annotations
from typing import Callable
import torch


def backward_mean_loss(loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                       x: torch.Tensor, y: torch.Tensor, microbatch: int) -> float:
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("microbatch must be a positive integer")
    if x.ndim < 1 or y.ndim < 1 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("nonempty aligned optimizer batch required")
    total = 0.0
    for start in range(0, len(x), microbatch):
        end = min(start + microbatch, len(x))
        loss = loss_fn(x[start:end], y[start:end])
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ValueError("mean loss must be a finite scalar")
        weight = (end - start) / len(x)
        (loss * weight).backward()
        total += float(loss.detach()) * weight
    return total
