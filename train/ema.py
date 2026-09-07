"""Exponential moving average of model parameters."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import torch
import torch.nn as nn


class EMA:
    """Tracks an exponential moving average of a model's trainable parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Lerp shadow weights toward current raw training parameters."""
        current = {
            name: p for name, p in model.named_parameters() if p.requires_grad
        }
        if set(current) != set(self.shadow):
            raise RuntimeError("EMA/model trainable-parameter names changed")
        d = self.decay
        for name, p in current.items():
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap EMA weights into ``model`` and restore raw weights."""
        current = {
            name: p for name, p in model.named_parameters() if p.requires_grad
        }
        if set(current) != set(self.shadow):
            raise RuntimeError("EMA/model trainable-parameter names changed")
        backup = {name: p.detach().clone() for name, p in current.items()}
        try:
            for name, p in current.items():
                p.data.copy_(self.shadow[name])
            yield
        finally:
            for name, p in current.items():
                p.data.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return detached clones so serialization cannot alias live EMA state."""
        return {name: tensor.detach().clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Strictly restore EMA state; never substitute raw weights."""
        if not isinstance(state, Mapping):
            raise TypeError("EMA state must be a mapping")
        expected = set(self.shadow)
        actual = set(state)
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"EMA state keys mismatch: missing={missing}, extra={extra}")
        for name in sorted(expected):
            tensor = state[name]
            if not torch.is_tensor(tensor):
                raise TypeError(f"EMA entry {name!r} is not a tensor")
            if tuple(tensor.shape) != tuple(self.shadow[name].shape):
                raise ValueError(
                    f"EMA shape mismatch for {name}: "
                    f"checkpoint={tuple(tensor.shape)}, current={tuple(self.shadow[name].shape)}"
                )
            self.shadow[name].copy_(
                tensor.to(device=self.shadow[name].device, dtype=self.shadow[name].dtype)
            )
