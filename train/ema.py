"""Exponential moving average of model parameters (BLUEPRINT section 11.3).

EMA is mandatory for SPECTRA: the raw model is used for training, but a slowly-
averaged copy is used for validation and checkpointing, which smooths out the
noise that ternary/recursive training injects. Default decay 0.999.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class EMA:
    """Tracks an exponential moving average of a model's trainable parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Lerp the shadow weights toward the current model parameters."""
        d = self.decay
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap EMA weights into ``model`` (e.g. for validation)."""
        backup = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        try:
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(self.shadow[name])
            yield
        finally:
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, tensor in state.items():
            self.shadow[name].copy_(tensor)
