"""Ternary (W1.58) linear layer with a straight-through estimator.

Implements BLUEPRINT section 18 (FakeBitLinear) with two additions used by the
Stability Shield (section 11):

  * ``quant_strength`` (rho): the soft-ternary warmup of section 11.6,
    ``W_forward = (1 - rho) W + rho * (gamma * W_q)``. ``rho = 1`` is the hard
    ternary forward of section 18; ``rho = 0`` is full precision. Ramping rho up
    over training avoids the shock of switching weights to ternary all at once.
  * ``ternary_stats`` for the saturation collapse detector (section 11.7).

Weights stay full precision (the master weight); only the *forward* uses ternary
values, with gradients flowing to the master weight via the STE.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeBitLinear(nn.Module):
    """Drop-in ``nn.Linear`` replacement with ternary {-1, 0, +1} forward weights.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include a (full-precision) bias.
        eps: Floor for the per-channel scale to avoid divide-by-zero.
        quant_strength: Initial soft-ternary interpolation rho in [0, 1].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
        quant_strength: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Non-persistent buffer: rides along with .to(device) but is not a weight.
        self.register_buffer(
            "quant_strength", torch.tensor(float(quant_strength)), persistent=False
        )

    def _ternarize_hard(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Absmean ternarization: returns (w_q in {-1,0,1}, per-channel scale)."""
        scale = w.abs().mean(dim=1, keepdim=True).clamp_min(self.eps)
        w_q = torch.clamp(torch.round(w / scale), -1, 1)
        return w_q, scale

    def ternarize(self, w: torch.Tensor) -> torch.Tensor:
        """Soft/hard ternary weight with STE (forward ternary, grad to master)."""
        w_q, scale = self._ternarize_hard(w)
        w_hard = w_q * scale  # gamma * W_q
        rho = self.quant_strength
        w_soft = (1.0 - rho) * w + rho * w_hard
        # Straight-through estimator: identity gradient w.r.t. the master weight.
        return w + (w_soft - w).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.ternarize(self.weight), self.bias)

    @torch.no_grad()
    def ternary_stats(self) -> dict[str, float]:
        """Fraction of hard-ternary weights that are negative / zero / positive."""
        w_q, _ = self._ternarize_hard(self.weight)
        total = w_q.numel()
        return {
            "neg": (w_q < 0).sum().item() / total,
            "zero": (w_q == 0).sum().item() / total,
            "pos": (w_q > 0).sum().item() / total,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
