"""INT8 activation fake-quantization (BLUEPRINT section 19).

The deployment target is W1.58A8: ternary weights with **INT8 activations**. To
validate that INT8 latent-state recursion still trains, we fake-quantize the
recurrent states inside the loop with a straight-through estimator -- forward
uses the INT8-rounded value, gradients pass through unchanged.

Scale is per-token (per last-dim vector), recomputed dynamically from the running
activation so it tracks the changing latent magnitude during recursion.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FakeActQuant(nn.Module):
    """Per-token symmetric INT8 fake-quantizer with an STE.

    Args:
        bits: Quantization bit-width (8 for the A8 target).
        eps: Floor on the scale to avoid divide-by-zero on all-zero activations.
    """

    def __init__(self, bits: int = 8, eps: float = 1e-6):
        super().__init__()
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1
        self.eps = eps
        # Last per-token scale mean, for activation-scale logging (section 29).
        self.register_buffer("last_scale_mean", torch.zeros(()), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic per-token scale: max-abs over the feature dim mapped to qmax.
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(self.eps) / self.qmax
        x_q = torch.clamp(torch.round(x / scale), self.qmin, self.qmax)
        x_deq = x_q * scale
        with torch.no_grad():
            self.last_scale_mean = scale.mean()
        # STE: forward is the INT8-rounded value, backward is identity.
        return x + (x_deq - x).detach()
