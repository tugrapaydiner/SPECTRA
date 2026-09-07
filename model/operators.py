"""Operator blocks used inside the shared recursive function f_theta.

The recursive core (``model/trm.py``) repeatedly applies a small stack of these
blocks. For the FP16 MVP (Phase 1) we use a standard pre-norm transformer block
(``SwapBlock``, BLUEPRINT section 15.1). Later phases swap the internal
``nn.Linear`` projections for ternary ``FakeBitLinear`` layers (section 11.5
progressive ternarization) -- which is why the projection layer is injectable via
``linear_cls`` rather than hard-coded.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# A factory that builds a linear projection. ``nn.Linear`` for FP16; later phases
# pass ``FakeBitLinear`` to ternarize the FFN/projection weights in place.
LinearFactory = Callable[..., nn.Module]


class SelfAttention(nn.Module):
    """Multi-head self-attention with projection layers built from ``linear_cls``.

    Used for the ternary recursive core (pass ``FakeBitLinear`` as ``linear_cls``)
    so the q/k/v/output projections are ternarized along with the FFN. The FP16
    core instead uses ``nn.MultiheadAttention`` (kept as the validated default in
    ``SwapBlock``); this hand-written variant exists so attention weights can be
    ternarized (BLUEPRINT section 11.5 stage 3 / operator ablations 27.2).
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        linear_cls: LinearFactory = nn.Linear,
        bias: bool = False,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        # Match nn.MultiheadAttention's metadata spelling. This is a plain
        # integer alias only; it does not alter attention math or checkpoint state.
        self.num_heads = heads
        self.head_dim = dim // heads
        self.q = linear_cls(dim, dim, bias=bias)
        self.k = linear_cls(dim, dim, bias=bias)
        self.v = linear_cls(dim, dim, bias=bias)
        self.proj = linear_cls(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        # [B, L, D] -> [B, heads, L, head_dim] for scaled dot-product attention.
        q = self.q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class SwapBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm -> MHSA -> residual -> RMSNorm -> FFN.

    Named per BLUEPRINT section 15.1. Self-attention lets tokens exchange
    information ("swap") each time the operator is applied during recursion.

    Args:
        dim: Hidden dimension D.
        heads: Number of attention heads (``dim`` must be divisible by ``heads``).
        mlp_ratio: FFN expansion factor (4x by default).
        linear_cls: Factory for the FFN projections. Defaults to ``nn.Linear``;
            pass ``FakeBitLinear`` in later phases to ternarize them.
        ternary_attn: If True, use the ``linear_cls``-based :class:`SelfAttention`
            so attention projections are ternarized too; if False, use the
            validated FP16 ``nn.MultiheadAttention`` default.
        dropout: Attention dropout (0.0 by default; recursion is the regulariser).
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        mlp_ratio: int = 4,
        linear_cls: LinearFactory = nn.Linear,
        ternary_attn: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")

        self.ternary_attn = ternary_attn
        self.norm1 = nn.RMSNorm(dim)
        if ternary_attn:
            self.attn: nn.Module = SelfAttention(dim, heads, linear_cls=linear_cls)
        else:
            self.attn = nn.MultiheadAttention(
                dim, heads, dropout=dropout, batch_first=True
            )
        self.norm2 = nn.RMSNorm(dim)
        self.ff = nn.Sequential(
            linear_cls(dim, mlp_ratio * dim),
            nn.GELU(),
            linear_cls(mlp_ratio * dim, dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply attention then FFN with residual connections.

        Args:
            h: Tensor of shape ``[B, L, D]``.

        Returns:
            Tensor of shape ``[B, L, D]``.
        """
        q = self.norm1(h)
        if self.ternary_attn:
            attn_out = self.attn(q)
        else:
            # need_weights=False skips materialising the attention matrix (faster).
            attn_out, _ = self.attn(q, q, q, need_weights=False)
        h = h + attn_out
        h = h + self.ff(self.norm2(h))
        return h
