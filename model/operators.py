"""Operator blocks used inside the shared recursive function f_theta.

The recursive core repeatedly applies a small stack of these blocks.  M11 adds a
batch-size-one *active-query* execution path.  The sparse path is deliberately
narrow: query/output/FFN rows may be compacted, while K/V remain dense so frozen
tokens stay in the attention context.  Full density calls the ordinary dense
forward directly and is therefore an exact regression anchor.
"""

from __future__ import annotations

from typing import Callable, MutableMapping

import torch
import torch.nn as nn
import torch.nn.functional as F

LinearFactory = Callable[..., nn.Module]


def _bump(work: MutableMapping[str, int] | None, key: str, amount: int = 1) -> None:
    if work is not None:
        work[key] = int(work.get(key, 0)) + int(amount)


class SelfAttention(nn.Module):
    """Multi-head self-attention with projection layers built from ``linear_cls``."""

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
        self.head_dim = dim // heads
        self.q = linear_cls(dim, dim, bias=bias)
        self.k = linear_cls(dim, dim, bias=bias)
        self.v = linear_cls(dim, dim, bias=bias)
        self.proj = linear_cls(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class SwapBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm -> MHSA -> residual -> RMSNorm -> FFN.

    ``forward_active`` is the M11 faithful sparse path.  It only computes query,
    attention-output and FFN rows for active tokens, but computes K/V over the
    complete sequence.  Consequently an active query attends to exactly the same
    set of token states as the dense block.  The method is intended for a
    one-block adaptive operator; ``TRM`` falls back to dense execution for partial
    masks when multiple blocks would require inactive intermediate states.
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
            self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.RMSNorm(dim)
        self.ff = nn.Sequential(
            linear_cls(dim, mlp_ratio * dim),
            nn.GELU(),
            linear_cls(mlp_ratio * dim, dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.norm1(h)
        if self.ternary_attn:
            attn_out = self.attn(q)
        else:
            attn_out, _ = self.attn(q, q, q, need_weights=False)
        h = h + attn_out
        h = h + self.ff(self.norm2(h))
        return h

    def _record_dense_work(self, b: int, n: int, work: MutableMapping[str, int] | None) -> None:
        vectors = b * n
        _bump(work, "block_applications", 1)
        _bump(work, "attention_q_vectors", vectors)
        _bump(work, "attention_k_vectors", vectors)
        _bump(work, "attention_v_vectors", vectors)
        _bump(work, "attention_output_vectors", vectors)
        _bump(work, "ffn_input_vectors", vectors)
        _bump(work, "ffn_output_vectors", vectors)
        heads = self.attn.heads if self.ternary_attn else self.attn.num_heads
        _bump(work, "attention_query_key_pairs", b * heads * n * n)

    def _active_attention_ternary(
        self,
        q_dense: torch.Tensor,
        active_idx: torch.Tensor,
        work: MutableMapping[str, int] | None,
    ) -> torch.Tensor:
        attn = self.attn
        assert isinstance(attn, SelfAttention)
        b, n, d = q_dense.shape
        a = int(active_idx.numel())
        q_in = q_dense.index_select(1, active_idx)
        q = attn.q(q_in).view(b, a, attn.heads, attn.head_dim).transpose(1, 2)
        k = attn.k(q_dense).view(b, n, attn.heads, attn.head_dim).transpose(1, 2)
        v = attn.v(q_dense).view(b, n, attn.heads, attn.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, a, d)
        out = attn.proj(out)
        _bump(work, "attention_q_vectors", b * a)
        _bump(work, "attention_k_vectors", b * n)
        _bump(work, "attention_v_vectors", b * n)
        _bump(work, "attention_output_vectors", b * a)
        _bump(work, "attention_query_key_pairs", b * attn.heads * a * n)
        return out

    def _active_attention_mha(
        self,
        q_dense: torch.Tensor,
        active_idx: torch.Tensor,
        work: MutableMapping[str, int] | None,
    ) -> torch.Tensor:
        attn = self.attn
        assert isinstance(attn, nn.MultiheadAttention)
        if attn.in_proj_weight is None:
            raise RuntimeError("M11 sparse MHA requires packed in_proj_weight")
        b, n, d = q_dense.shape
        a = int(active_idx.numel())
        q_in = q_dense.index_select(1, active_idx)
        qw, kw, vw = attn.in_proj_weight.chunk(3, dim=0)
        if attn.in_proj_bias is None:
            qb = kb = vb = None
        else:
            qb, kb, vb = attn.in_proj_bias.chunk(3, dim=0)
        q_lin = F.linear(q_in, qw, qb)
        k_lin = F.linear(q_dense, kw, kb)
        v_lin = F.linear(q_dense, vw, vb)
        heads = int(attn.num_heads)
        head_dim = d // heads
        q = q_lin.view(b, a, heads, head_dim).transpose(1, 2)
        k = k_lin.view(b, n, heads, head_dim).transpose(1, 2)
        v = v_lin.view(b, n, heads, head_dim).transpose(1, 2)
        dropout_p = float(attn.dropout) if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(b, a, d)
        out = attn.out_proj(out)
        _bump(work, "attention_q_vectors", b * a)
        _bump(work, "attention_k_vectors", b * n)
        _bump(work, "attention_v_vectors", b * n)
        _bump(work, "attention_output_vectors", b * a)
        _bump(work, "attention_query_key_pairs", b * heads * a * n)
        return out

    def forward_active(
        self,
        h: torch.Tensor,
        active_mask: torch.Tensor,
        work: MutableMapping[str, int] | None = None,
    ) -> torch.Tensor:
        """Faithful active-query execution for a batch-size-one partial mask.

        Inactive output rows are copied from the input because the caller will not
        commit them to recurrent state.  Full density calls ``forward`` directly.
        """
        if h.ndim != 3:
            raise ValueError("SwapBlock.forward_active expects [B,L,D]")
        b, n, _ = h.shape
        if active_mask.shape not in {(b, n), (b, n, 1)}:
            raise ValueError("active_mask must have shape [B,L] or [B,L,1]")
        mask2 = active_mask[..., 0] if active_mask.ndim == 3 else active_mask
        mask2 = mask2.bool()
        active_count = int(mask2.sum())

        if active_count == b * n:
            self._record_dense_work(b, n, work)
            return self.forward(h)
        if b != 1:
            raise ValueError("partial active-query execution is currently batch-size-one only")
        if active_count == 0:
            _bump(work, "block_applications", 0)
            return h

        _bump(work, "block_applications", 1)
        active_idx = torch.nonzero(mask2[0], as_tuple=False).flatten()
        q_dense = self.norm1(h)  # dense: K/V context needs every token
        _bump(work, "attention_norm_vectors_dense", n)
        if self.ternary_attn:
            attn_active = self._active_attention_ternary(q_dense, active_idx, work)
        else:
            attn_active = self._active_attention_mha(q_dense, active_idx, work)

        h_active = h.index_select(1, active_idx) + attn_active
        ff_in = self.norm2(h_active)
        ff = self.ff[0](ff_in)
        ff = self.ff[1](ff)
        ff = self.ff[2](ff)
        active_out = h_active + ff
        _bump(work, "ffn_input_vectors", active_count)
        _bump(work, "ffn_output_vectors", active_count)
        _bump(work, "ffn_norm_vectors", active_count)

        out = h.clone()
        out[:, active_idx, :] = active_out
        return out
