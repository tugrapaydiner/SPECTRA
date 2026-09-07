"""Tiny Recursive Model (TRM): precision-agnostic recursive reasoning core.

M11 adds an explicit execution-state interface so inference can perform exactly
one deep-supervision step at a time.  This is the primitive used by real online
halting and faithful active-token execution; later steps are never precomputed.
The ordinary ``forward`` API remains compatible and is implemented by repeatedly
calling the same step primitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, MutableMapping

import torch
import torch.nn as nn

from model.bitlinear import FakeBitLinear
from model.fake_quant import FakeActQuant
from model.operators import SwapBlock
from model.spatial_encoder import SpatialEncoder

LinearFactory = Callable[..., nn.Module]


def _zero_work() -> dict[str, int]:
    return {
        "input_embedding_tokens": 0,
        "supervision_steps": 0,
        "recursive_cycles": 0,
        "recursive_cycles_skipped": 0,
        "block_applications": 0,
        "attention_q_vectors": 0,
        "attention_k_vectors": 0,
        "attention_v_vectors": 0,
        "attention_output_vectors": 0,
        "attention_query_key_pairs": 0,
        "attention_norm_vectors_dense": 0,
        "ffn_input_vectors": 0,
        "ffn_output_vectors": 0,
        "ffn_norm_vectors": 0,
        "recurrent_norm_vectors": 0,
        "a8_vectors": 0,
        "active_token_updates": 0,
        "frozen_token_copies": 0,
        "all_frozen_step_skips": 0,
        "sparse_fallback_dense_block_applications": 0,
        "output_head_vectors": 0,
        "model_halt_head_calls": 0,
    }


def _delta(after: MutableMapping[str, int], before: MutableMapping[str, int]) -> dict[str, int]:
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in set(after) | set(before)}


@dataclass
class TRMExecutionState:
    """Observable recurrent state for incremental inference."""

    x_emb: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    next_step: int = 0
    active_mask: torch.Tensor | None = None
    ever_frozen: torch.Tensor | None = None
    work: dict[str, int] = field(default_factory=_zero_work)


class TRM(nn.Module):
    """Recursive reasoning core with deep supervision and a halting head."""

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        seq_len: int,
        n_layers: int = 2,
        n: int = 6,
        T: int = 3,
        N_sup: int = 16,
        heads: int = 8,
        alpha_y: float = 0.1,
        alpha_z: float = 0.1,
        max_grid_size: int = 32,
        linear_cls: LinearFactory = nn.Linear,
        ternary: bool = False,
        act8: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.n = n
        self.T = T
        self.N_sup = N_sup
        self.max_grid_size = max_grid_size
        self.ternary = ternary
        self.act8 = act8
        self.act_quant: nn.Module = FakeActQuant() if act8 else nn.Identity()

        if ternary:
            linear_cls = FakeBitLinear

        self.token_embed = nn.Embedding(num_tokens, dim)
        self.pos_encoder = SpatialEncoder(dim, max_grid_size)
        self.blocks = nn.ModuleList(
            [SwapBlock(dim, heads=heads, linear_cls=linear_cls, ternary_attn=ternary) for _ in range(n_layers)]
        )
        self.out_head: nn.Module = FakeBitLinear(dim, num_tokens, bias=True) if ternary else nn.Linear(dim, num_tokens)
        self.halt_head = nn.Linear(dim, 1)
        self.norm_y = nn.RMSNorm(dim)
        self.norm_z = nn.RMSNorm(dim)
        self.alpha_y = nn.Parameter(torch.tensor(float(alpha_y)))
        self.alpha_z = nn.Parameter(torch.tensor(float(alpha_z)))

    def encode_positions(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return self.pos_encoder(x.shape[1], width, x.device)

    def f(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return h

    def recursive_cycle(
        self,
        x_emb: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Historical dense-compute cycle; mask freezes state but does not save work."""
        for _ in range(self.n):
            update_z = self.f(x_emb + y + z)
            z_new = self.act_quant(self.norm_z(z + self.alpha_z * update_z))
            z = z_new if mask is None else mask * z_new + (1.0 - mask) * z
        update_y = self.f(y + z)
        y_new = self.act_quant(self.norm_y(y + self.alpha_y * update_y))
        y = y_new if mask is None else mask * y_new + (1.0 - mask) * y
        return y, z

    # ------------------------------------------------------------------
    # M11 incremental / adaptive execution
    # ------------------------------------------------------------------
    def init_execution_state(self, x: torch.Tensor, height: int = 9, width: int = 9) -> TRMExecutionState:
        """Encode the problem once and create zero recurrent state."""
        if x.ndim != 2:
            raise ValueError("x must have shape [B,L]")
        if x.shape[1] != height * width:
            raise ValueError("x sequence length must equal height*width")
        x_emb = self.token_embed(x) + self.encode_positions(x, height, width)
        y = torch.zeros_like(x_emb)
        z = torch.zeros_like(x_emb)
        active = torch.ones(x.shape[0], x.shape[1], 1, dtype=torch.bool, device=x.device)
        ever_frozen = torch.zeros_like(active)
        work = _zero_work()
        work["input_embedding_tokens"] = int(x.shape[0] * x.shape[1])
        return TRMExecutionState(x_emb=x_emb, y=y, z=z, active_mask=active, ever_frozen=ever_frozen, work=work)

    @staticmethod
    def _mask_for_state(state: TRMExecutionState, mask: torch.Tensor | None) -> torch.Tensor:
        b, n, _ = state.y.shape
        if mask is None:
            return torch.ones(b, n, 1, dtype=torch.bool, device=state.y.device)
        if mask.shape == (b, n):
            mask = mask.unsqueeze(-1)
        if mask.shape != (b, n, 1):
            raise ValueError("active mask must have shape [B,L] or [B,L,1]")
        if mask.dtype is torch.bool:
            return mask
        if not torch.isfinite(mask).all() or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("active mask must contain only 0/1 values")
        return mask.bool()

    def _effective_mask(
        self,
        state: TRMExecutionState,
        proposed: torch.Tensor | None,
        reactivation_policy: str,
    ) -> torch.Tensor:
        if reactivation_policy not in {"allow", "sticky"}:
            raise ValueError("reactivation_policy must be 'allow' or 'sticky'")
        mask = self._mask_for_state(state, proposed)
        if reactivation_policy == "sticky":
            assert state.ever_frozen is not None
            mask = mask & ~state.ever_frozen
        return mask

    @staticmethod
    def _record_dense_block_work(block: SwapBlock, b: int, n: int, work: dict[str, int]) -> None:
        """Record dense work while still invoking ``block(...)`` through Module.__call__.

        Using the ordinary module call matters: existing telemetry and third-party
        instrumentation install forward hooks on ``SwapBlock``.  M11 full-density
        and dense-fallback execution therefore preserve those hooks exactly.
        """
        block._record_dense_work(b, n, work)

    def _f_active(self, h: torch.Tensor, mask: torch.Tensor, work: dict[str, int]) -> torch.Tensor:
        b, n, _ = h.shape
        partial = int(mask.sum()) != b * n

        # Full density is deliberately the historical module call, not a direct
        # call to forward_active.  This preserves exact forward-hook semantics and
        # gives telemetry an independent regression anchor.
        if not partial:
            out = h
            for block in self.blocks:
                self._record_dense_block_work(block, b, n, work)
                out = block(out)
            return out

        # With >1 block, inactive block-1 outputs would be needed as exact K/V
        # context for block 2.  Fall back to dense rather than changing semantics.
        if b != 1 or len(self.blocks) != 1:
            work["sparse_fallback_dense_block_applications"] += len(self.blocks)
            out = h
            for block in self.blocks:
                self._record_dense_block_work(block, b, n, work)
                out = block(out)
            return out

        return self.blocks[0].forward_active(h, mask, work)

    def _commit_active_update(
        self,
        base: torch.Tensor,
        update: torch.Tensor,
        alpha: torch.Tensor,
        norm: nn.Module,
        mask: torch.Tensor,
        work: dict[str, int],
    ) -> torch.Tensor:
        b, n, _ = base.shape
        active_count = int(mask.sum())
        total = b * n
        if active_count == 0:
            work["frozen_token_copies"] += total
            return base
        if active_count == total:
            out = self.act_quant(norm(base + alpha * update))
            work["recurrent_norm_vectors"] += total
            work["a8_vectors"] += total if self.act8 else 0
            work["active_token_updates"] += total
            return out
        if b != 1:
            dense = self.act_quant(norm(base + alpha * update))
            work["recurrent_norm_vectors"] += total
            work["a8_vectors"] += total if self.act8 else 0
            work["active_token_updates"] += active_count
            work["frozen_token_copies"] += total - active_count
            return torch.where(mask, dense, base)

        idx = torch.nonzero(mask[0, :, 0], as_tuple=False).flatten()
        b_active = base.index_select(1, idx)
        u_active = update.index_select(1, idx)
        committed = self.act_quant(norm(b_active + alpha * u_active))
        out = base.clone()
        out[:, idx, :] = committed
        work["recurrent_norm_vectors"] += active_count
        work["a8_vectors"] += active_count if self.act8 else 0
        work["active_token_updates"] += active_count
        work["frozen_token_copies"] += total - active_count
        return out

    def recursive_cycle_adaptive(
        self,
        x_emb: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        work: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One cycle whose partial-density work follows the M11 sparse contract."""
        for _ in range(self.n):
            update_z = self._f_active(x_emb + y + z, mask, work)
            z = self._commit_active_update(z, update_z, self.alpha_z, self.norm_z, mask, work)
        update_y = self._f_active(y + z, mask, work)
        y = self._commit_active_update(y, update_y, self.alpha_y, self.norm_y, mask, work)
        return y, z

    def run_execution_step(
        self,
        state: TRMExecutionState,
        *,
        active_mask: torch.Tensor | None = None,
        reactivation_policy: str = "allow",
    ) -> dict[str, Any]:
        """Execute exactly one deep-supervision step and return observable state/work.

        No later step is materialised.  An empty active set skips every recursive
        cycle in this supervision step and carries ``y/z`` forward unchanged.
        """
        if state.next_step < 0 or state.next_step >= self.N_sup:
            raise RuntimeError("execution state is already exhausted")
        before = dict(state.work)
        mask = self._effective_mask(state, active_mask, reactivation_policy)
        b, n, _ = state.y.shape
        active_count = int(mask.sum())
        total = b * n

        if active_count == 0:
            state.work["all_frozen_step_skips"] += 1
            state.work["recursive_cycles_skipped"] += int(self.T)
        else:
            for _ in range(self.T):
                state.y, state.z = self.recursive_cycle_adaptive(
                    state.x_emb, state.y, state.z, mask, state.work
                )
                state.work["recursive_cycles"] += 1

        logits = self.out_head(state.y)
        halt_logit = self.halt_head(state.y.mean(dim=1)).squeeze(-1)
        state.work["output_head_vectors"] += total
        state.work["model_halt_head_calls"] += b
        state.work["supervision_steps"] += 1

        state.active_mask = mask
        assert state.ever_frozen is not None
        state.ever_frozen = state.ever_frozen | ~mask
        step_index = state.next_step
        state.next_step += 1

        return {
            "step": step_index,
            "logits": logits,
            "halt_logit": halt_logit,
            "y": state.y,
            "z": state.z,
            "mask": mask,
            "active_tokens": active_count,
            "active_density": active_count / max(1, total),
            "reactivation_policy": reactivation_policy,
            "work_delta": _delta(state.work, before),
            "work_cumulative": dict(state.work),
        }

    def forward(
        self,
        x: torch.Tensor,
        height: int = 9,
        width: int = 9,
        y_target: torch.Tensor | None = None,
        router: Any | None = None,
        device_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Compatibility forward implemented through the explicit step interface."""
        del y_target
        state = self.init_execution_state(x, height=height, width=width)
        step_outputs: list[dict[str, Any]] = []

        for k in range(self.N_sup):
            mask = logprob = None
            if router is not None:
                route_logits = self.out_head(state.y)
                mask, logprob = router(k, state.y, state.z, route_logits, device_state)
            out = self.run_execution_step(state, active_mask=mask, reactivation_policy="allow")
            if router is not None:
                out["router_logprob"] = logprob
            step_outputs.append(out)
            state.y = state.y.detach()
            state.z = state.z.detach()

        return step_outputs[-1]["logits"], step_outputs

    @property
    def effective_depth(self) -> int:
        return self.T * (self.n + 1) * len(self.blocks)
