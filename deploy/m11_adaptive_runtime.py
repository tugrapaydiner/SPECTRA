"""M11 adaptive execution on the accepted M10 packed-ternary native primitive.

No new arithmetic kernel is introduced.  Pointwise sparse projections gather the
active token rows and call ``CPURecursiveRuntime._linear``, which in turn calls the
same checked M10 packed-ternary FP32 C++ operator.  Attention keys/values remain
dense over the complete sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from deploy.m10_runtime import CPURecursiveRuntime


def _semantic_zero() -> dict[str, int]:
    return {
        "supervision_steps": 0,
        "recursive_cycles": 0,
        "recursive_cycles_skipped": 0,
        "block_applications": 0,
        "attention_q_vectors": 0,
        "attention_k_vectors": 0,
        "attention_v_vectors": 0,
        "attention_output_vectors": 0,
        "attention_query_key_pairs": 0,
        "ffn_input_vectors": 0,
        "ffn_output_vectors": 0,
        "recurrent_norm_vectors": 0,
        "a8_vectors": 0,
        "active_token_updates": 0,
        "frozen_token_copies": 0,
        "all_frozen_step_skips": 0,
        "output_head_vectors": 0,
    }


@dataclass
class NativeExecutionState:
    x_emb: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    next_step: int = 0
    active_mask: torch.Tensor | None = None
    ever_frozen: torch.Tensor | None = None
    semantic_work: dict[str, int] = field(default_factory=_semantic_zero)


class AdaptiveCPURecursiveRuntime(CPURecursiveRuntime):
    """Batch-size-one incremental/sparse runtime for an M10 v1 artifact."""

    def _reset_work(self) -> None:
        super()._reset_work()
        self._semantic_work = _semantic_zero()

    def _record_dense_f(self, n: int) -> None:
        heads = int(self.arch["heads"])
        self._semantic_work["block_applications"] += 1
        self._semantic_work["attention_q_vectors"] += n
        self._semantic_work["attention_k_vectors"] += n
        self._semantic_work["attention_v_vectors"] += n
        self._semantic_work["attention_output_vectors"] += n
        self._semantic_work["attention_query_key_pairs"] += heads * n * n
        self._semantic_work["ffn_input_vectors"] += n
        self._semantic_work["ffn_output_vectors"] += n

    def f(self, h: torch.Tensor) -> torch.Tensor:
        self._record_dense_f(int(h.shape[1]))
        return super().f(h)

    @staticmethod
    def _mask(state: NativeExecutionState, proposed: torch.Tensor | None, policy: str) -> torch.Tensor:
        if policy not in {"allow", "sticky"}:
            raise ValueError("reactivation_policy must be 'allow' or 'sticky'")
        b, n, _ = state.y.shape
        if proposed is None:
            mask = torch.ones(b, n, 1, dtype=torch.bool)
        else:
            if proposed.shape == (b, n):
                proposed = proposed.unsqueeze(-1)
            if proposed.shape != (b, n, 1):
                raise ValueError("active mask must have shape [B,L] or [B,L,1]")
            if proposed.dtype is torch.bool:
                mask = proposed
            else:
                if not torch.isfinite(proposed).all() or not bool(((proposed == 0) | (proposed == 1)).all()):
                    raise ValueError("active mask must contain only 0/1")
                mask = proposed.bool()
        if policy == "sticky":
            assert state.ever_frozen is not None
            mask = mask & ~state.ever_frozen
        return mask

    def init_execution_state(self, x: torch.Tensor) -> NativeExecutionState:
        self._reset_work()
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError("M11 native adaptive path is batch-size-one")
        x_emb = self.encode_input(x)
        y = torch.zeros_like(x_emb)
        z = torch.zeros_like(x_emb)
        active = torch.ones(1, x.shape[1], 1, dtype=torch.bool)
        return NativeExecutionState(
            x_emb=x_emb,
            y=y,
            z=z,
            active_mask=active,
            ever_frozen=torch.zeros_like(active),
            semantic_work=self._semantic_work,
        )

    def _attention_active(self, h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        b, n, d = h.shape
        a = int(idx.numel())
        heads = int(self.arch["heads"])
        head_dim = d // heads
        q_in = h.index_select(1, idx)
        q = self._linear("blocks.0.attn.q", q_in).view(b, a, heads, head_dim).transpose(1, 2)
        k = self._linear("blocks.0.attn.k", h).view(b, n, heads, head_dim).transpose(1, 2)
        v = self._linear("blocks.0.attn.v", h).view(b, n, heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, a, d)
        out = self._linear("blocks.0.attn.proj", out)
        self._semantic_work["attention_q_vectors"] += a
        self._semantic_work["attention_k_vectors"] += n
        self._semantic_work["attention_v_vectors"] += n
        self._semantic_work["attention_output_vectors"] += a
        self._semantic_work["attention_query_key_pairs"] += heads * a * n
        return out

    def block_active(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n = int(h.shape[1])
        active = int(mask.sum())
        if active == n:
            return self.f(h)
        if active == 0:
            return h
        self._semantic_work["block_applications"] += 1
        idx = torch.nonzero(mask[0, :, 0], as_tuple=False).flatten()
        norm1 = self._rms_norm(h, "blocks.0.norm1.weight")  # dense K/V context
        attn = self._attention_active(norm1, idx)
        h_active = h.index_select(1, idx) + attn
        ff_in = F.rms_norm(
            h_active,
            (self.dim,),
            self.fp["blocks.0.norm2.weight"],
            eps=None,
        )
        ff = self._linear("blocks.0.ff.0", ff_in)
        ff = F.gelu(ff, approximate="none")
        ff = self._linear("blocks.0.ff.2", ff)
        active_out = h_active + ff
        self._semantic_work["ffn_input_vectors"] += active
        self._semantic_work["ffn_output_vectors"] += active
        out = h.clone()
        out[:, idx, :] = active_out
        return out

    def _commit(
        self,
        base: torch.Tensor,
        update: torch.Tensor,
        alpha_name: str,
        norm_name: str,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        n = int(base.shape[1])
        active = int(mask.sum())
        if active == 0:
            self._semantic_work["frozen_token_copies"] += n
            return base
        if active == n:
            out = self._a8(self._rms_norm(base + self.fp[alpha_name] * update, norm_name))
            self._semantic_work["recurrent_norm_vectors"] += n
            self._semantic_work["a8_vectors"] += n
            self._semantic_work["active_token_updates"] += n
            return out
        idx = torch.nonzero(mask[0, :, 0], as_tuple=False).flatten()
        b = base.index_select(1, idx)
        u = update.index_select(1, idx)
        committed = self._a8(F.rms_norm(b + self.fp[alpha_name] * u, (self.dim,), self.fp[norm_name], eps=None))
        out = base.clone()
        out[:, idx, :] = committed
        self._semantic_work["recurrent_norm_vectors"] += active
        self._semantic_work["a8_vectors"] += active
        self._semantic_work["active_token_updates"] += active
        self._semantic_work["frozen_token_copies"] += n - active
        return out

    def recursive_cycle_active(
        self,
        x_emb: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(int(self.arch["n"])):
            update_z = self.block_active(x_emb + y + z, mask)
            z = self._commit(z, update_z, "alpha_z", "norm_z.weight", mask)
        update_y = self.block_active(y + z, mask)
        y = self._commit(y, update_y, "alpha_y", "norm_y.weight", mask)
        return y, z

    @torch.inference_mode()
    def run_execution_step(
        self,
        state: NativeExecutionState,
        *,
        active_mask: torch.Tensor | None = None,
        reactivation_policy: str = "allow",
    ) -> dict[str, Any]:
        if state.next_step >= int(self.arch["N_sup"]):
            raise RuntimeError("native execution state is exhausted")
        before_sem = dict(self._semantic_work)
        before_vectors = self._native_vectors
        before_products = self._native_scalar_products
        before_calls = int(sum(self._layer_calls.values()))
        mask = self._mask(state, active_mask, reactivation_policy)
        n = int(state.y.shape[1])
        active = int(mask.sum())

        if active == 0:
            self._semantic_work["all_frozen_step_skips"] += 1
            self._semantic_work["recursive_cycles_skipped"] += int(self.arch["T"])
        else:
            for _ in range(int(self.arch["T"])):
                if active == n:
                    state.y, state.z = super().recursive_cycle(state.x_emb, state.y, state.z)
                    # super().recursive_cycle invokes self.f, which records dense semantic work.
                    updates = int(self.arch["n"]) + 1
                    self._semantic_work["recurrent_norm_vectors"] += updates * n
                    self._semantic_work["a8_vectors"] += updates * n
                    self._semantic_work["active_token_updates"] += updates * n
                else:
                    state.y, state.z = self.recursive_cycle_active(state.x_emb, state.y, state.z, mask)
                self._semantic_work["recursive_cycles"] += 1

        logits = self._linear("out_head", state.y)
        halt = self._halt(state.y)
        self._semantic_work["output_head_vectors"] += n
        self._semantic_work["supervision_steps"] += 1
        state.active_mask = mask
        assert state.ever_frozen is not None
        state.ever_frozen = state.ever_frozen | ~mask
        step_index = state.next_step
        state.next_step += 1
        sem_delta = {k: int(self._semantic_work[k] - before_sem.get(k, 0)) for k in self._semantic_work}
        return {
            "step": step_index,
            "logits": logits,
            "halt_logit": halt,
            "y": state.y,
            "z": state.z,
            "mask": mask,
            "active_tokens": active,
            "active_density": active / max(1, n),
            "reactivation_policy": reactivation_policy,
            "work_delta": {
                **sem_delta,
                "native_linear_calls": int(sum(self._layer_calls.values())) - before_calls,
                "native_input_vectors": int(self._native_vectors - before_vectors),
                "native_scalar_products": int(self._native_scalar_products - before_products),
            },
        }

    @torch.inference_mode()
    def forward_incremental(self, x: torch.Tensor) -> dict[str, Any]:
        state = self.init_execution_state(x)
        steps = []
        for _ in range(int(self.arch["N_sup"])):
            steps.append(self.run_execution_step(state))
        answer = self.decode(steps[-1]["logits"])
        return {
            "logits": steps[-1]["logits"],
            "answer": answer,
            "steps": steps,
            "state": state,
            "work": self.adaptive_work_record(),
        }

    def adaptive_work_record(self) -> dict[str, Any]:
        return {
            **dict(self._semantic_work),
            "native_linear_calls": int(sum(self._layer_calls.values())),
            "native_calls_by_layer": dict(sorted(self._layer_calls.items())),
            "native_input_vectors": int(self._native_vectors),
            "native_scalar_products": int(self._native_scalar_products),
            "a8_kernel_calls": int(self._a8_calls),
            "halt_head_fp32_calls": int(self._halt_head_calls),
            "decode_calls": int(self._decode_calls),
            "sparse_contract": "active_q_attn_out_ffn__dense_kv_v1",
            "frozen_tokens_remain_attention_kv": True,
        }
