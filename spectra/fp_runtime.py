"""Owning, prepared FP32 inference for the audited 4x4 Sudoku TRM graph.

Optional native API. No changes to weights, training, recurrent budget or the
exact historical Sudoku checker. The ATen graph is environment-bound, not a
cross-hardware bitwise guarantee. Preparation copies weights; it is not zero-copy.
"""
from __future__ import annotations
import functools
import hashlib
from pathlib import Path
from typing import Any
import torch
from torch import nn
from deploy.m10_native import load_extension as load_checker
from model.operators import SwapBlock
from model.spatial_encoder import SpatialEncoder
from model.trm import TRM
from spectra.blocked_runtime import compile_flags


@functools.lru_cache(maxsize=1)
def load_extension() -> Any:
    from torch.utils.cpp_extension import load
    return load(name="spectra_prepared_fp_v1",
                sources=[str(Path(__file__).parent / "_native" / "fp_step.cpp")],
                extra_cflags=compile_flags(), verbose=False)


def _environment() -> None:
    if not str(torch.__version__).split("+")[0].startswith("2.10."):
        raise RuntimeError("prepared FP currently requires the audited PyTorch 2.10.x environment")
    if not torch.backends.mha.get_fastpath_enabled():
        raise ValueError("prepared FP fidelity requires the eager MHA fast path")
    if torch.is_autocast_enabled("cpu"):
        raise ValueError("CPU autocast is outside the prepared FP32 contract")
    hooks = torch.nn.modules.module
    if hooks._global_forward_hooks or hooks._global_forward_pre_hooks:
        raise ValueError("global forward hooks are outside the prepared graph contract")


def _plain(module: nn.Module, kind: type) -> None:
    if type(module) is not kind:
        raise TypeError(f"expected the unmodified {kind.__name__} module")
    if module.training or module._forward_hooks or module._forward_pre_hooks:
        raise ValueError("all graph modules must be eval-mode and hook-free")
    if any(name in vars(module) for name in
           ("forward", "__call__", "_call_impl", "f", "recursive_cycle", "encode_positions")):
        raise ValueError("instance-overridden graph methods are unsupported")


def _norm(module: nn.Module, dim: int) -> None:
    _plain(module, nn.RMSNorm)
    if module.weight is None or module.eps is not None or tuple(module.normalized_shape) != (dim,):
        raise ValueError("expected default-epsilon hidden-dimension RMSNorm")


def _linear(module: nn.Module) -> None:
    _plain(module, nn.Linear)
    if module.bias is None:
        raise ValueError("this FP graph requires biased linear layers")


def _validate(model: TRM) -> None:
    _environment(); _plain(model, TRM)
    if (model.n != 1 or model.T != 1 or model.ternary or model.act8 or
            model.seq_len != 16 or model.num_tokens != 5 or model.max_grid_size < 4):
        raise ValueError("expected the CPU FP n=T=1, 4x4 Sudoku TRM graph")
    if type(model.N_sup) is not int or not 1 <= model.N_sup <= 256:
        raise ValueError("invalid trained supervision budget")
    _plain(model.act_quant, nn.Identity)
    _plain(model.token_embed, nn.Embedding)
    _plain(model.pos_encoder, SpatialEncoder)
    _plain(model.pos_encoder.row_embed, nn.Embedding)
    _plain(model.pos_encoder.col_embed, nn.Embedding)
    for emb in (model.token_embed, model.pos_encoder.row_embed, model.pos_encoder.col_embed):
        if emb.max_norm is not None or emb.padding_idx is not None:
            raise ValueError("embedding renormalization/padding is unsupported")
    _plain(model.blocks, nn.ModuleList)
    if not 1 <= len(model.blocks) <= 32:
        raise ValueError("invalid block count")
    _norm(model.norm_y, model.dim); _norm(model.norm_z, model.dim)
    _linear(model.out_head); _linear(model.halt_head)
    expected = {"", "act_quant", "token_embed", "pos_encoder", "pos_encoder.row_embed",
                "pos_encoder.col_embed", "blocks", "norm_y", "norm_z", "out_head", "halt_head"}
    for i, block in enumerate(model.blocks):
        _plain(block, SwapBlock)
        if block.ternary_attn:
            raise ValueError("ternary attention is unsupported")
        _norm(block.norm1, model.dim); _norm(block.norm2, model.dim)
        a = block.attn
        _plain(a, nn.MultiheadAttention)
        _plain(a.out_proj, nn.modules.linear.NonDynamicallyQuantizableLinear)
        if (not a.batch_first or not a._qkv_same_embed_dim or a.num_heads % 2 or
                a.add_zero_attn or a.bias_k is not None or a.bias_v is not None or
                a.in_proj_bias is None or a.out_proj.bias is None or a.embed_dim != model.dim):
            raise ValueError("attention is not eligible for the audited native MHA graph")
        _plain(block.ff, nn.Sequential)
        if len(block.ff) != 3:
            raise ValueError("expected linear/GELU/linear FFN")
        _linear(block.ff[0]); _linear(block.ff[2]); _plain(block.ff[1], nn.GELU)
        if block.ff[1].approximate != "none":
            raise ValueError("expected exact GELU approximation setting")
        expected.update(f"blocks.{i}{suffix}" for suffix in
                        ("", ".norm1", ".norm2", ".attn", ".attn.out_proj", ".ff", ".ff.0", ".ff.1", ".ff.2"))
    if {name for name, _ in model.named_modules()} != expected:
        raise ValueError("unexpected modules in the source graph")
    for p in model.parameters():
        if p.device.type != "cpu" or p.dtype != torch.float32 or p.requires_grad:
            raise ValueError("source parameters must be frozen CPU FP32 tensors")
        if not bool(torch.isfinite(p).all()):
            raise ValueError("source parameters must be finite")


class PreparedFPSudoku:
    """Snapshot a supported trained FP model for reference-faithful B=1 solves.

    Source model: eval-mode, CPU FP32, frozen and unmodified. It is not retained
    or mutated. Later source mutations do not alter this owning snapshot.
    Preparation (including native loading) is separate from complete solve cost.
    Each solve constructs a fresh exact checker and zero recurrent state.
    """
    @torch.inference_mode()
    def __init__(self, model: TRM):
        _validate(model)
        self._trained_steps = int(model.N_sup)
        self._block_count = len(model.blocks)
        state = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            state.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}\n".encode())
            state.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        self._source_state_sha256 = state.hexdigest()
        dummy = torch.zeros((1, 16), dtype=torch.int64)
        common = [model.token_embed.weight, model.encode_positions(dummy, 4, 4),
                  model.norm_y.weight, model.norm_z.weight, model.alpha_y, model.alpha_z,
                  model.out_head.weight, model.out_head.bias]
        blocks = []; heads = []
        for b in model.blocks:
            blocks.append([b.norm1.weight, b.attn.in_proj_weight, b.attn.in_proj_bias,
                           b.attn.out_proj.weight, b.attn.out_proj.bias, b.norm2.weight,
                           b.ff[0].weight, b.ff[0].bias, b.ff[2].weight, b.ff[2].bias])
            heads.append(b.attn.num_heads)
        self._attention_heads = tuple(heads)
        self._dimension = int(model.dim)
        self._handle = load_extension().PreparedFPStep(common, blocks, heads)
        self._checker_type = load_checker().SudokuProblem

    @property
    def trained_steps(self) -> int:
        return self._trained_steps

    @property
    def block_count(self) -> int:
        return self._block_count

    @property
    def source_state_sha256(self) -> str:
        return self._source_state_sha256

    def _start(self, x: torch.Tensor, max_steps: int):
        _environment()
        if type(max_steps) is not int or not 1 <= max_steps <= self.trained_steps:
            raise ValueError("max_steps must be inside the trained supervision budget")
        if (type(x) is not torch.Tensor or x.dtype != torch.int64 or x.device.type != "cpu" or
                x.layout != torch.strided or x.shape != (1, 16)):
            raise ValueError("input must be a CPU int64 tensor with shape [1,16]")
        x = x.contiguous()
        checker = self._checker_type(x, 2)
        x_emb = self._handle.encode(x)
        return checker, x_emb, torch.zeros_like(x_emb), torch.zeros_like(x_emb)

    @torch.inference_mode()
    def solve(self, x: torch.Tensor, max_steps: int = 4):
        checker, x_emb, y, z = self._start(x, max_steps)
        for step in range(1, max_steps + 1):
            y, z, logits = self._handle.step(x_emb, y, z)
            answer, valid = checker.decode(logits)
            if valid:
                break
        return answer, {"executed_steps": step, "block_applications": step * 2 * self.block_count,
                        "semantic_checks": step, "checker_constructions": 1,
                        "final_semantic": bool(valid), "target_used": False,
                        "checker": "native_exact_sudoku_v1",
                        "stop_reason": "semantic_valid" if valid else "budget_exhausted"}

    @torch.inference_mode()
    def trace(self, x: torch.Tensor, max_steps: int = 4):
        """Diagnostic full-budget states; never used inside measured solves."""
        _, x_emb, y, z = self._start(x, max_steps)
        records = []
        for _ in range(max_steps):
            y, z, logits = self._handle.step(x_emb, y, z)
            records.append((y.clone(), z.clone(), logits.clone()))
        return x_emb, records

    def identity(self) -> dict[str, Any]:
        return {"runtime": "prepared_fp_sudoku_v1", "source_state_sha256": self.source_state_sha256,
                "trained_steps": self.trained_steps, "block_count": self.block_count,
                "dimension": self._dimension, "attention_heads": list(self._attention_heads),
                "sequence_length": 16, "vocabulary_size": 5, "n": 1, "T": 1,
                "owned_tensor_bytes": self._handle.owned_bytes(), "weight_scope": "owned payload; not RSS",
                "torch": str(torch.__version__), "precision": "FP32", "quantized": False,
                "attention": "aten::_native_multi_head_attention", "positions": "input-independent only"}
