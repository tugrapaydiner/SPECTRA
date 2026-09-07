"""Faithful M10 mixed-precision CPU runtime for the version-1 deployment artifact."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from deploy.bitnet_cpp_adapter import bitnet_cpp_root
from deploy.m10_artifact import LoadedCPUArtifact, load_cpu_artifact
from deploy import m10_native


@dataclass
class RuntimeOutput:
    logits: torch.Tensor
    answer: torch.Tensor
    step_outputs: list[dict[str, torch.Tensor]]
    work: dict[str, Any]


class CPURecursiveRuntime:
    """Execute the supported M10 TRM graph from the deployment artifact only."""

    FLOAT_WORK = [
        "token_embedding_lookup_fp32",
        "row_col_positional_embedding_lookup_add_fp32",
        "rmsnorm_fp32",
        "attention_qkv_reshape_transpose_fp32",
        "scaled_dot_product_attention_fp32",
        "gelu_approximate_none_fp32",
        "residual_add_and_alpha_scale_fp32",
        "recurrent_a8_scale_round_clamp_dequant_fp32_scale",
        "halt_head_dense_fp32",
        "argmax_decode",
    ]

    def __init__(self, artifact: LoadedCPUArtifact | str):
        self.artifact = load_cpu_artifact(artifact) if isinstance(artifact, str) else artifact
        self.payload = self.artifact.payload
        self.arch = dict(self.payload["architecture"])
        self.task = dict(self.payload["task"])
        self.fp: dict[str, torch.Tensor] = dict(self.payload["fp_tensors"])
        self.linears: dict[str, dict[str, Any]] = dict(self.payload["packed_linears"])
        self.dim = int(self.arch["dim"])
        self.num_tokens = int(self.arch["num_tokens"])
        self._reset_work()

    def _reset_work(self) -> None:
        self._layer_calls: Counter[str] = Counter()
        self._native_vectors = 0
        self._native_scalar_products = 0
        self._a8_calls = 0
        self._a8_scale_means: list[float] = []
        self._halt_head_calls = 0
        self._decode_calls = 0

    def backend_report(self) -> dict[str, Any]:
        configured = bitnet_cpp_root()
        return {
            **m10_native.backend_identity(),
            "runtime": "spectra_cpu_recursive_v1",
            "artifact_format": self.payload["format"],
            "artifact_version": self.payload["version"],
            "backend_contract": self.payload["backend_contract"],
            "mixed_precision": True,
            "float_work": list(self.FLOAT_WORK),
            "bitnet_cpp_configured_root": str(configured) if configured is not None else None,
            "bitnet_cpp_supported": False,
            "bitnet_cpp_reason": (
                "No compatible SPECTRA artifact loader and recursive inference integration "
                "has been implemented/tested for bitnet.cpp; a configured directory is not availability."
            ),
        }

    def _linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self.linears:
            raise RuntimeError(f"artifact has no packed linear {name}")
        if x.device.type != "cpu" or x.dtype is not torch.float32:
            raise TypeError(f"native linear {name} requires CPU FP32 input")
        entry = self.linears[name]
        in_features = int(entry["in_features"])
        out_features = int(entry["out_features"])
        if int(x.shape[-1]) != in_features:
            raise ValueError(f"native linear {name} input feature mismatch")
        leading = x.shape[:-1]
        x2 = x.reshape(-1, in_features).contiguous()
        y2 = m10_native.dense_ternary_linear_fp32(
            x2,
            entry["packed"],
            entry["scale"],
            entry["bias"],
            out_features,
        )
        vectors = int(x2.shape[0])
        self._layer_calls[name] += 1
        self._native_vectors += vectors
        self._native_scalar_products += vectors * in_features * out_features
        return y2.reshape(*leading, out_features)

    def _rms_norm(self, x: torch.Tensor, weight_name: str) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.fp[weight_name], eps=None)

    def _attention(self, h: torch.Tensor) -> torch.Tensor:
        b, n, d = h.shape
        heads = int(self.arch["heads"])
        head_dim = d // heads
        q = self._linear("blocks.0.attn.q", h).view(b, n, heads, head_dim).transpose(1, 2)
        k = self._linear("blocks.0.attn.k", h).view(b, n, heads, head_dim).transpose(1, 2)
        v = self._linear("blocks.0.attn.v", h).view(b, n, heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self._linear("blocks.0.attn.proj", out)

    def block(self, h: torch.Tensor) -> torch.Tensor:
        q = self._rms_norm(h, "blocks.0.norm1.weight")
        h = h + self._attention(q)
        ff_in = self._rms_norm(h, "blocks.0.norm2.weight")
        ff = self._linear("blocks.0.ff.0", ff_in)
        ff = F.gelu(ff, approximate="none")
        ff = self._linear("blocks.0.ff.2", ff)
        return h + ff

    def f(self, h: torch.Tensor) -> torch.Tensor:
        # The version-1 artifact supports exactly one block.
        return self.block(h)

    def _a8(self, x: torch.Tensor) -> torch.Tensor:
        eps = float(self.arch["act_quant_eps"])
        qmin = int(self.arch["act_quant_qmin"])
        qmax = int(self.arch["act_quant_qmax"])
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        self._a8_calls += 1
        self._a8_scale_means.append(float(scale.mean()))
        return q * scale

    def recursive_cycle(
        self,
        x_emb: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(int(self.arch["n"])):
            update_z = self.f(x_emb + y + z)
            z = self._a8(
                self._rms_norm(z + self.fp["alpha_z"] * update_z, "norm_z.weight")
            )
        update_y = self.f(y + z)
        y = self._a8(
            self._rms_norm(y + self.fp["alpha_y"] * update_y, "norm_y.weight")
        )
        return y, z

    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.device.type != "cpu" or x.dtype is not torch.long:
            raise TypeError("M10 runtime input must be CPU torch.long")
        if x.ndim != 2 or int(x.shape[1]) != int(self.arch["seq_len"]):
            raise ValueError("M10 runtime input must have shape [B, seq_len]")
        if x.numel() and (int(x.min()) < 0 or int(x.max()) >= self.num_tokens):
            raise ValueError("input token id outside artifact vocabulary")
        height, width = int(self.task["height"]), int(self.task["width"])
        if height * width != int(x.shape[1]):
            raise ValueError("runtime task geometry disagrees with input length")
        token = F.embedding(x, self.fp["token_embed.weight"])
        positions = torch.arange(x.shape[1], dtype=torch.long)
        rows = positions // width
        cols = positions % width
        pos = (
            self.fp["pos_encoder.row_embed.weight"][rows]
            + self.fp["pos_encoder.col_embed.weight"][cols]
        )[None, :, :]
        return token + pos

    def _halt(self, y: torch.Tensor) -> torch.Tensor:
        self._halt_head_calls += 1
        return F.linear(
            y.mean(dim=1), self.fp["halt_head.weight"], self.fp["halt_head.bias"]
        ).squeeze(-1)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        self._decode_calls += 1
        return logits.argmax(dim=-1)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> RuntimeOutput:
        self._reset_work()
        x_emb = self.encode_input(x)
        y = torch.zeros_like(x_emb)
        z = torch.zeros_like(x_emb)
        steps: list[dict[str, torch.Tensor]] = []
        for _ in range(int(self.arch["N_sup"])):
            for _ in range(int(self.arch["T"])):
                y, z = self.recursive_cycle(x_emb, y, z)
            logits = self._linear("out_head", y)
            halt = self._halt(y)
            steps.append({"logits": logits, "halt_logit": halt, "y": y, "z": z})
        answer = self.decode(steps[-1]["logits"])
        return RuntimeOutput(steps[-1]["logits"], answer, steps, self.work_record())

    def expected_native_calls_per_forward(self) -> int:
        per_f = 6  # q,k,v,proj,ff0,ff2 for one supported block
        return int(self.arch["N_sup"]) * (
            int(self.arch["T"]) * (int(self.arch["n"]) + 1) * per_f + 1  # out_head
        )

    def work_record(self) -> dict[str, Any]:
        return {
            "native_linear_calls": int(sum(self._layer_calls.values())),
            "native_calls_by_layer": dict(sorted(self._layer_calls.items())),
            "native_input_vectors": int(self._native_vectors),
            "native_scalar_products": int(self._native_scalar_products),
            "expected_native_linear_calls": self.expected_native_calls_per_forward(),
            "native_call_count_matches_architecture": int(sum(self._layer_calls.values())) == self.expected_native_calls_per_forward(),
            "a8_quantization_calls": int(self._a8_calls),
            "a8_scale_mean_min": min(self._a8_scale_means) if self._a8_scale_means else None,
            "a8_scale_mean_max": max(self._a8_scale_means) if self._a8_scale_means else None,
            "halt_head_fp32_calls": int(self._halt_head_calls),
            "decode_calls": int(self._decode_calls),
            "floating_point_work": list(self.FLOAT_WORK),
        }
