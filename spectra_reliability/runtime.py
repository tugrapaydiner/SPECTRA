"""Integration adapters for the existing SPECTRA inference graph.

No substitute activation, norm, attention, recurrence or training checkpoint.
The ternary adapter changes only linear execution; native semantic exit changes
only the exact checker implementation relative to accepted M14 Attempt 5.
"""
from __future__ import annotations
from pathlib import Path
import torch
from deploy.m10_artifact import LoadedCPUArtifact
from deploy.m10_runtime import CPURecursiveRuntime
from .native import NativeCPU


class ValidatedCPURecursiveRuntime(CPURecursiveRuntime):
    def __init__(self, artifact: LoadedCPUArtifact | str, native: NativeCPU, *, avx2: bool = True):
        if type(avx2) is not bool: raise ValueError("avx2 must be boolean")
        if avx2 and not native.has_avx2: raise RuntimeError("AVX2 requested but unavailable")
        super().__init__(artifact)
        self.native = native; self.avx2 = avx2; self.handles = {}
        self.fp = {name: tensor.detach().clone() for name, tensor in self.fp.items()}
        try:
            for name, entry in self.linears.items():
                self.handles[name] = native.weight(entry["packed"].numpy(), entry["scale"].numpy(),
                                                   entry["bias"].numpy(), int(entry["in_features"]))
        except Exception:
            self.close(); raise

    def close(self):
        for handle in self.handles.values(): handle.close()

    def _linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self.handles: raise RuntimeError(f"no validated native linear {name}")
        if x.device.type != "cpu" or x.dtype != torch.float32 or x.requires_grad:
            raise TypeError("validated runtime requires CPU FP32 inference without gradients")
        handle = self.handles[name]
        if x.shape[-1] != handle.hidden: raise ValueError("input feature mismatch")
        leading = x.shape[:-1]; x2 = x.reshape(-1, handle.hidden).contiguous()
        y2 = torch.from_numpy(handle.linear(x2.numpy(), avx2=self.avx2))
        vectors = int(x2.shape[0]); self._layer_calls[name] += 1
        self._native_vectors += vectors; self._native_scalar_products += vectors * handle.hidden * handle.out
        return y2.reshape(*leading, handle.out)

    def backend_report(self):
        memory = {name: handle.memory for name, handle in self.handles.items()}
        return {"backend": "spectra_m16_validated_cpu", "runtime": "m10_graph_m16_linears_v1",
            "operator": "ordered_fp32_output_avx2" if self.avx2 else "ordered_fp32_cached_scalar",
            "vectorized": self.avx2, "native_library": self.native.path.name,
            "mixed_precision": True, "float_work": list(self.FLOAT_WORK),
            "packed_only_execution": False, "bitnet_cpp_supported": False,
            "weight_payload_bytes": sum(v["tensor_payload_bytes"] for v in memory.values()),
            "additional_int8_layout_bytes": sum(v["unpacked_transposed_int8_bytes"] for v in memory.values()),
            "weight_memory_by_layer": memory, "cache_residency_established": False,
            "remaining_original_artifact_storage_excluded_from_weight_handle_bytes": True}


@torch.inference_mode()
def semantic_exit_native(model, x: torch.Tensor, max_steps: int, native: NativeCPU):
    """Same accepted FP64 dual-stream solve, with an equivalent native checker."""
    if x.device.type != "cpu" or x.dtype != torch.int64 or x.shape != (1, 16):
        raise ValueError("requires CPU int64 x[1,16]")
    if type(max_steps) is not int or not 1 <= max_steps <= 4: raise ValueError("max_steps in [1,4] required")
    if model.ternary or model.act8 or model.dim != 64 or model.n != 1 or model.T != 1:
        raise ValueError("requires accepted FP64 dual-stream n1/t1 contract")
    model.eval(); x_emb = model.token_embed(x) + model.encode_positions(x, 4, 4)
    y = torch.zeros_like(x_emb); z = torch.zeros_like(x_emb); given = x != 0
    xp = x.reshape(-1).contiguous().numpy(); ok = False; answer = None; steps = 0
    for _ in range(max_steps):
        y, z = model.recursive_cycle(x_emb, y, z); steps += 1
        answer = torch.where(given, x, model.out_head(y).argmax(dim=-1))
        ok = native.sudoku_valid(xp, answer.reshape(-1).contiguous().numpy(), 2)
        if ok: break
    return answer, {"executed_steps": steps, "semantic_checks": steps, "final_semantic": ok,
        "block_applications": steps * (model.n + 1) * len(model.blocks), "target_used": False,
        "checker": "native_sudoku_bitmask_v1", "stop_reason": "semantic_valid" if ok else "budget_exhausted"}
