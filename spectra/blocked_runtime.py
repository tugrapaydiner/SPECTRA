"""Opt-in ordered packed-linear blocking for the version-1 CPU artifact.

Each native handle owns validated weight copies. Four independent input vectors
share weight decoding, but each accumulator visits hidden coordinates in the
historical order. The original runtimes and native operators are unchanged.
Importing this optional module requires Torch; compilation is explicit at runtime
construction. One runtime instance per worker is required for mutable telemetry.
"""
from __future__ import annotations

import functools
from pathlib import Path
import sys
from typing import Any

import torch

from deploy.m10_artifact import LoadedCPUArtifact
from deploy.m10_runtime import CPURecursiveRuntime


def compile_flags() -> list[str]:
    """Do not enable FP contraction or reduction reassociation."""
    if sys.platform == "win32":
        return ["/O2", "/std:c++20", "/fp:strict"]
    return ["-O3", "-std=c++20", "-fno-fast-math", "-ffp-contract=off"]


@functools.lru_cache(maxsize=1)
def load_extension() -> Any:
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent / "_native" / "blocked_linear.cpp"
    try:
        return load(
            name="spectra_ordered_blocked_v1",
            sources=[str(source)],
            extra_cflags=compile_flags(),
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to build/load ordered blocked CPU kernel: {exc}") from exc


class BlockedCPURecursiveRuntime(CPURecursiveRuntime):
    """Same recurrent graph; owned four-vector blocked packed-linear handles.

    Weight storage duplicates the artifact's packed tensors and scales/biases.
    This is not a zero-copy, all-integer, energy, or general solver improvement.
    """

    def __init__(self, artifact: LoadedCPUArtifact | str):
        super().__init__(artifact)
        extension = load_extension()
        self._geometry: dict[str, tuple[int, int]] = {}
        self._handles: dict[str, Any] = {}
        for name, entry in self.linears.items():
            inputs, outputs = int(entry["in_features"]), int(entry["out_features"])
            self._geometry[name] = (inputs, outputs)
            self._handles[name] = extension.BlockedPackedLinear(
                entry["packed"], entry["scale"], entry["bias"], outputs, inputs
            )

    def backend_report(self) -> dict[str, Any]:
        # Avoid compiling the old operator merely to describe this runtime.
        return {
            "backend": "spectra_ordered_blocked_native",
            "runtime": "spectra_blocked_cpu_recursive_v1",
            "operator": "packed_ternary_fp32_block4_v1",
            "artifact_format": self.payload["format"],
            "artifact_version": self.payload["version"],
            "backend_contract": self.payload["backend_contract"],
            "mixed_precision": True,
            "float_work": list(self.FLOAT_WORK),
            "weight_validation": "once_on_private_owned_copy",
            "input_vector_block": 4,
            "hidden_reduction_order": "increasing_index_without_fp_contraction",
            "compile_flags": compile_flags(),
            "additional_owned_weight_bytes": sum(h.storage_bytes() for h in self._handles.values()),
            "weight_memory_scope": "owned payload bytes, not allocator overhead or process RSS",
        }

    def _linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self._handles:
            raise RuntimeError(f"artifact has no packed linear {name}")
        if not isinstance(x, torch.Tensor) or x.device.type != "cpu" or x.dtype is not torch.float32:
            raise TypeError("native input must be CPU FP32")
        inputs, outputs = self._geometry[name]
        if x.ndim < 1 or int(x.shape[-1]) != inputs:
            raise ValueError("input feature mismatch")
        leading = x.shape[:-1]
        x2 = x.reshape(-1, inputs).contiguous()
        y = self._handles[name].forward(x2)
        vectors = int(x2.shape[0])
        self._layer_calls[name] += 1
        self._native_vectors += vectors
        self._native_scalar_products += vectors * inputs * outputs
        return y.reshape(*leading, outputs)
