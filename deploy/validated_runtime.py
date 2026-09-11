"""Optional M16 runtime with owning, once-validated native packed weights.

The original checked operator remains available. Arithmetic is the same ordered
scalar FP32 helper, not a new SIMD or all-integer implementation. Additional
owned-weight memory is reported; this is not a zero-copy runtime.
"""
from __future__ import annotations

import torch

from deploy import m10_native
from deploy.m10_runtime import CPURecursiveRuntime


class ValidatedCPURecursiveRuntime(CPURecursiveRuntime):
    def __init__(self, artifact):
        super().__init__(artifact)
        extension = m10_native.load_extension()
        self._geometry = {}
        self._handles = {}
        for name, entry in self.linears.items():
            inputs, outputs = int(entry["in_features"]), int(entry["out_features"])
            self._geometry[name] = (inputs, outputs)
            self._handles[name] = extension.ValidatedPackedLinear(
                entry["packed"], entry["scale"], entry["bias"], outputs, inputs)

    def backend_report(self):
        report = super().backend_report()
        report.update(runtime="spectra_validated_cpu_recursive_v1",
                      operator="packed_ternary_fp32_validated_handle_v1",
                      weight_validation="once_on_private_owned_copy",
                      additional_owned_weight_bytes=sum(h.storage_bytes() for h in self._handles.values()))
        return report

    def _linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self._handles:
            raise RuntimeError(f"artifact has no packed linear {name}")
        if x.device.type != "cpu" or x.dtype is not torch.float32:
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
