"""Optional output-only execution for the supported mixed-precision CPU artifact.

Import this module only with the research/native extras. The dependency-light
spectra and spectra.cnf imports do not import it. Historical training, tracing
and replay runtimes are unchanged. A runtime instance is not thread-safe: use a
separate instance per concurrent worker, as with CPURecursiveRuntime.forward.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from deploy.m10_runtime import CPURecursiveRuntime
from deploy.validated_runtime import ValidatedCPURecursiveRuntime
from spectra.blocked_runtime import BlockedCPURecursiveRuntime


_SUPPORTED_RUNTIME_TYPES = (
    CPURecursiveRuntime, ValidatedCPURecursiveRuntime, BlockedCPURecursiveRuntime
)


@dataclass(frozen=True)
class FinalPrediction:
    logits: torch.Tensor
    answer: torch.Tensor
    halt_logit: torch.Tensor
    work: dict[str, Any]


@torch.inference_mode()
def predict_final(runtime: CPURecursiveRuntime, x: torch.Tensor) -> FinalPrediction:
    """Return final outputs without retaining diagnostic recurrent trajectories.

    Intermediate heads cannot affect the recurrence in this exact runtime, so
    they are omitted. Recurrence order, precision, quantization and final head
    operations are unchanged. This does NOT implement early stopping and makes
    no quality claim beyond equivalence to the historical final outputs. It is
    deliberately limited to the three audited concrete runtime types, not arbitrary
    subclass hooks. Validated and blocked handles own private packed-weight copies;
    final-only execution does not bypass their input validation.
    """
    if type(runtime) not in _SUPPORTED_RUNTIME_TYPES:
        raise TypeError("an unmodified supported CPU runtime is required; custom subclasses are not supported")
    runtime._reset_work()
    x_emb = runtime.encode_input(x)
    y = torch.zeros_like(x_emb)
    z = torch.zeros_like(x_emb)
    for _ in range(int(runtime.arch["N_sup"])):
        for _ in range(int(runtime.arch["T"])):
            y, z = runtime.recursive_cycle(x_emb, y, z)
    logits = runtime._linear("out_head", y)
    halt = runtime._halt(y)
    answer = runtime.decode(logits)
    work = runtime.work_record()
    historical_expected = runtime.expected_native_calls_per_forward()
    expected = historical_expected - int(runtime.arch["N_sup"]) + 1
    work.update(expected_native_linear_calls=expected,
                native_call_count_matches_architecture=work["native_linear_calls"] == expected,
                historical_trace_native_linear_calls=historical_expected,
                execution_mode="final_outputs_only", retained_recurrent_steps=0)
    return FinalPrediction(logits, answer, halt, work)
