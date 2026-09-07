"""Auditable arithmetic-intensity helpers for SPECTRA kernel measurements.

M13 convention: one multiply-accumulate (MAC) is two arithmetic operations.
Throughput in GOP/s and arithmetic intensity in operations/byte use the same
numerator. Traffic is a declared *logical/external first-touch model*, not measured
DRAM traffic. No cache-residency or bandwidth-bottleneck claim follows from these
helpers or from a flat throughput curve.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class TrafficModel:
    macs: int
    operations: int
    packed_weight_bytes: int
    input_activation_bytes: int
    output_activation_bytes: int
    requant_parameter_bytes: int
    total_external_first_touch_bytes: int
    decoded_row_scratch_bytes: int
    traffic_scope: str = "kernel_external_first_touch_logical_bytes_not_measured_dram"

    def record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RooflinePoint:
    arithmetic_intensity: float
    ridge_point: float
    attainable_gops: float
    bound: str
    classification_scope: str = "model_from_supplied_peak_and_bandwidth_not_cache_evidence"


def precomputed_input_reuse_traffic(
    out_dim: int,
    hidden: int,
    *,
    weight_bits: float = 2.0,
    reuse: int = 1,
    input_element_bytes: int = 1,
    output_element_bytes: int = 1,
    requant_bytes_per_output: int = 4,
) -> TrafficModel:
    """External first-touch model for the checked `[K,H]` weight-stationary call.

    `reuse` is K precomputed input vectors passed into one kernel invocation. This
    is not a model of sequential recurrent generation and does not assert a cache
    level. Packed rows use row padding to whole bytes for the 2-bit representation.
    """
    if out_dim <= 0 or hidden <= 0 or reuse <= 0:
        raise ValueError("out_dim, hidden and reuse must be positive")
    if weight_bits <= 0 or input_element_bytes <= 0 or output_element_bytes <= 0:
        raise ValueError("bit/element widths must be positive")
    packed_row = int(math.ceil(hidden * weight_bits / 8.0))
    packed = out_dim * packed_row
    inputs = reuse * hidden * input_element_bytes
    outputs = reuse * out_dim * output_element_bytes
    requant = out_dim * requant_bytes_per_output
    macs = reuse * out_dim * hidden
    operations = 2 * macs
    total = packed + inputs + outputs + requant
    # The C++ implementation expands one row to int8 scratch before applying K
    # precomputed vectors. Scratch is disclosed separately, not called DRAM traffic.
    scratch = hidden
    return TrafficModel(
        macs=int(macs), operations=int(operations), packed_weight_bytes=int(packed),
        input_activation_bytes=int(inputs), output_activation_bytes=int(outputs),
        requant_parameter_bytes=int(requant), total_external_first_touch_bytes=int(total),
        decoded_row_scratch_bytes=int(scratch),
    )


def arithmetic_intensity(
    out_dim: int,
    hidden: int,
    weight_bits: float = 2.0,
    reuse: int = 1,
    act_bytes: int = 1,
) -> float:
    """Operations per declared external first-touch byte for precomputed K inputs."""
    t = precomputed_input_reuse_traffic(
        out_dim, hidden, weight_bits=weight_bits, reuse=reuse,
        input_element_bytes=act_bytes, output_element_bytes=act_bytes,
    )
    return t.operations / t.total_external_first_touch_bytes


def classify(
    out_dim: int,
    hidden: int,
    peak_gops: float,
    bandwidth_gbs: float,
    weight_bits: float = 2.0,
    reuse: int = 1,
) -> RooflinePoint:
    """Hypothetical roofline placement from explicitly supplied peak/bandwidth.

    This is a model, not a hardware classification inferred from the throughput
    curve. M13 retained measurements do not use it as evidence of cache residency.
    """
    if peak_gops <= 0 or bandwidth_gbs <= 0:
        raise ValueError("peak_gops and bandwidth_gbs must be positive")
    ai = arithmetic_intensity(out_dim, hidden, weight_bits, reuse)
    ridge = peak_gops / bandwidth_gbs
    return RooflinePoint(ai, ridge, min(peak_gops, ai * bandwidth_gbs),
                         "compute" if ai >= ridge else "memory")


def roofline_report(
    out_dim: int,
    hidden: int,
    peak_gops: float,
    bandwidth_gbs: float,
    recursion_steps: int,
) -> dict:
    """Compatibility report with corrected units and explicit model-only scope."""
    fp32 = classify(out_dim, hidden, peak_gops, bandwidth_gbs, 32, 1)
    w158 = classify(out_dim, hidden, peak_gops, bandwidth_gbs, 2, 1)
    precomputed = classify(out_dim, hidden, peak_gops, bandwidth_gbs, 2, recursion_steps)
    return {
        "operation_convention": "1_MAC_equals_2_arithmetic_operations",
        "traffic_scope": "external_first_touch_logical_bytes_not_measured_dram",
        "classification_scope": "hypothetical_from_supplied_peak_and_bandwidth",
        "ridge_point": w158.ridge_point,
        "fp32_ai": fp32.arithmetic_intensity, "fp32_bound": fp32.bound,
        "w158_ai": w158.arithmetic_intensity, "w158_bound": w158.bound,
        "precomputed_k_input_ai": precomputed.arithmetic_intensity,
        "precomputed_k_input_modelled_bound": precomputed.bound,
        "K": int(recursion_steps),
        "cache_residency_established": False,
        "bandwidth_bottleneck_established": False,
    }
