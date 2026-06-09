"""Roofline / arithmetic-intensity analysis for the W1.58A8 B=1 GEMV kernel.

THE BRUTAL PHYSICS (read this before claiming the kernel "defeats" DRAM):

At batch size 1 a matrix-vector product reads each weight EXACTLY ONCE and uses
it in EXACTLY ONE multiply-accumulate. There is no reuse dimension, so:

    Arithmetic Intensity (AI) = ops / DRAM_bytes
                              = (O*H) / (O*H * bits/8)
                              = 8 / bits   [ops per byte]

For FP32 weights:  AI = 8/32 = 0.25 ops/byte   -> hopelessly memory-bound.
For W1.58 (2-bit): AI = 8/2  = 4.0  ops/byte   -> 16x better, but STILL low.

You CANNOT raise the AI of a single B=1 GEMV by cache-blocking: there is nothing
to reuse. Quantization helps only by cutting the bytes 16x (the time at the
memory-bound limit is ``weight_bytes / bandwidth``, so 16x fewer bytes = 16x
faster). Whether AI=4 crosses the machine's ridge point depends on the device;
on a bandwidth-starved single legacy core it sits right AT the ridge.

The ONLY physically valid way to become compute-bound is to introduce reuse, and
recursion provides it: the SAME ~1.4 MB ternary core is re-applied K = T*n*N_sup
times. If those weights stay resident in L2/L3 across the recursion, DRAM pays for
them ONCE and amortises over K applications:

    AI_recursion = (K * O*H) / (O*H * bits/8) = K * 8/bits = 4K  for 2-bit.

That is the real "cache-resident" claim, and it is sound -- realised by the
weight-stationary recursion kernel (``spectra_weight_stationary_gemv``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RooflinePoint:
    arithmetic_intensity: float  # ops / DRAM byte
    ridge_point: float           # peak_ops / bandwidth (ops/byte)
    attainable_gops: float       # min(peak, AI * bandwidth)
    bound: str                   # "memory" or "compute"


def arithmetic_intensity(
    out_dim: int,
    hidden: int,
    weight_bits: float = 2.0,
    reuse: int = 1,
    act_bytes: int = 1,
) -> float:
    """Ops-per-DRAM-byte of a (possibly reuse-``reuse``) ternary GEMV.

    ``reuse`` = number of times the (resident) weight matrix is re-applied before
    being evicted -- 1 for a single GEMV, K for a recursion held in cache.
    """
    ops = reuse * out_dim * hidden
    weight_bytes = out_dim * hidden * weight_bits / 8.0      # streamed from DRAM ONCE
    # In a B=1 recursion the activation/answer states are generated on-chip and
    # stay L1/L2-resident across the K applications, so they are first-touch only.
    resident_bytes = hidden * act_bytes + out_dim
    dram = weight_bytes + resident_bytes
    return ops / dram


def classify(
    out_dim: int,
    hidden: int,
    peak_gops: float,
    bandwidth_gbs: float,
    weight_bits: float = 2.0,
    reuse: int = 1,
) -> RooflinePoint:
    """Place the kernel on the roofline for a given device."""
    ai = arithmetic_intensity(out_dim, hidden, weight_bits, reuse)
    ridge = peak_gops / bandwidth_gbs
    attainable = min(peak_gops, ai * bandwidth_gbs)
    return RooflinePoint(
        arithmetic_intensity=ai,
        ridge_point=ridge,
        attainable_gops=attainable,
        bound="compute" if ai >= ridge else "memory",
    )


def roofline_report(
    out_dim: int,
    hidden: int,
    peak_gops: float,
    bandwidth_gbs: float,
    recursion_steps: int,
) -> dict:
    """Compare FP32 / W1.58 single-GEMV / W1.58 recursion-resident on one device."""
    fp32 = classify(out_dim, hidden, peak_gops, bandwidth_gbs, weight_bits=32, reuse=1)
    w158 = classify(out_dim, hidden, peak_gops, bandwidth_gbs, weight_bits=2, reuse=1)
    recur = classify(out_dim, hidden, peak_gops, bandwidth_gbs, weight_bits=2, reuse=recursion_steps)
    return {
        "ridge_point": w158.ridge_point,
        "fp32_ai": fp32.arithmetic_intensity, "fp32_bound": fp32.bound,
        "w158_ai": w158.arithmetic_intensity, "w158_bound": w158.bound,
        "recursion_ai": recur.arithmetic_intensity, "recursion_bound": recur.bound,
        "recursion_steps": recursion_steps,
        # The decisive number: how much faster the recursion-resident kernel is than
        # FP32 at the memory-bound limit (= bytes ratio, since both are streamed).
        "dram_byte_reduction_vs_fp32": (32 / 2) * recursion_steps,
    }
