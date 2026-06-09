"""GT1 tests: the roofline / arithmetic-intensity physics is stated honestly.

Proves: a single B=1 W1.58 GEMV is ~4 ops/byte (16x FP32 but still low / memory-
bound on a starved core), and the recursion-resident reuse is what makes it
compute-bound -- not cache-blocking a single GEMV (which is physically impossible
at B=1).
"""

from eval.roofline import arithmetic_intensity, classify, roofline_report


def test_fp32_is_memory_bound_and_ternary_is_16x_higher():
    O = H = 512
    ai_fp32 = arithmetic_intensity(O, H, weight_bits=32, reuse=1)
    ai_w158 = arithmetic_intensity(O, H, weight_bits=2, reuse=1)
    assert ai_fp32 < 0.30                      # ~0.25 ops/byte: hopeless
    assert 3.5 < ai_w158 < 4.1                 # ~4 ops/byte
    assert ai_w158 / ai_fp32 > 14              # ~16x = 32-bit / 2-bit


def test_recursion_residency_scales_intensity_linearly():
    O = H = 512
    ai1 = arithmetic_intensity(O, H, weight_bits=2, reuse=1)
    ai40 = arithmetic_intensity(O, H, weight_bits=2, reuse=40)
    assert ai40 / ai1 > 35                     # AI ~ 4*reuse (weights streamed once)


def test_single_gemv_memory_bound_but_recursion_compute_bound():
    # Catastrophic edge: ~10 GB/s single-thread DDR3, ~40 GOPS int8 -> ridge = 4.
    O, H, peak, bw = 512, 512, 40.0, 10.0
    fp32 = classify(O, H, peak, bw, weight_bits=32, reuse=1)
    single = classify(O, H, peak, bw, weight_bits=2, reuse=1)
    recur = classify(O, H, peak, bw, weight_bits=2, reuse=42)
    assert fp32.bound == "memory"
    assert recur.bound == "compute"            # residency crosses the ridge
    # A single ternary GEMV sits right AT the ridge (AI ~4 vs ridge 4).
    assert abs(single.arithmetic_intensity - single.ridge_point) < 1.0


def test_roofline_report():
    rep = roofline_report(512, 512, 40.0, 10.0, recursion_steps=42)
    assert rep["fp32_bound"] == "memory"
    assert rep["w158_bound"] in ("memory", "compute")  # borderline at the ridge
    assert rep["recursion_bound"] == "compute"
    assert rep["dram_byte_reduction_vs_fp32"] == 16 * 42
