"""M13 arithmetic-intensity contracts.

These tests verify units/traffic bookkeeping only. They do not treat a modelled
roofline classification or a throughput curve as evidence of cache residency.
"""
from eval.roofline import (
    arithmetic_intensity,
    classify,
    precomputed_input_reuse_traffic,
    roofline_report,
)


def test_fp32_and_ternary_share_two_ops_per_mac_convention():
    O = H = 512
    fp = precomputed_input_reuse_traffic(O, H, weight_bits=32, reuse=1)
    tq = precomputed_input_reuse_traffic(O, H, weight_bits=2, reuse=1)
    assert fp.operations == 2 * fp.macs
    assert tq.operations == 2 * tq.macs
    assert 0.45 < arithmetic_intensity(O, H, 32, 1) < 0.55
    assert 7.0 < arithmetic_intensity(O, H, 2, 1) < 8.1
    # Approximately the bit-width ratio once activations/requant metadata are included.
    assert arithmetic_intensity(O, H, 2, 1) / arithmetic_intensity(O, H, 32, 1) > 14


def test_precomputed_k_input_model_includes_activation_growth():
    O = H = 512
    t1 = precomputed_input_reuse_traffic(O, H, weight_bits=2, reuse=1)
    t40 = precomputed_input_reuse_traffic(O, H, weight_bits=2, reuse=40)
    assert t40.input_activation_bytes == 40 * t1.input_activation_bytes
    assert t40.output_activation_bytes == 40 * t1.output_activation_bytes
    ai1 = arithmetic_intensity(O, H, 2, 1)
    ai40 = arithmetic_intensity(O, H, 2, 40)
    assert ai40 > ai1
    assert ai40 / ai1 < 40  # activations/output traffic prevents fictitious exact linear AI


def test_classify_is_explicit_model_not_hardware_cache_evidence():
    point = classify(512, 512, peak_gops=40.0, bandwidth_gbs=10.0,
                     weight_bits=2, reuse=1)
    assert point.bound in {"memory", "compute"}
    assert "not_cache_evidence" in point.classification_scope


def test_roofline_report_refuses_cache_and_bandwidth_claims():
    rep = roofline_report(512, 512, 40.0, 10.0, recursion_steps=42)
    assert rep["operation_convention"] == "1_MAC_equals_2_arithmetic_operations"
    assert rep["traffic_scope"] == "external_first_touch_logical_bytes_not_measured_dram"
    assert "precomputed_k_input_ai" in rep
    assert rep["cache_residency_established"] is False
    assert rep["bandwidth_bottleneck_established"] is False
