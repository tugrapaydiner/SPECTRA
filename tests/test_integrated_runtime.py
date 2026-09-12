"""Exact native/operator/runtime contracts, independent of performance outcomes."""
from pathlib import Path

import pytest
import torch

from deploy import m10_native
from deploy.m10_artifact import export_cpu_artifact, load_cpu_artifact
from deploy.m10_runtime import CPURecursiveRuntime
from deploy.pack_ternary import pack_ternary_rows
from deploy.validated_runtime import ValidatedCPURecursiveRuntime
from model.trm import TRM
from spectra.blocked_runtime import BlockedCPURecursiveRuntime, load_extension
from spectra.inference import predict_final


def assert_bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    if a.is_floating_point():
        assert torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))
    else:
        assert torch.equal(a, b)


def native_fixture(inputs, outputs=7, bias=True):
    rng = torch.Generator().manual_seed(7000 + inputs)
    weights = torch.randint(-1, 2, (outputs, inputs), generator=rng)
    packed = torch.from_numpy(pack_ternary_rows(weights.numpy())[0])
    scale = torch.rand(outputs, generator=rng)
    scale[0] = 0.0  # Zero scales must not turn skipped zero weights into products.
    biases = torch.randn(outputs, generator=rng) if bias else torch.empty(0)
    return packed, scale, biases


@pytest.mark.parametrize("vectors", [1, 2, 3, 4, 5, 7, 8, 9, 17, 64])
@pytest.mark.parametrize("inputs", [1, 3, 4, 5, 16, 63, 64, 65])
@pytest.mark.parametrize("bias", [False, True])
def test_blocked_matches_historical_operator_bits(vectors, inputs, bias):
    packed, scale, biases = native_fixture(inputs, bias=bias)
    rng = torch.Generator().manual_seed(vectors + 7091)
    x = torch.randn(vectors, inputs, generator=rng)
    handle = load_extension().BlockedPackedLinear(packed, scale, biases, 7, inputs)
    expected = m10_native.dense_ternary_linear_fp32(x, packed, scale, biases, 7)
    assert_bits(expected, handle.forward(x))


def test_blocked_native_owns_validated_copies_and_distinct_outputs():
    packed, scale, bias = native_fixture(65)
    x = torch.randn(9, 65)
    handle = load_extension().BlockedPackedLinear(packed, scale, bias, 7, 65)
    expected = handle.forward(x)
    bytes_owned = packed.numel() + 4 * (scale.numel() + bias.numel())
    assert handle.storage_bytes() == bytes_owned
    packed.fill_(255); scale.fill_(float("nan")); bias.fill_(123)
    got = handle.forward(x)
    assert_bits(expected, got)
    assert got.data_ptr() != expected.data_ptr()
    got.fill_(123)
    assert_bits(expected, handle.forward(x))


@pytest.mark.parametrize("kind", ["reserved", "padding", "shape", "scale", "negative_scale", "bias", "dtype", "rank", "overflow"])
def test_blocked_native_rejects_bad_weight_contracts(kind):
    p, s, b = torch.tensor([1, 0], dtype=torch.uint8), torch.ones(1), torch.zeros(1)
    outputs, inputs = 1, 5
    if kind == "reserved": p[0] = 3
    elif kind == "padding": p[1] = 4
    elif kind == "shape": p = p[:1]
    elif kind == "scale": s[0] = float("nan")
    elif kind == "negative_scale": s[0] = -1
    elif kind == "bias": b[0] = float("inf")
    elif kind == "dtype": p = p.long()
    elif kind == "rank": p = p[None]
    elif kind == "overflow": inputs, outputs = 2**62, 2**62
    with pytest.raises(RuntimeError):
        load_extension().BlockedPackedLinear(p, s, b, outputs, inputs)


@pytest.mark.parametrize("kind", ["empty", "shape", "dtype", "rank", "strided"])
def test_blocked_native_rejects_bad_inputs(kind):
    p, s, b = native_fixture(5)
    handle = load_extension().BlockedPackedLinear(p, s, b, 7, 5)
    x = {
        "empty": torch.empty(0, 5), "shape": torch.ones(4, 4),
        "dtype": torch.ones(4, 5, dtype=torch.float64), "rank": torch.ones(5),
        "strided": torch.ones(5, 4).T,
    }[kind]
    with pytest.raises(RuntimeError): handle.forward(x)


@pytest.mark.parametrize("value", [0.0, -0.0, 1e-35, 1e20])
def test_blocked_edge_arithmetic(value):
    p, s, b = native_fixture(65)
    x = torch.full((9, 65), value)
    x[:, ::2] = -x[:, ::2]
    expected = m10_native.dense_ternary_linear_fp32(x, p, s, b, 7)
    actual = load_extension().BlockedPackedLinear(p, s, b, 7, 65).forward(x)
    assert_bits(expected, actual)


def artifact(path: Path, *, depth=4, n=1, T=1):
    torch.manual_seed(71001)
    model = TRM(dim=16, num_tokens=5, seq_len=16, n_layers=1, n=n, T=T,
                N_sup=depth, heads=4, max_grid_size=8, ternary=True, act8=True).eval()
    export_cpu_artifact(model, path, height=4, width=4, box=2,
        source_checkpoint_sha256="unit-fixture", source_checkpoint_tensor_sha256="unit-fixture",
        training_seed=71001, training_step=0, data_provenance={"unit_fixture": True},
        export_git_sha="unit-fixture")
    return load_cpu_artifact(path)


@pytest.mark.parametrize("cls", [ValidatedCPURecursiveRuntime, BlockedCPURecursiveRuntime])
@pytest.mark.parametrize("depth,n,T", [(1, 1, 1), (4, 1, 1), (16, 1, 1), (3, 2, 1), (2, 1, 3), (2, 2, 2)])
@pytest.mark.parametrize("batch", [1, 3])
def test_supported_runtime_composition_is_exact(tmp_path, cls, depth, n, T, batch):
    a = artifact(tmp_path / "model.pt", depth=depth, n=n, T=T)
    reference, candidate = CPURecursiveRuntime(a), cls(a)
    x = torch.randint(0, 5, (batch, 16))
    historical = reference.forward(x)
    before = candidate.forward(x)
    assert before.work == historical.work
    for original, current in zip(historical.step_outputs, before.step_outputs, strict=True):
        for key in original: assert_bits(original[key], current[key])
    final = predict_final(candidate, x)
    assert_bits(historical.logits, final.logits)
    assert_bits(historical.answer, final.answer)
    assert_bits(historical.step_outputs[-1]["halt_logit"], final.halt_logit)
    assert final.work["retained_recurrent_steps"] == 0
    assert final.work["native_linear_calls"] == before.work["native_linear_calls"] - depth + 1
    assert final.work["native_call_count_matches_architecture"]
    assert final.work["halt_head_fp32_calls"] == final.work["decode_calls"] == 1
    assert final.work["a8_quantization_calls"] == before.work["a8_quantization_calls"]
    assert final.work["a8_scale_mean_min"] == before.work["a8_scale_mean_min"]
    assert final.work["a8_scale_mean_max"] == before.work["a8_scale_mean_max"]
    # Another input and another runtime cannot contaminate this one's answer/work.
    predict_final(cls(a), torch.flip(x, [1]))
    predict_final(candidate, torch.flip(x, [1]))
    after = candidate.forward(x)
    assert_bits(before.logits, after.logits)
    assert before.work == after.work
    assert_bits(final.logits, after.logits)  # old output still owns its storage


@pytest.mark.parametrize("cls", [ValidatedCPURecursiveRuntime, BlockedCPURecursiveRuntime])
def test_custom_subclasses_still_rejected(tmp_path, cls):
    class Custom(cls):
        pass
    with pytest.raises(TypeError, match="custom subclasses"):
        predict_final(Custom(artifact(tmp_path / "model.pt")), torch.ones(1, 16, dtype=torch.long))


@pytest.mark.parametrize("cls", [ValidatedCPURecursiveRuntime, BlockedCPURecursiveRuntime])
def test_integrated_runtime_preserves_input_validation(tmp_path, cls):
    runtime = cls(artifact(tmp_path / "model.pt"))
    for x in [None, torch.ones(1, 16)]:
        with pytest.raises(TypeError): predict_final(runtime, x)
    for x in [torch.ones(1, 15, dtype=torch.long), torch.full((1, 16), 5), torch.full((1, 16), -1)]:
        with pytest.raises(ValueError): predict_final(runtime, x)
    with pytest.raises(RuntimeError): predict_final(runtime, torch.empty(0, 16, dtype=torch.long))
