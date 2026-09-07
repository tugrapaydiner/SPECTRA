from __future__ import annotations

from pathlib import Path

import pytest
import torch

from deploy.m10_artifact import (
    DeploymentArtifactError,
    export_cpu_artifact,
    load_cpu_artifact,
)
from deploy.m10_runtime import CPURecursiveRuntime
from deploy import m10_native
from model.bitlinear import FakeBitLinear
from model.stability import quant_strength_state
from model.trm import TRM


def make_model(*, ternary=True, act8=True) -> TRM:
    torch.manual_seed(101)
    model = TRM(
        dim=16, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1, N_sup=2,
        heads=4, alpha_y=0.1, alpha_z=0.1, max_grid_size=8,
        ternary=ternary, act8=act8,
    ).cpu().eval()
    return model


def export(model: TRM, path: Path):
    return export_cpu_artifact(
        model, path, height=4, width=4, box=2,
        source_checkpoint_sha256="checkpoint-sha",
        source_checkpoint_tensor_sha256="checkpoint-tensor-sha",
        training_seed=101, training_step=1,
        data_provenance={"test": "unused"}, export_git_sha="git-sha",
    )


def hard_weight(module: FakeBitLinear) -> torch.Tensor:
    with torch.no_grad():
        q, scale = module._ternarize_hard(module.weight)
        return q * scale


def test_export_rejects_nonternary_and_non_a8(tmp_path: Path):
    with pytest.raises(DeploymentArtifactError, match="ternary=true"):
        export(make_model(ternary=False, act8=False), tmp_path / "dense.pt")
    with pytest.raises(DeploymentArtifactError, match="FakeActQuant A8"):
        export(make_model(ternary=True, act8=False), tmp_path / "noa8.pt")


def test_export_rejects_soft_ternary_state(tmp_path: Path):
    model = make_model()
    first = next(m for m in model.modules() if isinstance(m, FakeBitLinear))
    first.quant_strength.fill_(0.5)
    with pytest.raises(DeploymentArtifactError, match="soft-ternary export rejected"):
        export(model, tmp_path / "soft.pt")


def test_artifact_is_self_contained_and_packed_reconstruction_exact(tmp_path: Path):
    model = make_model()
    assert all(v == 1.0 for v in quant_strength_state(model).values())
    path = tmp_path / "artifact.pt"
    report = export(model, path)
    loaded = load_cpu_artifact(path)
    assert report["packed_linear_count"] == 7
    assert report["fp_tensor_count"] == 11
    assert loaded.payload["architecture"]["gelu_approximate"] == "none"
    assert loaded.payload["precision"]["native_linear_accumulator"] == "float32_scalar_ordered"
    assert "model_state" not in loaded.payload
    assert all(entry["reconstruction_exact"] is True for entry in loaded.payload["packed_linears"].values())


def test_loader_rejects_reserved_packed_code_and_hash_corruption(tmp_path: Path):
    model = make_model(); path = tmp_path / "artifact.pt"; export(model, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["packed_linears"]["blocks.0.attn.q"]["packed"][0] |= 0b11
    bad = tmp_path / "bad.pt"; torch.save(payload, bad)
    with pytest.raises(DeploymentArtifactError, match="reserved code|hash mismatch"):
        load_cpu_artifact(bad)


def test_native_dense_linear_matches_hard_fakebitlinear(tmp_path: Path):
    model = make_model(); path = tmp_path / "artifact.pt"; export(model, path)
    loaded = load_cpu_artifact(path)
    entry = loaded.payload["packed_linears"]["blocks.0.ff.0"]
    module = model.blocks[0].ff[0]
    torch.manual_seed(202)
    x = torch.randn(37, module.in_features, dtype=torch.float32).contiguous()
    ref = torch.nn.functional.linear(x, hard_weight(module), module.bias)
    got = m10_native.dense_ternary_linear_fp32(
        x, entry["packed"], entry["scale"], entry["bias"], entry["out_features"]
    )
    err = (ref - got).abs()
    assert float(err.max()) <= 5e-5
    assert float(err.mean()) <= 5e-6
    assert m10_native.operator_identity() == "packed_ternary_fp32_scalar_linear_v1"


def test_runtime_block_cycle_and_full_forward_match_reference(tmp_path: Path):
    model = make_model(); path = tmp_path / "artifact.pt"; export(model, path)
    runtime = CPURecursiveRuntime(load_cpu_artifact(path))
    torch.manual_seed(303)
    x = torch.randint(0, 5, (2, 16), dtype=torch.long)
    x_emb = model.token_embed(x) + model.encode_positions(x, 4, 4)
    y = torch.randn_like(x_emb) * 0.1
    z = torch.randn_like(x_emb) * 0.1

    h = torch.randn_like(x_emb)
    ref_block = model.blocks[0](h)
    got_block = runtime.block(h)
    block_err = (ref_block - got_block).abs()
    assert float(block_err.max()) <= 2e-4
    assert float(block_err.mean()) <= 2e-5

    ref_y, ref_z = model.recursive_cycle(x_emb, y, z)
    got_y, got_z = runtime.recursive_cycle(x_emb, y, z)
    for ref, got in ((ref_y, got_y), (ref_z, got_z)):
        err = (ref - got).abs()
        assert float(err.max()) <= 5e-4
        assert float(err.mean()) <= 5e-5

    ref_logits, ref_steps = model(x, height=4, width=4)
    out = runtime.forward(x)
    err = (ref_logits - out.logits).abs()
    assert float(err.max()) <= 1e-3
    assert float(err.mean()) <= 1e-4
    assert torch.equal(ref_logits.argmax(-1), out.answer)
    assert len(ref_steps) == len(out.step_outputs) == 2
    for a, b in zip(ref_steps, out.step_outputs):
        assert torch.equal(a["logits"].argmax(-1), b["logits"].argmax(-1))
    assert out.work["native_call_count_matches_architecture"] is True
    assert out.work["native_linear_calls"] == runtime.expected_native_calls_per_forward()
    assert out.work["a8_quantization_calls"] == 4


def test_runtime_backend_report_never_promotes_bitnet_directory(monkeypatch, tmp_path: Path):
    model = make_model(); path = tmp_path / "artifact.pt"; export(model, path)
    fake_root = tmp_path / "bitnet.cpp"; fake_root.mkdir()
    monkeypatch.setenv("SPECTRA_BITNET_CPP", str(fake_root))
    report = CPURecursiveRuntime(load_cpu_artifact(path)).backend_report()
    assert report["bitnet_cpp_configured_root"] == str(fake_root)
    assert report["bitnet_cpp_supported"] is False
    assert report["vectorized"] is False


def test_runtime_rejects_bad_input_contract(tmp_path: Path):
    model = make_model(); path = tmp_path / "artifact.pt"; export(model, path)
    runtime = CPURecursiveRuntime(load_cpu_artifact(path))
    with pytest.raises(TypeError):
        runtime.forward(torch.zeros(1, 16, dtype=torch.float32))
    with pytest.raises(ValueError):
        runtime.forward(torch.zeros(1, 15, dtype=torch.long))
