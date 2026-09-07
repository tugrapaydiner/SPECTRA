from __future__ import annotations

from pathlib import Path

import torch

from deploy.m10_artifact import export_cpu_artifact, load_cpu_artifact
from deploy.m10_runtime import CPURecursiveRuntime
from deploy.m11_adaptive_runtime import AdaptiveCPURecursiveRuntime
from model.halting import DEVICE_DIM, HaltingPolicy, run_with_halting
from model.trm import TRM


def _model(*, n_sup=4, ternary=False, act8=False):
    torch.manual_seed(123)
    return TRM(
        dim=16,
        num_tokens=5,
        seq_len=16,
        n_layers=1,
        n=1,
        T=1,
        N_sup=n_sup,
        heads=2,
        max_grid_size=8,
        ternary=ternary,
        act8=act8,
    ).eval()


def _legacy_truncated(model: TRM, x: torch.Tensor, steps: int):
    x_emb = model.token_embed(x) + model.encode_positions(x, 4, 4)
    y = torch.zeros_like(x_emb)
    z = torch.zeros_like(x_emb)
    rows = []
    for _ in range(steps):
        for _ in range(model.T):
            y, z = model.recursive_cycle(x_emb, y, z)
        logits = model.out_head(y)
        halt = model.halt_head(y.mean(dim=1)).squeeze(-1)
        rows.append((logits, halt, y, z))
        y = y.detach(); z = z.detach()
    return rows[-1]


def _constant_policy(dim: int, halt: bool) -> HaltingPolicy:
    p = HaltingPolicy(dim).eval()
    with torch.no_grad():
        for q in p.parameters():
            q.zero_()
        p.net[-1].bias.fill_(20.0 if halt else -20.0)
    return p


def test_full_density_step_interface_matches_independent_dense_reference():
    model = _model(n_sup=3, ternary=True)
    x = torch.randint(0, 5, (1, 16))
    legacy = _legacy_truncated(model, x, 3)
    state = model.init_execution_state(x, 4, 4)
    out = None
    for _ in range(3):
        out = model.run_execution_step(state)
        state.y = state.y.detach(); state.z = state.z.detach()
    assert out is not None
    assert torch.equal(out["logits"], legacy[0])
    assert torch.equal(out["halt_logit"], legacy[1])
    assert torch.equal(state.y, legacy[2])
    assert torch.equal(state.z, legacy[3])
    assert state.work["supervision_steps"] == 3


def test_online_policy_halt_executes_fewer_real_steps_and_matches_truncation():
    model = _model(n_sup=4, ternary=True)
    x = torch.randint(0, 5, (1, 16))
    d = torch.zeros(1, DEVICE_DIM)
    early = run_with_halting(
        model, x, _constant_policy(model.dim, True), d, 4, 4, return_details=True
    )
    full = run_with_halting(
        model, x, _constant_policy(model.dim, False), d, 4, 4, return_details=True
    )
    legacy1 = _legacy_truncated(model, x, 1)
    assert early.stop_reason == "policy_halt"
    assert early.executed_steps == 1
    assert early.halt_step.item() == 0
    assert full.stop_reason == "model_exhausted"
    assert full.executed_steps == 4
    assert full.halt_step.item() == 3
    assert early.work["block_applications"] < full.work["block_applications"]
    assert early.work["attention_q_vectors"] < full.work["attention_q_vectors"]
    assert torch.equal(early.logits, legacy1[0])
    assert torch.equal(early.y, legacy1[2])
    assert torch.equal(early.z, legacy1[3])


def test_forced_budget_stop_is_observable_and_truncated():
    model = _model(n_sup=4)
    x = torch.randint(0, 5, (1, 16))
    d = torch.zeros(1, DEVICE_DIM)
    result = run_with_halting(
        model, x, None, d, 4, 4, max_steps=2, return_details=True
    )
    legacy2 = _legacy_truncated(model, x, 2)
    assert result.stop_reason == "budget_exhausted"
    assert result.executed_steps == 2
    assert result.halt_step.item() == 1
    # The FP MultiheadAttention path may choose numerically different but
    # equivalent fused kernels under inference/no-grad.  The M11 contract is a
    # fixed absolute tolerance, not bit identity, for this FP reference case.
    assert torch.allclose(result.logits, legacy2[0], atol=2e-6, rtol=0)


def test_partial_active_path_matches_dense_masked_reference_and_saves_declared_rows():
    model = _model(n_sup=1, ternary=True)
    x = torch.randint(0, 5, (1, 16))
    mask = torch.zeros(1, 16, 1)
    mask[:, ::2] = 1
    a = int(mask.sum())

    # Historical reference: compute the complete block then freeze masked states.
    x_emb = model.token_embed(x) + model.encode_positions(x, 4, 4)
    y0 = torch.zeros_like(x_emb); z0 = torch.zeros_like(x_emb)
    y_ref, z_ref = model.recursive_cycle(x_emb, y0, z0, mask)
    logits_ref = model.out_head(y_ref)

    state = model.init_execution_state(x, 4, 4)
    out = model.run_execution_step(state, active_mask=mask)
    assert torch.allclose(state.y, y_ref, atol=2e-6, rtol=0)
    assert torch.allclose(state.z, z_ref, atol=2e-6, rtol=0)
    assert torch.allclose(out["logits"], logits_ref, atol=2e-6, rtol=0)

    blocks = (model.n + 1) * model.T
    assert state.work["attention_q_vectors"] == a * blocks
    assert state.work["attention_output_vectors"] == a * blocks
    assert state.work["ffn_input_vectors"] == a * blocks
    assert state.work["attention_k_vectors"] == 16 * blocks
    assert state.work["attention_v_vectors"] == 16 * blocks
    assert state.work["attention_q_vectors"] < state.work["attention_k_vectors"]


def test_all_frozen_skips_recursive_core_and_preserves_state_exactly():
    model = _model(n_sup=2, ternary=True, act8=True)
    x = torch.randint(0, 5, (1, 16))
    state = model.init_execution_state(x, 4, 4)
    model.run_execution_step(state)  # make recurrent state non-trivial
    y = state.y.clone(); z = state.z.clone()
    before_blocks = state.work["block_applications"]
    out = model.run_execution_step(state, active_mask=torch.zeros(1, 16, 1))
    assert torch.equal(state.y, y)
    assert torch.equal(state.z, z)
    assert out["active_tokens"] == 0
    assert out["work_delta"]["all_frozen_step_skips"] == 1
    assert out["work_delta"]["recursive_cycles"] == 0
    assert state.work["block_applications"] == before_blocks


def test_reactivation_policy_allow_vs_sticky_is_explicit():
    x = torch.randint(0, 5, (1, 16))
    first = torch.zeros(1, 16, 1); first[:, :8] = 1
    opposite = 1 - first

    allow_model = _model(n_sup=2, ternary=True)
    allow = allow_model.init_execution_state(x, 4, 4)
    allow_model.run_execution_step(allow, active_mask=first, reactivation_policy="allow")
    before = allow.y.clone()
    out_allow = allow_model.run_execution_step(allow, active_mask=opposite, reactivation_policy="allow")
    assert out_allow["active_tokens"] == 8
    assert not torch.equal(allow.y[:, 8:], before[:, 8:])

    sticky_model = _model(n_sup=2, ternary=True)
    sticky = sticky_model.init_execution_state(x, 4, 4)
    sticky_model.run_execution_step(sticky, active_mask=first, reactivation_policy="sticky")
    y_before = sticky.y.clone(); z_before = sticky.z.clone()
    out_sticky = sticky_model.run_execution_step(sticky, active_mask=opposite, reactivation_policy="sticky")
    assert out_sticky["active_tokens"] == 0
    assert torch.equal(sticky.y, y_before)
    assert torch.equal(sticky.z, z_before)


def _export_tiny_m10(model: TRM, path: Path):
    export_cpu_artifact(
        model,
        path,
        height=4,
        width=4,
        box=2,
        source_checkpoint_sha256="0" * 64,
        source_checkpoint_tensor_sha256="1" * 64,
        training_seed=123,
        training_step=0,
        data_provenance={"purpose": "m11_contract"},
        export_git_sha="m11-test",
    )
    return load_cpu_artifact(path)


def test_native_full_density_partial_sparse_and_empty_active_agree_with_reference(tmp_path: Path):
    model = _model(n_sup=2, ternary=True, act8=True)
    x = torch.randint(0, 5, (1, 16), dtype=torch.long)
    loaded = _export_tiny_m10(model, tmp_path / "tiny_cpu.pt")

    dense_native = CPURecursiveRuntime(loaded).forward(x)
    adaptive_native_runtime = AdaptiveCPURecursiveRuntime(loaded)
    incremental = adaptive_native_runtime.forward_incremental(x)
    assert torch.allclose(incremental["logits"], dense_native.logits, atol=1e-6, rtol=0)
    assert torch.equal(incremental["answer"], dense_native.answer)

    mask = torch.zeros(1, 16, 1); mask[:, ::2] = 1
    ref_state = model.init_execution_state(x, 4, 4)
    ref = model.run_execution_step(ref_state, active_mask=mask)

    native = AdaptiveCPURecursiveRuntime(loaded)
    native_state = native.init_execution_state(x)
    nout = native.run_execution_step(native_state, active_mask=mask)
    assert torch.allclose(nout["logits"], ref["logits"], atol=1e-3, rtol=0)
    assert torch.allclose(native_state.y, ref_state.y, atol=5e-4, rtol=0)
    assert torch.allclose(native_state.z, ref_state.z, atol=5e-4, rtol=0)
    assert nout["work_delta"]["attention_q_vectors"] < nout["work_delta"]["attention_k_vectors"]

    full = AdaptiveCPURecursiveRuntime(loaded)
    fs = full.init_execution_state(x)
    fout = full.run_execution_step(fs)
    assert nout["work_delta"]["native_input_vectors"] < fout["work_delta"]["native_input_vectors"]
    assert nout["work_delta"]["native_scalar_products"] < fout["work_delta"]["native_scalar_products"]

    empty = AdaptiveCPURecursiveRuntime(loaded)
    es = empty.init_execution_state(x)
    y0 = es.y.clone(); z0 = es.z.clone()
    eout = empty.run_execution_step(es, active_mask=torch.zeros(1, 16, 1))
    assert torch.equal(es.y, y0)
    assert torch.equal(es.z, z0)
    assert eout["active_tokens"] == 0
    assert eout["work_delta"]["recursive_cycles"] == 0
    assert eout["work_delta"]["all_frozen_step_skips"] == 1
    assert eout["work_delta"]["attention_q_vectors"] == 0
    assert eout["work_delta"]["attention_k_vectors"] == 0
    assert eout["work_delta"]["attention_v_vectors"] == 0
    assert eout["work_delta"]["ffn_input_vectors"] == 0
    # The step boundary still returns logits through the native out_head; no
    # recursive q/k/v/ffn work is executed.
    assert eout["work_delta"]["native_linear_calls"] == 1
    assert eout["work_delta"]["native_input_vectors"] == 16
