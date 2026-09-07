#!/usr/bin/env python3
"""Milestone 11 retained experiment: real adaptive execution and overhead."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.seed import set_seed
from data.datasets import build_sudoku_arrays
from deploy.m10_artifact import export_cpu_artifact, load_cpu_artifact, module_tensor_state_sha256
from deploy.m10_runtime import CPURecursiveRuntime
from deploy.m11_adaptive_runtime import AdaptiveCPURecursiveRuntime
from model.halting import DEVICE_DIM, HaltingPolicy, run_with_halting
from model.lazy_router import ConfidenceRouter
from model.trm import TRM
from model.verifier import sudoku_score
from train.losses import deep_supervision_loss

SEED = 20260911
TRAIN_STEPS = 120
BATCH = 64
TIMING_REPS = 12


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def constant_policy(dim: int, halt: bool) -> HaltingPolicy:
    p = HaltingPolicy(dim).eval()
    with torch.no_grad():
        for q in p.parameters(): q.zero_()
        p.net[-1].bias.fill_(20.0 if halt else -20.0)
    return p


def build_model() -> TRM:
    return TRM(
        dim=32, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1, N_sup=4,
        heads=4, max_grid_size=8, ternary=True, act8=True,
    )


def train_model(model: TRM, tx: torch.Tensor, ty: torch.Tensor) -> list[dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    curve = []
    model.train()
    gen = torch.Generator().manual_seed(SEED + 1)
    for step in range(1, TRAIN_STEPS + 1):
        idx = torch.randint(0, tx.shape[0], (BATCH,), generator=gen)
        _, rows = model(tx[idx], height=4, width=4)
        loss = deep_supervision_loss(rows, ty[idx])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in {1, 20, 40, 60, 80, 100, 120}:
            curve.append({"step": step, "loss": float(loss.detach())})
    model.eval()
    return curve


def mean_score(x: torch.Tensor, answer: torch.Tensor) -> float:
    return float(sudoku_score(x, answer, box=2).mean())


def timed(fn, reps: int = TIMING_REPS):
    for _ in range(3): fn()
    vals = []
    last = None
    for _ in range(reps):
        t0 = time.perf_counter(); last = fn(); vals.append(time.perf_counter() - t0)
    return {
        "mean_seconds": float(np.mean(vals)),
        "median_seconds": float(np.median(vals)),
        "min_seconds": float(np.min(vals)),
        "max_seconds": float(np.max(vals)),
        "repetitions": reps,
    }, last


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/m11_adaptive_execution")
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    set_seed(SEED, deterministic=True)

    rng = np.random.default_rng(SEED)
    tx, ty, h, w = build_sudoku_arrays(2, 512, 8, rng, require_unique=True, augment=True)
    vx, vy, _, _ = build_sudoku_arrays(2, 64, 8, rng, require_unique=True, augment=True)
    tx = torch.from_numpy(tx).long(); ty = torch.from_numpy(ty).long()
    vx = torch.from_numpy(vx).long(); vy = torch.from_numpy(vy).long()

    model = build_model()
    before_train = module_tensor_state_sha256(model)
    train_t0 = time.perf_counter(); curve = train_model(model, tx, ty); train_seconds = time.perf_counter() - train_t0
    after_train = module_tensor_state_sha256(model)

    x = vx[:1]
    d = torch.zeros(1, DEVICE_DIM)
    halt_first = constant_policy(model.dim, True)
    continue_policy = constant_policy(model.dim, False)
    half_router = ConfidenceRouter(threshold=0.0, warmup_steps=0, min_active_frac=0.5)
    frozen_router = ConfidenceRouter(threshold=0.0, warmup_steps=0, min_active_frac=0.0)

    # Dense reference and full-density incremental path.
    dense_timing, dense_output = timed(lambda: model(x, height=4, width=4))

    def full_state_run():
        s = model.init_execution_state(x, 4, 4); row = None
        for _ in range(model.N_sup): row = model.run_execution_step(s)
        return s, row
    state_timing, full_state = timed(full_state_run)
    full_state_obj, full_state_row = full_state
    full_density_equivalent = bool(torch.equal(full_state_row["logits"], dense_output[0]))

    # Actual policy halt at step 0 and actual full-depth continue policy.
    early_timing, early = timed(lambda: run_with_halting(
        model, x, halt_first, d, 4, 4, return_details=True
    ))
    full_halt_timing, full_halt = timed(lambda: run_with_halting(
        model, x, continue_policy, d, 4, 4, return_details=True
    ))

    # Partial active execution using the existing confidence router.  The chosen
    # threshold forces the min-active set to exactly half of the 16 tokens.
    adaptive_timing, adaptive = timed(lambda: run_with_halting(
        model, x, continue_policy, d, 4, 4, router=half_router,
        reactivation_policy="allow", return_details=True
    ))
    frozen_timing, frozen = timed(lambda: run_with_halting(
        model, x, continue_policy, d, 4, 4, router=frozen_router,
        reactivation_policy="sticky", max_steps=1, return_details=True
    ))

    # Independent score comparison on held-out examples.  This is descriptive;
    # M11 is an execution gate and does not tune thresholds on these examples.
    score_rows = []
    for i in range(32):
        xi = vx[i:i+1]
        dense_answer = model(xi, 4, 4)[0].argmax(dim=-1)
        e = run_with_halting(model, xi, halt_first, d, 4, 4, return_details=True)
        a = run_with_halting(
            model, xi, continue_policy, d, 4, 4, router=half_router,
            reactivation_policy="allow", return_details=True
        )
        score_rows.append({
            "index": i,
            "dense_score": mean_score(xi, dense_answer),
            "early_score": mean_score(xi, e.answer),
            "sparse_score": mean_score(xi, a.answer),
            "early_same_answer": bool(torch.equal(dense_answer, e.answer)),
            "sparse_same_answer": bool(torch.equal(dense_answer, a.answer)),
        })
    write_json(out / "score_rows.json", score_rows)

    # Export the same hard-ternary model and exercise M10's verified native
    # primitive through M11 compaction.
    artifact_path = out / "m11_cpu_artifact.pt"
    export_cpu_artifact(
        model, artifact_path, height=4, width=4, box=2,
        source_checkpoint_sha256="0" * 64,
        source_checkpoint_tensor_sha256=after_train,
        training_seed=SEED, training_step=TRAIN_STEPS,
        data_provenance={"task": "generated_4x4_sudoku", "seed": SEED},
        export_git_sha=os.environ.get("GITHUB_SHA", "local"),
    )
    loaded = load_cpu_artifact(artifact_path)
    base_native = CPURecursiveRuntime(loaded)
    # Warm/compile outside measured native timings.
    base_native.forward(x)
    native_dense_timing, native_dense = timed(lambda: CPURecursiveRuntime(loaded).forward(x), reps=6)

    def native_full_incremental():
        r = AdaptiveCPURecursiveRuntime(loaded)
        s = r.init_execution_state(x); row = None
        for _ in range(int(r.arch["N_sup"])): row = r.run_execution_step(s)
        return r, s, row
    native_inc_timing, native_inc = timed(native_full_incremental, reps=6)
    native_inc_runtime, native_inc_state, native_inc_row = native_inc

    fixed_half = torch.zeros(1, 16, 1); fixed_half[:, ::2] = 1
    def native_sparse():
        r = AdaptiveCPURecursiveRuntime(loaded)
        s = r.init_execution_state(x); row = None
        for _ in range(int(r.arch["N_sup"])):
            row = r.run_execution_step(s, active_mask=fixed_half, reactivation_policy="allow")
        return r, s, row
    native_sparse_timing, native_sparse_out = timed(native_sparse, reps=6)
    native_sparse_runtime, native_sparse_state, native_sparse_row = native_sparse_out

    # Corresponding PyTorch reference with the identical fixed mask.
    ref_sparse_state = model.init_execution_state(x, 4, 4); ref_sparse_row = None
    for _ in range(model.N_sup):
        ref_sparse_row = model.run_execution_step(ref_sparse_state, active_mask=fixed_half)
    assert ref_sparse_row is not None

    native_full_equiv = bool(torch.allclose(native_inc_row["logits"], native_dense.logits, atol=1e-6, rtol=0))
    native_reference_sparse_equiv = bool(
        torch.allclose(native_sparse_row["logits"], ref_sparse_row["logits"], atol=1e-3, rtol=0)
    )

    dense_scores = np.asarray([r["dense_score"] for r in score_rows])
    early_scores = np.asarray([r["early_score"] for r in score_rows])
    sparse_scores = np.asarray([r["sparse_score"] for r in score_rows])

    early_fewer = bool(
        early.executed_steps < full_halt.executed_steps
        and early.work["block_applications"] < full_halt.work["block_applications"]
        and early.work["attention_q_vectors"] < full_halt.work["attention_q_vectors"]
    )
    sparse_skips = bool(
        adaptive.work["attention_q_vectors"] < full_halt.work["attention_q_vectors"]
        and adaptive.work["ffn_input_vectors"] < full_halt.work["ffn_input_vectors"]
        and adaptive.work["attention_k_vectors"] >= adaptive.work["attention_q_vectors"]
    )
    native_saved = bool(
        native_sparse_runtime.adaptive_work_record()["native_input_vectors"]
        < native_inc_runtime.adaptive_work_record()["native_input_vectors"]
        and native_sparse_runtime.adaptive_work_record()["native_scalar_products"]
        < native_inc_runtime.adaptive_work_record()["native_scalar_products"]
    )
    overhead_measured = bool(
        adaptive.timing["router_seconds"] >= 0
        and adaptive.timing["halter_seconds"] >= 0
        and adaptive.work["router_calls"] > 0
        and adaptive.work["halter_calls"] > 0
    )

    summary = {
        "status": "complete" if all([
            full_density_equivalent, early_fewer, sparse_skips, native_full_equiv,
            native_reference_sparse_equiv, native_saved, overhead_measured,
        ]) else "failed",
        "milestone": 11,
        "scope": "real_adaptive_execution_batch1",
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
        "training": {
            "seconds": train_seconds,
            "steps": TRAIN_STEPS,
            "curve": curve,
            "initial_tensor_hash": before_train,
            "trained_tensor_hash": after_train,
        },
        "acceptance": {
            "full_density_equivalence": full_density_equivalent,
            "earlier_halting_fewer_steps_and_work": early_fewer,
            "sparse_execution_skips_declared_work": sparse_skips,
            "native_full_density_equivalence": native_full_equiv,
            "native_sparse_reference_equivalence": native_reference_sparse_equiv,
            "native_sparse_reduces_real_work": native_saved,
            "router_and_halter_overhead_measured": overhead_measured,
        },
        "early_exit": {
            "stop_reason": early.stop_reason,
            "executed_steps": early.executed_steps,
            "full_executed_steps": full_halt.executed_steps,
            "work": early.work,
            "full_work": full_halt.work,
            "timing_internal": early.timing,
        },
        "adaptive_router_halter": {
            "work": adaptive.work,
            "timing_internal": adaptive.timing,
            "active_densities": [r["active_density"] for r in adaptive.step_records],
        },
        "all_frozen": {
            "work": frozen.work,
            "timing_internal": frozen.timing,
            "active_densities": [r["active_density"] for r in frozen.step_records],
        },
        "wall_timing": {
            "dense_direct": dense_timing,
            "full_density_state_interface": state_timing,
            "policy_halt_first": early_timing,
            "full_continue_with_halter": full_halt_timing,
            "half_active_router": adaptive_timing,
            "all_frozen_router": frozen_timing,
            "native_dense": native_dense_timing,
            "native_full_incremental": native_inc_timing,
            "native_half_active": native_sparse_timing,
        },
        "native_work": {
            "full": native_inc_runtime.adaptive_work_record(),
            "half_active": native_sparse_runtime.adaptive_work_record(),
        },
        "heldout_behavior": {
            "examples": len(score_rows),
            "dense_mean_structural_score": float(dense_scores.mean()),
            "early_mean_structural_score": float(early_scores.mean()),
            "sparse_mean_structural_score": float(sparse_scores.mean()),
            "early_minus_dense": float((early_scores - dense_scores).mean()),
            "sparse_minus_dense": float((sparse_scores - dense_scores).mean()),
            "early_answer_agreement": float(np.mean([r["early_same_answer"] for r in score_rows])),
            "sparse_answer_agreement": float(np.mean([r["sparse_same_answer"] for r in score_rows])),
        },
        "interpretation": {
            "whole_model_linear_density_claim": False,
            "attention_kv_remain_dense": True,
            "batch_compaction_claim": False,
            "timing_is_descriptive": True,
            "adaptive_can_be_slower_due_to_control_gather_scatter": True,
            "accuracy_can_drop_under_early_or_sparse_execution": True,
        },
    }
    write_json(out / "summary.json", summary)
    write_json(out / "measurement.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
