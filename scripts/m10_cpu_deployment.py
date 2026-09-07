#!/usr/bin/env python3
"""Milestone 10: train one W1.58A8 TRM and validate faithful CPU deployment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config
from common.seed import set_seed
from deploy.m10_artifact import (
    export_cpu_artifact,
    load_cpu_artifact,
    module_tensor_state_sha256,
    sha256_file,
)
from deploy.m10_runtime import CPURecursiveRuntime
from deploy import m10_native
from eval.metrics import task_metrics
from model.bitlinear import FakeBitLinear
from model.stability import QuantWarmup, load_quant_strength_state, quant_strength_state
from model.trm import TRM
from model.verifier import sudoku_correct
from scripts._common import build_data_splits
from train.losses import deep_supervision_loss

DATA_SEED = 20260910
MODEL_SEED = 10101
TRAIN_N, VAL_N, TEST_N = 384, 96, 128
STEPS = 200
BATCH = 32
LR = 1e-3
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
QUANT_WARMUP = 50
FIDELITY_N = 16
WARM_REPEATS = 5

LINEAR_MAX = 5e-5
LINEAR_MEAN = 5e-6
BLOCK_MAX = 2e-4
BLOCK_MEAN = 2e-5
CYCLE_MAX = 5e-4
CYCLE_MEAN = 5e-5
FULL_MAX = 1e-3
FULL_MEAN = 1e-4


def plain(v: Any) -> Any:
    if isinstance(v, dict): return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [plain(x) for x in v]
    if isinstance(v, np.generic): return v.item()
    if torch.is_tensor(v):
        return v.detach().cpu().item() if v.ndim == 0 else v.detach().cpu().tolist()
    if isinstance(v, Path): return str(v)
    if isinstance(v, float) and not math.isfinite(v): return None
    return v


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(plain(payload), sort_keys=True) + "\n")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def id_hash(ids: Iterable[str]) -> str:
    h = hashlib.sha256()
    for value in ids:
        b = str(value).encode("utf-8")
        h.update(len(b).to_bytes(8, "little")); h.update(b)
    return h.hexdigest()


def environment() -> dict[str, Any]:
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    return {
        "git_sha": git_sha(), "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "cpu_model": cpu, "logical_cpus": os.cpu_count(),
        "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(), "torch_num_threads": torch.get_num_threads(),
        "device": "cpu",
    }


def make_cfg():
    return load_config("config/sudoku.yaml", overrides=[
        f"seed={DATA_SEED}", "device=cpu",
        "data.min_clues=30", "data.max_clues=35", "data.augment=true",
        "train.precision=fp32", "train.backend=pytorch_eager", "train.deterministic=true",
    ])


def make_model() -> TRM:
    set_seed(MODEL_SEED, deterministic=True)
    return TRM(
        dim=48, num_tokens=10, seq_len=81, n_layers=1, n=1, T=1, N_sup=2,
        heads=4, alpha_y=0.1, alpha_z=0.1, max_grid_size=16,
        ternary=True, act8=True,
    ).cpu()


def tensors(ds):
    return torch.from_numpy(ds.inputs).long(), torch.from_numpy(ds.targets).long()


def save_source_checkpoint(path: Path, model: TRM, opt, *, step: int) -> dict[str, Any]:
    payload = {
        "format": "spectra.m10_trained_source", "version": 1,
        "model_seed": MODEL_SEED, "training_step": int(step),
        "architecture": {
            "dim": 48, "num_tokens": 10, "seq_len": 81, "n_layers": 1,
            "heads": 4, "n": 1, "T": 1, "N_sup": 2, "max_grid_size": 16,
            "ternary": True, "act8": True,
        },
        "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
        "quant_strength": quant_strength_state(model), "git_sha": git_sha(),
    }
    torch.save(payload, path)
    return {"path": str(path), "sha256": sha256_file(path), "tensor_sha256": module_tensor_state_sha256(model)}


def reload_source_checkpoint(path: Path) -> TRM:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "spectra.m10_trained_source" or payload.get("version") != 1:
        raise RuntimeError("invalid M10 source checkpoint")
    model = make_model()
    model.load_state_dict(payload["model_state"], strict=True)
    load_quant_strength_state(model, payload["quant_strength"])
    state = quant_strength_state(model)
    if not state or any(float(v) != 1.0 for v in state.values()):
        raise RuntimeError(f"source reload is not hard ternary: {state}")
    return model.eval()


def train(out: Path, train_ds, val_ds) -> tuple[TRM, dict[str, Any]]:
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warm = QuantWarmup(QUANT_WARMUP)
    x, y = tensors(train_ds)
    vx, vy = tensors(val_ds)
    rng = np.random.default_rng(MODEL_SEED + 17)
    curve = out / "training_curve.jsonl"
    if curve.exists(): curve.unlink()
    first_loss = last_loss = None
    max_grad = 0.0
    t0 = time.perf_counter()
    model.train()
    for step in range(1, STEPS + 1):
        warm.apply(model, step)
        idx = torch.from_numpy(rng.integers(0, len(train_ds), size=BATCH, dtype=np.int64))
        _, outputs = model(x[idx], height=9, width=9)
        loss = deep_supervision_loss(outputs, y[idx], lambda_h=0.5, lambda_improve=0.1, margin=0.01)
        if not torch.isfinite(loss): raise RuntimeError(f"nonfinite train loss at {step}")
        opt.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        if not torch.isfinite(torch.as_tensor(grad)): raise RuntimeError(f"nonfinite grad at {step}")
        opt.step()
        first_loss = float(loss.detach()) if first_loss is None else first_loss
        last_loss = float(loss.detach()); max_grad = max(max_grad, float(grad))
        if step == 1 or step % 20 == 0 or step == STEPS:
            model.eval()
            with torch.inference_mode():
                vl, _ = model(vx[:32], height=9, width=9)
                vacc = float((vl.argmax(-1) == vy[:32]).float().mean())
            append_jsonl(curve, {
                "step": step, "train_loss": float(loss.detach()), "grad_norm_preclip": float(grad),
                "validation_cell_accuracy_32": vacc, "quant_strength": quant_strength_state(model),
            })
            model.train()
    elapsed = time.perf_counter() - t0
    model.eval()
    q = quant_strength_state(model)
    if not q or any(float(v) != 1.0 for v in q.values()):
        raise RuntimeError(f"training ended without hard ternary state: {q}")
    ckpt = out / "trained_source.pt"
    checkpoint = save_source_checkpoint(ckpt, model, opt, step=STEPS)
    reloaded = reload_source_checkpoint(ckpt)
    report = {
        "steps": STEPS, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "quant_warmup": QUANT_WARMUP, "first_loss": first_loss, "last_loss": last_loss,
        "max_grad_norm_preclip": max_grad, "elapsed_seconds": elapsed,
        "checkpoint": checkpoint, "quant_strength": quant_strength_state(reloaded),
        "curve": str(curve),
    }
    write_json(out / "training.json", report)
    return reloaded, report


def hard_weight(module: FakeBitLinear) -> torch.Tensor:
    with torch.no_grad():
        q, scale = module._ternarize_hard(module.weight)
        return q * scale


def err_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    e = (a.detach().cpu().float() - b.detach().cpu().float()).abs()
    return {"max_abs": float(e.max()) if e.numel() else 0.0, "mean_abs": float(e.mean()) if e.numel() else 0.0}


def capture_linear_inputs(model: TRM, x: torch.Tensor) -> dict[str, torch.Tensor]:
    wanted = {
        "blocks.0.attn.q", "blocks.0.attn.k", "blocks.0.attn.v", "blocks.0.attn.proj",
        "blocks.0.ff.0", "blocks.0.ff.2", "out_head",
    }
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in model.named_modules():
        if name in wanted:
            def hook(mod, args, name=name):
                if name not in captured:
                    captured[name] = args[0].detach().cpu().float().clone()
            handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode(): model(x, height=9, width=9)
    finally:
        for h in handles: h.remove()
    if set(captured) != wanted:
        raise RuntimeError(f"failed to capture all trained native-linear inputs: {sorted(wanted-set(captured))}")
    return captured


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/m10_cpu_deployment")
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    write_json(out / "environment.json", environment())

    cfg = make_cfg()
    datasets, manifest = build_data_splits(
        cfg, TRAIN_N, VAL_N, TEST_N, seed=DATA_SEED, manifest_path=out / "data_manifest.json"
    )
    model, training = train(out, datasets["train"], datasets["validation"])
    source_tensor_hash = module_tensor_state_sha256(model)

    provenance = {
        "data_seed": DATA_SEED,
        "train_id_sha256": id_hash(datasets["train"].ids),
        "validation_id_sha256": id_hash(datasets["validation"].ids),
        "test_id_sha256": id_hash(datasets["test"].ids),
        "duplicate_audit": manifest["duplicate_audit"],
        "test_used_for_training": False,
    }

    artifact_path = out / "spectra_cpu_v1.pt"
    t0 = time.perf_counter()
    artifact_report = export_cpu_artifact(
        model, artifact_path, height=9, width=9, box=3,
        source_checkpoint_sha256=training["checkpoint"]["sha256"],
        source_checkpoint_tensor_sha256=training["checkpoint"]["tensor_sha256"],
        training_seed=MODEL_SEED, training_step=STEPS,
        data_provenance=provenance, export_git_sha=git_sha(),
    )
    export_seconds = time.perf_counter() - t0

    t0 = time.perf_counter(); loaded = load_cpu_artifact(artifact_path); load_seconds = time.perf_counter() - t0
    if loaded.payload["source"]["model_tensor_state_sha256"] != source_tensor_hash:
        raise RuntimeError("deployment artifact source tensor hash does not match trained model")

    # Native extension compile/load is measured separately from artifact loading.
    t0 = time.perf_counter(); m10_native.load_extension(); native_compile_seconds = time.perf_counter() - t0
    runtime = CPURecursiveRuntime(loaded)
    backend = runtime.backend_report()
    write_json(out / "backend_identity.json", backend)

    test_x, test_y = tensors(datasets["test"])
    probe_x = test_x[:1]

    # Exact packed reconstruction is already enforced by export; re-check inventory declaration.
    reconstruction = {
        name: bool(entry["reconstruction_exact"])
        for name, entry in loaded.payload["packed_linears"].items()
    }
    if not all(reconstruction.values()): raise RuntimeError("packed reconstruction declaration failed")

    # Layer-level fidelity on actual trained-model inputs from a held-out puzzle.
    captured = capture_linear_inputs(model, probe_x)
    module_map = dict(model.named_modules())
    linear_results = {}
    for name, inp in captured.items():
        module = module_map[name]
        entry = loaded.payload["packed_linears"][name]
        flat = inp.reshape(-1, int(entry["in_features"])).contiguous()
        with torch.inference_mode():
            ref = F.linear(flat, hard_weight(module), module.bias)
            got = m10_native.dense_ternary_linear_fp32(
                flat, entry["packed"], entry["scale"], entry["bias"], int(entry["out_features"])
            )
        stats = err_stats(ref, got)
        stats["pass"] = stats["max_abs"] <= LINEAR_MAX and stats["mean_abs"] <= LINEAR_MEAN
        linear_results[name] = stats

    # Block and one-cycle fidelity from the actual root state of a held-out puzzle.
    with torch.inference_mode():
        x_emb = model.token_embed(probe_x) + model.encode_positions(probe_x, 9, 9)
        y0 = torch.zeros_like(x_emb); z0 = torch.zeros_like(x_emb)
        ref_block = model.blocks[0](x_emb)
        got_block = runtime.block(x_emb)
        block_stats = err_stats(ref_block, got_block)
        ref_yc, ref_zc = model.recursive_cycle(x_emb, y0, z0)
        got_yc, got_zc = runtime.recursive_cycle(x_emb, y0, z0)
        cycle_y = err_stats(ref_yc, got_yc); cycle_z = err_stats(ref_zc, got_zc)
    block_stats["pass"] = block_stats["max_abs"] <= BLOCK_MAX and block_stats["mean_abs"] <= BLOCK_MEAN
    for s in (cycle_y, cycle_z): s["pass"] = s["max_abs"] <= CYCLE_MAX and s["mean_abs"] <= CYCLE_MEAN

    # Full held-out fidelity. This is the first use of test data beyond the one fixed fidelity probe.
    hx = test_x[:FIDELITY_N]
    hy = test_y[:FIDELITY_N]
    with torch.inference_mode():
        ref_logits, ref_steps = model(hx, height=9, width=9)
        deployed = runtime.forward(hx)
    full_stats = err_stats(ref_logits, deployed.logits)
    ref_pred = ref_logits.argmax(-1)
    dep_pred = deployed.answer
    final_exact = bool(torch.equal(ref_pred, dep_pred))
    step_exact = [
        bool(torch.equal(a["logits"].argmax(-1), b["logits"].argmax(-1)))
        for a, b in zip(ref_steps, deployed.step_outputs)
    ]
    ref_sem = sudoku_correct(hx, ref_pred, 3).bool()
    dep_sem = sudoku_correct(hx, dep_pred, 3).bool()
    semantic_exact = bool(torch.equal(ref_sem, dep_sem))
    halt_stats = [err_stats(a["halt_logit"], b["halt_logit"]) for a, b in zip(ref_steps, deployed.step_outputs)]
    full_stats.update({
        "pass_numeric": full_stats["max_abs"] <= FULL_MAX and full_stats["mean_abs"] <= FULL_MEAN,
        "decoded_exact_all": final_exact,
        "step_decoded_exact": step_exact,
        "semantic_decision_exact_all": semantic_exact,
        "reference_semantic_valid_count": int(ref_sem.sum()),
        "deployed_semantic_valid_count": int(dep_sem.sum()),
        "halt_logit_error_by_step": halt_stats,
    })
    full_stats["pass"] = bool(full_stats["pass_numeric"] and final_exact and all(step_exact) and semantic_exact)
    heldout_metrics = {
        "reference": task_metrics("sudoku", hx, ref_pred, hy, box=3),
        "deployed": task_metrics("sudoku", hx, dep_pred, hy, box=3),
    }

    work = deployed.work
    if not work["native_call_count_matches_architecture"]:
        raise RuntimeError(f"native call structure mismatch: {work}")
    if work["native_linear_calls"] <= 0:
        raise RuntimeError("recursive deployment made no native calls")

    fidelity = {
        "tolerances": {
            "linear": {"max_abs": LINEAR_MAX, "mean_abs": LINEAR_MEAN},
            "block": {"max_abs": BLOCK_MAX, "mean_abs": BLOCK_MEAN},
            "cycle": {"max_abs": CYCLE_MAX, "mean_abs": CYCLE_MEAN},
            "full": {"max_abs": FULL_MAX, "mean_abs": FULL_MEAN},
        },
        "packed_reconstruction_exact": reconstruction,
        "native_linears": linear_results,
        "block": block_stats,
        "cycle_y": cycle_y, "cycle_z": cycle_z,
        "full_heldout_16": full_stats,
        "heldout_metrics_16": heldout_metrics,
        "runtime_work_heldout_16": work,
    }
    fidelity_pass = bool(
        all(v["pass"] for v in linear_results.values())
        and block_stats["pass"] and cycle_y["pass"] and cycle_z["pass"] and full_stats["pass"]
    )
    write_json(out / "fidelity.json", fidelity)

    # Warm-up and warmed latency are single-puzzle and exclude compile/artifact packing.
    with torch.inference_mode(): runtime.forward(probe_x)
    warm_times = []
    for _ in range(WARM_REPEATS):
        t0 = time.perf_counter()
        with torch.inference_mode(): runtime.forward(probe_x)
        warm_times.append(time.perf_counter() - t0)

    # Required profiler trace from a trained held-out puzzle solve attempt.
    trace_path = out / "profiler_trace.json"
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        prof_out = runtime.forward(probe_x)
    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
    (out / "profiler_table.txt").write_text(table + "\n", encoding="utf-8")
    profiler_work = prof_out.work
    if profiler_work["native_linear_calls"] != runtime.expected_native_calls_per_forward():
        raise RuntimeError("profiled solve did not execute expected native linears")

    timing = {
        "training_seconds_excluded_from_deployment": training["elapsed_seconds"],
        "export_pack_write_seconds": export_seconds,
        "cold_artifact_load_validate_seconds": load_seconds,
        "cold_native_compile_load_seconds": native_compile_seconds,
        "warmup_inference_count": 1,
        "warmed_single_puzzle_seconds": warm_times,
        "warmed_mean_seconds": statistics.mean(warm_times),
        "warmed_median_seconds": statistics.median(warm_times),
        "performance_claim": False,
    }
    write_json(out / "timing.json", timing)

    profile_record = {
        "trace": str(trace_path), "table": str(out / "profiler_table.txt"),
        "solve_input_id": datasets["test"].ids[0],
        "reference_prediction": ref_pred[0].tolist(),
        "deployed_prediction": prof_out.answer[0].tolist(),
        "prediction_matches_reference": bool(torch.equal(ref_pred[0], prof_out.answer[0])),
        "runtime_work": profiler_work,
        "native_scope_name": "spectra::dense_ternary_linear_fp32",
    }
    write_json(out / "profiler_record.json", profile_record)

    artifact_manifest = {
        **artifact_report,
        "architecture": loaded.payload["architecture"],
        "task": loaded.payload["task"],
        "precision": loaded.payload["precision"],
        "source": loaded.payload["source"],
        "inventory": loaded.payload["inventory"],
        "loader_sha256": loaded.sha256,
    }
    write_json(out / "artifact_manifest.json", artifact_manifest)

    exact_commands = [
        "python scripts/m10_cpu_deployment.py --out outputs/m10_cpu_deployment",
        "python -m pytest tests/test_m10_cpu_deployment.py -q",
        "python -m pytest -m 'not slow' -ra",
    ]
    write_json(out / "exact_commands.json", exact_commands)

    acceptance = bool(
        fidelity_pass
        and loaded.sha256 == artifact_report["sha256"]
        and work["native_linear_calls"] > 0
        and work["native_call_count_matches_architecture"]
        and backend["bitnet_cpp_supported"] is False
        and Path(trace_path).is_file() and Path(trace_path).stat().st_size > 0
    )
    summary = {
        "status": "complete" if acceptance else "failed_fidelity_gate",
        "acceptance_pass": acceptance,
        "trained_checkpoint": training["checkpoint"],
        "hard_ternary_verified": all(float(v) == 1.0 for v in training["quant_strength"].values()),
        "artifact": artifact_report,
        "backend": backend,
        "fidelity_pass": fidelity_pass,
        "fidelity": fidelity,
        "timing": timing,
        "profile": profile_record,
        "bitnet_cpp_supported": False,
        "mixed_precision": True,
        "full_integer_inference_claim": False,
        "gelu_preserved": True,
        "historical_fused_int8_ffn_used": False,
        "exact_commands": exact_commands,
    }
    write_json(out / "summary.json", summary)
    md = [
        "# M10 CPU Deployment Summary", "",
        f"- acceptance: **{'PASS' if acceptance else 'FAIL'}**",
        f"- artifact: `{artifact_report['sha256']}`",
        f"- native operator: `{backend['operator']}`",
        f"- native calls per profiled solve: `{profiler_work['native_linear_calls']}`",
        f"- full held-out max/mean logit error: `{full_stats['max_abs']:.8g}` / `{full_stats['mean_abs']:.8g}`",
        f"- held-out decoded exact: `{final_exact}`",
        f"- semantic-decision exact: `{semantic_exact}`",
        f"- bitnet.cpp supported: `{backend['bitnet_cpp_supported']}`",
        "- execution: mixed precision; native packed ternary FP32 linears + explicit FP32 attention/GELU/RMSNorm/residual/halt work + model A8 recurrent round/dequant.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(plain(summary), indent=2, sort_keys=True))
    return 0 if acceptance else 3


if __name__ == "__main__":
    raise SystemExit(main())
