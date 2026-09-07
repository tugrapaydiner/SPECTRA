#!/usr/bin/env python3
"""Milestone 06: one preregistered controlled trained baseline experiment.

This runner intentionally contains no MCTS/router/PRM/distillation/scaling-law work.
It executes the protocol frozen in docs/M06_PROTOCOL.md and retains raw evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config  # noqa: E402
from common.seed import set_seed  # noqa: E402
from data.datasets import GridDataset, build_sudoku_arrays  # noqa: E402
from eval.metrics import task_metrics  # noqa: E402
from model.stability import (  # noqa: E402
    QuantWarmup,
    load_quant_strength_state,
    quant_strength_state,
    ternary_report,
)
from model.system1_student import System1Student  # noqa: E402
from model.trm import TRM  # noqa: E402
from model.verifier import sudoku_correct  # noqa: E402
from scripts._common import build_data_splits  # noqa: E402
from train.losses import deep_supervision_loss  # noqa: E402

DATA_SEED = 20260907
TRAIN_N, VAL_N, TEST_N = 384, 96, 128
BATCH_SIZE = 32
TARGET_STEPS = 200
TIMING_STEPS = 5
MAIN_TRAIN_BUDGET_SECONDS = 1080.0
DIAGNOSTIC_BUDGET_SECONDS = 240.0
MODEL_SEEDS = [1101, 2202]
LR = 1e-3
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
KINDS = ("fp_recursive", "ternary_recursive", "single_pass")


class ExperimentStop(RuntimeError):
    pass


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if torch.is_tensor(x):
        if x.ndim == 0:
            return x.item()
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return x


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def environment_record() -> dict[str, Any]:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    try:
        import psutil
        memory_bytes = int(psutil.virtual_memory().total)
    except Exception:
        memory_bytes = None
    return {
        "git_sha": git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "memory_bytes": memory_bytes,
        "device": "cpu",
    }


def count_params(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def make_model(
    kind: str,
    seed: int,
    *,
    seq_len: int = 81,
    num_tokens: int = 10,
    minimal: bool = False,
) -> torch.nn.Module:
    set_seed(int(seed), deterministic=True)
    if kind in {"fp_recursive", "ternary_recursive"}:
        dim = 32 if minimal else 48
        model = TRM(
            dim=dim,
            num_tokens=num_tokens,
            seq_len=seq_len,
            n_layers=1,
            n=1,
            T=1,
            N_sup=2,
            heads=4,
            alpha_y=0.1,
            alpha_z=0.1,
            max_grid_size=16,
            ternary=(kind == "ternary_recursive"),
            act8=(kind == "ternary_recursive"),
        )
        return model.cpu()
    if kind == "single_pass":
        return System1Student(
            dim=48 if minimal else 96,
            num_tokens=num_tokens,
            seq_len=seq_len,
            n_layers=1 if minimal else 2,
            heads=4,
            max_grid_size=16,
        ).cpu()
    raise ValueError(f"unknown model kind {kind!r}")


def model_architecture(kind: str, model: torch.nn.Module) -> dict[str, Any]:
    if kind in {"fp_recursive", "ternary_recursive"}:
        assert isinstance(model, TRM)
        return {
            "kind": kind,
            "class": "TRM",
            "dim": model.dim,
            "n_layers": len(model.blocks),
            "heads": int(model.blocks[0].attn.num_heads),
            "n": model.n,
            "T": model.T,
            "N_sup": model.N_sup,
            "ternary": model.ternary,
            "act8": model.act8,
            "trainable_params": count_params(model),
            "forward_shared_block_applications_per_example": (
                model.N_sup * model.T * (model.n + 1) * len(model.blocks)
            ),
        }
    assert isinstance(model, System1Student)
    return {
        "kind": kind,
        "class": "System1Student",
        "dim": int(model.token_embed.embedding_dim),
        "n_layers": len(model.blocks),
        "heads": int(model.blocks[0].attn.num_heads),
        "trainable_params": count_params(model),
        "forward_shared_block_applications_per_example": len(model.blocks),
        "single_pass": True,
    }


def _loss(model: torch.nn.Module, kind: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if kind == "single_pass":
        logits, _ = model(x, height=9, width=9)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    _, steps = model(x, height=9, width=9)
    return deep_supervision_loss(steps, y, lambda_h=0.5, lambda_improve=0.1, margin=0.01)


def _predict(model: torch.nn.Module, kind: str, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if kind == "single_pass":
        logits, _ = model(x, height=h, width=w)
    else:
        logits, _ = model(x, height=h, width=w)
    return logits.argmax(-1)


def _dataset_tensors(ds: GridDataset) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(ds.inputs).long(), torch.from_numpy(ds.targets).long()


def evaluate(
    model: torch.nn.Module,
    kind: str,
    ds: GridDataset,
    *,
    predictions_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    x_all, y_all = _dataset_tensors(ds)
    preds: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(ds), BATCH_SIZE):
            xb = x_all[start : start + BATCH_SIZE]
            preds.append(_predict(model, kind, xb, ds.height, ds.width).cpu())
    pred = torch.cat(preds, dim=0) if preds else torch.empty_like(y_all)
    metrics = task_metrics("sudoku", x_all, pred, y_all, box=3)

    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as f:
            semantic = sudoku_correct(x_all, pred, 3).bool()
            for i in range(len(ds)):
                blank = x_all[i] == 0
                blank_acc = (
                    float((pred[i][blank] == y_all[i][blank]).float().mean().item())
                    if int(blank.sum()) else float("nan")
                )
                row = {
                    "id": ds.ids[i],
                    "group_id": ds.group_ids[i],
                    "input": x_all[i].tolist(),
                    "target": y_all[i].tolist(),
                    "prediction": pred[i].tolist(),
                    "exact_reference_match": bool(torch.equal(pred[i], y_all[i])),
                    "semantic_valid": bool(semantic[i].item()),
                    "blank_cell_accuracy": blank_acc,
                }
                f.write(json.dumps(_plain(row), sort_keys=True) + "\n")
    return {k: float(v) for k, v in metrics.items()}


def timing_probe(kind: str, train_ds: GridDataset) -> dict[str, Any]:
    seed = 9090
    model = make_model(kind, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x, y = _dataset_tensors(train_ds)
    rng = np.random.default_rng(77123)
    warmup = QuantWarmup(1) if kind == "ternary_recursive" else None
    elapsed = 0.0
    finite = True
    model.train()
    for step in range(1, TIMING_STEPS + 1):
        if warmup is not None:
            warmup.apply(model, step)
        idx = torch.from_numpy(rng.integers(0, len(train_ds), size=BATCH_SIZE, dtype=np.int64))
        t0 = time.perf_counter()
        loss = _loss(model, kind, x[idx], y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()
        elapsed += time.perf_counter() - t0
        if not torch.isfinite(loss) or not torch.isfinite(torch.as_tensor(grad)):
            finite = False
            break
    return {
        "kind": kind,
        "steps": TIMING_STEPS,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(1, TIMING_STEPS),
        "finite": finite,
        "trainable_params": count_params(model),
    }


def derive_budget(timing: list[dict[str, Any]]) -> dict[str, Any]:
    if not timing or not all(bool(r["finite"]) for r in timing):
        return {"run_main": False, "reason": "nonfinite_timing_probe"}
    s_max = max(float(r["seconds_per_step"]) for r in timing)
    if 6 * TARGET_STEPS * s_max <= MAIN_TRAIN_BUDGET_SECONDS:
        seeds = MODEL_SEEDS
        steps = TARGET_STEPS
    elif 3 * TARGET_STEPS * s_max <= MAIN_TRAIN_BUDGET_SECONDS:
        seeds = MODEL_SEEDS[:1]
        steps = TARGET_STEPS
    else:
        seeds = MODEL_SEEDS[:1]
        steps = int((MAIN_TRAIN_BUDGET_SECONDS / (3 * s_max)) // 10) * 10
        steps = min(TARGET_STEPS, steps)
    return {
        "run_main": steps >= 40,
        "reason": None if steps >= 40 else "timing_implies_fewer_than_40_steps",
        "slowest_seconds_per_step": s_max,
        "main_training_budget_seconds": MAIN_TRAIN_BUDGET_SECONDS,
        "target_steps": TARGET_STEPS,
        "selected_steps_per_model": steps,
        "selected_seeds": seeds if steps >= 40 else [],
        "repeated_seed": len(seeds) > 1 if steps >= 40 else False,
        "estimated_training_seconds": 3 * len(seeds) * steps * s_max if steps >= 40 else None,
    }


def _snapshot_steps(steps: int) -> set[int]:
    vals = {1, steps}
    for frac in (0.25, 0.5, 0.75):
        vals.add(max(1, int(round(steps * frac))))
    return vals


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    *,
    kind: str,
    seed: int,
    step: int,
    train_seconds: float,
    architecture: dict[str, Any],
) -> None:
    payload = {
        "format": "spectra.m06_controlled_baseline",
        "version": 1,
        "kind": kind,
        "seed": int(seed),
        "step": int(step),
        "train_seconds": float(train_seconds),
        "architecture": architecture,
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "quant_strength": quant_strength_state(model) if kind == "ternary_recursive" else {},
        "git_sha": git_sha(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint_for_eval(path: Path) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "spectra.m06_controlled_baseline" or payload.get("version") != 1:
        raise ExperimentStop(f"invalid M06 checkpoint format: {path}")
    kind = str(payload["kind"])
    model = make_model(kind, int(payload["seed"]))
    model.load_state_dict(payload["model_state"], strict=True)
    qstate = dict(payload.get("quant_strength", {}))
    if kind == "ternary_recursive":
        load_quant_strength_state(model, qstate)
        restored = quant_strength_state(model)
        if not restored or any(abs(float(v) - 1.0) > 1e-8 for v in restored.values()):
            raise ExperimentStop(
                f"ternary evaluation rejected: checkpoint quantization strength is not 1.0: {restored}"
            )
    model.eval()
    return model, kind, payload


def train_main_model(
    kind: str,
    seed: int,
    steps: int,
    train_ds: GridDataset,
    val_ds: GridDataset,
    out_dir: Path,
    budget_tracker: dict[str, float],
) -> dict[str, Any]:
    model = make_model(kind, seed)
    arch = model_architecture(kind, model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x, y = _dataset_tensors(train_ds)
    rng = np.random.default_rng(int(seed) + 50000)
    quant_warmup_steps = max(1, steps // 4)
    qwarm = QuantWarmup(quant_warmup_steps) if kind == "ternary_recursive" else None
    curve_path = out_dir / "curves" / f"{kind}_seed{seed}.jsonl"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = _snapshot_steps(steps)
    losses: list[float] = []
    grad_norms: list[float] = []
    train_seconds = 0.0

    model.train()
    with curve_path.open("w", encoding="utf-8") as curve:
        for step in range(1, steps + 1):
            if qwarm is not None:
                rho = qwarm.apply(model, step)
            else:
                rho = None
            idx = torch.from_numpy(
                rng.integers(0, len(train_ds), size=BATCH_SIZE, dtype=np.int64)
            )
            t0 = time.perf_counter()
            loss = _loss(model, kind, x[idx], y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
            dt = time.perf_counter() - t0
            train_seconds += dt
            budget_tracker["used"] += dt
            lv = float(loss.detach().item())
            gv = float(torch.as_tensor(grad).detach().item())
            losses.append(lv)
            grad_norms.append(gv)
            if not math.isfinite(lv):
                raise ExperimentStop(f"{kind}/seed{seed}: non-finite loss at step {step}")
            if not math.isfinite(gv):
                raise ExperimentStop(f"{kind}/seed{seed}: non-finite gradient norm at step {step}")
            if budget_tracker["used"] > MAIN_TRAIN_BUDGET_SECONDS:
                raise ExperimentStop(
                    f"main training wall budget exceeded after {kind}/seed{seed} step {step}"
                )

            if step in snapshots:
                val_metrics = evaluate(model, kind, val_ds)
                row = {
                    "kind": kind,
                    "seed": seed,
                    "step": step,
                    "train_loss": lv,
                    "grad_norm_preclip": gv,
                    "quant_strength": rho,
                    "validation": val_metrics,
                    "cumulative_model_train_seconds": train_seconds,
                }
                curve.write(json.dumps(_plain(row), sort_keys=True) + "\n")
                curve.flush()
                model.train()

    if kind == "ternary_recursive":
        qstate = quant_strength_state(model)
        if not qstate or any(abs(float(v) - 1.0) > 1e-8 for v in qstate.values()):
            raise ExperimentStop(f"ternary final quantization strength is not 1.0: {qstate}")
    else:
        qstate = {}

    window = min(10, max(2, len(losses) // 4))
    first_mean = mean(losses[:window])
    last_mean = mean(losses[-window:])
    relative_improvement = (first_mean - last_mean) / max(abs(first_mean), 1e-12)
    learning_failure = relative_improvement < 0.05 or max(grad_norms) <= 1e-12

    ckpt = out_dir / "checkpoints" / f"{kind}_seed{seed}.pt"
    save_checkpoint(
        ckpt,
        model,
        opt,
        kind=kind,
        seed=seed,
        step=steps,
        train_seconds=train_seconds,
        architecture=arch,
    )
    return {
        "kind": kind,
        "seed": seed,
        "steps": steps,
        "batch_size": BATCH_SIZE,
        "examples_sampled": steps * BATCH_SIZE,
        "architecture": arch,
        "train_seconds": train_seconds,
        "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY},
        "clip_grad_norm": CLIP_NORM,
        "loss_first_window_mean": first_mean,
        "loss_last_window_mean": last_mean,
        "relative_loss_improvement": relative_improvement,
        "max_grad_norm_preclip": max(grad_norms),
        "learning_failure": learning_failure,
        "checkpoint": str(ckpt),
        "quant_warmup_steps": quant_warmup_steps if qwarm is not None else None,
        "quant_strength": qstate,
        "ternary_report": ternary_report(model) if kind == "ternary_recursive" else None,
    }


def gradient_inspection(kind: str, train_ds: GridDataset, seed: int) -> dict[str, Any]:
    model = make_model(kind, seed + 700000)
    if kind == "ternary_recursive":
        QuantWarmup(1).apply(model, 1)
    x, y = _dataset_tensors(train_ds)
    xb, yb = x[:BATCH_SIZE], y[:BATCH_SIZE]
    loss = _loss(model, kind, xb, yb)
    loss.backward()
    total = nonzero = finite = 0
    norms: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += g.numel()
        nonzero += int((g != 0).sum().item())
        finite += int(torch.isfinite(g).sum().item())
        norms[name] = float(g.norm().item())
    return {
        "loss": float(loss.detach().item()),
        "gradient_elements": total,
        "nonzero_gradient_elements": nonzero,
        "finite_gradient_elements": finite,
        "all_gradient_finite": finite == total,
        "nonzero_fraction": nonzero / max(1, total),
        "parameter_grad_norms": norms,
    }


def diagnostic_overfit(
    kind: str,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    h: int,
    w: int,
    seed: int,
    minimal: bool,
    max_steps: int = 80,
) -> dict[str, Any]:
    model = make_model(
        kind,
        seed + (810000 if minimal else 800000),
        seq_len=x.shape[1],
        num_tokens=int(y.max().item()) + 1,
        minimal=minimal,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    qwarm = QuantWarmup(max(1, max_steps // 4)) if kind == "ternary_recursive" else None
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(1, max_steps + 1):
        if time.perf_counter() - started > DIAGNOSTIC_BUDGET_SECONDS:
            break
        if qwarm is not None:
            qwarm.apply(model, step)
        if kind == "single_pass":
            logits, _ = model(x, height=h, width=w)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        else:
            _, outs = model(x, height=h, width=w)
            loss = deep_supervision_loss(outs, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()
        losses.append(float(loss.detach().item()))
    model.eval()
    with torch.inference_mode():
        pred = _predict(model, kind, x, h, w)
    blank = x == 0
    blank_acc = float((pred[blank] == y[blank]).float().mean().item()) if int(blank.sum()) else float("nan")
    exact = float((pred == y).all(dim=1).float().mean().item())
    return {
        "minimal_4x4": minimal,
        "steps_completed": len(losses),
        "loss_initial": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "blank_cell_accuracy": blank_acc,
        "exact_reference_match": exact,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_failure_diagnostics(
    kind: str,
    seed: int,
    train_ds: GridDataset,
) -> dict[str, Any]:
    x9, y9 = _dataset_tensors(train_ds)
    x9, y9 = x9[:8], y9[:8]
    rng = np.random.default_rng(seed + 123)
    x4_np, y4_np, h4, w4 = build_sudoku_arrays(
        2, n=8, num_clues=8, rng=rng, require_unique=True, augment=False
    )
    x4 = torch.from_numpy(x4_np).long()
    y4 = torch.from_numpy(y4_np).long()
    return {
        "gradient_inspection_9x9": gradient_inspection(kind, train_ds, seed),
        "overfit_8_examples_9x9": diagnostic_overfit(
            kind, x9, y9, h=9, w=9, seed=seed, minimal=False
        ),
        "minimal_pipeline_debug_4x4": diagnostic_overfit(
            kind, x4, y4, h=h4, w=w4, seed=seed, minimal=True
        ),
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in KINDS:
        rows = [r for r in runs if r["kind"] == kind and "test_metrics" in r]
        if not rows:
            out[kind] = {"n": 0}
            continue
        metrics = sorted(rows[0]["test_metrics"])
        agg = {"n": len(rows), "seeds": [r["seed"] for r in rows]}
        for m in metrics:
            vals = [float(r["test_metrics"][m]) for r in rows]
            agg[m] = {
                "mean": mean(vals),
                "std_population": pstdev(vals) if len(vals) > 1 else 0.0,
                "values": vals,
            }
        out[kind] = agg
    return out


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Milestone 06 Controlled Baseline — Observed Summary",
        "",
        f"Status: **{summary['status']}**",
        f"Seed label: **{'repeated-seed pilot' if summary.get('repeated_seed') else 'single-seed pilot'}**",
        "",
        "Primary endpoint is strict 9×9 Sudoku semantic solve rate. Zero is an admissible result.",
        "",
        "| model | seeds | semantic solve | exact match | blank-cell acc | cell acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in KINDS:
        a = summary.get("aggregate", {}).get(kind, {})
        if not a or not a.get("n"):
            lines.append(f"| {kind} | 0 | n/a | n/a | n/a | n/a |")
            continue
        def fmt(metric: str) -> str:
            row = a[metric]
            if a["n"] > 1:
                return f"{row['mean']:.4f} ± {row['std_population']:.4f}"
            return f"{row['mean']:.4f} (single seed)"
        lines.append(
            f"| {kind} | {a['n']} | {fmt('semantic_validity')} | "
            f"{fmt('exact_reference_match')} | {fmt('blank_cell_accuracy')} | {fmt('cell_accuracy')} |"
        )
    lines += [
        "",
        "Interpretation is intentionally limited to this fixed generated-data pilot; see the JSON evidence for training time, parameter counts, quantization state, predictions, and any failures.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".m06")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "environment.json", environment_record())

    cfg = load_config(
        "config/sudoku.yaml",
        ["seed=20260907", "data.min_clues=30", "data.max_clues=35"],
    )
    datasets, manifest = build_data_splits(
        cfg,
        TRAIN_N,
        VAL_N,
        TEST_N,
        seed=DATA_SEED,
        manifest_path=out / "data_manifest.json",
    )
    audit = manifest["duplicate_audit"]
    if any(audit["cross_split_group_overlap"].values()) or any(audit["cross_split_exact_overlap"].values()):
        raise ExperimentStop(f"data overlap audit failed: {audit}")
    train_ds, val_ds, test_ds = datasets["train"], datasets["validation"], datasets["test"]

    timing = [timing_probe(kind, train_ds) for kind in KINDS]
    write_json(out / "timing_probe.json", timing)
    params = {r["kind"]: int(r["trainable_params"]) for r in timing}
    if params["fp_recursive"] != params["ternary_recursive"]:
        raise ExperimentStop(f"matched recursive parameter counts differ: {params}")
    if params["single_pass"] <= params["fp_recursive"]:
        raise ExperimentStop(f"single-pass baseline is not larger: {params}")

    budget = derive_budget(timing)
    budget["parameter_counts"] = params
    write_json(out / "budget_decision.json", budget)
    if not budget.get("run_main"):
        summary = {
            "status": "stopped_compute_infeasible",
            "repeated_seed": False,
            "budget": budget,
            "runs": [],
            "aggregate": {},
        }
        write_json(out / "summary.json", summary)
        (out / "SUMMARY.md").write_text(summary_markdown(summary), encoding="utf-8")
        print(json.dumps(_plain(summary), indent=2, sort_keys=True))
        return 0

    selected_steps = int(budget["selected_steps_per_model"])
    selected_seeds = [int(v) for v in budget["selected_seeds"]]
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    tracker = {"used": 0.0}

    for seed in selected_seeds:
        # Floating-point recursive reference is always trained first.
        for kind in KINDS:
            try:
                record = train_main_model(
                    kind, seed, selected_steps, train_ds, val_ds, out, tracker
                )
                eval_model, eval_kind, payload = load_checkpoint_for_eval(Path(record["checkpoint"]))
                if eval_kind != kind:
                    raise ExperimentStop("checkpoint kind mismatch")
                prediction_path = out / "predictions" / f"{kind}_seed{seed}.jsonl"
                test_metrics = evaluate(
                    eval_model, kind, test_ds, predictions_path=prediction_path
                )
                record["test_metrics"] = test_metrics
                record["evaluation_quant_strength"] = (
                    quant_strength_state(eval_model) if kind == "ternary_recursive" else {}
                )
                record["prediction_file"] = str(prediction_path)
                runs.append(record)

                if record["learning_failure"]:
                    diagnostics[f"{kind}_seed{seed}"] = run_failure_diagnostics(
                        kind, seed, train_ds
                    )
            except Exception as exc:
                failures.append({
                    "kind": kind,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "training_budget_used_seconds": tracker["used"],
                })
                # Preserve fairness: if the global training budget is exhausted,
                # do not silently continue models outside the preregistered budget.
                if tracker["used"] > MAIN_TRAIN_BUDGET_SECONDS:
                    break
        if tracker["used"] > MAIN_TRAIN_BUDGET_SECONDS:
            break

    write_json(out / "runs.json", runs)
    write_json(out / "failures.json", failures)
    write_json(out / "diagnostics.json", diagnostics)
    aggregate = aggregate_runs(runs)
    status = "complete" if not failures else "complete_with_recorded_failures"
    summary = {
        "status": status,
        "task": "9x9 Sudoku, 30-35 clues, unique solutions",
        "primary_metric": "semantic_validity",
        "data_seed": DATA_SEED,
        "data_counts": {"train": TRAIN_N, "validation": VAL_N, "test": TEST_N},
        "selected_steps_per_model": selected_steps,
        "selected_seeds": selected_seeds,
        "repeated_seed": len(selected_seeds) > 1,
        "training_budget_used_seconds": tracker["used"],
        "main_training_budget_seconds": MAIN_TRAIN_BUDGET_SECONDS,
        "parameter_counts": params,
        "aggregate": aggregate,
        "runs": runs,
        "failures": failures,
        "diagnostics_triggered": sorted(diagnostics),
    }
    write_json(out / "summary.json", summary)
    (out / "SUMMARY.md").write_text(summary_markdown(summary), encoding="utf-8")
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentStop as exc:
        print(f"M06 STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)
