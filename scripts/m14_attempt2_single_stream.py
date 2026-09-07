#!/usr/bin/env python3
"""M14 Attempt 2: input-conditioned single-stream recurrent intervention.

Protocol: docs/M14_ATTEMPT2_PROTOCOL.md

Attempt 1 remains immutable evidence.  This runner reuses only the established
M14 data/statistics/measurement contracts, retrains the unchanged primary
baseline, and does not generate confirmation data unless the Attempt-2 candidate
passes the unchanged development gate.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.measurement_env import measurement_environment
from common.seed import set_seed
from data.datasets import GridDataset
from eval.edge_energy import energy_counter_inventory, measure_energy_record
from model.single_stream_trm import InputConditionedSingleStreamTRM
from model.system1_student import System1Student
from model.verifier import sudoku_correct
from scripts.m14_primary_experiment import (
    BATCH_SIZE,
    BOOTSTRAP_SEED,
    CONFIRMATION_COST_N,
    CONFIRM_DATA_SEED,
    DEVELOPMENT_COST_N,
    DEV_N,
    LATENCY_REPEATS,
    LATER_WEIGHTS,
    LR,
    MODEL_SEEDS,
    PRIMARY_BASELINE,
    RESERVE_CONFIRM_DATA_SEED,
    TRAIN_N,
    TRAIN_STEPS,
    VAL_N,
    VALIDATION_COST_N,
    WEIGHT_DECAY,
    CLIP_NORM,
    all_aggregates,
    append_jsonl,
    build_confirmation_data,
    build_development_data,
    clamp_givens,
    dynamic_int8_baseline,
    git_sha,
    paired_effect,
    practical_gate,
    percentile,
    plain,
    sha256_file,
    symbolic_bundle,
    write_csv,
    write_json,
    SolverBundle,
)

ATTEMPT1_RUN = 34159474745
ATTEMPT1_HEAD = "66acbc62f48b3e5b1e81080897b89562edfd8bd8"
ATTEMPT1_ARTIFACT_ID = 10032762057
ATTEMPT1_ARTIFACT_SHA256 = "04a5b810d697227c4ccf88a365cf529647734ea0d95deec363f84fcaae885522"
ATTEMPT2_PROTOCOL_COMMIT = "1282d026c89a6af696a225947aa67bd00e970917"
CANDIDATE_KIND = "input_conditioned_single_stream"
CANDIDATE_DIM = 96
CANDIDATE_BUDGETS = [1, 2, 3, 4]


class Attempt2Stop(RuntimeError):
    pass


def model_params(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def make_candidate(seed: int) -> InputConditionedSingleStreamTRM:
    set_seed(int(seed), deterministic=True)
    return InputConditionedSingleStreamTRM(
        dim=CANDIDATE_DIM,
        num_tokens=5,
        seq_len=16,
        n_layers=1,
        T=1,
        N_sup=4,
        heads=4,
        alpha_y=0.1,
        max_grid_size=8,
    ).cpu()


def make_baseline(seed: int) -> System1Student:
    set_seed(int(seed), deterministic=True)
    model = System1Student(
        dim=96, num_tokens=5, seq_len=16, n_layers=2, heads=4, max_grid_size=8
    ).cpu()
    for p in model.conf_head.parameters():
        p.requires_grad_(False)
    return model


def candidate_architecture(model: InputConditionedSingleStreamTRM) -> dict[str, Any]:
    return {
        "kind": CANDIDATE_KIND,
        "class": type(model).__name__,
        "semantics": model.SEMANTICS,
        "dim": int(model.dim),
        "n_layers": len(model.blocks),
        "heads": 4,
        "T": int(model.T),
        "training_N_sup": int(model.N_sup),
        "recurrent_streams": "y_only_z_compatibility_zero",
        "block_apps_per_training_example": int(model.N_sup * model.T * len(model.blocks)),
        "trainable_params": model_params(model),
    }


def baseline_architecture(model: System1Student) -> dict[str, Any]:
    return {
        "kind": "single_pass",
        "class": type(model).__name__,
        "dim": 96,
        "n_layers": len(model.blocks),
        "heads": 4,
        "single_pass": True,
        "confidence_head_frozen": True,
        "block_apps_per_training_example": len(model.blocks),
        "trainable_params": model_params(model),
    }


def blank_ce(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = x == 0
    if not bool(mask.any()):
        raise Attempt2Stop("training batch unexpectedly contains no blank cells")
    return torch.nn.functional.cross_entropy(logits[mask], y[mask])


def candidate_training_loss(
    model: InputConditionedSingleStreamTRM, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    _, steps = model(x, height=4, width=4)
    if len(steps) != len(LATER_WEIGHTS):
        raise Attempt2Stop(f"candidate expected four supervision outputs, got {len(steps)}")
    weights = [float(v) / float(sum(LATER_WEIGHTS)) for v in LATER_WEIGHTS]
    return sum(w * blank_ce(step["logits"], x, y) for w, step in zip(weights, steps))


def baseline_training_loss(model: System1Student, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits, _ = model(x, height=4, width=4)
    return blank_ce(logits, x, y)


def predict_candidate(
    model: InputConditionedSingleStreamTRM, x: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.inference_mode():
        logits, steps = model(x, height=4, width=4)
    pred = clamp_givens(x, logits.argmax(dim=-1))
    work = dict(steps[-1]["work_cumulative"]) if steps else {}
    return pred, work


def predict_baseline(model: System1Student, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        logits, _ = model(x, height=4, width=4)
    return clamp_givens(x, logits.argmax(dim=-1))


def quick_metrics_candidate(model: InputConditionedSingleStreamTRM, ds: GridDataset) -> dict[str, float]:
    model.eval()
    xs = torch.from_numpy(ds.inputs).long(); ys = torch.from_numpy(ds.targets).long()
    preds = []
    for i in range(0, len(ds), 128):
        p, _ = predict_candidate(model, xs[i:i+128]); preds.append(p.cpu())
    pred = torch.cat(preds)
    valid = sudoku_correct(xs, pred, 2).float()
    blank = xs == 0
    return {
        "semantic_validity": float(valid.mean()),
        "exact_reference_match": float((pred == ys).all(dim=1).float().mean()),
        "blank_cell_accuracy": float((pred[blank] == ys[blank]).float().mean()),
    }


def quick_metrics_baseline(model: System1Student, ds: GridDataset) -> dict[str, float]:
    model.eval()
    xs = torch.from_numpy(ds.inputs).long(); ys = torch.from_numpy(ds.targets).long()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(ds), 128):
            preds.append(predict_baseline(model, xs[i:i+128]).cpu())
    pred = torch.cat(preds)
    valid = sudoku_correct(xs, pred, 2).float()
    blank = xs == 0
    return {
        "semantic_validity": float(valid.mean()),
        "exact_reference_match": float((pred == ys).all(dim=1).float().mean()),
        "blank_cell_accuracy": float((pred[blank] == ys[blank]).float().mean()),
    }


def save_model(
    path: Path,
    model: nn.Module,
    *,
    kind: str,
    seed: int,
    architecture: dict[str, Any],
    training: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "spectra.m14_attempt2_model",
        "version": 1,
        "kind": kind,
        "seed": int(seed),
        "architecture": architecture,
        "training": training,
        "model_state": model.state_dict(),
        "git_sha": git_sha(),
        "attempt2_protocol_commit": ATTEMPT2_PROTOCOL_COMMIT,
    }, path)


def load_model(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "spectra.m14_attempt2_model" or payload.get("version") != 1:
        raise Attempt2Stop(f"invalid Attempt-2 checkpoint {path}")
    kind = str(payload["kind"]); seed = int(payload["seed"])
    if kind == CANDIDATE_KIND:
        model: nn.Module = make_candidate(seed)
    elif kind == "single_pass":
        model = make_baseline(seed)
    else:
        raise Attempt2Stop(f"unsupported checkpoint kind {kind}")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def train_one(
    kind: str,
    seed: int,
    train_ds: GridDataset,
    val_ds: GridDataset,
    out: Path,
) -> dict[str, Any]:
    if kind == CANDIDATE_KIND:
        model: nn.Module = make_candidate(seed)
        architecture = candidate_architecture(model)
        loss_fn = candidate_training_loss
        metric_fn = quick_metrics_candidate
    elif kind == "single_pass":
        model = make_baseline(seed)
        architecture = baseline_architecture(model)
        loss_fn = baseline_training_loss
        metric_fn = quick_metrics_baseline
    else:
        raise ValueError(kind)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    x = torch.from_numpy(train_ds.inputs).long(); y = torch.from_numpy(train_ds.targets).long()
    rng = np.random.default_rng(seed + 141400)
    curve = out / "curves" / f"{kind}_seed{seed}.jsonl"
    curve.parent.mkdir(parents=True, exist_ok=True)
    if curve.exists(): curve.unlink()
    snapshots = {1, TRAIN_STEPS // 2, 3 * TRAIN_STEPS // 4, TRAIN_STEPS}
    losses: list[float] = []
    max_grad = 0.0
    train_seconds = 0.0
    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        idx = torch.from_numpy(
            rng.integers(0, len(train_ds), size=BATCH_SIZE, dtype=np.int64)
        )
        t0 = time.perf_counter()
        loss = loss_fn(model, x[idx], y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
        optimizer.step()
        train_seconds += time.perf_counter() - t0
        lv = float(loss.detach()); gv = float(torch.as_tensor(grad).detach())
        if not math.isfinite(lv) or not math.isfinite(gv):
            raise Attempt2Stop(f"nonfinite training state {kind}/seed{seed}/step{step}")
        losses.append(lv); max_grad = max(max_grad, gv)
        if step == 1 or step % 100 == 0 or step in snapshots:
            val = None
            if step in snapshots:
                model.eval(); val = metric_fn(model, val_ds); model.train()
            append_jsonl(curve, {
                "kind": kind,
                "seed": seed,
                "step": step,
                "train_blank_ce": lv,
                "grad_norm_preclip": gv,
                "validation": val,
                "cumulative_train_seconds": train_seconds,
            })
    model.eval()
    window = min(50, len(losses))
    first = float(np.mean(losses[:window])); last = float(np.mean(losses[-window:]))
    record = {
        "kind": kind,
        "seed": seed,
        "steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "examples_sampled": TRAIN_STEPS * BATCH_SIZE,
        "train_seconds": train_seconds,
        "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY},
        "clip_grad_norm": CLIP_NORM,
        "objective": (
            "four_step_weighted_blank_cell_cross_entropy" if kind == CANDIDATE_KIND
            else "blank_cell_cross_entropy"
        ),
        "initial_loss_mean": first,
        "final_loss_mean": last,
        "relative_loss_improvement": (first - last) / max(abs(first), 1e-12),
        "max_grad_norm_preclip": max_grad,
        "architecture": architecture,
        "training_block_applications": (
            TRAIN_STEPS * BATCH_SIZE * int(architecture["block_apps_per_training_example"])
        ),
        "validation_final": metric_fn(model, val_ds),
    }
    path = out / "checkpoints" / f"{kind}_seed{seed}.pt"
    save_model(path, model, kind=kind, seed=seed, architecture=architecture, training=record)
    record["checkpoint"] = str(path)
    record["checkpoint_sha256"] = sha256_file(path)
    append_jsonl(out / "training_runs.jsonl", record)
    return record


def train_all(train_ds: GridDataset, val_ds: GridDataset, out: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # Candidate and primary baseline each use all five seeds.  No other learned
    # family is introduced in Attempt 2.
    for kind in [CANDIDATE_KIND, "single_pass"]:
        for seed in MODEL_SEEDS:
            records.append(train_one(kind, seed, train_ds, val_ds, out))
    c_params = {r["architecture"]["trainable_params"] for r in records if r["kind"] == CANDIDATE_KIND}
    b_params = {r["architecture"]["trainable_params"] for r in records if r["kind"] == "single_pass"}
    if len(c_params) != 1 or len(b_params) != 1 or not (next(iter(c_params)) < next(iter(b_params))):
        raise Attempt2Stop(f"candidate must be smaller than primary baseline: candidate={c_params}, baseline={b_params}")
    write_json(out / "parameter_audit.json", {
        "candidate_trainable_params": next(iter(c_params)),
        "baseline_trainable_params": next(iter(b_params)),
        "candidate_smaller": True,
    })
    return records


def candidate_bundle(seed: int, budget: int, out: Path) -> SolverBundle:
    ck = out / "checkpoints" / f"{CANDIDATE_KIND}_seed{seed}.pt"
    model, payload = load_model(ck)
    if not isinstance(model, InputConditionedSingleStreamTRM):
        raise Attempt2Stop("candidate checkpoint did not restore single-stream model")
    model.N_sup = int(budget); model.eval()
    last_work: dict[str, Any] = {}

    def solve(x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            logits, steps = model(x, height=4, width=4)
        last_work.clear()
        if steps:
            last_work.update(dict(steps[-1]["work_cumulative"]))
        return clamp_givens(x, logits.argmax(dim=-1))

    return SolverBundle(
        config_id=f"single_stream_n{budget}",
        family="single_stream_recursive",
        seed=seed,
        solve=solve,
        work=lambda: dict(last_work),
        setup={
            "checkpoint": str(ck),
            "checkpoint_sha256": sha256_file(ck),
            "N_sup": int(budget),
            "semantics": payload["architecture"]["semantics"],
        },
    )


def baseline_bundle(seed: int, out: Path, *, int8: bool = False, audit_sink: Path | None = None) -> SolverBundle | None:
    ck = out / "checkpoints" / f"single_pass_seed{seed}.pt"
    model, _ = load_model(ck)
    if not isinstance(model, System1Student):
        raise Attempt2Stop("baseline checkpoint did not restore System1Student")
    audit = {"seed": seed, "source_checkpoint": str(ck), "source_sha256": sha256_file(ck)}
    if int8:
        q, qa = dynamic_int8_baseline(model); audit.update(qa)
        if audit_sink is not None: append_jsonl(audit_sink, audit)
        if q is None: return None
        model = q
    cid = "single_pass_int8" if int8 else PRIMARY_BASELINE

    def solve(x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            logits, _ = model(x, height=4, width=4)
        return clamp_givens(x, logits.argmax(dim=-1))

    return SolverBundle(
        cid, cid, seed, solve,
        work=lambda: {"block_applications": 2, "supervision_steps": 1},
        setup=audit,
    )


def evaluate_bundle(
    bundle: SolverBundle,
    ds: GridDataset,
    split: str,
    *,
    cost_n: int,
    repeats: int,
    out_rows: Path,
) -> list[dict[str, Any]]:
    xs = torch.from_numpy(ds.inputs).long(); ys = torch.from_numpy(ds.targets).long()
    n_cost = min(int(cost_n), len(ds))
    for i in range(min(4, len(ds))):
        a = bundle.solve(xs[i:i+1]); _ = sudoku_correct(xs[i:i+1], a, 2)
    rows: list[dict[str, Any]] = []
    for i in range(len(ds)):
        xi = xs[i:i+1]
        latency = None; answer = None; semantic = None
        if i < n_cost:
            times: list[float] = []
            for _r in range(max(1, int(repeats))):
                t0 = time.perf_counter_ns()
                a = bundle.solve(xi)
                ok = sudoku_correct(xi, a, 2).bool()
                times.append((time.perf_counter_ns() - t0) / 1e6)
                if answer is None:
                    answer = a.detach().cpu(); semantic = bool(ok.item())
            latency = float(statistics.median(times))
        else:
            answer = bundle.solve(xi).detach().cpu()
            semantic = bool(sudoku_correct(xi, answer, 2).item())
        assert answer is not None and semantic is not None
        target = ys[i:i+1]
        blank = xi.cpu() == 0
        work = bundle.work() if bundle.work is not None else {}
        row = {
            "attempt": 2,
            "split": split,
            "config_id": bundle.config_id,
            "family": bundle.family,
            "seed": bundle.seed,
            "example_index": i,
            "example_id": ds.ids[i],
            "semantic_success": int(semantic),
            "exact_reference_match": int(bool((answer == target).all().item())),
            "blank_cell_accuracy": float((answer[blank] == target[blank]).float().mean()),
            "latency_ms": latency,
            "latency_repeats": repeats if i < n_cost else 0,
            "complete_solve_timing_includes_semantic_check": True,
            "target_used_inside_solver": False,
            "work_block_applications": plain(work.get("block_applications")),
            "work_supervision_steps": plain(work.get("supervision_steps")),
            "work_recursive_cycles": plain(work.get("recursive_cycles")),
        }
        append_jsonl(out_rows, row); rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return all_aggregates(rows)


def choose_candidate(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["config_id"]: r for r in aggregates}
    base = by.get(PRIMARY_BASELINE)
    if base is None or base.get("latency_median_ms") is None:
        raise Attempt2Stop("primary baseline validation row missing")
    base_q = float(base["semantic_validity"]); base_lat = float(base["latency_median_ms"])
    candidates = []
    for budget in CANDIDATE_BUDGETS:
        cid = f"single_stream_n{budget}"
        row = by.get(cid)
        if not row or not row.get("n") or row.get("latency_median_ms") is None:
            continue
        row = dict(row)
        row["validation_quality_difference"] = float(row["semantic_validity"]) - base_q
        row["validation_latency_ratio_vs_baseline"] = float(row["latency_median_ms"]) / base_lat
        row["path_b_point_eligible"] = (
            row["validation_quality_difference"] >= -0.02
            and row["validation_latency_ratio_vs_baseline"] <= 0.65
        )
        row["path_a_point_eligible"] = (
            row["validation_quality_difference"] >= 0.03
            and row["validation_latency_ratio_vs_baseline"] <= 1.15
        )
        ratio = row["validation_latency_ratio_vs_baseline"]
        row["latency_match_status"] = "matched" if 0.85 <= ratio <= 1.15 else "unmatched"
        candidates.append(row)
    if not candidates:
        raise Attempt2Stop("no Attempt-2 candidate operating points")

    b_pool = [r for r in candidates if r["path_b_point_eligible"]]
    a_pool = [r for r in candidates if r["path_a_point_eligible"]]
    if b_pool:
        pool = b_pool; rule = "path_b_point_eligible_priority"
    elif a_pool:
        pool = a_pool; rule = "path_a_point_eligible_fallback"
    else:
        matched = [r for r in candidates if r["validation_latency_ratio_vs_baseline"] <= 1.15]
        pool = matched if matched else candidates
        rule = "diagnostic_fallback_no_point_gate_relaxation"
    chosen = sorted(
        pool,
        key=lambda r: (
            -float(r["semantic_validity"]),
            float(r["latency_median_ms"]),
            int(str(r["config_id"]).rsplit("n", 1)[1]),
        ),
    )[0]
    return {
        "selected_config_id": chosen["config_id"],
        "selection_rule": rule,
        "selected_validation": chosen,
        "primary_baseline_validation": base,
        "all_candidate_validation": candidates,
        "selection_used_development": False,
        "selection_used_confirmation": False,
        "thresholds_changed": False,
    }


def run_validation(val_ds: GridDataset, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = out / "validation_rows.jsonl"
    if path.exists(): path.unlink()
    rows: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        for budget in CANDIDATE_BUDGETS:
            rows.extend(evaluate_bundle(
                candidate_bundle(seed, budget, out), val_ds, "validation",
                cost_n=VALIDATION_COST_N, repeats=LATENCY_REPEATS, out_rows=path,
            ))
        rows.extend(evaluate_bundle(
            baseline_bundle(seed, out, int8=False), val_ds, "validation",
            cost_n=VALIDATION_COST_N, repeats=LATENCY_REPEATS, out_rows=path,
        ))
        q = baseline_bundle(
            seed, out, int8=True, audit_sink=out / "int8_conversion_audit.jsonl"
        )
        if q is not None:
            rows.extend(evaluate_bundle(
                q, val_ds, "validation", cost_n=VALIDATION_COST_N,
                repeats=LATENCY_REPEATS, out_rows=path,
            ))
    rows.extend(evaluate_bundle(
        symbolic_bundle(), val_ds, "validation", cost_n=VALIDATION_COST_N,
        repeats=LATENCY_REPEATS, out_rows=path,
    ))
    aggs = aggregate(rows)
    selection = choose_candidate(aggs)
    write_json(out / "validation_operating_points.json", aggs)
    write_csv(out / "validation_operating_points.csv", aggs)
    write_json(out / "candidate_selection.json", selection)
    return rows, selection


def evaluate_surface(
    candidate: str,
    ds: GridDataset,
    split: str,
    out: Path,
    *,
    cost_n: int,
    repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    path = out / f"{split}_rows.jsonl"
    if path.exists(): path.unlink()
    budget = int(candidate.rsplit("n", 1)[1])
    rows: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        rows.extend(evaluate_bundle(
            candidate_bundle(seed, budget, out), ds, split,
            cost_n=cost_n, repeats=repeats, out_rows=path,
        ))
        rows.extend(evaluate_bundle(
            baseline_bundle(seed, out, int8=False), ds, split,
            cost_n=cost_n, repeats=repeats, out_rows=path,
        ))
        q = baseline_bundle(seed, out, int8=True, audit_sink=out / "int8_conversion_audit.jsonl")
        if q is not None:
            rows.extend(evaluate_bundle(
                q, ds, split, cost_n=cost_n, repeats=repeats, out_rows=path,
            ))
    rows.extend(evaluate_bundle(
        symbolic_bundle(), ds, split, cost_n=cost_n, repeats=repeats, out_rows=path,
    ))
    cand = [r for r in rows if r["config_id"] == candidate]
    base = [r for r in rows if r["config_id"] == PRIMARY_BASELINE]
    effect = paired_effect(
        cand, base,
        seed=(BOOTSTRAP_SEED + 1 if split == "confirmation" else BOOTSTRAP_SEED),
    )
    gate = practical_gate(effect)
    write_json(out / f"{split}_operating_points.json", aggregate(rows))
    write_json(out / f"{split}_effect.json", effect)
    write_json(out / f"{split}_decision.json", gate)
    return rows, effect, gate


def representative_energy(candidate: str, ds: GridDataset, out: Path, split: str) -> dict[str, Any]:
    budget = int(candidate.rsplit("n", 1)[1])
    records: dict[str, Any] = {
        "scope": "cpu_package_rapl_not_gpu_not_whole_system",
        "split": split,
        "configs": {},
    }
    bundles = {
        candidate: candidate_bundle(MODEL_SEEDS[0], budget, out),
        PRIMARY_BASELINE: baseline_bundle(MODEL_SEEDS[0], out, int8=False),
    }
    xs = torch.from_numpy(ds.inputs[:32]).long()
    for cid, bundle in bundles.items():
        assert bundle is not None
        def run(bundle=bundle):
            for i in range(len(xs)):
                a = bundle.solve(xs[i:i+1]); _ = sudoku_correct(xs[i:i+1], a, 2)
        rec = measure_energy_record(run, n_runs=1)
        records["configs"][cid] = {
            **rec,
            "problems_per_window": len(xs),
            "joules_per_complete_solve": (
                float(rec["energy_joules"]) / len(xs)
                if rec.get("available") and rec.get("energy_joules") is not None
                else None
            ),
        }
    write_json(out / f"{split}_energy.json", records)
    return records


def freeze_candidate(candidate: str, out: Path) -> dict[str, Any]:
    files = []
    for seed in MODEL_SEEDS:
        cp = out / "checkpoints" / f"{CANDIDATE_KIND}_seed{seed}.pt"
        bp = out / "checkpoints" / f"single_pass_seed{seed}.pt"
        files.append({"role": "candidate", "seed": seed, "path": str(cp), "sha256": sha256_file(cp)})
        files.append({"role": "primary_baseline", "seed": seed, "path": str(bp), "sha256": sha256_file(bp)})
    rec = {
        "attempt": 2,
        "candidate_config_id": candidate,
        "architecture_semantics": InputConditionedSingleStreamTRM.SEMANTICS,
        "files": files,
        "frozen_before_confirmation": True,
        "confirmation_seed": CONFIRM_DATA_SEED,
        "git_sha": git_sha(),
        "timestamp_unix": time.time(),
    }
    write_json(out / "confirmation_freeze_manifest.json", rec)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".m14a2/experiment")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass

    write_json(out / "prior_attempt.json", {
        "attempt": 1,
        "status": "INCOMPLETE",
        "run": ATTEMPT1_RUN,
        "head": ATTEMPT1_HEAD,
        "artifact_id": ATTEMPT1_ARTIFACT_ID,
        "artifact_zip_sha256": ATTEMPT1_ARTIFACT_SHA256,
        "confirmation_opened": False,
        "development_selected_candidate": "fp64_n1",
        "development_quality_difference": -0.005078125,
        "development_latency_ratio": 1.0943352911334046,
    })
    write_json(out / "environment.json", measurement_environment(
        backend="pytorch_cpu_complete_solve",
        compiler_flags=None,
        extra={
            "milestone": 14,
            "attempt": 2,
            "git_sha": git_sha(),
            "protocol": "docs/M14_ATTEMPT2_PROTOCOL.md",
            "primary_cost": "complete_solve_latency",
        },
    ))
    write_json(out / "energy_inventory.json", energy_counter_inventory())

    train_ds, val_ds, dev_ds, dev_manifest = build_development_data(out)
    training = train_all(train_ds, val_ds, out)
    _, selection = run_validation(val_ds, out)
    candidate = str(selection["selected_config_id"])
    dev_rows, dev_effect, dev_gate = evaluate_surface(
        candidate, dev_ds, "development", out,
        cost_n=DEVELOPMENT_COST_N, repeats=LATENCY_REPEATS,
    )
    representative_energy(candidate, dev_ds, out, "development")

    confirmation_opened = False
    confirmation_effect = None
    confirmation_gate = None
    confirmation_audit = None
    if dev_gate["pass"]:
        freeze_candidate(candidate, out)
        confirm_ds, _, confirmation_audit = build_confirmation_data(out, dev_manifest)
        confirmation_opened = True
        _, confirmation_effect, confirmation_gate = evaluate_surface(
            candidate, confirm_ds, "confirmation", out,
            cost_n=CONFIRMATION_COST_N, repeats=LATENCY_REPEATS,
        )
        representative_energy(candidate, confirm_ds, out, "confirmation")

    status = "COMPLETE" if confirmation_gate is not None and confirmation_gate["pass"] else "INCOMPLETE"
    result = {
        "milestone": 14,
        "attempt": 2,
        "status": status,
        "claim_confirmed": bool(status == "COMPLETE"),
        "protocol_committed_before_attempt2_results": True,
        "protocol_commit": ATTEMPT2_PROTOCOL_COMMIT,
        "prior_attempt_count": 1,
        "prior_attempt_adaptive_development_accounted": True,
        "prior_attempt_run": ATTEMPT1_RUN,
        "primary_metric": "strict_sudoku_semantic_validity",
        "primary_baseline": PRIMARY_BASELINE,
        "candidate_family": CANDIDATE_KIND,
        "selected_candidate": candidate,
        "development_gate": dev_gate,
        "development_effect": dev_effect,
        "confirmation_opened": confirmation_opened,
        "confirmation_gate": confirmation_gate,
        "confirmation_effect": confirmation_effect,
        "confirmation_seed": CONFIRM_DATA_SEED if confirmation_opened else None,
        "reserve_confirmation_seed_unopened": RESERVE_CONFIRM_DATA_SEED,
        "confirmation_overlap_audit": confirmation_audit,
        "n_training_seeds": len(MODEL_SEEDS),
        "training_seeds": MODEL_SEEDS,
        "training_runs": len(training),
        "train_examples": TRAIN_N,
        "validation_examples": VAL_N,
        "development_examples": DEV_N,
        "confirmation_examples": 1024 if confirmation_opened else 0,
        "training_steps_per_model": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "thresholds_changed_after_attempt1": False,
        "baseline_weakened": False,
        "favorable_seed_filtering": False,
        "hidden_width_sweep": False,
        "learning_rate_sweep": False,
        "loss_weight_sweep": False,
        "scaling_law_claim": False,
        "energy_scope": "cpu_package_if_valid_otherwise_unavailable_not_gpu_not_whole_system",
        "latency_match_tolerance": "validation matched label remains +/-15%; path-B faster points are explicitly unmatched, not iso-budget",
    }
    write_json(out / "summary.json", result)
    return 0 if status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
