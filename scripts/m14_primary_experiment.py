#!/usr/bin/env python3
"""Milestone 14: preregistered primary controlled experiment.

The protocol is docs/M14_PROTOCOL.md.  This runner intentionally keeps validation,
development, and confirmation roles separate.  The confirmation manifest is not
generated unless a validation-selected candidate first passes the development gate.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config
from common.measurement_env import measurement_environment
from common.seed import set_seed
from data import sudoku as sudoku_np
from data.datasets import GridDataset
from eval.edge_energy import energy_counter_inventory, measure_energy_record
from eval.grounded_targets import generate_trajectory_states, tensor_state_sha256
from eval.latent_mcts import LatentNativeMCTS, _LatentNode
from model.grounded_verifier import GroundedStateVerifier
from model.latent_action import LatentActionCodebook
from model.stability import QuantWarmup, load_quant_strength_state, quant_strength_state, ternary_report
from model.system1_student import System1Student
from model.trm import TRM
from model.verifier import sudoku_correct
from scripts._common import build_data_splits


# ---------------------------------------------------------------------------
# Frozen protocol constants.  Keep in sync with docs/M14_PROTOCOL.md.
# ---------------------------------------------------------------------------
DEV_DATA_SEED = 2026091401
CONFIRM_DATA_SEED = 2026091402
RESERVE_CONFIRM_DATA_SEED = 2026091403
MODEL_SEEDS = [1401, 2402, 3403, 4404, 5405]
TRAIN_N = 4096
VAL_N = 512
DEV_N = 512
CONFIRM_N = 1024
TRAIN_STEPS = 1800
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
QUANT_WARMUP = 450
RECURSIVE_TRAIN_NSUP = 4
RECURSIVE_BUDGETS = [1, 2, 3, 4]
SEARCH_ROLLOUTS = [1, 2, 4]
SEARCH_ACTIONS = 3
SEARCH_MAX_DEPTH = 4
VERIFIER_STEPS = 200
VERIFIER_BATCH = 64
VERIFIER_TRAIN_PUZZLES = 256
VERIFIER_VAL_PUZZLES = 128
VALIDATION_COST_N = 64
DEVELOPMENT_COST_N = 128
CONFIRMATION_COST_N = 256
LATENCY_REPEATS = 2
BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 2026091499
LATER_WEIGHTS = [0.1, 0.2, 0.3, 0.4]

PRIMARY_BASELINE = "single_pass_fp"
CONTEXT_CONFIGS = [
    "fp_n4", "ternary_n4", "fp_search_r2", "ternary_search_r2",
    "single_pass_fp", "single_pass_int8", "symbolic_exact",
]


class ExperimentStop(RuntimeError):
    pass


def plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if torch.is_tensor(v):
        return v.detach().cpu().item() if v.ndim == 0 else v.detach().cpu().tolist()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(plain(payload), sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows([{k: row.get(k) for k in keys} for row in rows])


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tensor(t: torch.Tensor) -> str:
    c = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(c.dtype).encode()); h.update(np.asarray(c.shape, dtype=np.int64).tobytes())
    h.update(c.numpy().tobytes())
    return h.hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def model_params(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def make_cfg(seed: int):
    return load_config(
        "config/sudoku.yaml",
        overrides=[
            f"seed={seed}", "device=cpu",
            "data.box=2", "data.num_tokens=5", "data.seq_len=16",
            "data.height=4", "data.width=4", "data.min_clues=6", "data.max_clues=10",
            "data.require_unique=true", "data.augment=false",
            "data.solution_method=random_backtracking",
        ],
    )


def manifest_fingerprints(manifest: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for split in manifest.get("splits", {}).values():
        for row in split.get("examples", []):
            out.add(str(row["fingerprint"]))
    return out


def build_development_data(out: Path) -> tuple[GridDataset, GridDataset, GridDataset, dict[str, Any]]:
    cfg = make_cfg(DEV_DATA_SEED)
    ds, manifest = build_data_splits(
        cfg, TRAIN_N, VAL_N, DEV_N, seed=DEV_DATA_SEED,
        manifest_path=out / "manifests" / "train_validation_development.json",
    )
    audit = manifest["duplicate_audit"]
    if any(audit["cross_split_group_overlap"].values()) or any(audit["cross_split_exact_overlap"].values()):
        raise ExperimentStop(f"development hierarchy duplicate audit failed: {audit}")
    return ds["train"], ds["validation"], ds["test"], manifest


def build_confirmation_data(out: Path, prior_manifest: dict[str, Any]) -> tuple[GridDataset, dict[str, Any], dict[str, Any]]:
    cfg = make_cfg(CONFIRM_DATA_SEED)
    ds, manifest = build_data_splits(
        cfg, 0, 0, CONFIRM_N, seed=CONFIRM_DATA_SEED,
        manifest_path=out / "manifests" / "confirmation_1.json",
    )
    prior = manifest_fingerprints(prior_manifest)
    conf = manifest_fingerprints(manifest)
    overlap = sorted(prior & conf)
    audit = {
        "prior_fingerprints": len(prior),
        "confirmation_fingerprints": len(conf),
        "overlap_count": len(overlap),
        "overlap_examples": overlap[:10],
        "confirmation_seed": CONFIRM_DATA_SEED,
        "reserve_confirmation_seed_named_but_unopened": RESERVE_CONFIRM_DATA_SEED,
    }
    write_json(out / "manifests" / "confirmation_overlap_audit.json", audit)
    if overlap:
        raise ExperimentStop("confirmation fingerprint overlap with development hierarchy")
    return ds["test"], manifest, audit


# ---------------------------------------------------------------------------
# Models and training
# ---------------------------------------------------------------------------
def make_model(kind: str, seed: int, *, dim_override: int | None = None) -> nn.Module:
    set_seed(seed, deterministic=True)
    if kind in {"fp_recursive", "ternary_recursive", "fp_recursive_dim64"}:
        dim = 64 if kind == "fp_recursive_dim64" else int(dim_override or 48)
        ternary = kind == "ternary_recursive"
        return TRM(
            dim=dim, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1,
            N_sup=RECURSIVE_TRAIN_NSUP, heads=4, alpha_y=0.1, alpha_z=0.1,
            max_grid_size=8, ternary=ternary, act8=ternary,
        ).cpu()
    if kind == "single_pass":
        m = System1Student(
            dim=96, num_tokens=5, seq_len=16, n_layers=2, heads=4, max_grid_size=8
        ).cpu()
        for p in m.conf_head.parameters():
            p.requires_grad_(False)
        return m
    raise ValueError(kind)


def family_record(kind: str, model: nn.Module) -> dict[str, Any]:
    if isinstance(model, TRM):
        return {
            "kind": kind, "class": "TRM", "dim": model.dim, "n_layers": len(model.blocks),
            "heads": 4, "n": model.n, "T": model.T, "training_N_sup": model.N_sup,
            "ternary": model.ternary, "act8": model.act8,
            "trainable_params": model_params(model),
            "block_apps_per_training_example": model.N_sup * model.T * (model.n + 1) * len(model.blocks),
        }
    return {
        "kind": kind, "class": "System1Student", "dim": 96, "n_layers": 2,
        "heads": 4, "single_pass": True, "confidence_head_frozen": True,
        "trainable_params": model_params(model), "block_apps_per_training_example": 2,
    }


def blank_ce(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = x == 0
    if not bool(mask.any()):
        raise ExperimentStop("training batch unexpectedly contains no blank cells")
    return F.cross_entropy(logits[mask], y[mask])


def training_loss(model: nn.Module, kind: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if kind == "single_pass":
        logits, _ = model(x, height=4, width=4)
        return blank_ce(logits, x, y)
    _, steps = model(x, height=4, width=4)
    if len(steps) != len(LATER_WEIGHTS):
        raise ExperimentStop(f"recursive training expected four supervision steps, got {len(steps)}")
    weights = [v / sum(LATER_WEIGHTS) for v in LATER_WEIGHTS]
    return sum(float(w) * blank_ce(s["logits"], x, y) for w, s in zip(weights, steps))


def clamp_givens(x: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.where(x != 0, x, pred)


def predict_loaded(model: nn.Module, kind: str, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        if kind == "single_pass":
            logits, _ = model(x, height=4, width=4)
        else:
            logits, _ = model(x, height=4, width=4)
        return clamp_givens(x, logits.argmax(dim=-1))


def quick_metrics(model: nn.Module, kind: str, ds: GridDataset) -> dict[str, float]:
    model.eval()
    xs = torch.from_numpy(ds.inputs).long(); ys = torch.from_numpy(ds.targets).long()
    answers: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(0, len(ds), 128):
            answers.append(predict_loaded(model, kind, xs[i:i+128]).cpu())
    pred = torch.cat(answers)
    valid = sudoku_correct(xs, pred, 2).float()
    blank = xs == 0
    return {
        "semantic_validity": float(valid.mean()),
        "exact_reference_match": float((pred == ys).all(dim=1).float().mean()),
        "blank_cell_accuracy": float((pred[blank] == ys[blank]).float().mean()),
    }


def save_checkpoint(path: Path, model: nn.Module, kind: str, seed: int, train_record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "spectra.m14_model", "version": 1, "kind": kind, "seed": seed,
        "model_state": model.state_dict(), "quant_strength": quant_strength_state(model) if kind == "ternary_recursive" else {},
        "family": family_record(kind, model), "training": train_record, "git_sha": git_sha(),
    }, path)


def load_checkpoint(path: Path) -> tuple[nn.Module, str, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "spectra.m14_model" or payload.get("version") != 1:
        raise ExperimentStop(f"invalid M14 checkpoint {path}")
    kind = str(payload["kind"]); seed = int(payload["seed"])
    model = make_model(kind, seed)
    model.load_state_dict(payload["model_state"], strict=True)
    if kind == "ternary_recursive":
        load_quant_strength_state(model, payload.get("quant_strength", {}))
        q = quant_strength_state(model)
        if not q or any(abs(float(v) - 1.0) > 1e-8 for v in q.values()):
            raise ExperimentStop(f"ternary checkpoint is not at full quantization strength: {q}")
    model.eval()
    return model, kind, payload


def train_one(kind: str, seed: int, train_ds: GridDataset, val_ds: GridDataset, out: Path) -> dict[str, Any]:
    model = make_model(kind, seed)
    arch = family_record(kind, model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    q = QuantWarmup(QUANT_WARMUP) if kind == "ternary_recursive" else None
    x = torch.from_numpy(train_ds.inputs).long(); y = torch.from_numpy(train_ds.targets).long()
    rng = np.random.default_rng(seed + 141400)
    curve = out / "curves" / f"{kind}_seed{seed}.jsonl"
    curve.parent.mkdir(parents=True, exist_ok=True)
    if curve.exists(): curve.unlink()
    snapshots = {1, QUANT_WARMUP, TRAIN_STEPS // 2, 3 * TRAIN_STEPS // 4, TRAIN_STEPS}
    losses: list[float] = []
    train_seconds = 0.0
    max_grad = 0.0
    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        rho = q.apply(model, step) if q is not None else None
        idx = torch.from_numpy(rng.integers(0, len(train_ds), size=BATCH_SIZE, dtype=np.int64))
        t0 = time.perf_counter()
        loss = training_loss(model, "single_pass" if kind == "single_pass" else kind, x[idx], y[idx])
        opt.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, CLIP_NORM); opt.step()
        dt = time.perf_counter() - t0; train_seconds += dt
        lv = float(loss.detach()); gv = float(torch.as_tensor(grad).detach())
        if not math.isfinite(lv) or not math.isfinite(gv):
            raise ExperimentStop(f"nonfinite training state {kind}/seed{seed}/step{step}")
        losses.append(lv); max_grad = max(max_grad, gv)
        if step == 1 or step % 100 == 0 or step in snapshots:
            val = None
            if step in snapshots:
                model.eval(); val = quick_metrics(model, "single_pass" if kind == "single_pass" else kind, val_ds); model.train()
            append_jsonl(curve, {
                "kind": kind, "seed": seed, "step": step, "train_blank_ce": lv,
                "grad_norm_preclip": gv, "quant_strength": rho, "validation": val,
                "cumulative_train_seconds": train_seconds,
            })
    model.eval()
    if kind == "ternary_recursive":
        qstate = quant_strength_state(model)
        if not qstate or any(abs(float(v) - 1.0) > 1e-8 for v in qstate.values()):
            raise ExperimentStop(f"ternary final quantization strength invalid: {qstate}")
    window = min(50, len(losses))
    record = {
        "kind": kind, "seed": seed, "steps": TRAIN_STEPS, "batch_size": BATCH_SIZE,
        "examples_sampled": TRAIN_STEPS * BATCH_SIZE, "train_seconds": train_seconds,
        "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY},
        "clip_grad_norm": CLIP_NORM, "objective": "blank_cell_cross_entropy",
        "initial_loss_mean": float(np.mean(losses[:window])), "final_loss_mean": float(np.mean(losses[-window:])),
        "relative_loss_improvement": float((np.mean(losses[:window]) - np.mean(losses[-window:])) / max(abs(np.mean(losses[:window])), 1e-12)),
        "max_grad_norm_preclip": max_grad, "architecture": arch,
        "training_block_applications": TRAIN_STEPS * BATCH_SIZE * arch["block_apps_per_training_example"],
        "validation_final": quick_metrics(model, "single_pass" if kind == "single_pass" else kind, val_ds),
        "tensor_state_sha256": tensor_state_sha256(model),
        "ternary_report": ternary_report(model) if kind == "ternary_recursive" else None,
    }
    ckpt = out / "checkpoints" / f"{kind}_seed{seed}.pt"
    save_checkpoint(ckpt, model, kind, seed, record)
    record["checkpoint"] = str(ckpt); record["checkpoint_sha256"] = sha256_file(ckpt)
    append_jsonl(out / "training_runs.jsonl", record)
    return record


def train_main_families(train_ds: GridDataset, val_ds: GridDataset, out: Path) -> list[dict[str, Any]]:
    runs = []
    for kind in ["fp_recursive", "ternary_recursive", "single_pass"]:
        for seed in MODEL_SEEDS:
            runs.append(train_one(kind, seed, train_ds, val_ds, out))
    return runs


# ---------------------------------------------------------------------------
# Grounded verifier + faithful full-state serial search experiment adapter.
# ---------------------------------------------------------------------------
class FullStateM14MCTS(LatentNativeMCTS):
    """M08 serial tree mechanics with M07 full-state verifier values."""
    def _value(self, x: torch.Tensor, node: _LatentNode) -> float:
        with torch.inference_mode():
            v = self.verifier.value_state(x, node.y, node.latent(), self.width)
        return float(v.detach().mean())

    def _value_batch(self, x_rep, z_batch):  # pragma: no cover - M14 uses serial only
        del x_rep, z_batch
        raise RuntimeError("M14 full-state search is serial because batched leaves require paired y states")


def rows_to_tensors(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.stack([r["x"] for r in rows]).long(),
        torch.stack([r["y"] for r in rows]).float(),
        torch.stack([r["z"] for r in rows]).float(),
        torch.tensor([float(r["label"]) for r in rows], dtype=torch.float32),
    )


def fit_verifier(reasoner_path: Path, family: str, seed: int, train_ds: GridDataset, val_ds: GridDataset, out: Path) -> dict[str, Any]:
    reasoner, kind, _ = load_checkpoint(reasoner_path)
    if not isinstance(reasoner, TRM): raise ExperimentStop("search verifier requires TRM reasoner")
    for p in reasoner.parameters(): p.requires_grad_(False)
    reasoner.eval()
    tx = torch.from_numpy(train_ds.inputs[:VERIFIER_TRAIN_PUZZLES]).long()
    vx = torch.from_numpy(val_ds.inputs[:VERIFIER_VAL_PUZZLES]).long()
    tr = generate_trajectory_states(reasoner, tx, train_ds.ids[:VERIFIER_TRAIN_PUZZLES], max_depth=4, height=4, width=4, box=2)
    vr = generate_trajectory_states(reasoner, vx, val_ds.ids[:VERIFIER_VAL_PUZZLES], max_depth=4, height=4, width=4, box=2)
    x, y, z, labels = rows_to_tensors(tr); vxx, vyy, vzz, vlabels = rows_to_tensors(vr)
    vseed = seed + (710000 if family == "fp" else 720000)
    set_seed(vseed, deterministic=True)
    verifier = GroundedStateVerifier(num_tokens=5, dim=reasoner.dim, n_layers=1, heads=4, max_grid_size=8, act_bits=8, include_y=True)
    opt = torch.optim.AdamW(verifier.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(vseed + 19)
    curve = out / "verifiers" / f"{family}_seed{seed}_curve.jsonl"
    curve.parent.mkdir(parents=True, exist_ok=True)
    if curve.exists(): curve.unlink()
    first = last = None
    for step in range(1, VERIFIER_STEPS + 1):
        idx = torch.from_numpy(rng.integers(0, len(tr), size=VERIFIER_BATCH, dtype=np.int64))
        verifier.train(); probs = verifier(x[idx], y[idx], z[idx], 4)
        loss = F.binary_cross_entropy(probs, labels[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); grad = torch.nn.utils.clip_grad_norm_(verifier.parameters(), 1.0); opt.step()
        if not torch.isfinite(loss) or not torch.isfinite(torch.as_tensor(grad)):
            raise ExperimentStop("nonfinite M14 verifier training")
        first = float(loss) if first is None else first; last = float(loss)
        if step == 1 or step % 50 == 0 or step == VERIFIER_STEPS:
            verifier.eval()
            with torch.inference_mode(): vp = verifier(vxx, vyy, vzz, 4)
            append_jsonl(curve, {
                "step": step, "train_bce": float(loss.detach()), "validation_bce": float(F.binary_cross_entropy(vp, vlabels)),
                "validation_accuracy": float(((vp >= .5) == (vlabels >= .5)).float().mean()),
                "positive_train": int(labels.sum()), "positive_validation": int(vlabels.sum()),
            })
    verifier.eval()
    vp = verifier(vxx, vyy, vzz, 4).detach()
    path = out / "verifiers" / f"{family}_seed{seed}.pt"
    torch.save({
        "format": "spectra.m14_grounded_verifier", "version": 1, "family": family, "reasoner_seed": seed,
        "reasoner_checkpoint_sha256": sha256_file(reasoner_path), "state": verifier.state_dict(),
        "dim": reasoner.dim, "train_steps": VERIFIER_STEPS,
    }, path)
    # Fixed, explicitly untrained action codebook retained for reproducibility.
    action_seed = seed + (910000 if family == "fp" else 920000)
    set_seed(action_seed, deterministic=True)
    codebook = LatentActionCodebook(reasoner.dim, n_actions=SEARCH_ACTIONS, scale=0.5)
    for p in codebook.parameters(): p.requires_grad_(False)
    action_path = out / "verifiers" / f"{family}_seed{seed}_fixed_actions.pt"
    torch.save({
        "format": "spectra.m14_fixed_actions", "version": 1, "trained": False,
        "seed": action_seed, "n_actions": SEARCH_ACTIONS, "scale": 0.5,
        "state": codebook.state_dict(),
    }, action_path)
    rec = {
        "family": family, "seed": seed, "reasoner_kind": kind, "verifier_seed": vseed,
        "train_rows": len(tr), "validation_rows": len(vr),
        "train_positive": int(labels.sum()), "validation_positive": int(vlabels.sum()),
        "first_bce": first, "last_bce": last, "validation_bce": float(F.binary_cross_entropy(vp, vlabels)),
        "validation_accuracy": float(((vp >= .5) == (vlabels >= .5)).float().mean()),
        "verifier_checkpoint": str(path), "verifier_sha256": sha256_file(path),
        "fixed_action_checkpoint": str(action_path), "fixed_action_sha256": sha256_file(action_path),
        "fixed_actions_trained": False,
    }
    append_jsonl(out / "verifier_runs.jsonl", rec)
    return rec


def fit_main_verifiers(train_ds: GridDataset, val_ds: GridDataset, out: Path) -> list[dict[str, Any]]:
    recs = []
    for family, kind in [("fp", "fp_recursive"), ("ternary", "ternary_recursive")]:
        for seed in MODEL_SEEDS:
            recs.append(fit_verifier(out / "checkpoints" / f"{kind}_seed{seed}.pt", family, seed, train_ds, val_ds, out))
    return recs


def load_verifier_and_actions(family: str, seed: int, reasoner: TRM, out: Path) -> tuple[GroundedStateVerifier, LatentActionCodebook]:
    vp = torch.load(out / "verifiers" / f"{family}_seed{seed}.pt", map_location="cpu", weights_only=False)
    verifier = GroundedStateVerifier(num_tokens=5, dim=reasoner.dim, n_layers=1, heads=4, max_grid_size=8, act_bits=8, include_y=True)
    verifier.load_state_dict(vp["state"], strict=True); verifier.eval()
    ap = torch.load(out / "verifiers" / f"{family}_seed{seed}_fixed_actions.pt", map_location="cpu", weights_only=False)
    codebook = LatentActionCodebook(reasoner.dim, n_actions=int(ap["n_actions"]), scale=float(ap["scale"]))
    codebook.load_state_dict(ap["state"], strict=True); codebook.eval()
    for p in codebook.parameters(): p.requires_grad_(False)
    return verifier, codebook


# ---------------------------------------------------------------------------
# INT8 conventional baseline audit.
# ---------------------------------------------------------------------------
def dynamic_int8_baseline(model: System1Student) -> tuple[nn.Module | None, dict[str, Any]]:
    started = time.perf_counter()
    try:
        from torch.ao.quantization import quantize_dynamic
        q = quantize_dynamic(copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8, inplace=False)
        quantized = []
        for name, module in q.named_modules():
            mod = type(module).__module__.lower()
            if "quantized" in mod and type(module).__name__.lower() == "linear":
                quantized.append(name)
        if not quantized:
            return None, {"supported": False, "reason": "no_actual_quantized_linear_modules", "conversion_seconds": time.perf_counter()-started}
        smoke = torch.from_numpy(np.asarray([[1, 0, 0, 4, 0, 4, 1, 0, 0, 1, 4, 0, 4, 0, 0, 1]], dtype=np.int64)).long()
        with torch.inference_mode():
            logits, _ = q(smoke, height=4, width=4)
        if not torch.isfinite(logits).all():
            return None, {"supported": False, "reason": "nonfinite_quantized_smoke_output", "quantized_modules": quantized}
        return q, {
            "supported": True, "quantized_modules": quantized,
            "quantized_module_count": len(quantized), "attention_note": "unsupported attention internals may remain floating point",
            "conversion_seconds": time.perf_counter()-started,
        }
    except Exception as exc:
        return None, {"supported": False, "reason": f"{type(exc).__name__}: {exc}", "conversion_seconds": time.perf_counter()-started}


# ---------------------------------------------------------------------------
# Solver constructors and raw evaluation.
# ---------------------------------------------------------------------------
@dataclass
class SolverBundle:
    config_id: str
    family: str
    seed: int | str
    solve: Callable[[torch.Tensor], torch.Tensor]
    work: Callable[[], dict[str, Any]] | None = None
    setup: dict[str, Any] | None = None


def recursive_bundle(kind: str, seed: int, budget: int, out: Path, *, reserve: bool = False) -> SolverBundle:
    ck = out / "checkpoints" / f"{kind}_seed{seed}.pt"
    model, loaded_kind, _ = load_checkpoint(ck)
    assert isinstance(model, TRM)
    model.N_sup = int(budget); model.eval()
    prefix = "fp64" if reserve else ("fp" if kind == "fp_recursive" else "ternary")
    cid = f"{prefix}_n{budget}"
    def solve(x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            logits, _ = model(x, height=4, width=4)
            return clamp_givens(x, logits.argmax(-1))
    return SolverBundle(cid, prefix, seed, solve, setup={"checkpoint": str(ck), "checkpoint_sha256": sha256_file(ck), "N_sup": budget, "kind": loaded_kind})


def search_bundle(family: str, seed: int, rollouts: int, out: Path) -> SolverBundle:
    kind = "fp_recursive" if family == "fp" else "ternary_recursive"
    ck = out / "checkpoints" / f"{kind}_seed{seed}.pt"
    model, _, _ = load_checkpoint(ck); assert isinstance(model, TRM)
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    verifier, codebook = load_verifier_and_actions(family, seed, model, out)
    controller = FullStateM14MCTS(model, verifier, codebook, 4, 4, n_rollouts=int(rollouts), c_puct=1.5, max_depth=SEARCH_MAX_DEPTH)
    cid = f"{family}_search_r{rollouts}"
    def solve(x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            node = controller.search(x)
            pred = model.out_head(node.y).argmax(-1)
            return clamp_givens(x, pred)
    return SolverBundle(cid, f"{family}_search", seed, solve, work=lambda: dict(controller.last_search_stats), setup={
        "reasoner_checkpoint": str(ck), "reasoner_sha256": sha256_file(ck),
        "verifier": str(out / "verifiers" / f"{family}_seed{seed}.pt"),
        "fixed_actions": str(out / "verifiers" / f"{family}_seed{seed}_fixed_actions.pt"),
        "fixed_actions_trained": False, "rollouts": rollouts,
    })


def single_pass_bundle(seed: int, out: Path, *, int8: bool = False, audit_sink: Path | None = None) -> SolverBundle | None:
    ck = out / "checkpoints" / f"single_pass_seed{seed}.pt"
    model, _, _ = load_checkpoint(ck); assert isinstance(model, System1Student)
    audit = {"seed": seed, "source_checkpoint": str(ck), "source_sha256": sha256_file(ck)}
    if int8:
        q, qa = dynamic_int8_baseline(model); audit.update(qa)
        if audit_sink is not None: append_jsonl(audit_sink, audit)
        if q is None: return None
        model = q
    cid = "single_pass_int8" if int8 else "single_pass_fp"
    def solve(x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            logits, _ = model(x, height=4, width=4)
            return clamp_givens(x, logits.argmax(-1))
    return SolverBundle(cid, cid, seed, solve, setup=audit)


def symbolic_bundle() -> SolverBundle:
    def solve(x: torch.Tensor) -> torch.Tensor:
        rows = []
        for row in x.detach().cpu().numpy():
            ans = sudoku_np.solve(row.reshape(4, 4), 2)
            if ans is None: raise ExperimentStop("symbolic solver failed on generated valid puzzle")
            rows.append(torch.from_numpy(ans.reshape(-1)).long())
        return torch.stack(rows)
    return SolverBundle("symbolic_exact", "symbolic_exact", "symbolic", solve, setup={"algorithm": "exact_mrv_backtracking", "learned_training_steps": 0})


def evaluate_bundle(bundle: SolverBundle, ds: GridDataset, split: str, *, cost_n: int, repeats: int, out_rows: Path) -> list[dict[str, Any]]:
    xs = torch.from_numpy(ds.inputs).long(); ys = torch.from_numpy(ds.targets).long()
    n_cost = min(cost_n, len(ds))
    # Distinct warmups are not timed.
    for i in range(min(4, len(ds))):
        ans = bundle.solve(xs[i:i+1]); _ = sudoku_correct(xs[i:i+1], ans, 2)
    rows: list[dict[str, Any]] = []
    for i in range(len(ds)):
        xi = xs[i:i+1]
        latency = None; answer = None; semantic = None
        if i < n_cost:
            times: list[float] = []
            for _r in range(max(1, repeats)):
                t0 = time.perf_counter_ns()
                a = bundle.solve(xi)
                ok = sudoku_correct(xi, a, 2).bool()
                dt = (time.perf_counter_ns() - t0) / 1e6
                times.append(dt)
                if answer is None:
                    answer = a.detach().cpu(); semantic = bool(ok.item())
            latency = float(statistics.median(times))
        else:
            answer = bundle.solve(xi).detach().cpu(); semantic = bool(sudoku_correct(xi, answer, 2).item())
        assert answer is not None and semantic is not None
        target = ys[i:i+1]
        blank = xi.cpu() == 0
        work = bundle.work() if bundle.work is not None else {}
        row = {
            "split": split, "config_id": bundle.config_id, "family": bundle.family,
            "seed": bundle.seed, "example_index": i, "example_id": ds.ids[i],
            "semantic_success": int(semantic),
            "exact_reference_match": int(bool((answer == target).all().item())),
            "blank_cell_accuracy": float((answer[blank] == target[blank]).float().mean()),
            "latency_ms": latency, "latency_repeats": repeats if i < n_cost else 0,
            "complete_solve_timing_includes_semantic_check": True,
            "target_used_inside_solver": False,
        }
        for key in ["transition_calls", "recursive_cycle_calls", "verifier_calls", "verifier_evaluations", "decode_calls", "best_path"]:
            if key in work: row[f"work_{key}"] = plain(work[key])
        append_jsonl(out_rows, row); rows.append(row)
    return rows


def aggregate_config(rows: list[dict[str, Any]], cid: str) -> dict[str, Any]:
    rr = [r for r in rows if r["config_id"] == cid]
    if not rr: return {"config_id": cid, "n": 0}
    task = [r for r in rr if r["seed"] != "symbolic"] or rr
    lats = [float(r["latency_ms"]) for r in rr if r.get("latency_ms") is not None]
    seeds = sorted({str(r["seed"]) for r in rr})
    seed_rates = {}
    for s in seeds:
        sr = [r for r in rr if str(r["seed"]) == s]
        seed_rates[s] = float(np.mean([r["semantic_success"] for r in sr]))
    return {
        "config_id": cid, "family": rr[0]["family"], "n": len(task), "seeds": seeds,
        "n_training_seeds": len([s for s in seeds if s != "symbolic"]),
        "semantic_validity": float(np.mean([r["semantic_success"] for r in task])),
        "exact_reference_match": float(np.mean([r["exact_reference_match"] for r in task])),
        "blank_cell_accuracy": float(np.mean([r["blank_cell_accuracy"] for r in task])),
        "seed_semantic_validity": seed_rates,
        "latency_n": len(lats), "latency_median_ms": float(np.median(lats)) if lats else None,
        "latency_p95_ms": percentile(lats, .95) if lats else None,
    }


def all_aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = sorted({str(r["config_id"]) for r in rows})
    return [aggregate_config(rows, cid) for cid in ids]


def candidate_ids(*, reserve: bool = False) -> list[str]:
    if reserve: return [f"fp64_n{n}" for n in RECURSIVE_BUDGETS]
    out = [f"fp_n{n}" for n in RECURSIVE_BUDGETS] + [f"ternary_n{n}" for n in RECURSIVE_BUDGETS]
    out += [f"fp_search_r{r}" for r in SEARCH_ROLLOUTS] + [f"ternary_search_r{r}" for r in SEARCH_ROLLOUTS]
    return out


def choose_candidate(aggregates: list[dict[str, Any]], *, reserve: bool = False) -> dict[str, Any]:
    by = {r["config_id"]: r for r in aggregates}
    base = by.get(PRIMARY_BASELINE)
    if base is None or base.get("latency_median_ms") is None: raise ExperimentStop("primary baseline validation latency missing")
    base_lat = float(base["latency_median_ms"])
    candidates = [by[c] for c in candidate_ids(reserve=reserve) if c in by and by[c].get("n")]
    if not candidates: raise ExperimentStop("no validation candidate rows")
    for c in candidates:
        lat = c.get("latency_median_ms")
        c["validation_latency_ratio_vs_baseline"] = None if lat is None else float(lat) / base_lat
        ratio = c["validation_latency_ratio_vs_baseline"]
        c["latency_match_status"] = "matched" if ratio is not None and 0.85 <= ratio <= 1.15 else "unmatched"
    eligible = [c for c in candidates if c["validation_latency_ratio_vs_baseline"] is not None and c["validation_latency_ratio_vs_baseline"] <= 1.15]
    pool = eligible if eligible else candidates
    chosen = sorted(pool, key=lambda r: (-float(r["semantic_validity"]), float(r.get("latency_median_ms") or 1e30), str(r["config_id"])))[0]
    return {
        "selected_config_id": chosen["config_id"],
        "selection_pool": "latency_le_1.15x_baseline" if eligible else "all_recursive_unmatched_fallback",
        "selected_validation": chosen,
        "primary_baseline_validation": base,
        "all_candidate_validation": candidates,
        "selection_used_development": False, "selection_used_confirmation": False,
    }


def bundle_for_config(cid: str, seed: int, out: Path) -> SolverBundle | None:
    if cid.startswith("fp64_n"):
        return recursive_bundle("fp_recursive_dim64", seed, int(cid.split("n")[-1]), out, reserve=True)
    if cid.startswith("fp_n"):
        return recursive_bundle("fp_recursive", seed, int(cid.split("n")[-1]), out)
    if cid.startswith("ternary_n"):
        return recursive_bundle("ternary_recursive", seed, int(cid.split("n")[-1]), out)
    if cid.startswith("fp_search_r"):
        return search_bundle("fp", seed, int(cid.rsplit("r", 1)[1]), out)
    if cid.startswith("ternary_search_r"):
        return search_bundle("ternary", seed, int(cid.rsplit("r", 1)[1]), out)
    if cid == "single_pass_fp": return single_pass_bundle(seed, out, int8=False)
    if cid == "single_pass_int8": return single_pass_bundle(seed, out, int8=True, audit_sink=out / "int8_conversion_audit.jsonl")
    if cid == "symbolic_exact": return symbolic_bundle()
    raise ValueError(cid)


def evaluate_config_set(configs: Iterable[str], ds: GridDataset, split: str, out: Path, *, cost_n: int, repeats: int) -> list[dict[str, Any]]:
    rows_path = out / f"{split}_rows.jsonl"
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    configs = [c for c in configs if not (c in seen or seen.add(c))]
    for cid in configs:
        if cid == "symbolic_exact":
            rows.extend(evaluate_bundle(symbolic_bundle(), ds, split, cost_n=cost_n, repeats=repeats, out_rows=rows_path)); continue
        for seed in MODEL_SEEDS:
            bundle = bundle_for_config(cid, seed, out)
            if bundle is None:
                continue
            rows.extend(evaluate_bundle(bundle, ds, split, cost_n=cost_n, repeats=repeats, out_rows=rows_path))
    return rows


# ---------------------------------------------------------------------------
# Paired hierarchical uncertainty and practical gate.
# ---------------------------------------------------------------------------
def paired_arrays(candidate_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], *, latency_only: bool = False) -> dict[int, tuple[np.ndarray, np.ndarray | None, np.ndarray | None]]:
    base = {(int(r["seed"]), str(r["example_id"])): r for r in baseline_rows if r["seed"] != "symbolic"}
    by_seed: dict[int, list[tuple[float, float | None, float | None]]] = {}
    for r in candidate_rows:
        if r["seed"] == "symbolic": continue
        key = (int(r["seed"]), str(r["example_id"]))
        b = base.get(key)
        if b is None: continue
        cl = r.get("latency_ms"); bl = b.get("latency_ms")
        if latency_only and (cl is None or bl is None): continue
        by_seed.setdefault(key[0], []).append((float(r["semantic_success"])-float(b["semantic_success"]), None if cl is None else float(cl), None if bl is None else float(bl)))
    out = {}
    for seed, vals in by_seed.items():
        q = np.asarray([v[0] for v in vals], dtype=np.float64)
        c = np.asarray([v[1] for v in vals if v[1] is not None and v[2] is not None], dtype=np.float64)
        b = np.asarray([v[2] for v in vals if v[1] is not None and v[2] is not None], dtype=np.float64)
        out[seed] = (q, c if len(c) else None, b if len(b) else None)
    return out


def paired_effect(candidate_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], *, bootstraps: int = BOOTSTRAPS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    data = paired_arrays(candidate_rows, baseline_rows)
    seeds = sorted(data)
    if len(seeds) != len(MODEL_SEEDS):
        raise ExperimentStop(f"paired effect expected five training seeds, got {seeds}")
    quality_point = float(np.mean(np.concatenate([data[s][0] for s in seeds])))
    c_lats = np.concatenate([data[s][1] for s in seeds if data[s][1] is not None])
    b_lats = np.concatenate([data[s][2] for s in seeds if data[s][2] is not None])
    if not len(c_lats) or not len(b_lats): raise ExperimentStop("paired latency rows missing")
    latency_point = float(np.median(c_lats) / np.median(b_lats))
    rng = np.random.default_rng(seed)
    q_boot = np.empty(bootstraps, dtype=np.float64); r_boot = np.empty(bootstraps, dtype=np.float64)
    seed_arr = np.asarray(seeds, dtype=np.int64)
    for bi in range(bootstraps):
        picked = rng.choice(seed_arr, size=len(seed_arr), replace=True)
        q_parts=[]; c_parts=[]; b_parts=[]
        for s0 in picked:
            q, c, b = data[int(s0)]
            qi = rng.integers(0, len(q), size=len(q)); q_parts.append(q[qi])
            if c is not None and b is not None:
                li = rng.integers(0, len(c), size=len(c)); c_parts.append(c[li]); b_parts.append(b[li])
        q_boot[bi] = float(np.mean(np.concatenate(q_parts)))
        r_boot[bi] = float(np.median(np.concatenate(c_parts)) / np.median(np.concatenate(b_parts)))
    return {
        "n_training_seeds": len(seeds), "training_seeds": seeds,
        "paired_examples_per_seed": {str(s): int(len(data[s][0])) for s in seeds},
        "paired_latency_examples_per_seed": {str(s): int(0 if data[s][1] is None else len(data[s][1])) for s in seeds},
        "quality_difference": quality_point,
        "quality_ci95": [float(np.quantile(q_boot, .025)), float(np.quantile(q_boot, .975))],
        "latency_ratio_candidate_over_baseline": latency_point,
        "latency_ratio_ci95": [float(np.quantile(r_boot, .025)), float(np.quantile(r_boot, .975))],
        "bootstrap_replicates": bootstraps, "bootstrap_seed": seed,
        "quality_effect_unit": "paired_semantic_success_difference",
        "latency_effect_unit": "paired_complete_solve_median_ratio",
    }


def practical_gate(effect: dict[str, Any]) -> dict[str, Any]:
    q = float(effect["quality_difference"]); qlo = float(effect["quality_ci95"][0])
    r = float(effect["latency_ratio_candidate_over_baseline"]); rhi = float(effect["latency_ratio_ci95"][1])
    path_a = q >= .03 and qlo > 0.0 and r <= 1.15 and rhi <= 1.20
    path_b = q >= -.02 and qlo >= -.03 and r <= .65 and rhi <= .75
    return {
        "pass": bool(path_a or path_b), "path": "quality_superiority" if path_a else ("quality_cost_tradeoff" if path_b else None),
        "path_a": {"pass": path_a, "min_quality_diff": .03, "quality_ci_lower_gt": 0.0, "max_latency_ratio": 1.15, "max_latency_ci_upper": 1.20},
        "path_b": {"pass": path_b, "min_quality_diff": -.02, "min_quality_ci_lower": -.03, "max_latency_ratio": .65, "max_latency_ci_upper": .75},
        "observed": effect,
    }


# ---------------------------------------------------------------------------
# Validation, development, reserve intervention, confirmation.
# ---------------------------------------------------------------------------
def run_validation(val_ds: GridDataset, out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows_path = out / "validation_rows.jsonl"
    if rows_path.exists(): rows_path.unlink()
    rows: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        for n in RECURSIVE_BUDGETS:
            rows.extend(evaluate_bundle(recursive_bundle("fp_recursive", seed, n, out), val_ds, "validation", cost_n=VALIDATION_COST_N, repeats=LATENCY_REPEATS, out_rows=rows_path))
            rows.extend(evaluate_bundle(recursive_bundle("ternary_recursive", seed, n, out), val_ds, "validation", cost_n=VALIDATION_COST_N, repeats=LATENCY_REPEATS, out_rows=rows_path))
        for r in SEARCH_ROLLOUTS:
            rows.extend(evaluate_bundle(search_bundle("fp", seed, r, out), val_ds, "validation", cost_n=VALIDATION_COST_N, repeats=LATENCY_REPEATS, out_rows=rows_path))
            rows.extend(evaluate_bundle(search_bundle("ternary", seed, r, out), val_ds, "validation", cost_n=VALIDATION_COST_N, repeats=LATENCY_REPEATS, out_rows=rows_path))
        rows.extend(evaluate_bundle(single_pass_bundle(seed, out, int8=False), val_ds, "validation", cost_n=VALIDATION_COST_N, repeats=LATENCY_REPEATS, out_rows=rows_path))
        q = single_pass_bundle(seed, out, int8=True, audit_sink=out / "int8_conversion_audit.jsonl")
        if q is not None:
            rows.extend(evaluate_bundle(q, val_ds, "validation", cost_n=VALIDATION_COST_N, repeats=LATENCY_REPEATS, out_rows=rows_path))
    rows.extend(evaluate_bundle(symbolic_bundle(), val_ds, "validation", cost_n=VALIDATION_COST_N, repeats=LATENCY_REPEATS, out_rows=rows_path))
    agg = all_aggregates(rows); selection = choose_candidate(agg)
    write_json(out / "validation_operating_points.json", agg); write_csv(out / "validation_operating_points.csv", agg)
    write_json(out / "candidate_selection.json", selection)
    return rows, agg, selection


def evaluate_development_candidate(candidate: str, dev_ds: GridDataset, out: Path, *, tag: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    path = out / f"development_rows_{tag}.jsonl"
    if path.exists(): path.unlink()
    configs = list(CONTEXT_CONFIGS) + [candidate]
    rows: list[dict[str, Any]] = []
    for cid in dict.fromkeys(configs):
        if cid == "single_pass_int8":
            # It may be unsupported per seed; bundle_for_config handles that.
            pass
        if cid == "symbolic_exact":
            rows.extend(evaluate_bundle(symbolic_bundle(), dev_ds, "development", cost_n=DEVELOPMENT_COST_N, repeats=LATENCY_REPEATS, out_rows=path)); continue
        for seed in MODEL_SEEDS:
            b = bundle_for_config(cid, seed, out)
            if b is not None:
                rows.extend(evaluate_bundle(b, dev_ds, "development", cost_n=DEVELOPMENT_COST_N, repeats=LATENCY_REPEATS, out_rows=path))
    cand = [r for r in rows if r["config_id"] == candidate]
    base = [r for r in rows if r["config_id"] == PRIMARY_BASELINE]
    effect = paired_effect(cand, base); gate = practical_gate(effect)
    write_json(out / f"development_effect_{tag}.json", effect); write_json(out / f"development_decision_{tag}.json", gate)
    write_json(out / f"development_operating_points_{tag}.json", all_aggregates(rows))
    return rows, effect, gate


def train_reserve(train_ds: GridDataset, val_ds: GridDataset, out: Path) -> list[dict[str, Any]]:
    runs=[]
    for seed in MODEL_SEEDS: runs.append(train_one("fp_recursive_dim64", seed, train_ds, val_ds, out))
    return runs


def reserve_validation(val_ds: GridDataset, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = out / "reserve_validation_rows.jsonl"
    if path.exists(): path.unlink()
    rows=[]
    for seed in MODEL_SEEDS:
        for n in RECURSIVE_BUDGETS:
            rows.extend(evaluate_bundle(recursive_bundle("fp_recursive_dim64", seed, n, out, reserve=True), val_ds, "validation_reserve", cost_n=VALIDATION_COST_N, repeats=LATENCY_REPEATS, out_rows=path))
        rows.extend(evaluate_bundle(single_pass_bundle(seed, out, int8=False), val_ds, "validation_reserve", cost_n=VALIDATION_COST_N, repeats=LATENCY_REPEATS, out_rows=path))
    agg=all_aggregates(rows); sel=choose_candidate(agg, reserve=True)
    write_json(out / "reserve_validation_operating_points.json", agg); write_json(out / "reserve_candidate_selection.json", sel)
    return rows, sel


def freeze_candidate(candidate: str, out: Path) -> dict[str, Any]:
    files=[]
    for seed in MODEL_SEEDS:
        if candidate.startswith("fp64_"): ck = out / "checkpoints" / f"fp_recursive_dim64_seed{seed}.pt"
        elif candidate.startswith("fp_"): ck = out / "checkpoints" / f"fp_recursive_seed{seed}.pt"
        elif candidate.startswith("ternary_"): ck = out / "checkpoints" / f"ternary_recursive_seed{seed}.pt"
        else: raise ExperimentStop(f"cannot freeze candidate {candidate}")
        files.append({"role": "candidate_reasoner", "seed": seed, "path": str(ck), "sha256": sha256_file(ck)})
        if "search" in candidate:
            family = "fp" if candidate.startswith("fp_") else "ternary"
            vp=out/"verifiers"/f"{family}_seed{seed}.pt"; ap=out/"verifiers"/f"{family}_seed{seed}_fixed_actions.pt"
            files += [
                {"role":"candidate_verifier","seed":seed,"path":str(vp),"sha256":sha256_file(vp)},
                {"role":"candidate_fixed_actions","seed":seed,"path":str(ap),"sha256":sha256_file(ap)},
            ]
        bp=out/"checkpoints"/f"single_pass_seed{seed}.pt"
        files.append({"role":"primary_baseline","seed":seed,"path":str(bp),"sha256":sha256_file(bp)})
    rec={"candidate_config_id":candidate,"files":files,"frozen_before_confirmation":True,"git_sha":git_sha(),"timestamp_unix":time.time()}
    write_json(out/"confirmation_freeze_manifest.json",rec); return rec


def evaluate_confirmation(candidate: str, ds: GridDataset, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    path=out/"confirmation_rows.jsonl"
    if path.exists(): path.unlink()
    configs=list(CONTEXT_CONFIGS)+[candidate]
    rows=[]
    for cid in dict.fromkeys(configs):
        if cid=="symbolic_exact":
            rows.extend(evaluate_bundle(symbolic_bundle(),ds,"confirmation",cost_n=CONFIRMATION_COST_N,repeats=LATENCY_REPEATS,out_rows=path)); continue
        for seed in MODEL_SEEDS:
            b=bundle_for_config(cid,seed,out)
            if b is not None: rows.extend(evaluate_bundle(b,ds,"confirmation",cost_n=CONFIRMATION_COST_N,repeats=LATENCY_REPEATS,out_rows=path))
    cand=[r for r in rows if r["config_id"]==candidate]; base=[r for r in rows if r["config_id"]==PRIMARY_BASELINE]
    effect=paired_effect(cand,base,seed=BOOTSTRAP_SEED+1); gate=practical_gate(effect)
    write_json(out/"confirmation_effect.json",effect); write_json(out/"confirmation_decision.json",gate)
    write_json(out/"confirmation_operating_points.json",all_aggregates(rows))
    return rows,effect,gate


def representative_energy(candidate: str, ds: GridDataset, out: Path, split: str) -> dict[str, Any]:
    records={"scope":"cpu_package_rapl_not_gpu_not_whole_system","split":split,"configs":{}}
    for cid in [candidate,PRIMARY_BASELINE]:
        bundle=bundle_for_config(cid,MODEL_SEEDS[0],out)
        if bundle is None: continue
        xs=torch.from_numpy(ds.inputs[:32]).long()
        def run():
            for i in range(len(xs)):
                a=bundle.solve(xs[i:i+1]); _=sudoku_correct(xs[i:i+1],a,2)
        rec=measure_energy_record(run,n_runs=1)
        records["configs"][cid]={**rec,"problems_per_window":len(xs),"joules_per_complete_solve":(float(rec["energy_joules"])/len(xs) if rec.get("available") and rec.get("energy_joules") is not None else None)}
    write_json(out/f"{split}_energy.json",records); return records


def failure_diagnosis(selection: dict[str, Any], effect: dict[str, Any], gate: dict[str, Any], out: Path, *, reserve: bool=False) -> dict[str, Any]:
    obs=effect
    reason=[]
    if float(obs["quality_difference"])<-.02: reason.append("candidate_quality_deficit_exceeds_tradeoff")
    if float(obs["quality_ci95"][0])<-.03: reason.append("quality_uncertainty_allows_large_deficit")
    if float(obs["latency_ratio_candidate_over_baseline"])>.65: reason.append("latency_reduction_below_tradeoff_target")
    if float(obs["latency_ratio_ci95"][1])>.75: reason.append("latency_uncertainty_misses_tradeoff_target")
    rec={
        "reserve_intervention":reserve,"selected":selection.get("selected_config_id"),"gate_pass":gate["pass"],
        "diagnostic_reasons":reason,
        "falsifiable_intervention_if_main_failure":("increase recursive hidden dim 48->64 at fixed data/steps/objective while leaving baseline unchanged" if not reserve else None),
        "thresholds_changed":False,"baseline_changed":False,
    }
    write_json(out/("reserve_failure_diagnosis.json" if reserve else "main_failure_diagnosis.json"),rec); return rec


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=".m14/experiment"); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    env=measurement_environment(backend="pytorch_cpu_complete_solve",compiler_flags=None,extra={
        "milestone":14,"git_sha":git_sha(),"primary_cost":"complete_solve_latency","protocol":"docs/M14_PROTOCOL.md",
    })
    write_json(out/"environment.json",env); write_json(out/"energy_inventory.json",energy_counter_inventory())
    train_ds,val_ds,dev_ds,dev_manifest=build_development_data(out)

    training=train_main_families(train_ds,val_ds,out)
    verifier_runs=fit_main_verifiers(train_ds,val_ds,out)
    _,val_agg,selection=run_validation(val_ds,out)
    candidate=str(selection["selected_config_id"])
    dev_rows,dev_effect,dev_gate=evaluate_development_candidate(candidate,dev_ds,out,tag="main")
    representative_energy(candidate,dev_ds,out,"development_main")
    final_candidate=candidate; final_dev_effect=dev_effect; final_dev_gate=dev_gate; intervention_used=False

    if not dev_gate["pass"]:
        failure_diagnosis(selection,dev_effect,dev_gate,out,reserve=False)
        intervention_used=True
        reserve_runs=train_reserve(train_ds,val_ds,out)
        _,reserve_sel=reserve_validation(val_ds,out)
        reserve_candidate=str(reserve_sel["selected_config_id"])
        _,reserve_effect,reserve_gate=evaluate_development_candidate(reserve_candidate,dev_ds,out,tag="reserve_dim64")
        representative_energy(reserve_candidate,dev_ds,out,"development_reserve_dim64")
        failure_diagnosis(reserve_sel,reserve_effect,reserve_gate,out,reserve=True)
        final_candidate=reserve_candidate; final_dev_effect=reserve_effect; final_dev_gate=reserve_gate
    else:
        reserve_runs=[]

    confirmation_opened=False; confirmation_gate=None; confirmation_effect=None; confirmation_manifest=None; confirmation_audit=None
    if final_dev_gate["pass"]:
        freeze_candidate(final_candidate,out)
        confirm_ds,confirmation_manifest,confirmation_audit=build_confirmation_data(out,dev_manifest)
        confirmation_opened=True
        _,confirmation_effect,confirmation_gate=evaluate_confirmation(final_candidate,confirm_ds,out)
        representative_energy(final_candidate,confirm_ds,out,"confirmation")

    status="COMPLETE" if confirmation_gate is not None and confirmation_gate["pass"] else "INCOMPLETE"
    result={
        "milestone":14,"status":status,"claim_confirmed":bool(status=="COMPLETE"),
        "protocol_committed_before_results":True,"primary_metric":"strict_sudoku_semantic_validity",
        "primary_baseline":PRIMARY_BASELINE,"selected_candidate":final_candidate,
        "development_gate":final_dev_gate,"development_effect":final_dev_effect,
        "reserve_intervention_used":intervention_used,"reserve_runs":reserve_runs,
        "confirmation_opened":confirmation_opened,"confirmation_gate":confirmation_gate,
        "confirmation_effect":confirmation_effect,
        "confirmation_seed":CONFIRM_DATA_SEED if confirmation_opened else None,
        "reserve_confirmation_seed_unopened":RESERVE_CONFIRM_DATA_SEED,
        "n_training_seeds":len(MODEL_SEEDS),"training_seeds":MODEL_SEEDS,
        "train_examples":TRAIN_N,"validation_examples":VAL_N,"development_examples":DEV_N,
        "confirmation_examples":CONFIRM_N if confirmation_opened else 0,
        "training_steps_per_main_model":TRAIN_STEPS,"batch_size":BATCH_SIZE,
        "all_main_training_runs":len(training),"verifier_runs":len(verifier_runs),
        "energy_scope":"cpu_package_if_valid_otherwise_unavailable_not_gpu_not_whole_system",
        "latency_match_tolerance":"validation median within +/-15% only; unmatched points are not iso-budget",
        "scaling_law_claim":False,
        "confirmation_manifest_opened_only_after_development_pass":confirmation_opened,
        "thresholds_changed_after_results":False,"baseline_weakened":False,"favorable_seed_filtering":False,
    }
    write_json(out/"summary.json",result)
    return 0 if status=="COMPLETE" else 2


if __name__=="__main__":
    raise SystemExit(main())
