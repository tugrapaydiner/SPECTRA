#!/usr/bin/env python3
"""Prepare strict reasoner/verifier checkpoints and target-free inputs for M12.

This is fixture preparation, not RL training.  The actual adaptive-policy update is
performed only by ``scripts/train_adaptive_rl.py``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.seed import set_seed
from data.datasets import GridDataset, build_sudoku_arrays
from eval.checkpoint_eval import load_research_trm_checkpoint, sha256_file
from eval.grounded_checkpoint import (
    load_grounded_verifier_checkpoint,
    save_grounded_verifier_checkpoint,
)
from eval.grounded_targets import (
    generate_trajectory_states,
    grounding_metadata,
    tensor_state_sha256,
)
from model.grounded_verifier import GroundedStateVerifier
from model.trm import TRM
from train.checkpoint import atomic_torch_save
from train.distill import grounded_improvement_bce_loss
from train.trainer import TrainConfig, Trainer

SEED = 20260912
REASONER_SEED = 12101
VERIFIER_SEED = 12203
REASONER_STEPS = 120
VERIFIER_STEPS = 120
VERIFIER_BATCH = 64
REASONER_TRAIN = 384
REASONER_VAL = 96
RL_TRAIN = 256
RL_HELDOUT = 96
VERIFIER_PUZZLES = 160
VERIFIER_VAL_PUZZLES = 48


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def unique_rows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seen: set[bytes] = set()
    xs, ys = [], []
    for a, b in zip(x, y):
        key = np.ascontiguousarray(a).tobytes()
        if key in seen:
            continue
        seen.add(key); xs.append(a); ys.append(b)
    return np.stack(xs), np.stack(ys)


def make_datasets() -> tuple[dict[str, GridDataset], torch.Tensor, torch.Tensor, dict[str, int]]:
    rng = np.random.default_rng(SEED)
    # One deterministic generation stream, then an explicit disjoint row partition.
    # Generate excess rows so exact duplicate inputs can be discarded before split.
    x, y, h, w = build_sudoku_arrays(
        2, 1152, 8, rng, require_unique=True, augment=True,
        min_clues=7, max_clues=9,
    )
    x, y = unique_rows(x, y)
    needed = REASONER_TRAIN + REASONER_VAL + RL_TRAIN + RL_HELDOUT
    if len(x) < needed:
        raise RuntimeError(f"insufficient unique pilot puzzles: {len(x)} < {needed}")
    x, y = x[:needed], y[:needed]
    a = 0
    rt_x, rt_y = x[a:a+REASONER_TRAIN], y[a:a+REASONER_TRAIN]; a += REASONER_TRAIN
    rv_x, rv_y = x[a:a+REASONER_VAL], y[a:a+REASONER_VAL]; a += REASONER_VAL
    rl_x = torch.from_numpy(x[a:a+RL_TRAIN].copy()).long(); a += RL_TRAIN
    rh_x = torch.from_numpy(x[a:a+RL_HELDOUT].copy()).long(); a += RL_HELDOUT

    def ds(xx, yy, prefix):
        return GridDataset(
            xx, yy, h, w, task="sudoku", num_tokens=5, pad_token=None,
            ids=[f"{prefix}-{i}" for i in range(len(xx))],
            group_ids=[f"{prefix}-{i}" for i in range(len(xx))],
        )
    return {
        "train": ds(rt_x, rt_y, "reasoner-train"),
        "validation": ds(rv_x, rv_y, "reasoner-val"),
    }, rl_x, rh_x, {
        "generated": 1152,
        "unique": int(len(unique_rows(*build_sudoku_arrays(2, 1, 8, np.random.default_rng(SEED+999), True, False))[0])) if False else int(len(x)),
        "reasoner_train": REASONER_TRAIN,
        "reasoner_validation": REASONER_VAL,
        "rl_train": RL_TRAIN,
        "rl_heldout": RL_HELDOUT,
    }


def train_reasoner(out: Path, datasets: dict[str, GridDataset]):
    set_seed(REASONER_SEED, deterministic=True)
    model = TRM(
        dim=32, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1, N_sup=4,
        heads=4, alpha_y=0.1, alpha_z=0.1, max_grid_size=8,
        ternary=False, act8=False,
    )
    cfg = TrainConfig(
        lr=1e-3, weight_decay=0.01, batch_size=32, max_steps=REASONER_STEPS,
        lr_warmup_steps=12, quant_warmup_steps=0, clip_grad_norm=1.0,
        ema_decay=0.999, log_every=20, eval_every=60, eval_batches=2,
        ckpt_every=0, seed=REASONER_SEED, train_seed=REASONER_SEED+11,
        eval_seed=REASONER_SEED+29, device="cpu", precision="fp32",
        backend="pytorch_eager", deterministic=True,
    )
    run_config = {
        "seed": REASONER_SEED,
        "device": "cpu",
        "model": {
            "dim": 32, "n_layers": 1, "heads": 4, "n": 1, "T": 1,
            "N_sup": 4, "alpha_y": 0.1, "alpha_z": 0.1,
        },
        "data": {
            "num_tokens": 5, "seq_len": 16, "height": 4, "width": 4,
            "box": 2, "min_clues": 7, "max_clues": 9,
            "require_unique": True, "augment": True,
        },
        "train": {
            "lr": cfg.lr, "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size, "max_steps": cfg.max_steps,
        },
    }
    trainer = Trainer(model, datasets["train"], datasets["validation"], cfg, run_config=run_config)
    ckpt = out / "reasoner.pt"
    t0 = time.perf_counter(); report = trainer.fit(checkpoint_path=ckpt); elapsed = time.perf_counter() - t0
    core = load_research_trm_checkpoint(ckpt, device="cpu", weight_identity="recorded")
    for p in core.model.parameters(): p.requires_grad_(False)
    core.model.eval()
    return core, {"checkpoint": str(ckpt), "sha256": core.sha256, "elapsed_seconds": elapsed, "fit": report}


def rows_to_tensors(rows):
    return (
        torch.stack([r["x"] for r in rows]).long(),
        torch.stack([r["y"] for r in rows]).float(),
        torch.stack([r["z"] for r in rows]).float(),
        torch.tensor([r["label"] for r in rows], dtype=torch.float32),
    )


def train_verifier(out: Path, core, datasets: dict[str, GridDataset]):
    frozen_before = tensor_state_sha256(core.model)
    tx = torch.from_numpy(datasets["train"].inputs[:VERIFIER_PUZZLES]).long()
    vx = torch.from_numpy(datasets["validation"].inputs[:VERIFIER_VAL_PUZZLES]).long()
    train_rows = generate_trajectory_states(
        core.model, tx, [f"vtrain-{i}" for i in range(len(tx))],
        max_depth=4, height=4, width=4, box=2,
    )
    val_rows = generate_trajectory_states(
        core.model, vx, [f"vval-{i}" for i in range(len(vx))],
        max_depth=4, height=4, width=4, box=2,
    )
    x, y, z, labels = rows_to_tensors(train_rows)
    vx_t, vy_t, vz_t, vlabels = rows_to_tensors(val_rows)
    positives = int(labels.sum())
    counts = {"positive": positives, "negative": int(labels.numel()) - positives, "total": int(labels.numel())}
    if min(counts["positive"], counts["negative"]) == 0:
        raise RuntimeError(f"M12 verifier fixture target is one-class: {counts}")

    set_seed(VERIFIER_SEED, deterministic=True)
    verifier = GroundedStateVerifier(
        num_tokens=5, dim=32, n_layers=1, heads=4, max_grid_size=8,
        act_bits=8, include_y=True,
    )
    opt = torch.optim.AdamW(verifier.parameters(), lr=2e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(VERIFIER_SEED + 77)
    curve = []
    t0 = time.perf_counter()
    for step in range(1, VERIFIER_STEPS + 1):
        idx = torch.randint(0, len(train_rows), (VERIFIER_BATCH,), generator=gen)
        verifier.train()
        loss, _ = grounded_improvement_bce_loss(
            verifier, x[idx], y[idx], z[idx], labels[idx], width=4,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite verifier loss at step {step}")
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(verifier.parameters(), 1.0); opt.step()
        if step in {1, 30, 60, 90, 120}:
            verifier.eval()
            with torch.no_grad():
                vp = verifier(vx_t, vy_t, vz_t, width=4)
                vbce = torch.nn.functional.binary_cross_entropy(vp, vlabels)
                vacc = ((vp >= 0.5) == (vlabels >= 0.5)).float().mean()
            curve.append({"step": step, "train_bce": float(loss.detach()), "validation_bce": float(vbce), "validation_accuracy": float(vacc)})
    elapsed = time.perf_counter() - t0
    verifier.eval()
    provenance = {
        "purpose": "m12_strict_grounded_fixture",
        "data_seed": SEED,
        "reasoner_checkpoint_sha256": core.sha256,
        "reference_target_used": False,
        "train_class_counts": counts,
        "train_puzzles": VERIFIER_PUZZLES,
        "validation_puzzles": VERIFIER_VAL_PUZZLES,
        "trajectory_depths": 4,
    }
    ckpt = out / "grounded_verifier.pt"
    save_grounded_verifier_checkpoint(
        verifier, ckpt, core=core, trained_steps=VERIFIER_STEPS,
        grounding=grounding_metadata(), member_seeds=[VERIFIER_SEED],
        data_provenance=provenance,
    )
    loaded = load_grounded_verifier_checkpoint(ckpt, core)
    for p in loaded.module.parameters(): p.requires_grad_(False)
    loaded.module.eval()
    frozen_after = tensor_state_sha256(core.model)
    if frozen_before != frozen_after:
        raise RuntimeError("reasoner changed during grounded-verifier fixture training")
    return loaded, {
        "checkpoint": str(ckpt), "sha256": loaded.sha256,
        "elapsed_seconds": elapsed, "curve": curve, "class_counts": counts,
        "reasoner_hash_unchanged": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/m12_fixture")
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    set_seed(SEED, deterministic=True)

    datasets, rl_train, rl_heldout, split_counts = make_datasets()
    core, reasoner_report = train_reasoner(out, datasets)
    grounded, verifier_report = train_verifier(out, core, datasets)

    inputs_path = out / "rl_inputs.pt"
    atomic_torch_save({
        "format": "spectra.m12_rl_inputs",
        "version": 1,
        "task": "sudoku",
        "height": 4,
        "width": 4,
        "box": 2,
        "num_tokens": 5,
        "train_inputs": rl_train,
        "heldout_inputs": rl_heldout,
        "reference_targets_included": False,
        "data_seed": SEED,
        "split_policy": "single_unique_pool_then_disjoint_partition",
    }, inputs_path)
    summary = {
        "milestone": 12,
        "status": "fixture_ready",
        "reasoner": reasoner_report,
        "grounded_verifier": verifier_report,
        "inputs": {
            "path": str(inputs_path),
            "sha256": sha256_file(inputs_path),
            "reference_targets_included": False,
            "train_examples": int(rl_train.shape[0]),
            "heldout_examples": int(rl_heldout.shape[0]),
        },
        "splits": split_counts,
    }
    write_json(out / "fixture_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
