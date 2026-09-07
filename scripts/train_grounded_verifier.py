#!/usr/bin/env python3
"""M07: train and evaluate an independently grounded full-state verifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config
from common.seed import set_seed
from eval.checkpoint_eval import load_research_trm_checkpoint, sha256_file
from eval.grounded_checkpoint import (
    load_grounded_verifier_checkpoint,
    save_grounded_verifier_checkpoint,
    save_z_only_ablation_checkpoint,
)
from eval.grounded_targets import (
    LABEL_MARGIN,
    assert_frozen_reasoner,
    generate_trajectory_states,
    grounding_metadata,
    one_cycle_improvement_target,
    tensor_state_sha256,
)
from model.grounded_verifier import (
    EnsembleGroundedStateVerifier,
    GroundedStateVerifier,
)
from scripts._common import build_data_splits, build_trm
from train.distill import grounded_improvement_bce_loss
from train.trainer import TrainConfig, Trainer

DATA_SEED = 20260907
REASONER_SEED = 1701
VERIFIER_MEMBER_SEEDS = [7101, 7202, 7303]
ABLATION_MEMBER_SEEDS = [8101, 8202, 8303]
TRAIN_N, VAL_N, TEST_N = 256, 64, 96
TRAJ_TRAIN_PUZZLES = 192
TRAJ_VAL_PUZZLES = 48
TRAJ_TEST_PUZZLES = 64
TRAIN_DEPTHS = 4
TEST_DEPTHS = 8
REASONER_STEPS = 200
VERIFIER_STEPS = 300
VERIFIER_BATCH = 64
VERIFIER_LR = 2e-3
VERIFIER_WD = 0.01
PERTURB_SEED = 20260717
PERTURB_RELATIVE_SCALE = 0.10


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_plain(row), sort_keys=True) + "\n")


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
                cpu_model = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    return {
        "git_sha": git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "experiment_device": "cpu",
    }


def id_hash(ids: Iterable[str]) -> str:
    h = hashlib.sha256()
    for item in ids:
        b = str(item).encode("utf-8")
        h.update(len(b).to_bytes(8, "little")); h.update(b)
    return h.hexdigest()


def rows_to_tensors(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, ...]:
    if not rows:
        raise ValueError("empty verifier row collection")
    x = torch.stack([r["x"] for r in rows]).long()
    y = torch.stack([r["y"] for r in rows]).float()
    z = torch.stack([r["z"] for r in rows]).float()
    labels = torch.tensor([float(r["label"]) for r in rows], dtype=torch.float32)
    return x, y, z, labels


def class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    pos = sum(int(float(r["label"]) == 1.0) for r in rows)
    return {"positive": pos, "negative": len(rows) - pos, "total": len(rows)}


def _pairwise_auc(labels: torch.Tensor, probs: torch.Tensor) -> float:
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    comp = pos[:, None] - neg[None, :]
    return float(((comp > 0).float() + 0.5 * (comp == 0).float()).mean().item())


def _average_precision(labels: torch.Tensor, probs: torch.Tensor) -> float:
    n_pos = int((labels == 1).sum().item())
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(probs, descending=True)
    y = labels[order]
    tp = torch.cumsum(y, 0)
    precision = tp / torch.arange(1, len(y) + 1, dtype=torch.float32)
    return float((precision * y).sum().item() / n_pos)


def _ece(labels: torch.Tensor, probs: torch.Tensor, bins: int = 10) -> float:
    total = max(1, labels.numel())
    out = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        n = int(mask.sum().item())
        if n:
            conf = float(probs[mask].mean().item())
            acc = float(labels[mask].mean().item())
            out += n / total * abs(conf - acc)
    return out


def binary_metrics(labels: torch.Tensor, probs: torch.Tensor) -> dict[str, Any]:
    labels = labels.detach().cpu().float().reshape(-1)
    probs = probs.detach().cpu().float().reshape(-1)
    pred = probs >= 0.5
    truth = labels == 1
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    tn = int((~pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    predicted_pos = tp + fp
    actual_neg = tn + fp
    return {
        "n": int(labels.numel()),
        "positive": int(truth.sum()),
        "negative": int((~truth).sum()),
        "positive_rate": float(labels.mean().item()),
        "roc_auc": _pairwise_auc(labels, probs),
        "average_precision": _average_precision(labels, probs),
        "brier": float(((probs - labels) ** 2).mean().item()),
        "ece_10": _ece(labels, probs, 10),
        "accuracy_at_0_5": float((pred == truth).float().mean().item()),
        "false_acceptance_rate": float(fp / predicted_pos) if predicted_pos else float("nan"),
        "false_positive_rate": float(fp / actual_neg) if actual_neg else float("nan"),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "mean_probability": float(probs.mean().item()),
    }


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().float().reshape(-1)
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and float(values[order[j]]) == float(values[order[i]]):
            j += 1
        rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    if float(ra.std(unbiased=False)) == 0.0 or float(rb.std(unbiased=False)) == 0.0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1].item())


def make_cfg():
    overrides = [
        f"seed={DATA_SEED}", "device=cpu",
        "data.min_clues=30", "data.max_clues=35", "data.augment=true",
        "model.dim=48", "model.n_layers=1", "model.heads=4", "model.n=1",
        "model.T=1", "model.N_sup=2", "model.alpha_y=0.1", "model.alpha_z=0.1",
        "train.lr=0.001", "train.weight_decay=0.01", "train.batch_size=32",
        f"train.max_steps={REASONER_STEPS}", "train.lr_warmup_steps=20",
        "train.clip_grad_norm=1.0", "train.ema_decay=0.999",
        "train.eval_every=100", "train.eval_batches=2", "train.ckpt_every=0",
        "train.precision=fp32", "train.backend=pytorch_eager", "train.deterministic=true",
    ]
    return load_config("config/sudoku.yaml", overrides=overrides)


def train_reasoner(out: Path, cfg, train_ds, val_ds) -> tuple[Any, dict[str, Any]]:
    set_seed(REASONER_SEED, deterministic=True)
    model = build_trm(cfg, ternary=False, act8=False)
    tcfg = TrainConfig.from_config(cfg)
    tcfg.seed = REASONER_SEED
    tcfg.train_seed = REASONER_SEED + 11
    tcfg.eval_seed = REASONER_SEED + 29
    trainer = Trainer(model, train_ds, val_ds, tcfg, run_config=cfg)
    ckpt = out / "reasoner.pt"
    t0 = time.perf_counter()
    result = trainer.fit(checkpoint_path=ckpt)
    elapsed = time.perf_counter() - t0
    write_json(out / "reasoner_training.json", {"elapsed_seconds": elapsed, **result})
    core = load_research_trm_checkpoint(ckpt, device="cpu", weight_identity="recorded")
    for p in core.model.parameters():
        p.requires_grad_(False)
    core.model.eval()
    assert_frozen_reasoner(core.model)
    return core, {"checkpoint": str(ckpt), "sha256": core.sha256, "training": result, "elapsed_seconds": elapsed}


def _subset_rows(rows: list[dict[str, Any]], *, depths: set[int]) -> list[dict[str, Any]]:
    return [r for r in rows if int(r["depth"]) in depths]


def train_ensemble(
    *,
    include_y: bool,
    seeds: list[int],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    curve_dir: Path,
) -> tuple[EnsembleGroundedStateVerifier, dict[str, Any]]:
    x, y, z, labels = rows_to_tensors(train_rows)
    vx, vy, vz, vlabels = rows_to_tensors(val_rows)
    ensemble = EnsembleGroundedStateVerifier(
        num_tokens=10, dim=48, n_members=len(seeds), n_layers=1, heads=4,
        max_grid_size=32, act_bits=8, include_y=include_y,
    )
    member_reports = []
    for i, seed in enumerate(seeds):
        set_seed(seed, deterministic=True)
        member = GroundedStateVerifier(
            num_tokens=10, dim=48, n_layers=1, heads=4,
            max_grid_size=32, act_bits=8, include_y=include_y,
        )
        opt = torch.optim.AdamW(member.parameters(), lr=VERIFIER_LR, weight_decay=VERIFIER_WD)
        rng = np.random.default_rng(seed + 404)
        curve = curve_dir / f"{'xyz' if include_y else 'z_only'}_member{i}_seed{seed}.jsonl"
        if curve.exists(): curve.unlink()
        first_loss = last_loss = None
        for step in range(1, VERIFIER_STEPS + 1):
            idx = torch.from_numpy(rng.integers(0, len(train_rows), size=VERIFIER_BATCH, dtype=np.int64))
            member.train()
            loss, comp = grounded_improvement_bce_loss(
                member, x[idx], y[idx], z[idx], labels[idx], width=9
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite verifier loss member={i} step={step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(member.parameters(), 1.0)
            if not torch.isfinite(torch.as_tensor(grad)):
                raise RuntimeError(f"non-finite verifier gradient member={i} step={step}")
            opt.step()
            first_loss = float(loss) if first_loss is None else first_loss
            last_loss = float(loss)
            if step == 1 or step % 25 == 0 or step == VERIFIER_STEPS:
                member.eval()
                with torch.inference_mode():
                    vp = member(vx, vy, vz, 9)
                vm = binary_metrics(vlabels, vp)
                append_jsonl(curve, {
                    "step": step, "train_bce": float(loss.detach()),
                    "grad_norm_preclip": float(grad), "validation": vm,
                    "include_y": include_y, "member_seed": seed,
                })
        ensemble.members[i].load_state_dict(member.state_dict(), strict=True)
        member_reports.append({
            "member": i, "seed": seed, "first_loss": first_loss,
            "last_loss": last_loss, "curve": str(curve),
        })
    ensemble.eval()
    return ensemble, {"members": member_reports}


def ensemble_predictions(
    ensemble: EnsembleGroundedStateVerifier,
    rows: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, z, labels = rows_to_tensors(rows)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(rows), 128):
            chunks.append(ensemble(x[start:start+128], y[start:start+128], z[start:start+128], 9))
    members = torch.cat(chunks, dim=1)
    return labels, members.mean(dim=0), members.std(dim=0, unbiased=False)


def perturb_rows(core_model, rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    if field not in {"y", "z"}:
        raise ValueError(field)
    gen = torch.Generator().manual_seed(PERTURB_SEED + (1 if field == "y" else 2))
    out = []
    for r in rows:
        nr = dict(r)
        t = r[field].clone()
        rms = float(t.square().mean().sqrt().item())
        scale = PERTURB_RELATIVE_SCALE * max(rms, 0.05)
        noise = torch.randn(t.shape, generator=gen, dtype=t.dtype) * scale
        nr[field] = t + noise
        result = one_cycle_improvement_target(
            core_model, nr["x"].unsqueeze(0), nr["y"].unsqueeze(0), nr["z"].unsqueeze(0)
        )
        nr["label"] = float(result["label"][0])
        nr["score_before"] = float(result["score_before"][0])
        nr["score_after"] = float(result["score_after"][0])
        nr["variant"] = f"perturb_{field}"
        out.append(nr)
    return out


def aliasing_audit(core_model, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_id.setdefault(str(r["id"]), []).append(r)
    changed = total = 0
    deltas = []
    for group in by_id.values():
        group = sorted(group, key=lambda r: int(r["depth"]))
        if len(group) < 2:
            continue
        for i, r in enumerate(group):
            alt = group[(i + 1) % len(group)]
            result = one_cycle_improvement_target(
                core_model, r["x"].unsqueeze(0), alt["y"].unsqueeze(0), r["z"].unsqueeze(0)
            )
            alt_label = float(result["label"][0])
            changed += int(alt_label != float(r["label"]))
            deltas.append(float(result["score_after"][0] - result["score_before"][0]))
            total += 1
    return {
        "counterfactual_pairs": total,
        "same_xz_changed_y_label_change_count": changed,
        "same_xz_changed_y_label_change_rate": changed / max(1, total),
        "mean_counterfactual_score_delta": float(np.mean(deltas)) if deltas else None,
        "interpretation": "counterfactual state-sufficiency stress test; not a claim of natural exact-z collisions",
    }


def disagreement_report(labels: torch.Tensor, probs: torch.Tensor, disagreement: torch.Tensor) -> dict[str, Any]:
    error = (probs - labels).abs()
    corr = spearman(disagreement, error)
    order = torch.argsort(disagreement)
    mid = max(1, len(order) // 2)
    low, high = order[:mid], order[mid:]
    return {
        "spearman_disagreement_vs_absolute_error": corr,
        "mean_abs_error_low_disagreement_half": float(error[low].mean()) if len(low) else None,
        "mean_abs_error_high_disagreement_half": float(error[high].mean()) if len(high) else None,
        "interpretation": "heuristic only; no M07 MCTS uncertainty penalty enabled",
    }


def save_prediction_rows(path: Path, rows: list[dict[str, Any]], probs: torch.Tensor, std: torch.Tensor) -> None:
    if path.exists(): path.unlink()
    for r, p, s in zip(rows, probs.tolist(), std.tolist()):
        append_jsonl(path, {
            "id": r["id"], "depth": r["depth"], "variant": r.get("variant", "real"),
            "label": r["label"], "score_before": r["score_before"], "score_after": r["score_after"],
            "probability": p, "ensemble_disagreement": s,
            "reference_target_used": False,
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/m07_grounded_verifier")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    write_json(out / "environment.json", environment_record())

    cfg = make_cfg()
    datasets, manifest = build_data_splits(
        cfg, TRAIN_N, VAL_N, TEST_N, seed=DATA_SEED, manifest_path=out / "data_manifest.json"
    )
    reasoner, reasoner_report = train_reasoner(out, cfg, datasets["train"], datasets["validation"])
    frozen_hash_before = tensor_state_sha256(reasoner.model)

    train_puzzles = torch.from_numpy(datasets["train"].inputs[:TRAJ_TRAIN_PUZZLES]).long()
    val_puzzles = torch.from_numpy(datasets["validation"].inputs[:TRAJ_VAL_PUZZLES]).long()
    # Deliberately copy only held-out inputs and IDs. Test reference targets are never passed below.
    test_puzzles = torch.from_numpy(datasets["test"].inputs[:TRAJ_TEST_PUZZLES]).long()
    train_ids = datasets["train"].ids[:TRAJ_TRAIN_PUZZLES]
    val_ids = datasets["validation"].ids[:TRAJ_VAL_PUZZLES]
    test_ids = datasets["test"].ids[:TRAJ_TEST_PUZZLES]

    train_rows = generate_trajectory_states(reasoner.model, train_puzzles, train_ids, max_depth=TRAIN_DEPTHS)
    val_rows = generate_trajectory_states(reasoner.model, val_puzzles, val_ids, max_depth=TRAIN_DEPTHS)
    test_rows_all = generate_trajectory_states(reasoner.model, test_puzzles, test_ids, max_depth=TEST_DEPTHS)
    test_in = _subset_rows(test_rows_all, depths=set(range(TRAIN_DEPTHS)))
    test_deep = _subset_rows(test_rows_all, depths=set(range(TRAIN_DEPTHS, TEST_DEPTHS)))

    counts = {
        "train": class_counts(train_rows), "validation": class_counts(val_rows),
        "heldout_in_distribution": class_counts(test_in), "heldout_deeper": class_counts(test_deep),
    }
    if min(counts["train"]["positive"], counts["train"]["negative"]) == 0:
        write_json(out / "summary.json", {"status": "stopped_one_class_training_target", "class_counts": counts})
        return 2
    if min(counts["heldout_in_distribution"]["positive"], counts["heldout_in_distribution"]["negative"]) == 0:
        write_json(out / "summary.json", {"status": "stopped_one_class_heldout_target", "class_counts": counts})
        return 2

    grounding = grounding_metadata()
    provenance = {
        "grounding": grounding,
        "reasoner_checkpoint_sha256": reasoner.sha256,
        "reasoner_tensor_state_sha256": frozen_hash_before,
        "data_seed": DATA_SEED,
        "split_duplicate_audit": manifest["duplicate_audit"],
        "puzzle_id_hashes": {
            "train": id_hash(train_ids), "validation": id_hash(val_ids), "test": id_hash(test_ids),
        },
        "class_counts": counts,
        "test_reference_target_used": False,
        "train_depths": list(range(TRAIN_DEPTHS)),
        "heldout_deeper_depths": list(range(TRAIN_DEPTHS, TEST_DEPTHS)),
    }
    write_json(out / "target_provenance.json", provenance)

    xyz, xyz_train_report = train_ensemble(
        include_y=True, seeds=VERIFIER_MEMBER_SEEDS, train_rows=train_rows,
        val_rows=val_rows, curve_dir=out / "curves",
    )
    zonly, z_train_report = train_ensemble(
        include_y=False, seeds=ABLATION_MEMBER_SEEDS, train_rows=train_rows,
        val_rows=val_rows, curve_dir=out / "curves",
    )

    save_grounded_verifier_checkpoint(
        xyz, out / "grounded_verifier.pt", core=reasoner, trained_steps=VERIFIER_STEPS,
        grounding=grounding, member_seeds=VERIFIER_MEMBER_SEEDS,
        data_provenance=provenance,
    )
    save_z_only_ablation_checkpoint(
        zonly, out / "z_only_ablation.pt", core=reasoner, trained_steps=VERIFIER_STEPS,
        grounding=grounding, member_seeds=ABLATION_MEMBER_SEEDS,
    )
    loaded = load_grounded_verifier_checkpoint(out / "grounded_verifier.pt", reasoner)
    xyz = loaded.module

    perturb_y = perturb_rows(reasoner.model, test_in, "y")
    perturb_z = perturb_rows(reasoner.model, test_in, "z")
    sets = {
        "heldout_in_distribution": test_in,
        "heldout_deeper": test_deep,
        "heldout_perturb_y": perturb_y,
        "heldout_perturb_z": perturb_z,
    }
    metrics: dict[str, Any] = {}
    primary_labels = primary_probs = primary_std = None
    for name, rows in sets.items():
        labels, probs, std = ensemble_predictions(xyz, rows)
        metrics[name] = binary_metrics(labels, probs)
        metrics[name]["disagreement"] = disagreement_report(labels, probs, std)
        save_prediction_rows(out / "predictions" / f"{name}.jsonl", rows, probs, std)
        if name == "heldout_in_distribution":
            primary_labels, primary_probs, primary_std = labels, probs, std

    z_labels, z_probs, z_std = ensemble_predictions(zonly, test_in)
    z_metrics = binary_metrics(z_labels, z_probs)
    z_metrics["disagreement"] = disagreement_report(z_labels, z_probs, z_std)
    metrics["z_only_ablation_heldout"] = z_metrics

    depth_metrics = {}
    for depth in range(TEST_DEPTHS):
        rows = [r for r in test_rows_all if int(r["depth"]) == depth]
        labels, probs, std = ensemble_predictions(xyz, rows)
        depth_metrics[str(depth)] = binary_metrics(labels, probs)
        depth_metrics[str(depth)]["mean_disagreement"] = float(std.mean())
    write_json(out / "depth_metrics.json", depth_metrics)

    alias = aliasing_audit(reasoner.model, test_in)
    write_json(out / "aliasing_audit.json", alias)

    frozen_hash_after = tensor_state_sha256(reasoner.model)
    freeze_audit = {
        "before": frozen_hash_before,
        "after": frozen_hash_after,
        "unchanged": frozen_hash_before == frozen_hash_after,
        "all_parameters_requires_grad_false": not any(p.requires_grad for p in reasoner.model.parameters()),
        "model_training_flag": bool(reasoner.model.training),
    }
    if not freeze_audit["unchanged"] or not freeze_audit["all_parameters_requires_grad_false"] or freeze_audit["model_training_flag"]:
        raise RuntimeError(f"frozen reasoner contract failed: {freeze_audit}")
    write_json(out / "freeze_audit.json", freeze_audit)

    assert primary_labels is not None and primary_probs is not None and primary_std is not None
    primary = metrics["heldout_in_distribution"]
    useful = bool(
        primary["roc_auc"] is not None and primary["roc_auc"] >= 0.60
        and primary["average_precision"] is not None
        and primary["positive"] > 0 and primary["negative"] > 0
    )
    comparison = {
        "full_state_roc_auc": primary["roc_auc"],
        "z_only_roc_auc": z_metrics["roc_auc"],
        "roc_auc_delta_full_minus_z_only": (
            float(primary["roc_auc"] - z_metrics["roc_auc"])
            if primary["roc_auc"] is not None and z_metrics["roc_auc"] is not None else None
        ),
    }
    write_json(out / "metrics.json", {"sets": metrics, "depths": depth_metrics, "comparison": comparison})
    write_json(out / "training_report.json", {"full_state": xyz_train_report, "z_only": z_train_report})

    exact_commands = [
        "python scripts/train_grounded_verifier.py --out outputs/m07_grounded_verifier",
        "python -m pytest tests/test_m07_grounded_verifier.py -q",
        "python -m pytest -m 'not slow' -ra",
    ]
    write_json(out / "exact_commands.json", exact_commands)

    summary = {
        "status": "complete" if useful else "complete_but_acceptance_not_established",
        "acceptance_useful_meaning": useful,
        "target": grounding,
        "state_representation": "search_state_xyz_v1",
        "reasoner": reasoner_report,
        "reasoner_frozen": freeze_audit,
        "class_counts": counts,
        "primary_heldout": primary,
        "z_only_ablation": z_metrics,
        "full_vs_z_only": comparison,
        "aliasing": alias,
        "deeper": metrics["heldout_deeper"],
        "perturb_y": metrics["heldout_perturb_y"],
        "perturb_z": metrics["heldout_perturb_z"],
        "ensemble_disagreement": metrics["heldout_in_distribution"]["disagreement"],
        "verifier_checkpoint": str(out / "grounded_verifier.pt"),
        "verifier_checkpoint_sha256": sha256_file(out / "grounded_verifier.pt"),
        "z_only_checkpoint": str(out / "z_only_ablation.pt"),
        "test_reference_target_used": False,
        "mcts_bootstrap_used_as_ground_truth": False,
        "exact_commands": exact_commands,
    }
    write_json(out / "summary.json", summary)

    md = [
        "# M07 Grounded Verifier Summary", "",
        f"- status: **{summary['status']}**",
        f"- target: `{grounding['target_id']}`",
        "- representation: `search_state_xyz_v1`",
        f"- held-out ROC AUC: `{primary['roc_auc']:.6f}`",
        f"- held-out average precision: `{primary['average_precision']:.6f}`",
        f"- held-out Brier: `{primary['brier']:.6f}`",
        f"- held-out ECE(10): `{primary['ece_10']:.6f}`",
        f"- false-acceptance rate @0.5: `{primary['false_acceptance_rate']}`",
        f"- z-only ROC AUC: `{z_metrics['roc_auc']:.6f}`",
        f"- alias counterfactual label-change rate: `{alias['same_xz_changed_y_label_change_rate']:.6f}`",
        f"- test reference answers used for labels: `{summary['test_reference_target_used']}`",
        f"- MCTS bootstrap used as ground truth: `{summary['mcts_bootstrap_used_as_ground_truth']}`",
        "",
        "The verifier estimates one-cycle oracle structural-score improvement, not eventual solve probability.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if useful else 3


if __name__ == "__main__":
    raise SystemExit(main())
