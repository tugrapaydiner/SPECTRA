#!/usr/bin/env python3
"""Milestone 15: bounded mechanism-focused ablations.

Protocol: docs/M15_PROTOCOL.md and docs/M15_PROTOCOL_CLARIFICATION.md.

The primary question is whether native latent MCTS misuses a verifier trained to
predict *one-cycle improvement* as though that probability were an absolute state
value.  The experiment keeps the recursive cores frozen, uses reference-free
symbolic Sudoku checks only for independent analysis/terminal guards, and opens a
fresh confirmation set only after the development analysis is frozen.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config
from common.seed import set_seed
from eval.grounded_targets import (
    assert_frozen_reasoner,
    generate_trajectory_states,
    one_cycle_improvement_target,
    tensor_state_sha256,
)
from eval.action_search import StateConditionedLatentNativeMCTS
from eval.latent_mcts import _LatentNode, _quantize_int8, _dequantize_int8
from model.grounded_verifier import GroundedStateVerifier, EnsembleGroundedStateVerifier
from model.latent_action import LatentActionCodebook, StateConditionedLatentActionCodebook
from model.latent_vq import LatentVQ
from model.trm import TRM
from model.verifier import sudoku_correct, sudoku_score
from scripts._common import build_data_splits
import scripts.m14_primary_experiment as m14
from scripts.m14_attempt5_dual_stream_semantic_exit import dual_stream_semantic_exit_solve

PROTOCOL_COMMIT = "8caea3844a1eb9f855c4a06afe58b8853d4316a7"
CLARIFICATION_COMMIT = "4218e863acd55a6c461c11df15f66a0ef57f96ca"
CORE_SEEDS = [1401, 2402]
EXPECTED_FP64_TENSOR_SHA = {
    1401: "4cf4c93ec9d3bd688850394685924cb23d0762a8716eb3e2f42e42befd249f08",
    2402: "00a84312a4fba02e31b1954bddffe10b5933e01eaaae4226eb068490a3f97c0d",
}
EXPECTED_CONTEXT_TENSOR_SHA = {
    "fp_recursive": "e1c1416d3453bf42577827ebdeffa801e376d9afce3ed6dde7515072512291c0",
    "ternary_recursive": "881ec148c4514bab507eccd06e8f0cc69a21a5b480f7fd9663720eb5a01e5518",
}
M14_FILE_SHA = {
    1401: "d5d4769726e3e45822e84a947dd4e8cb007a35374a8a40ae61a786c4c2ba55a4",
    2402: "b43f111af13bc8f7b667c9b7e97558b2b9743f522945193a7a625db77ea194ff",
}
DEV_DATA_SEED = 2026091501
CONFIRM_DATA_SEED = 2026091502
SHIFT_DATA_SEED = 2026091503
TRAIN_N, VAL_N, DEV_N, CONFIRM_N, SHIFT_N = 2048, 256, 256, 512, 256
VERIFIER_TRAIN_PUZZLES = 384
ACTION_TRAIN_START, ACTION_TRAIN_PUZZLES = 384, 384
VERIFIER_TRAIN_DEPTH = 4
ACTION_TRAIN_DEPTH = 3
WEAK_STEPS = 60
STRONG_STEPS = 180
VERIFIER_BATCH = 64
VERIFIER_LR = 2e-3
VERIFIER_WD = 0.01
ACTION_BANK = 16
ACTION_KEEP = 3
ACTION_SCALE = 0.5
ACTION_POLICY_STEPS = 300
ACTION_POLICY_BATCH = 64
ACTION_POLICY_LR = 2e-3
ACTION_POLICY_WD = 0.01
UTILITY_TEMPERATURE = 0.02
SEARCH_ROLLOUTS = 12
SEARCH_CPUCT = 1.5
SEARCH_DEPTHS = [1, 2, 4]
BETA_GRID = [0.0, 0.5, 1.0, 2.0]
VQ_CODEBOOK = 32
VQ_STEPS = 200
VQ_BATCH_VECTORS = 1024
VQ_DEPTH = 8
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 2026091599
PRIMARY_SEARCH = "learned_strong_d4_int8"
NO_SEARCH = "semantic_exit_k4"
GUARDED_SEARCH = "learned_strong_guard_d4_int8"
ORACLE_SEARCH = "learned_oracle_d4_int8"
FP32_SEARCH = "learned_strong_d4_fp32"


class M15Stop(RuntimeError):
    pass


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
                keys.append(key); seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows([{k: row.get(k) for k in keys} for row in rows])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_record() -> dict[str, Any]:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    return {
        "git_sha": git_sha(), "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "cpu_model": cpu_model, "logical_cpus": os.cpu_count(),
        "torch": torch.__version__, "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(), "experiment_device": "cpu",
    }


def make_cfg(seed: int, min_clues: int = 6, max_clues: int = 10):
    return load_config(
        "config/sudoku.yaml",
        overrides=[
            f"seed={int(seed)}", "device=cpu", "data.box=2", "data.num_tokens=5",
            "data.seq_len=16", "data.height=4", "data.width=4",
            f"data.min_clues={int(min_clues)}", f"data.max_clues={int(max_clues)}",
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


def build_dev_hierarchy(out: Path):
    cfg = make_cfg(DEV_DATA_SEED)
    ds, manifest = build_data_splits(
        cfg, TRAIN_N, VAL_N, DEV_N, seed=DEV_DATA_SEED,
        manifest_path=out / "manifests" / "train_validation_development.json",
    )
    audit = manifest["duplicate_audit"]
    if any(audit["cross_split_group_overlap"].values()) or any(audit["cross_split_exact_overlap"].values()):
        raise M15Stop(f"M15 development duplicate audit failed: {audit}")
    return ds["train"], ds["validation"], ds["test"], manifest


def build_frozen_set(out: Path, *, seed: int, n: int, min_clues: int, max_clues: int,
                     prior_fingerprints: set[str], filename: str):
    cfg = make_cfg(seed, min_clues, max_clues)
    ds, manifest = build_data_splits(
        cfg, 0, 0, n, seed=seed, manifest_path=out / "manifests" / filename,
    )
    current = manifest_fingerprints(manifest)
    overlap = sorted(prior_fingerprints & current)
    audit = {
        "seed": seed, "n": n, "min_clues": min_clues, "max_clues": max_clues,
        "prior_fingerprints": len(prior_fingerprints), "current_fingerprints": len(current),
        "overlap_count": len(overlap), "overlap_examples": overlap[:10],
    }
    write_json(out / "manifests" / filename.replace(".json", "_overlap_audit.json"), audit)
    if overlap:
        raise M15Stop(f"frozen set overlaps prior hierarchy: {filename}")
    return ds["test"], manifest, audit


def freeze_model(model: TRM) -> TRM:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert_frozen_reasoner(model)
    return model


def reconstruct_sources(out: Path) -> tuple[dict[int, TRM], dict[str, TRM], dict[str, Any]]:
    source = out / "source_m14"; source.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, _, _ = m14.build_development_data(source / "data")
    primary: dict[int, TRM] = {}; records: list[dict[str, Any]] = []
    for seed in CORE_SEEDS:
        rec = m14.train_one("fp_recursive_dim64", seed, train_ds, val_ds, source)
        ck = Path(rec["checkpoint"]); model, kind, _ = m14.load_checkpoint(ck)
        if kind != "fp_recursive_dim64" or not isinstance(model, TRM):
            raise M15Stop("failed to reconstruct M14 FP64 source core")
        state_sha = tensor_state_sha256(model); expected = EXPECTED_FP64_TENSOR_SHA[seed]
        if state_sha != expected:
            raise M15Stop(f"M14 source tensor identity mismatch seed={seed}: observed={state_sha} expected={expected}")
        primary[seed] = freeze_model(model)
        records.append({
            "kind": kind, "seed": seed, "reconstructed_checkpoint": str(ck),
            "reconstructed_file_sha256": sha256_file(ck), "accepted_m14_file_sha256": M14_FILE_SHA[seed],
            "tensor_state_sha256": state_sha, "expected_tensor_state_sha256": expected,
            "tensor_identity_exact": True,
        })
    context: dict[str, TRM] = {}; context_records: list[dict[str, Any]] = []
    for kind in ["fp_recursive", "ternary_recursive"]:
        try:
            rec = m14.train_one(kind, 1401, train_ds, val_ds, source / "context")
            ck = Path(rec["checkpoint"]); model, got_kind, _ = m14.load_checkpoint(ck)
            state_sha = tensor_state_sha256(model); expected = EXPECTED_CONTEXT_TENSOR_SHA[kind]
            exact = got_kind == kind and state_sha == expected
            context_records.append({"kind": kind, "seed": 1401, "checkpoint": str(ck),
                                    "tensor_state_sha256": state_sha, "expected_tensor_state_sha256": expected,
                                    "tensor_identity_exact": exact})
            if exact: context[kind] = freeze_model(model)
        except Exception as exc:
            context_records.append({"kind": kind, "seed": 1401, "tensor_identity_exact": False, "error": repr(exc)})
    report = {"materialization": "deterministic_reconstruction_from_frozen_m14_recipe",
              "primary": records, "optional_precision_context": context_records,
              "original_file_hashes_are_provenance_not_required_for_serialized_byte_identity": True}
    write_json(out / "source_checkpoint_reconstruction.json", report)
    return primary, context, report


def binary_auc(labels: Iterable[float], scores: Iterable[float]) -> float | None:
    y = np.asarray(list(labels), dtype=np.int64); s = np.asarray(list(scores), dtype=np.float64)
    pos = s[y == 1]; neg = s[y == 0]
    if not len(pos) or not len(neg): return None
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def average_precision(labels: Iterable[float], scores: Iterable[float]) -> float | None:
    y = np.asarray(list(labels), dtype=np.int64); s = np.asarray(list(scores), dtype=np.float64)
    npos = int(y.sum())
    if npos == 0: return None
    order = np.argsort(-s, kind="stable"); yy = y[order]; prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / npos)


def ece10(labels: Iterable[float], probs: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.float64); p = np.asarray(list(probs), dtype=np.float64)
    out = 0.0
    for lo in np.linspace(0.0, 0.9, 10):
        hi = lo + 0.1; mask = (p >= lo) & (p < hi if hi < 1.0 else p <= 1.0)
        if mask.any(): out += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(out)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64); order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64); i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]: j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0; i = j
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    a = np.asarray(list(x), dtype=np.float64); b = np.asarray(list(y), dtype=np.float64)
    if len(a) < 2 or len(b) != len(a): return None
    ra, rb = rankdata(a), rankdata(b)
    if float(ra.std()) == 0.0 or float(rb.std()) == 0.0: return None
    return float(np.corrcoef(ra, rb)[0, 1])


def percentile(vals: list[float], q: float) -> float | None:
    return float(np.quantile(vals, q)) if vals else None


def bootstrap_mean_delta(pairs: list[tuple[float, float]], seed: int) -> dict[str, Any]:
    if not pairs: return {"n": 0, "mean_delta": None, "ci95": [None, None]}
    arr = np.asarray([[a, b] for a, b in pairs], dtype=np.float64); delta = arr[:, 0] - arr[:, 1]
    rng = np.random.default_rng(seed); reps = []
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(delta), len(delta)); reps.append(float(delta[idx].mean()))
    return {"n": len(delta), "mean_delta": float(delta.mean()),
            "ci95": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
            "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": seed}


def rows_to_tensors(rows):
    return (torch.stack([r["x"] for r in rows]).long(), torch.stack([r["y"] for r in rows]).float(),
            torch.stack([r["z"] for r in rows]).float(), torch.tensor([r["label"] for r in rows], dtype=torch.float32))


def fit_verifier_member(rows, *, seed: int, steps: int) -> tuple[GroundedStateVerifier, dict[str, Any]]:
    set_seed(seed, deterministic=True)
    model = GroundedStateVerifier(num_tokens=5, dim=64, n_layers=1, heads=4, max_grid_size=8, act_bits=8, include_y=True)
    opt = torch.optim.AdamW(model.parameters(), lr=VERIFIER_LR, weight_decay=VERIFIER_WD)
    owned = {id(p) for g in opt.param_groups for p in g["params"]}
    if owned != {id(p) for p in model.parameters()}: raise M15Stop("verifier optimizer ownership mismatch")
    x, y, z, labels = rows_to_tensors(rows); rng = np.random.default_rng(seed + 15000); curve = []
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}; model.train()
    for step in range(1, steps + 1):
        idx = torch.from_numpy(rng.integers(0, len(rows), size=VERIFIER_BATCH, dtype=np.int64))
        logits = model.forward_logits(x[idx], y[idx], z[idx], width=4)
        loss = F.binary_cross_entropy_with_logits(logits, labels[idx])
        if not torch.isfinite(loss): raise M15Stop("non-finite verifier loss")
        opt.zero_grad(set_to_none=True); loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(grad)): raise M15Stop("non-finite verifier grad")
        opt.step()
        if step == 1 or step % 20 == 0 or step == steps:
            curve.append({"step": step, "loss": float(loss.detach()), "grad_norm_preclip": float(grad)})
    movement = math.sqrt(sum(float((p.detach() - initial[n]).pow(2).sum()) for n, p in model.named_parameters()))
    if movement <= 0: raise M15Stop("verifier parameters did not update")
    model.eval()
    return model, {"seed": seed, "steps": steps, "batch": VERIFIER_BATCH, "lr": VERIFIER_LR,
                   "weight_decay": VERIFIER_WD, "parameter_count": sum(p.numel() for p in model.parameters()),
                   "parameter_movement_l2": movement, "optimizer_owns_exact_module": True, "curve": curve}


def fit_verifiers(core: TRM, core_seed: int, train_ds, val_ds, out: Path):
    train_inputs = torch.from_numpy(train_ds.inputs[:VERIFIER_TRAIN_PUZZLES]).long()
    train_rows = generate_trajectory_states(core, train_inputs, train_ds.ids[:VERIFIER_TRAIN_PUZZLES],
                                            max_depth=VERIFIER_TRAIN_DEPTH, height=4, width=4, box=2)
    val_inputs = torch.from_numpy(val_ds.inputs).long()
    val_rows = generate_trajectory_states(core, val_inputs, val_ds.ids, max_depth=8, height=4, width=4, box=2)
    weak, weak_rec = fit_verifier_member(train_rows, seed=core_seed + 151, steps=WEAK_STEPS)
    strong = EnsembleGroundedStateVerifier(num_tokens=5, dim=64, n_members=3, n_layers=1, heads=4,
                                            max_grid_size=8, act_bits=8, include_y=True)
    member_recs = []
    for mi, member_seed in enumerate([core_seed + 251, core_seed + 352, core_seed + 453]):
        member, rec = fit_verifier_member(train_rows, seed=member_seed, steps=STRONG_STEPS)
        strong.members[mi].load_state_dict(member.state_dict(), strict=True); member_recs.append(rec)
    strong.eval(); vdir = out / "aux"; vdir.mkdir(parents=True, exist_ok=True)
    weak_path = vdir / f"weak_verifier_seed{core_seed}.pt"; strong_path = vdir / f"strong_verifier_seed{core_seed}.pt"
    torch.save({"format":"spectra.m15_verifier","core_seed":core_seed,"strength":"weak","state":weak.state_dict()}, weak_path)
    torch.save({"format":"spectra.m15_verifier","core_seed":core_seed,"strength":"strong","state":strong.state_dict()}, strong_path)
    return weak, strong, train_rows, val_rows, {"core_seed": core_seed,
        "weak": {**weak_rec, "checkpoint": str(weak_path), "sha256": sha256_file(weak_path)},
        "strong_members": member_recs, "strong_checkpoint": str(strong_path), "strong_sha256": sha256_file(strong_path),
        "train_states": len(train_rows), "validation_states": len(val_rows)}


@torch.inference_mode()
def verifier_predictions(verifier, rows, batch=128):
    x, y, z, labels = rows_to_tensors(rows); probs, stds = [], []
    for start in range(0, len(rows), batch):
        xb, yb, zb = x[start:start+batch], y[start:start+batch], z[start:start+batch]
        if hasattr(verifier, "value_with_uncertainty_state"):
            p, s = verifier.value_with_uncertainty_state(xb, yb, zb, width=4)
        else:
            p = verifier.value_state(xb, yb, zb, width=4); s = torch.zeros_like(p)
        probs.extend(float(v) for v in p.cpu()); stds.extend(float(v) for v in s.cpu())
    return labels.tolist(), probs, stds


def verifier_metrics_by_depth(core: TRM, verifier, rows) -> dict[str, Any]:
    labels, probs, stds = verifier_predictions(verifier, rows); enriched = []
    for row, label, prob, std in zip(rows, labels, probs, stds):
        enriched.append({"depth": int(row["depth"]), "label": float(label), "prob": float(prob),
                         "std": float(std), "absolute_score": float(row["score_before"])})
    out = {}
    for depth in sorted({r["depth"] for r in enriched}):
        rr = [r for r in enriched if r["depth"] == depth]; y = [r["label"] for r in rr]; p = [r["prob"] for r in rr]
        out[str(depth)] = {"n": len(rr), "positive_rate": float(np.mean(y)), "roc_auc": binary_auc(y,p),
            "average_precision": average_precision(y,p), "brier": float(np.mean((np.asarray(p)-np.asarray(y))**2)),
            "ece10": ece10(y,p), "spearman_proxy_absolute_score": spearman(p,[r["absolute_score"] for r in rr]),
            "mean_disagreement": float(np.mean([r["std"] for r in rr]))}
    return {"by_depth": out, "rows": enriched}


def make_candidate_bank(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed); bank = torch.randn(ACTION_BANK, dim, generator=gen)
    return bank / bank.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.inference_mode()
def action_utility_table(core: TRM, rows, directions: torch.Tensor, batch_size=64):
    assert_frozen_reasoner(core); x, y, z, _ = rows_to_tensors(rows); cols = []; calls = 0
    for ai in range(directions.shape[0] + 1):
        chunks = []
        for start in range(0, len(rows), batch_size):
            xb, yb, zb = x[start:start+batch_size], y[start:start+batch_size], z[start:start+batch_size]
            if ai: zb = zb + ACTION_SCALE * directions[ai-1]
            xemb = core.token_embed(xb) + core.encode_positions(xb, 4, 4); yn, _ = core.recursive_cycle(xemb, yb, zb); calls += 1
            ans = m14.clamp_givens(xb, core.out_head(yn).argmax(dim=-1)); chunks.append(sudoku_score(xb, ans, box=2).cpu())
        cols.append(torch.cat(chunks))
    return torch.stack(cols, dim=1), {"states": len(rows), "actions_including_identity": directions.shape[0]+1,
        "state_action_equivalents": len(rows)*(directions.shape[0]+1), "batched_recursive_cycle_calls": calls}


def select_directions(full_u: torch.Tensor, k: int) -> tuple[list[int], dict[str, Any]]:
    best = full_u[:,0].clone(); selected=[]; available=list(range(full_u.shape[1]-1)); trace=[]
    for _ in range(k):
        candidates=[]
        for idx in available:
            mean=float(torch.maximum(best, full_u[:,idx+1]).mean()); candidates.append((mean,-idx,idx))
        _,_,idx=max(candidates); selected.append(idx); available.remove(idx); best=torch.maximum(best,full_u[:,idx+1]); trace.append(float(best.mean()))
    return selected, {"identity_mean": float(full_u[:,0].mean()), "selected_oracle_mean": float(best.mean()),
                      "coverage_gain": float(best.mean()-full_u[:,0].mean()), "coverage_trace": trace}


def fit_action_policy(core: TRM, core_seed: int, train_ds, out: Path):
    start=ACTION_TRAIN_START; end=start+ACTION_TRAIN_PUZZLES
    inputs=torch.from_numpy(train_ds.inputs[start:end]).long(); ids=train_ds.ids[start:end]
    rows=generate_trajectory_states(core, inputs, ids, max_depth=ACTION_TRAIN_DEPTH, height=4,width=4,box=2)
    bank=make_candidate_bank(64, 2026091511+core_seed); full_u, work=action_utility_table(core, rows, bank)
    selected, coverage=select_directions(full_u, ACTION_KEEP); dirs=bank[selected].clone()
    restricted=torch.cat([full_u[:,0:1], full_u[:,[i+1 for i in selected]]], dim=1)
    set_seed(core_seed+1551, deterministic=True)
    policy=StateConditionedLatentActionCodebook(dim=64,num_tokens=5,n_actions=4,scale=ACTION_SCALE,hidden_dim=64)
    with torch.no_grad(): policy.directions.copy_(dirs)
    policy.directions.requires_grad_(False); params=[p for p in policy.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params, lr=ACTION_POLICY_LR, weight_decay=ACTION_POLICY_WD)
    if {id(p) for g in opt.param_groups for p in g["params"]} != {id(p) for p in params}: raise M15Stop("action optimizer ownership mismatch")
    x,y,z,_=rows_to_tensors(rows); rng=np.random.default_rng(core_seed+1661); curve=[]; first_head=policy.policy_head.weight.detach().clone()
    for step in range(1,ACTION_POLICY_STEPS+1):
        idx=torch.from_numpy(rng.integers(0,len(rows),size=ACTION_POLICY_BATCH,dtype=np.int64)); logits=policy.policy_logits_for_state(x[idx],y[idx],z[idx]); u=restricted[idx]
        teacher=torch.softmax((u-u.max(dim=1,keepdim=True).values)/UTILITY_TEMPERATURE,dim=1)
        loss=-(teacher*F.log_softmax(logits,dim=1)).sum(dim=1).mean(); opt.zero_grad(set_to_none=True); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        if step==1 or step%50==0 or step==ACTION_POLICY_STEPS: curve.append({"step":step,"loss":float(loss.detach()),"grad_norm_preclip":float(grad)})
    movement=float((policy.policy_head.weight.detach()-first_head).norm())
    if movement<=0: raise M15Stop("action policy did not update")
    policy.eval(); uniform=LatentActionCodebook(dim=64,n_actions=4,scale=ACTION_SCALE)
    with torch.no_grad(): uniform.directions.copy_(dirs); uniform.prior_logits.zero_()
    for p in uniform.parameters(): p.requires_grad_(False)
    uniform.eval(); pdir=out/"aux"; pdir.mkdir(parents=True,exist_ok=True); ck=pdir/f"action_policy_seed{core_seed}.pt"
    torch.save({"format":"spectra.m15_action","core_seed":core_seed,"selected":selected,"directions":dirs,"state":policy.state_dict()},ck)
    return policy,uniform,dirs,{"core_seed":core_seed,"candidate_bank_seed":2026091511+core_seed,"candidate_bank":ACTION_BANK,
        "selected_indices":selected,"scale":ACTION_SCALE,"coverage":coverage,"utility_work":work,"policy_steps":ACTION_POLICY_STEPS,
        "policy_batch":ACTION_POLICY_BATCH,"lr":ACTION_POLICY_LR,"weight_decay":ACTION_POLICY_WD,"policy_head_movement_l2":movement,
        "curve":curve,"checkpoint":str(ck),"sha256":sha256_file(ck),"reference_target_used":False,"directions_identical_learned_and_unguided":True}


class M15Search(StateConditionedLatentNativeMCTS):
    def __init__(self,*args,value_mode="proxy",beta=0.0,terminal_guard=False,fp32_state=False,analysis_verifier=None,**kwargs):
        super().__init__(*args,**kwargs); self.value_mode=str(value_mode); self.beta=float(beta); self.terminal_guard=bool(terminal_guard)
        self.fp32_state=bool(fp32_state); self.analysis_verifier=analysis_verifier or self.verifier; self._valid_cache={}

    def _reset_search_state(self,mode):
        super()._reset_search_state(mode); self._valid_cache={}
        self.last_search_stats.update({"value_mode":self.value_mode,"m15_beta":self.beta,"terminal_guard":self.terminal_guard,
            "fp32_state":self.fp32_state,"semantic_guard_checks":0,"semantic_guard_terminal_nodes":0})

    def _make_zero_root(self,x_emb):
        if not self.fp32_state: return super()._make_zero_root(x_emb)
        z=torch.zeros_like(x_emb); scale=torch.ones((*z.shape[:-1],1),dtype=z.dtype,device=z.device)
        return _LatentNode(torch.zeros_like(x_emb),z,scale,depth=0,path=())

    def _step(self,x_emb,node,action):
        if not self.fp32_state: return super()._step(x_emb,node,action)
        if not self._is_int(action) or not 0 <= action < int(self.codebook.n_actions): raise ValueError("action out of range")
        self.last_search_stats["transition_calls"]=int(self.last_search_stats["transition_calls"])+1
        z=self.codebook.apply_action(node.latent(),action); y=node.y
        for _ in range(self.model.T):
            y,z=self.model.recursive_cycle(x_emb,y,z); self.last_search_stats["recursive_cycle_calls"]=int(self.last_search_stats["recursive_cycle_calls"])+1
        if self.latent_vq is not None:
            z=self.latent_vq.snap(z); self.last_search_stats["latent_vq_calls"]=int(self.last_search_stats["latent_vq_calls"])+1
        scale=torch.ones((*z.shape[:-1],1),dtype=z.dtype,device=z.device); return y,z,scale

    def _answer(self,x,node): return m14.clamp_givens(x,self.model.out_head(node.y).argmax(dim=-1))

    def _is_valid(self,x,node):
        key=id(node)
        if key not in self._valid_cache:
            ans=self._answer(x,node); ok=bool(sudoku_correct(x,ans,2).bool().item()); self._valid_cache[key]=ok
            if self.last_search_stats: self.last_search_stats["semantic_guard_checks"]=int(self.last_search_stats["semantic_guard_checks"])+1
        return self._valid_cache[key]

    def _expand(self,node,x_emb,*,initial=False):
        x=getattr(self,"_m09_search_x",None)
        if self.terminal_guard and x is not None and self._is_valid(x,node):
            self.last_search_stats["semantic_guard_terminal_nodes"]=int(self.last_search_stats["semantic_guard_terminal_nodes"])+1; return False
        return super()._expand(node,x_emb,initial=initial)

    def _value(self,x,node):
        if self.terminal_guard and self._is_valid(x,node): return 1.0
        if self.value_mode=="oracle_absolute": return float(sudoku_score(x,self._answer(x,node),box=2).detach().mean())
        if self.beta>0.0:
            if not hasattr(self.verifier,"value_with_uncertainty_state"): raise RuntimeError("beta requires ensemble grounded verifier")
            mean,std=self.verifier.value_with_uncertainty_state(x,node.y,node.latent(),width=4); return float((mean-self.beta*std).detach().mean())
        return float(self.verifier.value_state(x,node.y,node.latent(),width=4).detach().mean())

    @torch.inference_mode()
    def evaluated_analysis(self,x):
        if self.root is None: return []
        by_path={}; stack=[self.root]
        while stack:
            node=stack.pop(); by_path[tuple(node.path)]=node; stack.extend(node.children)
        out=[]
        for path_list in self.last_search_stats.get("evaluated_paths",[]):
            path=tuple(path_list); node=by_path[path]; ans=self._answer(x,node); absolute=float(sudoku_score(x,ans,box=2).item()); valid=bool(sudoku_correct(x,ans,2).bool().item())
            if hasattr(self.analysis_verifier,"value_with_uncertainty_state"):
                pm,ps=self.analysis_verifier.value_with_uncertainty_state(x,node.y,node.latent(),width=4); proxy=float(pm.item()); disagreement=float(ps.item())
            else:
                proxy=float(self.analysis_verifier.value_state(x,node.y,node.latent(),width=4).item()); disagreement=0.0
            target=one_cycle_improvement_target(self.model,x,node.y,node.latent(),height=4,width=4,box=2)
            out.append({"path":list(path),"depth":int(node.depth),"proxy":proxy,"disagreement":disagreement,"absolute_score":absolute,
                        "semantic_valid":valid,"improvement_label":float(target["label"].item()),"score_after_one_cycle":float(target["score_after"].item())})
        return out


def fixed_four_solve(model: TRM,x: torch.Tensor):
    xemb=model.token_embed(x)+model.encode_positions(x,4,4); y=torch.zeros_like(xemb); z=torch.zeros_like(xemb)
    for _ in range(4): y,z=model.recursive_cycle(xemb,y,z)
    ans=m14.clamp_givens(x,model.out_head(y).argmax(dim=-1)); return ans,{"executed_steps":4,"block_applications":8,"semantic_checks":1,"stop_reason":"fixed_depth"}


def run_no_search(model:TRM,x:torch.Tensor,kind:str):
    if kind==NO_SEARCH: return dual_stream_semantic_exit_solve(model,x,4)
    if kind=="fixed_depth4": return fixed_four_solve(model,x)
    raise ValueError(kind)


def searcher_for(model,verifier,codebook,*,depth,beta=0.0,guard=False,fp32=False,value_mode="proxy",analysis_verifier=None):
    return M15Search(model=model,energy_verifier=verifier,action_codebook=codebook,height=4,width=4,n_rollouts=SEARCH_ROLLOUTS,
        c_puct=SEARCH_CPUCT,uncertainty_beta=0.0,latent_vq=None,max_depth=depth,value_mode=value_mode,beta=beta,
        terminal_guard=guard,fp32_state=fp32,analysis_verifier=analysis_verifier)


def clue_count(x:torch.Tensor)->int: return int((x!=0).sum().item())


def evaluate_no_search(model,core_seed,ds,config_id,out_rows:Path):
    xs=torch.from_numpy(ds.inputs).long(); rows=[]
    for i in range(len(ds)):
        xi=xs[i:i+1]; t0=time.perf_counter_ns(); ans,work=run_no_search(model,xi,config_id); elapsed=(time.perf_counter_ns()-t0)/1e6
        semantic=bool(sudoku_correct(xi,ans,2).bool().item()); score=float(sudoku_score(xi,ans,box=2).item())
        row={"surface":"pending","config_id":config_id,"core_seed":core_seed,"example_index":i,"example_id":ds.ids[i],"clues":clue_count(xi),
             "semantic_success":int(semantic),"symbolic_score":score,"latency_ms":elapsed,"work":work,"target_used_inside_solver":False}
        append_jsonl(out_rows,row); rows.append(row)
    return rows


def evaluate_search(model,core_seed,ds,config_id,searcher,out_rows:Path,leaf_rows_path:Path):
    xs=torch.from_numpy(ds.inputs).long(); rows=[]
    for i in range(min(2,len(ds))): node=searcher.search(xs[i:i+1]); _=searcher.decode(node)
    for i in range(len(ds)):
        xi=xs[i:i+1]; t0=time.perf_counter_ns(); node=searcher.search(xi); ans=m14.clamp_givens(xi,searcher.decode(node)); semantic=bool(sudoku_correct(xi,ans,2).bool().item()); elapsed=(time.perf_counter_ns()-t0)/1e6
        score=float(sudoku_score(xi,ans,box=2).item()); work=copy.deepcopy(searcher.last_search_stats); analysis=searcher.evaluated_analysis(xi)
        selected_path=tuple(searcher.best_node.path if searcher.best_node is not None else ()); selected=next((r for r in analysis if tuple(r["path"])==selected_path),None)
        max_abs=max((r["absolute_score"] for r in analysis),default=score)
        exploitation=bool(selected is not None and selected["absolute_score"]+1e-12 < max_abs and all(selected["proxy"] >= r["proxy"]-1e-12 for r in analysis))
        row={"surface":"pending","config_id":config_id,"core_seed":core_seed,"example_index":i,"example_id":ds.ids[i],"clues":clue_count(xi),
             "semantic_success":int(semantic),"symbolic_score":score,"latency_ms":elapsed,"selected_path":list(selected_path),"selected_depth":len(selected_path),
             "proxy_exploitation":int(exploitation),"selected_proxy":selected["proxy"] if selected else None,"selected_absolute_score":selected["absolute_score"] if selected else score,
             "max_evaluated_absolute_score":max_abs,"work":work,"target_used_inside_solver":False}
        append_jsonl(out_rows,row); rows.append(row)
        for leaf in analysis:
            append_jsonl(leaf_rows_path,{**leaf,"surface":"pending","config_id":config_id,"core_seed":core_seed,"example_id":ds.ids[i],"clues":clue_count(xi),"selected":tuple(leaf["path"])==selected_path})
    return rows


def set_surface(rows,surface):
    for r in rows: r["surface"]=surface


def pair_map(rows): return {(int(r["core_seed"]),str(r["example_id"])):r for r in rows}


def config_summary(rows, baseline_rows=None):
    lat=[float(r["latency_ms"]) for r in rows]; sem=[int(r["semantic_success"]) for r in rows]; scores=[float(r["symbolic_score"]) for r in rows]
    out={"n":len(rows),"semantic_success":float(np.mean(sem)) if sem else None,"mean_symbolic_score":float(np.mean(scores)) if scores else None,
         "latency_mean_ms":float(np.mean(lat)) if lat else None,"latency_median_ms":float(np.median(lat)) if lat else None,"latency_p95_ms":percentile(lat,0.95),
         "proxy_exploitation_rate":float(np.mean([r.get("proxy_exploitation",0) for r in rows])) if rows else None}
    if baseline_rows is not None:
        b=pair_map(baseline_rows); regress=rescue=paired=0; deltas=[]
        for r in rows:
            key=(int(r["core_seed"]),str(r["example_id"])); br=b.get(key)
            if br is None: continue
            paired+=1; regress+=int(br["semantic_success"]==1 and r["semantic_success"]==0); rescue+=int(br["semantic_success"]==0 and r["semantic_success"]==1)
            deltas.append((float(r["symbolic_score"]),float(br["symbolic_score"])))
        out.update({"paired_n":paired,"search_induced_regressions":regress,"search_induced_regression_rate":regress/max(1,paired),
                    "search_induced_rescues":rescue,"search_induced_rescue_rate":rescue/max(1,paired),"score_delta":bootstrap_mean_delta(deltas,BOOTSTRAP_SEED)})
    return out


def select_beta(validation_by_beta, baseline):
    ranked=[]
    for beta,rows in validation_by_beta.items():
        s=config_summary(rows,baseline); ranked.append((s["search_induced_regression_rate"],-s["semantic_success"],s["latency_median_ms"],float(beta),s))
    ranked.sort(key=lambda x:(x[0],x[1],x[2],x[3])); best=ranked[0]
    return {"selected_beta":best[3],"selection_rule":"lowest_regression_then_higher_success_then_lower_median_latency_then_lower_beta",
            "candidates":{str(x[3]):x[4] for x in ranked}}


def mechanism_support(summaries):
    learned=summaries[PRIMARY_SEARCH]; oracle=summaries[ORACLE_SEARCH]; guard=summaries[GUARDED_SEARCH]; fp32=summaries[FP32_SEARCH]
    reg=int(learned["search_induced_regressions"]); oracle_reg=int(oracle["search_induced_regressions"]); guard_reg=int(guard["search_induced_regressions"]); fp32_reg=int(fp32["search_induced_regressions"])
    oracle_reduction=(reg-oracle_reg)/reg if reg else 0.0; guard_prevention=(reg-guard_reg)/reg if reg else 0.0; fp32_prevention=(reg-fp32_reg)/reg if reg else 0.0
    conditions={"material_regression_or_exploitation": learned["search_induced_regression_rate"]>=0.05 or learned["proxy_exploitation_rate"]>=0.10,
        "oracle_fewer_regressions_by_half": oracle_reduction>=0.50,"guard_prevents_half_regressions": guard_prevention>=0.50,
        "guard_success_not_down_gt_1pp": guard["semantic_success"] >= learned["semantic_success"]-0.01,"fp32_not_prevent_75pct": fp32_prevention < 0.75}
    return {"h1_supported":all(conditions.values()),"conditions":conditions,"unguarded_regressions":reg,
            "oracle_regression_reduction_fraction":oracle_reduction,"guard_prevention_fraction":guard_prevention,"fp32_prevention_fraction":fp32_prevention}


def fit_vq(core:TRM,core_seed:int,train_ds,out:Path):
    xs=torch.from_numpy(train_ds.inputs[:128]).long(); xemb=core.token_embed(xs)+core.encode_positions(xs,4,4)
    y=torch.zeros(xs.shape[0],16,64); z=torch.zeros_like(y); states=[]
    with torch.inference_mode():
        for _ in range(VQ_DEPTH): y,z=core.recursive_cycle(xemb,y,z); states.append(z.detach().reshape(-1,64).cpu())
    vectors=torch.cat(states); set_seed(core_seed+1771,deterministic=True); vq=LatentVQ(dim=64,codebook_size=VQ_CODEBOOK,commitment=0.25,decay=0.0)
    opt=torch.optim.AdamW(vq.parameters(),lr=5e-3,weight_decay=0.0); rng=np.random.default_rng(core_seed+1881); curve=[]; vq.train()
    for step in range(1,VQ_STEPS+1):
        idx=torch.from_numpy(rng.integers(0,len(vectors),size=VQ_BATCH_VECTORS,dtype=np.int64)); batch=vectors[idx].reshape(-1,1,64)
        _,_,loss=vq(batch); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step==1 or step%40==0 or step==VQ_STEPS: curve.append({"step":step,"loss":float(loss.detach())})
    vq.eval(); ck=out/"aux"/f"vq_seed{core_seed}.pt"; ck.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"format":"spectra.m15_vq","core_seed":core_seed,"state":vq.state_dict(),"codebook_size":VQ_CODEBOOK},ck)
    return vq,{"core_seed":core_seed,"codebook_size":VQ_CODEBOOK,"fit_vectors":len(vectors),"steps":VQ_STEPS,"curve":curve,"checkpoint":str(ck),"sha256":sha256_file(ck)}


def quant_dequant(z): c,s=_quantize_int8(z); return _dequantize_int8(c,s),c,s


@torch.inference_mode()
def trajectory_ablation(core:TRM,core_seed:int,ds,vq:LatentVQ,surface:str,out_rows:Path):
    xs=torch.from_numpy(ds.inputs).long(); xemb=core.token_embed(xs)+core.encode_positions(xs,4,4)
    yfp=torch.zeros(len(ds),16,64); zfp=torch.zeros_like(yfp); yi=torch.zeros_like(yfp); zi=torch.zeros_like(yfp); yv=torch.zeros_like(yfp); zv=torch.zeros_like(yfp)
    all_rows=[]
    for depth in range(1,VQ_DEPTH+1):
        yfp,zfp=core.recursive_cycle(xemb,yfp,zfp); yi,zpre_i=core.recursive_cycle(xemb,yi,zi); zi,codes_i,scale_i=quant_dequant(zpre_i)
        yv,zpre_v=core.recursive_cycle(xemb,yv,zv); idx=vq._nearest(zpre_v.reshape(-1,64)); zv=vq.codebook[idx].reshape_as(zpre_v)
        local_v=(zpre_v-zv).norm(dim=-1); div_i=(zi-zfp).norm(dim=-1); div_v=(zv-zfp).norm(dim=-1)
        ans_fp=m14.clamp_givens(xs,core.out_head(yfp).argmax(dim=-1)); ans_i=m14.clamp_givens(xs,core.out_head(yi).argmax(dim=-1)); ans_v=m14.clamp_givens(xs,core.out_head(yv).argmax(dim=-1))
        valid_fp=sudoku_correct(xs,ans_fp,2).bool(); valid_i=sudoku_correct(xs,ans_i,2).bool(); valid_v=sudoku_correct(xs,ans_v,2).bool()
        for variant,ans,valid,div,local in [("fp32_reference",ans_fp,valid_fp,torch.zeros_like(div_i),torch.zeros_like(local_v)),
            ("int8_requant",ans_i,valid_i,div_i,(zpre_i-zi).norm(dim=-1)),("vq32",ans_v,valid_v,div_v,local_v)]:
            agree=(ans==ans_fp).all(dim=1)
            for i in range(len(ds)):
                row={"surface":surface,"core_seed":core_seed,"example_id":ds.ids[i],"clues":clue_count(xs[i:i+1]),"depth":depth,"variant":variant,
                     "trajectory_divergence_mean_token_l2":float(div[i].mean()),"trajectory_divergence_p95_token_l2":float(torch.quantile(div[i],0.95)),
                     "local_projection_error_mean_token_l2":float(local[i].mean()),"answer_agreement_with_fp":int(agree[i]),"semantic_success":int(valid[i])}
                append_jsonl(out_rows,row); all_rows.append(row)
        counts=torch.bincount(idx.cpu(),minlength=VQ_CODEBOOK).float(); probs=counts/counts.sum().clamp_min(1); nz=probs[probs>0]
        util={"surface":surface,"core_seed":core_seed,"depth":depth,"variant":"vq32_utilization","unique_codes":int((counts>0).sum()),
              "utilization_fraction":float((counts>0).float().mean()),"perplexity":float(torch.exp(-(nz*nz.log()).sum())) if len(nz) else 0.0}
        append_jsonl(out_rows,util); all_rows.append(util)
    return all_rows


def trajectory_summary(rows):
    out=[]; keys=sorted({(r["core_seed"],r["depth"],r["variant"]) for r in rows if "example_id" in r})
    for seed,depth,var in keys:
        rr=[r for r in rows if r.get("core_seed")==seed and r.get("depth")==depth and r.get("variant")==var and "example_id" in r]
        out.append({"core_seed":seed,"depth":depth,"variant":var,"n":len(rr),
            "divergence_mean":float(np.mean([r["trajectory_divergence_mean_token_l2"] for r in rr])),
            "divergence_p95_example_mean":float(np.quantile([r["trajectory_divergence_mean_token_l2"] for r in rr],0.95)),
            "local_projection_error_mean":float(np.mean([r["local_projection_error_mean_token_l2"] for r in rr])),
            "answer_agreement":float(np.mean([r["answer_agreement_with_fp"] for r in rr])),"semantic_success":float(np.mean([r["semantic_success"] for r in rr]))})
    return out


def precision_context(context_models,dev_ds,out:Path):
    if set(context_models)!={"fp_recursive","ternary_recursive"}:
        result={"status":"omitted_exact_context_state_not_reconstructed","rows":[]}; write_json(out/"precision_context.json",result); return result
    xs=torch.from_numpy(dev_ds.inputs).long(); rows=[]
    for kind,model in context_models.items():
        with torch.inference_mode(): _,steps=model(xs,height=4,width=4)
        for d,step in enumerate(steps[:4],1):
            ans=m14.clamp_givens(xs,step["logits"].argmax(dim=-1)); valid=sudoku_correct(xs,ans,2).float()
            rows.append({"kind":kind,"depth":d,"semantic_success":float(valid.mean()),"n":len(dev_ds)})
    result={"status":"complete_matched_architecture_context","rows":rows}; write_json(out/"precision_context.json",result); return result


def leaf_mechanism_summary(leaf_rows):
    out={}
    for depth in sorted({int(r["depth"]) for r in leaf_rows}):
        rr=[r for r in leaf_rows if int(r["depth"])==depth]; labels=[r["improvement_label"] for r in rr]; proxy=[r["proxy"] for r in rr]; absolute=[r["absolute_score"] for r in rr]
        valid=[r for r in rr if r["semantic_valid"]]; invalid=[r for r in rr if not r["semantic_valid"]]
        out[str(depth)]={"n":len(rr),"improvement_auc":binary_auc(labels,proxy),"improvement_ap":average_precision(labels,proxy),
            "spearman_proxy_absolute_score":spearman(proxy,absolute),"mean_proxy_valid":float(np.mean([r["proxy"] for r in valid])) if valid else None,
            "mean_proxy_invalid":float(np.mean([r["proxy"] for r in invalid])) if invalid else None,
            "valid_minus_invalid_proxy":(float(np.mean([r["proxy"] for r in valid]))-float(np.mean([r["proxy"] for r in invalid]))) if valid and invalid else None,
            "valid_states":len(valid),"invalid_states":len(invalid)}
    return out


def difficult_failure_cases(all_configs,baseline_rows,limit=30):
    base=pair_map(baseline_rows); rows=[]
    for cid,crows in all_configs.items():
        if cid in {NO_SEARCH,"fixed_depth4"}: continue
        for r in crows:
            key=(int(r["core_seed"]),str(r["example_id"])); b=base.get(key)
            if b is None: continue
            regression=b["semantic_success"]==1 and r["semantic_success"]==0; rescue=b["semantic_success"]==0 and r["semantic_success"]==1; exploit=bool(r.get("proxy_exploitation",0)); difficult=int(r["clues"])<=7
            if regression or rescue or exploit:
                rows.append({"config_id":cid,"core_seed":r["core_seed"],"example_id":r["example_id"],"clues":r["clues"],
                    "difficult_low_clue_in_distribution":difficult,"search_regression":regression,"search_rescue":rescue,"proxy_exploitation":exploit,
                    "baseline_score":b["symbolic_score"],"search_score":r["symbolic_score"],"selected_proxy":r.get("selected_proxy"),
                    "selected_absolute_score":r.get("selected_absolute_score"),"max_evaluated_absolute_score":r.get("max_evaluated_absolute_score"),"selected_path":r.get("selected_path")})
    rows.sort(key=lambda r:(not r["search_regression"],not r["proxy_exploitation"],r["clues"],str(r["example_id"])))
    return rows[:limit]


def summarize_surface(configs,leaf_rows):
    baseline=configs[NO_SEARCH]; summaries={cid:config_summary(rows,None if cid==NO_SEARCH else baseline) for cid,rows in configs.items()}
    primary_leaves=[r for r in leaf_rows if r["config_id"]==PRIMARY_SEARCH]
    return summaries,leaf_mechanism_summary(primary_leaves),mechanism_support(summaries)


def freeze_manifest(out,cores,strongs,weaks,policies,vqs,beta,dev_summary):
    payload={"frozen_before_confirmation":True,"protocol_commit":PROTOCOL_COMMIT,"clarification_commit":CLARIFICATION_COMMIT,
        "core_tensor_hashes":{str(s):tensor_state_sha256(cores[s]) for s in CORE_SEEDS},"weak_verifier_tensor_hashes":{str(s):tensor_state_sha256(weaks[s]) for s in CORE_SEEDS},
        "strong_verifier_tensor_hashes":{str(s):tensor_state_sha256(strongs[s]) for s in CORE_SEEDS},"action_policy_tensor_hashes":{str(s):tensor_state_sha256(policies[s]) for s in CORE_SEEDS},
        "vq_tensor_hashes":{str(s):tensor_state_sha256(vqs[s]) for s in CORE_SEEDS},"selected_beta":beta,"search_rollouts":SEARCH_ROLLOUTS,
        "search_depths":SEARCH_DEPTHS,"primary_intervention":"terminal_validity_guard_reference_free","development_mechanism_support":dev_summary,"no_post_confirmation_selection":True}
    write_json(out/"confirmation_freeze_manifest.json",payload); return payload


def run_surface(surface,ds,cores,weaks,strongs,policies,uniforms,selected_beta,out:Path):
    configs=defaultdict(list); leaf_rows=[]; search_rows_path=out/f"{surface}_rows.jsonl"; leaf_path=out/f"{surface}_leaf_rows.jsonl"
    if search_rows_path.exists(): search_rows_path.unlink()
    if leaf_path.exists(): leaf_path.unlink()
    for seed in CORE_SEEDS:
        core=cores[seed]; weak=weaks[seed]; strong=strongs[seed]; policy=policies[seed]; uniform=uniforms[seed]
        for cid in [NO_SEARCH,"fixed_depth4"]:
            rr=evaluate_no_search(core,seed,ds,cid,search_rows_path); set_surface(rr,surface); configs[cid].extend(rr)
        variants=[("learned_strong_d1_int8",policy,strong,1,0.0,False,False,"proxy"),
            ("learned_strong_d2_int8",policy,strong,2,0.0,False,False,"proxy"),(PRIMARY_SEARCH,policy,strong,4,0.0,False,False,"proxy"),
            ("unguided_strong_d4_int8",uniform,strong,4,0.0,False,False,"proxy"),("learned_weak_d4_int8",policy,weak,4,0.0,False,False,"proxy"),
            (f"learned_lcb_beta{selected_beta:g}_d4_int8",policy,strong,4,selected_beta,False,False,"proxy"),(GUARDED_SEARCH,policy,strong,4,0.0,True,False,"proxy"),
            (ORACLE_SEARCH,policy,strong,4,0.0,False,False,"oracle_absolute"),(FP32_SEARCH,policy,strong,4,0.0,False,True,"proxy")]
        if surface=="shift_low_clue":
            keep={PRIMARY_SEARCH,"unguided_strong_d4_int8",GUARDED_SEARCH,ORACLE_SEARCH,FP32_SEARCH,f"learned_lcb_beta{selected_beta:g}_d4_int8"}; variants=[v for v in variants if v[0] in keep]
        for cid,cb,vf,depth,beta,guard,fp32,mode in variants:
            srch=searcher_for(core,vf,cb,depth=depth,beta=beta,guard=guard,fp32=fp32,value_mode=mode,analysis_verifier=strong)
            rr=evaluate_search(core,seed,ds,cid,srch,search_rows_path,leaf_path); set_surface(rr,surface); configs[cid].extend(rr)
    flat=[r for rr in configs.values() for r in rr]
    if search_rows_path.exists(): search_rows_path.unlink()
    for r in flat: append_jsonl(search_rows_path,r)
    if leaf_path.exists():
        parsed=[json.loads(x) for x in leaf_path.read_text().splitlines() if x.strip()]; leaf_path.unlink()
        for r in parsed: r["surface"]=surface; append_jsonl(leaf_path,r); leaf_rows.append(r)
    summaries,mechanism,hs=summarize_surface(configs,leaf_rows); return dict(configs),leaf_rows,summaries,mechanism,hs


def validate_beta(val_ds,cores,strongs,policies,out:Path):
    baseline=[]
    for seed in CORE_SEEDS: baseline.extend(evaluate_no_search(cores[seed],seed,val_ds,NO_SEARCH,out/"validation_no_search_rows.jsonl"))
    by_beta={}
    for beta in BETA_GRID:
        rows=[]
        for seed in CORE_SEEDS:
            srch=searcher_for(cores[seed],strongs[seed],policies[seed],depth=4,beta=beta,analysis_verifier=strongs[seed])
            rows.extend(evaluate_search(cores[seed],seed,val_ds,f"validation_beta{beta:g}",srch,out/"validation_search_rows.jsonl",out/"validation_leaf_rows.jsonl"))
        by_beta[beta]=rows
    selection=select_beta(by_beta,baseline); write_json(out/"validation_beta_selection.json",selection); return float(selection["selected_beta"]),selection


def overall_acceptance(dev_h1,conf_h1,dev_leaf,conf_leaf):
    h1=bool(dev_h1.get("h1_supported") and conf_h1.get("h1_supported"))
    def depth_collapse(leaf):
        d1=leaf.get("1",{}).get("improvement_auc"); d4=leaf.get("4",{}).get("improvement_auc")
        return d1 is not None and d4 is not None and d1>=0.60 and d4<=0.58
    h2=bool(depth_collapse(dev_leaf) and depth_collapse(conf_leaf))
    return {"pass":bool(h1 or h2),"confirmed_h1_target_value_mismatch":h1,"confirmed_h2_depth_ranking_collapse":h2}


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="outputs/m15_mechanism_ablations"); args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(2,os.cpu_count() or 1)); torch.set_num_interop_threads(min(2,os.cpu_count() or 1)); write_json(out/"environment.json",environment_record())
    cores,context_models,source_report=reconstruct_sources(out); train_ds,val_ds,dev_ds,dev_manifest=build_dev_hierarchy(out); prior_fp=manifest_fingerprints(dev_manifest)
    weaks={}; strongs={}; policies={}; uniforms={}; vqs={}; aux_records=[]
    for seed in CORE_SEEDS:
        weak,strong,_,val_traj,vrec=fit_verifiers(cores[seed],seed,train_ds,val_ds,out); policy,uniform,dirs,arec=fit_action_policy(cores[seed],seed,train_ds,out); vq,vqrec=fit_vq(cores[seed],seed,train_ds,out)
        weaks[seed]=weak; strongs[seed]=strong; policies[seed]=policy; uniforms[seed]=uniform; vqs[seed]=vq
        metrics=verifier_metrics_by_depth(cores[seed],strong,val_traj); write_json(out/f"verifier_validation_seed{seed}.json",metrics)
        aux_records.append({"core_seed":seed,"verifier":vrec,"action":arec,"vq":vqrec,"verifier_validation":metrics["by_depth"]})
    write_json(out/"auxiliary_training.json",aux_records); selected_beta,beta_selection=validate_beta(val_ds,cores,strongs,policies,out)
    dev_configs,dev_leaf_rows,dev_summaries,dev_leaf_summary,dev_h1=run_surface("development",dev_ds,cores,weaks,strongs,policies,uniforms,selected_beta,out)
    write_json(out/"development_summaries.json",dev_summaries); write_json(out/"development_leaf_mechanism.json",dev_leaf_summary); write_json(out/"development_hypothesis_test.json",dev_h1)
    write_json(out/"development_failure_cases.json",difficult_failure_cases(dev_configs,dev_configs[NO_SEARCH]))
    dev_vq_rows=[]
    for seed in CORE_SEEDS: dev_vq_rows.extend(trajectory_ablation(cores[seed],seed,dev_ds,vqs[seed],"development",out/"development_trajectory_rows.jsonl"))
    write_json(out/"development_trajectory_summary.json",trajectory_summary(dev_vq_rows)); context_report=precision_context(context_models,dev_ds,out)
    freeze=freeze_manifest(out,cores,strongs,weaks,policies,vqs,selected_beta,dev_h1)
    confirm_ds,confirm_manifest,confirm_overlap=build_frozen_set(out,seed=CONFIRM_DATA_SEED,n=CONFIRM_N,min_clues=6,max_clues=10,prior_fingerprints=prior_fp,filename="confirmation.json")
    combined_fp=prior_fp|manifest_fingerprints(confirm_manifest)
    shift_ds,shift_manifest,shift_overlap=build_frozen_set(out,seed=SHIFT_DATA_SEED,n=SHIFT_N,min_clues=4,max_clues=5,prior_fingerprints=combined_fp,filename="shift_low_clue.json")
    conf_configs,conf_leaf_rows,conf_summaries,conf_leaf_summary,conf_h1=run_surface("confirmation",confirm_ds,cores,weaks,strongs,policies,uniforms,selected_beta,out)
    write_json(out/"confirmation_summaries.json",conf_summaries); write_json(out/"confirmation_leaf_mechanism.json",conf_leaf_summary); write_json(out/"confirmation_hypothesis_test.json",conf_h1)
    write_json(out/"confirmation_failure_cases.json",difficult_failure_cases(conf_configs,conf_configs[NO_SEARCH]))
    shift_configs,shift_leaf_rows,shift_summaries,shift_leaf_summary,shift_h1=run_surface("shift_low_clue",shift_ds,cores,weaks,strongs,policies,uniforms,selected_beta,out)
    write_json(out/"shift_summaries.json",shift_summaries); write_json(out/"shift_leaf_mechanism.json",shift_leaf_summary); write_json(out/"shift_failure_cases.json",difficult_failure_cases(shift_configs,shift_configs[NO_SEARCH]))
    conf_vq_rows=[]
    for seed in CORE_SEEDS: conf_vq_rows.extend(trajectory_ablation(cores[seed],seed,confirm_ds,vqs[seed],"confirmation",out/"confirmation_trajectory_rows.jsonl"))
    write_json(out/"confirmation_trajectory_summary.json",trajectory_summary(conf_vq_rows))
    acceptance=overall_acceptance(dev_h1,conf_h1,dev_leaf_summary,conf_leaf_summary)
    preferred="target_value_mismatch" if acceptance["confirmed_h1_target_value_mismatch"] else ("depth_verifier_ranking_collapse" if acceptance["confirmed_h2_depth_ranking_collapse"] else "inconclusive")
    interpretation={"preferred_explanation":preferred,"development_h1":dev_h1,"confirmation_h1":conf_h1,
        "counterevidence":{"fp32_prevention_fraction_development":dev_h1.get("fp32_prevention_fraction"),"fp32_prevention_fraction_confirmation":conf_h1.get("fp32_prevention_fraction"),
            "oracle_regression_reduction_development":dev_h1.get("oracle_regression_reduction_fraction"),"oracle_regression_reduction_confirmation":conf_h1.get("oracle_regression_reduction_fraction"),
            "selected_uncertainty_beta":selected_beta},
        "limitations":["bounded two-core-seed 4x4 Sudoku pilot","terminal guard is task-specific symbolic verification, not generic learned halting",
                       "oracle absolute-quality search has different evaluator cost and is diagnostic only","M15 does not establish VQ preservation, OOD detection, a scaling law, or energy superiority"]}
    write_json(out/"causal_interpretation.json",interpretation)
    result_rows=[]
    for surface,summaries in [("development",dev_summaries),("confirmation",conf_summaries),("shift_low_clue",shift_summaries)]:
        for cid,s in summaries.items(): result_rows.append({"surface":surface,"config_id":cid,**s})
    write_csv(out/"result_table.csv",result_rows)
    summary={"milestone":15,"status":"COMPLETE" if acceptance["pass"] else "INCOMPLETE_MECHANISM_NOT_CONTROLLED","acceptance":acceptance,
        "protocol_commit":PROTOCOL_COMMIT,"clarification_commit":CLARIFICATION_COMMIT,"core_seeds":CORE_SEEDS,"selected_beta":selected_beta,
        "development_h1":dev_h1,"confirmation_h1":conf_h1,"preferred_explanation":preferred,"confirmation_opened_after_freeze":True,
        "confirmation_seed":CONFIRM_DATA_SEED,"shift_seed":SHIFT_DATA_SEED,"confirmation_overlap_count":confirm_overlap["overlap_count"],"shift_overlap_count":shift_overlap["overlap_count"],
        "source_tensor_identity_exact":all(r["tensor_identity_exact"] for r in source_report["primary"]),"precision_context_status":context_report["status"],"unsupported_theory_cleanup_required":True}
    write_json(out/"summary.json",summary); return 0 if acceptance["pass"] else 2


if __name__=="__main__":
    raise SystemExit(main())
