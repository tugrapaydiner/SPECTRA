#!/usr/bin/env python3
"""Milestone 09: train and evaluate a state-conditioned latent action mechanism."""
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config
from common.seed import set_seed
from eval.action_checkpoint import load_m09_action_checkpoint, save_m09_action_checkpoint
from eval.action_search import OracleScoreActionMCTS
from eval.checkpoint_eval import load_research_trm_checkpoint, sha256_file
from eval.grounded_targets import assert_frozen_reasoner, generate_trajectory_states, tensor_state_sha256
from model.latent_action import LatentActionCodebook, StateConditionedLatentActionCodebook
from model.verifier import sudoku_score
from scripts._common import build_data_splits, build_trm
from train.trainer import TrainConfig, Trainer

DATA_SEED = 20260909
REASONER_SEED = 1901
POLICY_SEED = 3901
CANDIDATE_SEED = 2901
BOOTSTRAP_SEED = 9901
TRAIN_N, VAL_N, TEST_N = 384, 96, 128
ACTION_FIT_PUZZLES = 240
ACTION_DEV_PUZZLES = 144
TRAJECTORY_DEPTHS = 4
REASONER_STEPS = 200
CANDIDATE_COUNT = 24
SELECTED_DIRECTIONS = 3
ACTION_SCALE = 0.5
POLICY_HIDDEN = 64
POLICY_STEPS = 400
POLICY_BATCH = 64
POLICY_LR = 2e-3
POLICY_WD = 0.01
DIRECTION_LOSS_WEIGHT = 0.5
UTILITY_TEMPERATURE = 0.02
SEARCH_ROLLOUTS = 2
SEARCH_DEPTH = 1
SEARCH_CPUCT = 1.5
BENEFIT_DELTA = 0.005
TARGET_EQUIVALENT_CAP = 25_000
PARAMETER_CAP = 50_000


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
        "git_sha": git_sha(), "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "cpu_model": cpu_model, "logical_cpus": os.cpu_count(),
        "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(), "torch_num_threads": torch.get_num_threads(),
        "experiment_device": "cpu",
    }


def id_hash(ids: Iterable[str]) -> str:
    h = hashlib.sha256()
    for item in ids:
        b = str(item).encode("utf-8")
        h.update(len(b).to_bytes(8, "little")); h.update(b)
    return h.hexdigest()


def tensor_sha256(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        cpu = tensor.detach().cpu().contiguous()
        h.update(str(cpu.dtype).encode("ascii"))
        h.update(np.asarray(cpu.shape, dtype="<i8").tobytes())
        h.update(cpu.numpy().tobytes())
    return h.hexdigest()


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


def train_reasoner(out: Path, cfg, train_ds, val_ds):
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
    return core, {"checkpoint": str(ckpt), "sha256": core.sha256, "elapsed_seconds": elapsed, "training": result}


def rows_to_tensors(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("empty trajectory row collection")
    return (
        torch.stack([r["x"] for r in rows]).long(),
        torch.stack([r["y"] for r in rows]).float(),
        torch.stack([r["z"] for r in rows]).float(),
    )


def make_candidate_bank(dim: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(CANDIDATE_SEED)
    bank = torch.randn(CANDIDATE_COUNT, dim, generator=gen)
    return bank / bank.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def utility_table(
    model,
    rows: list[dict[str, Any]],
    directions: torch.Tensor,
    *,
    scale: float,
    batch_size: int = 64,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Reference-free one-cycle utilities for identity + supplied directions."""
    assert_frozen_reasoner(model)
    x, y, z = rows_to_tensors(rows)
    columns: list[torch.Tensor] = []
    batched_cycles = 0
    decode_oracle_batches = 0
    action_count = int(directions.shape[0]) + 1
    for action in range(action_count):
        chunks = []
        for start in range(0, len(rows), batch_size):
            end = min(len(rows), start + batch_size)
            xb, yb, zb = x[start:end], y[start:end], z[start:end]
            x_emb = model.token_embed(xb) + model.encode_positions(xb, 9, 9)
            if action:
                zb = zb + float(scale) * directions[action - 1]
            y_next, _ = model.recursive_cycle(x_emb, yb, zb)
            batched_cycles += 1
            answer = model.out_head(y_next).argmax(dim=-1)
            chunks.append(sudoku_score(xb, answer, box=3).cpu())
            decode_oracle_batches += 1
        columns.append(torch.cat(chunks))
    table = torch.stack(columns, dim=1)
    return table, {
        "states": len(rows),
        "actions_including_identity": action_count,
        "state_action_equivalents": len(rows) * action_count,
        "batched_recursive_cycle_calls": batched_cycles,
        "batched_decode_oracle_calls": decode_oracle_batches,
        "batch_size": batch_size,
    }


def greedy_coverage_select(utilities: torch.Tensor, k: int) -> tuple[list[int], dict[str, float]]:
    if utilities.ndim != 2 or utilities.shape[1] != CANDIDATE_COUNT + 1:
        raise ValueError("expected identity + full candidate utility table")
    selected: list[int] = []  # 0-based candidate-bank indices
    best = utilities[:, 0].clone()
    identity_mean = float(best.mean())
    available = list(range(CANDIDATE_COUNT))
    coverage_trace = []
    for _ in range(k):
        best_idx = None
        best_mean = -math.inf
        for idx in available:
            candidate = torch.maximum(best, utilities[:, idx + 1])
            mean = float(candidate.mean())
            if mean > best_mean + 1e-15 or (abs(mean - best_mean) <= 1e-15 and (best_idx is None or idx < best_idx)):
                best_idx, best_mean = idx, mean
        assert best_idx is not None
        selected.append(best_idx)
        available.remove(best_idx)
        best = torch.maximum(best, utilities[:, best_idx + 1])
        coverage_trace.append(best_mean)
    return selected, {
        "identity_mean_utility": identity_mean,
        "selected_set_mean_best_utility": float(best.mean()),
        "coverage_gain": float(best.mean()) - identity_mean,
        "coverage_trace": coverage_trace,
    }


def restricted_utilities(full: torch.Tensor, selected: list[int]) -> torch.Tensor:
    cols = [0] + [idx + 1 for idx in selected]
    return full[:, cols]


def action_label_stats(utilities: torch.Tensor) -> dict[str, Any]:
    labels = utilities.argmax(dim=1)
    counts = torch.bincount(labels, minlength=utilities.shape[1]).float()
    probs = counts / max(1, labels.numel())
    nz = probs[probs > 0]
    entropy = float(-(nz * nz.log()).sum()) if nz.numel() else 0.0
    return {
        "counts": [int(v) for v in counts],
        "frequencies": probs.tolist(),
        "entropy_nats": entropy,
        "identity_best_rate": float((labels == 0).float().mean()),
    }


def _policy_parameter_ids(module: StateConditionedLatentActionCodebook) -> set[int]:
    return {id(p) for p in module.parameters()}


def fit_policy(
    *,
    version: str,
    train_rows: list[dict[str, Any]],
    train_utilities: torch.Tensor,
    target_directions: torch.Tensor,
    curve_path: Path,
) -> tuple[StateConditionedLatentActionCodebook, torch.optim.Optimizer, dict[str, Any]]:
    if version not in {"v1_hard_ce", "v2_utility_distillation"}:
        raise ValueError(version)
    set_seed(POLICY_SEED, deterministic=True)
    module = StateConditionedLatentActionCodebook(
        dim=48, num_tokens=10, n_actions=4, scale=ACTION_SCALE, hidden_dim=POLICY_HIDDEN
    )
    opt = torch.optim.AdamW(module.parameters(), lr=POLICY_LR, weight_decay=POLICY_WD)
    owned = {id(p) for group in opt.param_groups for p in group["params"]}
    if owned != _policy_parameter_ids(module):
        raise RuntimeError("optimizer does not own exactly the M09 action-module parameters")

    x, y, z = rows_to_tensors(train_rows)
    labels = train_utilities.argmax(dim=1)
    rng = np.random.default_rng(POLICY_SEED + (0 if version == "v1_hard_ce" else 1000))
    if curve_path.exists(): curve_path.unlink()

    initial_directions = module.directions.detach().clone()
    initial_policy = module.policy_head.weight.detach().clone()
    first_step = None
    first_loss = last_loss = None
    for step in range(1, POLICY_STEPS + 1):
        idx = torch.from_numpy(rng.integers(0, len(train_rows), size=POLICY_BATCH, dtype=np.int64))
        module.train()
        logits = module.policy_logits_for_state(x[idx], y[idx], z[idx])
        if version == "v1_hard_ce":
            policy_loss = F.cross_entropy(logits, labels[idx])
        else:
            u = train_utilities[idx]
            teacher = torch.softmax((u - u.max(dim=1, keepdim=True).values) / UTILITY_TEMPERATURE, dim=1)
            policy_loss = -(teacher * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        direction_loss = F.mse_loss(module.directions, target_directions)
        loss = policy_loss + DIRECTION_LOSS_WEIGHT * direction_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite M09 loss {version} step={step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        dir_grad = float(module.directions.grad.detach().norm()) if module.directions.grad is not None else 0.0
        head_grad = (
            float(module.policy_head.weight.grad.detach().norm())
            if module.policy_head.weight.grad is not None else 0.0
        )
        total_grad = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(total_grad)):
            raise RuntimeError("non-finite M09 gradient")
        opt.step()
        first_loss = float(loss.detach()) if first_loss is None else first_loss
        last_loss = float(loss.detach())
        if step == 1:
            first_step = {
                "direction_grad_norm_preclip": dir_grad,
                "policy_head_grad_norm_preclip": head_grad,
                "direction_change_after_step": float((module.directions.detach() - initial_directions).norm()),
                "policy_head_change_after_step": float((module.policy_head.weight.detach() - initial_policy).norm()),
            }
        if step == 1 or step % 25 == 0 or step == POLICY_STEPS:
            append_jsonl(curve_path, {
                "step": step, "version": version, "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()), "direction_loss": float(direction_loss.detach()),
                "direction_grad_norm_preclip": dir_grad,
                "policy_head_grad_norm_preclip": head_grad,
                "total_grad_norm_preclip": float(total_grad),
            })

    assert first_step is not None
    if first_step["direction_grad_norm_preclip"] <= 0 or first_step["policy_head_grad_norm_preclip"] <= 0:
        raise RuntimeError(f"M09 intended gradients were zero: {first_step}")
    if first_step["direction_change_after_step"] <= 0 or first_step["policy_head_change_after_step"] <= 0:
        raise RuntimeError(f"M09 intended parameters did not move: {first_step}")

    module.eval()
    cos = F.cosine_similarity(module.directions.detach(), target_directions, dim=1)
    param_count = sum(p.numel() for p in module.parameters())
    if param_count >= PARAMETER_CAP:
        raise RuntimeError(f"M09 action module exceeds parameter cap: {param_count}")
    return module, opt, {
        "version": version,
        "optimizer": "AdamW",
        "optimizer_owns_exact_action_module": True,
        "steps": POLICY_STEPS, "batch": POLICY_BATCH, "lr": POLICY_LR, "weight_decay": POLICY_WD,
        "first_loss": first_loss, "last_loss": last_loss, "first_step": first_step,
        "final_direction_change_l2": float((module.directions.detach() - initial_directions).norm()),
        "final_policy_head_change_l2": float((module.policy_head.weight.detach() - initial_policy).norm()),
        "direction_cosine_fidelity": cos.tolist(),
        "parameter_count": param_count,
        "curve": str(curve_path),
    }


@torch.no_grad()
def policy_diagnostics(
    module: StateConditionedLatentActionCodebook,
    rows: list[dict[str, Any]],
    utilities: torch.Tensor,
) -> dict[str, Any]:
    x, y, z = rows_to_tensors(rows)
    labels = utilities.argmax(dim=1)
    logits = []
    for start in range(0, len(rows), 128):
        logits.append(module.policy_logits_for_state(x[start:start+128], y[start:start+128], z[start:start+128]))
    logits = torch.cat(logits)
    pred = logits.argmax(dim=1)
    top2 = logits.topk(k=2, dim=1).indices
    return {
        "top1_accuracy": float((pred == labels).float().mean()),
        "top2_oracle_action_recall": float((top2 == labels[:, None]).any(dim=1).float().mean()),
        "predicted_action_frequencies": (
            torch.bincount(pred, minlength=utilities.shape[1]).float() / len(rows)
        ).tolist(),
        "target": action_label_stats(utilities),
    }


def make_uniform_codebook(directions: torch.Tensor) -> LatentActionCodebook:
    cb = LatentActionCodebook(dim=48, n_actions=4, scale=ACTION_SCALE)
    with torch.no_grad():
        cb.directions.copy_(directions)
        cb.prior_logits.zero_()
    for p in cb.parameters(): p.requires_grad_(False)
    return cb.eval()


def make_identity_codebook() -> LatentActionCodebook:
    cb = LatentActionCodebook(dim=48, n_actions=1, scale=ACTION_SCALE)
    with torch.no_grad(): cb.prior_logits.zero_()
    for p in cb.parameters(): p.requires_grad_(False)
    return cb.eval()


def aggregate_work(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = [
        "transition_calls", "recursive_cycle_calls", "verifier_calls", "verifier_evaluations",
        "state_conditioned_prior_calls", "global_prior_calls", "policy_forward_calls",
        "oracle_evaluator_calls", "oracle_decode_calls", "decode_calls",
    ]
    return {k: int(sum(int(r["work"].get(k, 0)) for r in rows)) for k in keys}


@torch.no_grad()
def run_search_baseline(model, puzzles: torch.Tensor, codebook, name: str) -> dict[str, Any]:
    rows = []
    for i in range(puzzles.shape[0]):
        x = puzzles[i:i+1]
        controller = OracleScoreActionMCTS(
            model, None, codebook, height=9, width=9,
            n_rollouts=SEARCH_ROLLOUTS, c_puct=SEARCH_CPUCT, max_depth=SEARCH_DEPTH,
        )
        best = controller.search(x)
        answer = controller.decode(best)
        score = float(sudoku_score(x, answer, box=3)[0])
        if controller.best_value is None or abs(score - float(controller.best_value)) > 1e-6:
            raise RuntimeError("final decode disagrees with M09 search evaluator value")
        rows.append({
            "index": i, "score": score, "chosen_path": list(best.path),
            "chosen_action": (int(best.path[0]) if best.path else None),
            "prediction": answer[0].cpu().tolist(), "work": dict(controller.last_search_stats),
        })
    return {"name": name, "mean_score": float(np.mean([r["score"] for r in rows])), "rows": rows,
            "work_totals": aggregate_work(rows)}


def paired_report(trained: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    a = np.asarray([r["score"] for r in trained["rows"]], dtype=np.float64)
    b = np.asarray([r["score"] for r in other["rows"]], dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired comparisons require aligned examples")
    d = a - b
    eps = 1e-12
    wins, losses = int((d > eps).sum()), int((d < -eps).sum())
    ties = int(len(d) - wins - losses)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = np.empty(2000, dtype=np.float64)
    for i in range(len(boots)):
        idx = rng.integers(0, len(d), size=len(d))
        boots[i] = d[idx].mean()
    return {
        "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
        "wins": wins, "ties": ties, "losses": losses,
        "bootstrap_95_mean_delta": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
    }


def comparison_suite(model, puzzles: torch.Tensor, trained_module, candidate_bank: torch.Tensor) -> dict[str, Any]:
    trained = run_search_baseline(model, puzzles, trained_module, "trained_state_conditioned")
    unguided = run_search_baseline(model, puzzles, make_uniform_codebook(trained_module.directions.detach()), "unguided_same_directions")
    fixed_random = run_search_baseline(model, puzzles, make_uniform_codebook(candidate_bank[:3]), "fixed_random_uniform")
    identity = run_search_baseline(model, puzzles, make_identity_codebook(), "identity_only")
    return {
        "baselines": {
            "trained": trained, "unguided": unguided,
            "fixed_random": fixed_random, "identity": identity,
        },
        "paired_vs_unguided": paired_report(trained, unguided),
        "paired_vs_fixed_random": paired_report(trained, fixed_random),
        "paired_vs_identity": paired_report(trained, identity),
    }


def benefit_gate(comp: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    b = comp["baselines"]
    paired = comp["paired_vs_unguided"]
    trained_work = b["trained"]["work_totals"]
    unguided_work = b["unguided"]["work_totals"]
    cost_equal = all(
        trained_work[k] == unguided_work[k]
        for k in ("transition_calls", "recursive_cycle_calls", "verifier_evaluations", "oracle_evaluator_calls")
    )
    checks = {
        "mean_delta_at_least_0_005": paired["mean_delta"] >= BENEFIT_DELTA,
        "paired_wins_gt_losses": paired["wins"] > paired["losses"],
        "not_worse_than_fixed_random": b["trained"]["mean_score"] >= b["fixed_random"]["mean_score"] - 1e-12,
        "not_worse_than_identity": b["trained"]["mean_score"] >= b["identity"]["mean_score"] - 1e-12,
        "equal_primary_search_work": cost_equal,
    }
    return all(checks.values()), checks


def transition_effect(model, module, row: dict[str, Any]) -> dict[str, float]:
    x = row["x"].unsqueeze(0); y = row["y"].unsqueeze(0); z = row["z"].unsqueeze(0)
    with torch.no_grad():
        x_emb = model.token_embed(x) + model.encode_positions(x, 9, 9)
        y0, z0 = model.recursive_cycle(x_emb, y, module.apply_action(z, 0))
        y1, z1 = model.recursive_cycle(x_emb, y, module.apply_action(z, 1))
    return {
        "action_input_delta_l2": float((module.apply_action(z, 1) - z).norm()),
        "transition_y_delta_l2": float((y1 - y0).norm()),
        "transition_z_delta_l2": float((z1 - z0).norm()),
    }


def failure_diagnosis(
    report: dict[str, Any], diagnostics: dict[str, Any], coverage: dict[str, Any], training: dict[str, Any]
) -> dict[str, Any]:
    paired = report["paired_vs_unguided"]
    return {
        "development_mean_delta": paired["mean_delta"],
        "development_wins_ties_losses": [paired["wins"], paired["ties"], paired["losses"]],
        "target_quality": {
            "coverage_gain": coverage["coverage_gain"],
            "identity_best_rate": diagnostics["target"]["identity_best_rate"],
            "label_entropy_nats": diagnostics["target"]["entropy_nats"],
        },
        "state_conditioning": {
            "top1_accuracy": diagnostics["top1_accuracy"],
            "top2_oracle_action_recall": diagnostics["top2_oracle_action_recall"],
            "predicted_action_frequencies": diagnostics["predicted_action_frequencies"],
        },
        "optimization": {
            "first_step": training["first_step"],
            "direction_cosine_fidelity": training["direction_cosine_fidelity"],
            "first_loss": training["first_loss"], "last_loss": training["last_loss"],
        },
        "search_integration": {
            "trained_work": report["baselines"]["trained"]["work_totals"],
            "unguided_work": report["baselines"]["unguided"]["work_totals"],
        },
        "revision": "v2 utility distillation on the unchanged utility/prototype table" ,
    }


def save_comparison_rows(path: Path, comp: dict[str, Any]) -> None:
    if path.exists(): path.unlink()
    names = ["trained", "unguided", "fixed_random", "identity"]
    baselines = comp["baselines"]
    n = len(baselines["trained"]["rows"])
    for i in range(n):
        append_jsonl(path, {
            "index": i,
            **{f"{name}_score": baselines[name]["rows"][i]["score"] for name in names},
            **{f"{name}_action": baselines[name]["rows"][i]["chosen_action"] for name in names},
            "trained_minus_unguided": (
                baselines["trained"]["rows"][i]["score"] - baselines["unguided"]["rows"][i]["score"]
            ),
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/m09_trained_actions")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    write_json(out / "environment.json", environment_record())

    cfg = make_cfg()
    datasets, manifest = build_data_splits(
        cfg, TRAIN_N, VAL_N, TEST_N, seed=DATA_SEED, manifest_path=out / "data_manifest.json"
    )
    core, reasoner_report = train_reasoner(out, cfg, datasets["train"], datasets["validation"])
    reasoner_hash_before = tensor_state_sha256(core.model)

    fit_inputs = torch.from_numpy(datasets["train"].inputs[:ACTION_FIT_PUZZLES]).long()
    fit_ids = datasets["train"].ids[:ACTION_FIT_PUZZLES]
    dev_inputs = torch.from_numpy(datasets["train"].inputs[ACTION_FIT_PUZZLES:ACTION_FIT_PUZZLES+ACTION_DEV_PUZZLES]).long()
    dev_ids = datasets["train"].ids[ACTION_FIT_PUZZLES:ACTION_FIT_PUZZLES+ACTION_DEV_PUZZLES]
    test_ids = datasets["test"].ids[:TEST_N]

    fit_rows = generate_trajectory_states(core.model, fit_inputs, fit_ids, max_depth=TRAJECTORY_DEPTHS)
    dev_rows = generate_trajectory_states(core.model, dev_inputs, dev_ids, max_depth=TRAJECTORY_DEPTHS)
    bank = make_candidate_bank(48)
    full_train_utilities, train_target_work = utility_table(core.model, fit_rows, bank, scale=ACTION_SCALE)
    if train_target_work["state_action_equivalents"] > TARGET_EQUIVALENT_CAP:
        raise RuntimeError(f"target-generation cap exceeded: {train_target_work}")
    selected, coverage = greedy_coverage_select(full_train_utilities, SELECTED_DIRECTIONS)
    targets = bank[selected].clone()
    train_u = restricted_utilities(full_train_utilities, selected)
    dev_u, dev_target_work = utility_table(core.model, dev_rows, targets, scale=ACTION_SCALE)
    utility_sha = tensor_sha256(full_train_utilities, bank, torch.tensor(selected, dtype=torch.int64))
    torch.save({
        "candidate_bank": bank, "full_train_utilities": full_train_utilities,
        "selected_candidate_indices": torch.tensor(selected), "selected_directions": targets,
        "restricted_train_utilities": train_u,
    }, out / "target_table.pt")
    target_report = {
        "reference_target_used": False,
        "candidate_bank_seed": CANDIDATE_SEED, "candidate_bank_count": CANDIDATE_COUNT,
        "selected_candidate_indices": selected, "candidate_scale": ACTION_SCALE,
        "train_utility_table_sha256": utility_sha,
        "train_work": train_target_work, "development_diagnostic_work": dev_target_work,
        "coverage": coverage, "train_action_labels": action_label_stats(train_u),
        "train_id_sha256": id_hash(fit_ids), "development_id_sha256": id_hash(dev_ids),
        "test_id_sha256": id_hash(test_ids),
    }
    write_json(out / "target_generation.json", target_report)

    versions = []
    selected_version = None
    selected_module = selected_optimizer = selected_training = None
    for version in ("v1_hard_ce", "v2_utility_distillation"):
        if version == "v2_utility_distillation" and versions and versions[0]["development_pass"]:
            break
        module, optimizer, train_report = fit_policy(
            version=version, train_rows=fit_rows, train_utilities=train_u,
            target_directions=targets, curve_path=out / "curves" / f"{version}.jsonl",
        )
        diagnostics = policy_diagnostics(module, dev_rows, dev_u)
        dev_comp = comparison_suite(core.model, dev_inputs, module, bank)
        dev_pass, dev_checks = benefit_gate(dev_comp)
        save_comparison_rows(out / "comparisons" / f"development_{version}.jsonl", dev_comp)
        row = {
            "version": version, "training": train_report, "diagnostics": diagnostics,
            "development": {
                "pass": dev_pass, "checks": dev_checks,
                "baseline_means": {k: v["mean_score"] for k, v in dev_comp["baselines"].items()},
                "paired_vs_unguided": dev_comp["paired_vs_unguided"],
                "work_totals": {k: v["work_totals"] for k, v in dev_comp["baselines"].items()},
            },
            "transition_effect": transition_effect(core.model, module, fit_rows[0]),
        }
        if not dev_pass:
            row["failure_diagnosis"] = failure_diagnosis(dev_comp, diagnostics, coverage, train_report)
        versions.append({"development_pass": dev_pass, **row})
        write_json(out / f"{version}_report.json", row)
        if dev_pass:
            selected_version, selected_module, selected_optimizer, selected_training = version, module, optimizer, train_report
            break

    write_json(out / "development_versions.json", versions)
    if selected_module is None:
        reasoner_hash_after = tensor_state_sha256(core.model)
        summary = {
            "status": "incomplete_development_benefit_not_demonstrated",
            "acceptance_pass": False, "reasoner": reasoner_report,
            "reasoner_frozen_unchanged": reasoner_hash_before == reasoner_hash_after,
            "target_generation": target_report, "versions": versions,
            "test_inputs_evaluated": False,
            "next_executable_experiment": (
                "Preserve the frozen utility table and investigate richer state-conditioned action values "
                "or a larger train-only prototype coverage set under a new preregistered milestone revision; "
                "do not inspect the M09 test split to tune this failed design."
            ),
        }
        write_json(out / "summary.json", summary)
        print(json.dumps(_plain(summary), indent=2, sort_keys=True))
        return 3

    assert selected_optimizer is not None and selected_training is not None and selected_version is not None
    provenance = {
        "reference_target_used": False,
        "candidate_bank_seed": CANDIDATE_SEED, "candidate_bank_count": CANDIDATE_COUNT,
        "selected_candidate_indices": selected,
        "target_utility_table_sha256": utility_sha,
        "reasoner_tensor_state_sha256": reasoner_hash_before,
        "train_id_sha256": id_hash(fit_ids), "development_id_sha256": id_hash(dev_ids),
        "test_id_sha256": id_hash(test_ids),
        "target_generation_work": train_target_work,
        "training_method": selected_version,
        "state_representation": StateConditionedLatentActionCodebook.STATE_REPRESENTATION,
        "prior_semantics": StateConditionedLatentActionCodebook.PRIOR_SEMANTICS,
        "candidate_scale": ACTION_SCALE,
        "utility_oracle": "model.verifier.sudoku_score",
        "search_contract": {"rollouts": SEARCH_ROLLOUTS, "max_depth": SEARCH_DEPTH, "c_puct": SEARCH_CPUCT},
    }
    action_ckpt = out / "action_policy.pt"
    save_m09_action_checkpoint(
        selected_module, action_ckpt, core=core, optimizer=selected_optimizer,
        trained_steps=POLICY_STEPS, fitted_version=selected_version, provenance=provenance,
    )
    loaded = load_m09_action_checkpoint(action_ckpt, core)

    # Untouched test inputs are first materialized for action evaluation here, after
    # version selection and checkpoint save/reload are complete.
    test_inputs = torch.from_numpy(datasets["test"].inputs[:TEST_N]).long()
    test_comp = comparison_suite(core.model, test_inputs, loaded.module, bank)
    test_pass, test_checks = benefit_gate(test_comp)
    save_comparison_rows(out / "comparisons" / "untouched_test.jsonl", test_comp)
    test_report = {
        "pass": test_pass, "checks": test_checks,
        "baseline_means": {k: v["mean_score"] for k, v in test_comp["baselines"].items()},
        "paired_vs_unguided": test_comp["paired_vs_unguided"],
        "paired_vs_fixed_random": test_comp["paired_vs_fixed_random"],
        "paired_vs_identity": test_comp["paired_vs_identity"],
        "work_totals": {k: v["work_totals"] for k, v in test_comp["baselines"].items()},
        "test_examples": TEST_N,
        "reference_target_used": False,
    }
    write_json(out / "untouched_test.json", test_report)

    reasoner_hash_after = tensor_state_sha256(core.model)
    freeze_audit = {
        "tensor_hash_before": reasoner_hash_before, "tensor_hash_after": reasoner_hash_after,
        "unchanged": reasoner_hash_before == reasoner_hash_after,
        "all_parameters_requires_grad_false": not any(p.requires_grad for p in core.model.parameters()),
        "training_flag": bool(core.model.training),
        "reasoner_parameter_in_action_optimizer": bool(
            {id(p) for p in core.model.parameters()} &
            {id(p) for group in selected_optimizer.param_groups for p in group["params"]}
        ),
    }
    if not freeze_audit["unchanged"] or freeze_audit["reasoner_parameter_in_action_optimizer"]:
        raise RuntimeError(f"M09 frozen reasoner contract failed: {freeze_audit}")
    write_json(out / "freeze_audit.json", freeze_audit)

    exact_commands = [
        "python scripts/train_action_policy.py --out outputs/m09_trained_actions",
        "python -m pytest tests/test_m09_trained_actions.py -q",
        "python -m pytest -m 'not slow' -ra",
    ]
    write_json(out / "exact_commands.json", exact_commands)
    summary = {
        "status": "complete" if test_pass else "incomplete_heldout_benefit_not_confirmed",
        "acceptance_pass": bool(test_pass),
        "selected_version": selected_version,
        "reasoner": reasoner_report, "reasoner_freeze_audit": freeze_audit,
        "target_generation": target_report,
        "selected_training": selected_training,
        "development": next(v for v in versions if v["version"] == selected_version)["development"],
        "untouched_test": test_report,
        "checkpoint": str(action_ckpt), "checkpoint_sha256": loaded.sha256,
        "checkpoint_format": loaded.payload["format"],
        "checkpoint_version": loaded.payload["version"],
        "test_inputs_evaluated_after_checkpoint_freeze": True,
        "reference_target_used": False,
        "exact_commands": exact_commands,
    }
    write_json(out / "summary.json", summary)
    md = [
        "# M09 Trained Actions Summary", "",
        f"- status: **{summary['status']}**",
        f"- selected version: `{selected_version}`",
        f"- checkpoint SHA-256: `{loaded.sha256}`",
        f"- development mean delta vs unguided: `{summary['development']['paired_vs_unguided']['mean_delta']}`",
        f"- test mean delta vs unguided: `{test_report['paired_vs_unguided']['mean_delta']}`",
        f"- test wins/ties/losses: `{test_report['paired_vs_unguided']['wins']}/{test_report['paired_vs_unguided']['ties']}/{test_report['paired_vs_unguided']['losses']}`",
        f"- test reference answers used: `{summary['reference_target_used']}`",
        "",
        "This result concerns the preregistered one-ply/two-evaluation symbolic-score action-isolation search only.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if test_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
