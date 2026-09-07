#!/usr/bin/env python3
"""M09 v3: budget-aligned root-state challenger policy.

This revision is defined in docs/M09_DEVELOPMENT_REVISION.md. It reuses the
original train-only candidate table/prototype construction, fits only on depth-0
train states, and never materializes untouched test inputs unless development
clears the unchanged practical gate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.seed import set_seed
from eval.action_checkpoint import load_m09_action_checkpoint, save_m09_action_checkpoint
from eval.grounded_targets import assert_frozen_reasoner, generate_trajectory_states, tensor_state_sha256
from model.latent_action import BudgetAlignedChallengerCodebook
from scripts._common import build_data_splits
from scripts.train_action_policy import (
    ACTION_DEV_PUZZLES,
    ACTION_FIT_PUZZLES,
    ACTION_SCALE,
    BENEFIT_DELTA,
    CANDIDATE_COUNT,
    CANDIDATE_SEED,
    DATA_SEED,
    DIRECTION_LOSS_WEIGHT,
    PARAMETER_CAP,
    POLICY_BATCH,
    POLICY_HIDDEN,
    POLICY_LR,
    POLICY_SEED,
    POLICY_STEPS,
    POLICY_WD,
    REASONER_STEPS,
    SELECTED_DIRECTIONS,
    TARGET_EQUIVALENT_CAP,
    TEST_N,
    TRAIN_N,
    TRAJECTORY_DEPTHS,
    UTILITY_TEMPERATURE,
    VAL_N,
    benefit_gate,
    comparison_suite,
    environment_record,
    greedy_coverage_select,
    id_hash,
    make_candidate_bank,
    make_cfg,
    restricted_utilities,
    save_comparison_rows,
    tensor_sha256,
    train_reasoner,
    transition_effect,
    utility_table,
    write_json,
    append_jsonl,
)

VERSION = "v3_budget_aligned_challenger"
FIRST_FAILED_RUN = 34138116310


def rows_to_tensors(rows):
    return (
        torch.stack([r["x"] for r in rows]).long(),
        torch.stack([r["y"] for r in rows]).float(),
        torch.stack([r["z"] for r in rows]).float(),
    )


def fit_challenger(train_rows, train_utilities, target_directions, curve_path: Path):
    """Fit the non-identity challenger distribution on root states only."""
    set_seed(POLICY_SEED, deterministic=True)
    module = BudgetAlignedChallengerCodebook(
        dim=48, num_tokens=10, n_actions=4, scale=ACTION_SCALE, hidden_dim=POLICY_HIDDEN
    )
    opt = torch.optim.AdamW(module.parameters(), lr=POLICY_LR, weight_decay=POLICY_WD)
    owned = {id(p) for group in opt.param_groups for p in group["params"]}
    expected = {id(p) for p in module.parameters()}
    if owned != expected:
        raise RuntimeError("v3 optimizer does not own exactly the action-module parameters")

    x, y, z = rows_to_tensors(train_rows)
    nonidentity_u = train_utilities[:, 1:]
    labels = nonidentity_u.argmax(dim=1)
    rng = np.random.default_rng(POLICY_SEED + 3000)
    if curve_path.exists(): curve_path.unlink()
    initial_dirs = module.directions.detach().clone()
    initial_head = module.policy_head.weight.detach().clone()
    first_step = None
    first_loss = last_loss = None

    for step in range(1, POLICY_STEPS + 1):
        idx = torch.from_numpy(rng.integers(0, len(train_rows), size=POLICY_BATCH, dtype=np.int64))
        module.train()
        logits = module.policy_logits_for_state(x[idx], y[idx], z[idx])
        u = nonidentity_u[idx]
        teacher = torch.softmax((u - u.max(dim=1, keepdim=True).values) / UTILITY_TEMPERATURE, dim=1)
        policy_loss = -(teacher * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        direction_loss = F.mse_loss(module.directions, target_directions)
        loss = policy_loss + DIRECTION_LOSS_WEIGHT * direction_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite v3 loss step={step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        dgrad = float(module.directions.grad.detach().norm()) if module.directions.grad is not None else 0.0
        pgrad = float(module.policy_head.weight.grad.detach().norm()) if module.policy_head.weight.grad is not None else 0.0
        total_grad = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(total_grad)):
            raise RuntimeError("non-finite v3 gradient")
        opt.step()
        first_loss = float(loss.detach()) if first_loss is None else first_loss
        last_loss = float(loss.detach())
        if step == 1:
            first_step = {
                "direction_grad_norm_preclip": dgrad,
                "policy_head_grad_norm_preclip": pgrad,
                "direction_change_after_step": float((module.directions.detach() - initial_dirs).norm()),
                "policy_head_change_after_step": float((module.policy_head.weight.detach() - initial_head).norm()),
            }
        if step == 1 or step % 25 == 0 or step == POLICY_STEPS:
            append_jsonl(curve_path, {
                "step": step, "version": VERSION, "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()), "direction_loss": float(direction_loss.detach()),
                "direction_grad_norm_preclip": dgrad, "policy_head_grad_norm_preclip": pgrad,
                "total_grad_norm_preclip": float(total_grad),
            })

    assert first_step is not None
    if min(first_step.values()) <= 0:
        raise RuntimeError(f"v3 intended gradients/updates were not all positive: {first_step}")
    module.eval()
    cos = F.cosine_similarity(module.directions.detach(), target_directions, dim=1)
    params = sum(p.numel() for p in module.parameters())
    if params >= PARAMETER_CAP:
        raise RuntimeError(f"v3 action module exceeds parameter cap: {params}")
    return module, opt, {
        "version": VERSION, "optimizer": "AdamW", "optimizer_owns_exact_action_module": True,
        "steps": POLICY_STEPS, "batch": POLICY_BATCH, "lr": POLICY_LR, "weight_decay": POLICY_WD,
        "first_loss": first_loss, "last_loss": last_loss, "first_step": first_step,
        "final_direction_change_l2": float((module.directions.detach() - initial_dirs).norm()),
        "final_policy_head_change_l2": float((module.policy_head.weight.detach() - initial_head).norm()),
        "direction_cosine_fidelity": cos.tolist(), "parameter_count": params,
        "curve": str(curve_path), "fit_depths": [0], "fit_states": len(train_rows),
    }


@torch.no_grad()
def challenger_diagnostics(module, rows, utilities):
    x, y, z = rows_to_tensors(rows)
    labels = utilities[:, 1:].argmax(dim=1)
    logits = []
    for start in range(0, len(rows), 128):
        logits.append(module.policy_logits_for_state(x[start:start+128], y[start:start+128], z[start:start+128]))
    logits = torch.cat(logits)
    pred = logits.argmax(dim=1)
    counts = torch.bincount(labels, minlength=3).float()
    probs = counts / max(1, len(labels))
    nz = probs[probs > 0]
    return {
        "challenger_top1_accuracy": float((pred == labels).float().mean()),
        "target_challenger_counts": [int(v) for v in counts],
        "target_challenger_frequencies": probs.tolist(),
        "target_challenger_entropy_nats": float(-(nz * nz.log()).sum()) if nz.numel() else 0.0,
        "predicted_challenger_frequencies": (
            torch.bincount(pred, minlength=3).float() / len(pred)
        ).tolist(),
    }


def scheduling_audit(comp: dict[str, Any]) -> dict[str, Any]:
    trained_rows = comp["baselines"]["trained"]["rows"]
    unguided_rows = comp["baselines"]["unguided"]["rows"]
    trained_ok = True
    unguided_ok = True
    trained_paths = {}
    for row in trained_rows:
        paths = row["work"].get("evaluated_paths", [])
        key = json.dumps(paths)
        trained_paths[key] = trained_paths.get(key, 0) + 1
        trained_ok &= (
            len(paths) == 2 and paths[0] == [0] and len(paths[1]) == 1 and paths[1][0] in {1,2,3}
        )
    for row in unguided_rows:
        paths = row["work"].get("evaluated_paths", [])
        unguided_ok &= (paths == [[0], [1]])
    return {
        "trained_identity_then_one_challenger": bool(trained_ok),
        "unguided_identity_then_action1": bool(unguided_ok),
        "trained_evaluated_path_patterns": trained_paths,
    }


def make_summary_failure(reasoner_report, target_report, training, diagnostics, dev_comp, schedule, freeze_hash):
    passed, checks = benefit_gate(dev_comp)
    checks = {**checks, "budget_aligned_schedule_realized": all([
        schedule["trained_identity_then_one_challenger"], schedule["unguided_identity_then_action1"]
    ])}
    return {
        "status": "incomplete_v3_development_benefit_not_demonstrated",
        "acceptance_pass": False, "selected_version": None,
        "reasoner": reasoner_report, "target_generation": target_report,
        "v3_training": training, "v3_diagnostics": diagnostics,
        "v3_development": {
            "pass": False, "checks": checks,
            "baseline_means": {k: v["mean_score"] for k,v in dev_comp["baselines"].items()},
            "paired_vs_unguided": dev_comp["paired_vs_unguided"],
            "work_totals": {k:v["work_totals"] for k,v in dev_comp["baselines"].items()},
            "scheduling_audit": schedule,
        },
        "prior_failed_run": FIRST_FAILED_RUN,
        "prior_failures": {
            "v1_mean_delta": -0.000850951402551598,
            "v2_mean_delta": 0.002762063251187404,
        },
        "reasoner_frozen_unchanged": freeze_hash,
        "test_inputs_evaluated": False,
        "next_executable_experiment": (
            "M09 remains incomplete. Preserve the candidate/prototype table and v3 root challenger result; "
            "a future preregistration should test richer root-state action-value supervision or a larger "
            "train-only direction basis before opening this untouched test split."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/m09_trained_actions_v3")
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    write_json(out / "environment.json", environment_record())

    cfg = make_cfg()
    datasets, _ = build_data_splits(cfg, TRAIN_N, VAL_N, TEST_N, seed=DATA_SEED, manifest_path=out/"data_manifest.json")
    core, reasoner_report = train_reasoner(out, cfg, datasets["train"], datasets["validation"])
    before = tensor_state_sha256(core.model)

    fit_inputs = torch.from_numpy(datasets["train"].inputs[:ACTION_FIT_PUZZLES]).long()
    fit_ids = datasets["train"].ids[:ACTION_FIT_PUZZLES]
    dev_inputs = torch.from_numpy(datasets["train"].inputs[ACTION_FIT_PUZZLES:ACTION_FIT_PUZZLES+ACTION_DEV_PUZZLES]).long()
    dev_ids = datasets["train"].ids[ACTION_FIT_PUZZLES:ACTION_FIT_PUZZLES+ACTION_DEV_PUZZLES]
    test_ids = datasets["test"].ids[:TEST_N]

    fit_rows = generate_trajectory_states(core.model, fit_inputs, fit_ids, max_depth=TRAJECTORY_DEPTHS)
    dev_rows = generate_trajectory_states(core.model, dev_inputs, dev_ids, max_depth=TRAJECTORY_DEPTHS)
    bank = make_candidate_bank(48)
    full_train_u, train_work = utility_table(core.model, fit_rows, bank, scale=ACTION_SCALE)
    if train_work["state_action_equivalents"] > TARGET_EQUIVALENT_CAP:
        raise RuntimeError("target-generation cap exceeded")
    selected, coverage = greedy_coverage_select(full_train_u, SELECTED_DIRECTIONS)
    target_dirs = bank[selected].clone()
    train_u = restricted_utilities(full_train_u, selected)
    dev_u, dev_work = utility_table(core.model, dev_rows, target_dirs, scale=ACTION_SCALE)

    # generate_trajectory_states is depth-major, so the first N rows are depth 0.
    root_fit_rows, root_train_u = fit_rows[:ACTION_FIT_PUZZLES], train_u[:ACTION_FIT_PUZZLES]
    root_dev_rows, root_dev_u = dev_rows[:ACTION_DEV_PUZZLES], dev_u[:ACTION_DEV_PUZZLES]
    if any(r["depth"] != 0 for r in root_fit_rows + root_dev_rows):
        raise RuntimeError("v3 root-state slice is not depth 0")

    utility_sha = tensor_sha256(full_train_u, bank, torch.tensor(selected, dtype=torch.int64))
    target_report = {
        "reference_target_used": False, "candidate_bank_seed": CANDIDATE_SEED,
        "candidate_bank_count": CANDIDATE_COUNT, "candidate_scale": ACTION_SCALE,
        "selected_candidate_indices": selected, "train_utility_table_sha256": utility_sha,
        "train_work": train_work, "development_diagnostic_work": dev_work,
        "coverage": coverage, "v3_policy_fit_states": ACTION_FIT_PUZZLES, "v3_policy_fit_depths": [0],
        "train_id_sha256": id_hash(fit_ids), "development_id_sha256": id_hash(dev_ids),
        "test_id_sha256": id_hash(test_ids),
    }
    torch.save({
        "candidate_bank": bank, "full_train_utilities": full_train_u,
        "selected_candidate_indices": torch.tensor(selected), "selected_directions": target_dirs,
        "restricted_train_utilities": train_u,
    }, out/"target_table.pt")
    write_json(out/"target_generation.json", target_report)

    module, optimizer, training = fit_challenger(
        root_fit_rows, root_train_u, target_dirs, out/"curves"/"v3_budget_aligned_challenger.jsonl"
    )
    diagnostics = challenger_diagnostics(module, root_dev_rows, root_dev_u)
    dev_comp = comparison_suite(core.model, dev_inputs, module, bank)
    save_comparison_rows(out/"comparisons"/"development_v3.jsonl", dev_comp)
    schedule = scheduling_audit(dev_comp)
    dev_pass, checks = benefit_gate(dev_comp)
    checks = {**checks, "budget_aligned_schedule_realized": bool(
        schedule["trained_identity_then_one_challenger"] and schedule["unguided_identity_then_action1"]
    )}
    dev_pass = bool(dev_pass and checks["budget_aligned_schedule_realized"])
    dev_report = {
        "pass": dev_pass, "checks": checks,
        "baseline_means": {k:v["mean_score"] for k,v in dev_comp["baselines"].items()},
        "paired_vs_unguided": dev_comp["paired_vs_unguided"],
        "paired_vs_fixed_random": dev_comp["paired_vs_fixed_random"],
        "paired_vs_identity": dev_comp["paired_vs_identity"],
        "work_totals": {k:v["work_totals"] for k,v in dev_comp["baselines"].items()},
        "scheduling_audit": schedule,
    }
    write_json(out/"v3_development.json", dev_report)

    after_dev = tensor_state_sha256(core.model)
    if not dev_pass:
        summary = make_summary_failure(
            reasoner_report, target_report, training, diagnostics, dev_comp, schedule, before == after_dev
        )
        write_json(out/"summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 3

    provenance = {
        "reference_target_used": False, "candidate_bank_seed": CANDIDATE_SEED,
        "candidate_bank_count": CANDIDATE_COUNT, "selected_candidate_indices": selected,
        "target_utility_table_sha256": utility_sha, "reasoner_tensor_state_sha256": before,
        "train_id_sha256": id_hash(fit_ids), "development_id_sha256": id_hash(dev_ids),
        "test_id_sha256": id_hash(test_ids), "target_generation_work": train_work,
        "training_method": VERSION, "state_representation": module.STATE_REPRESENTATION,
        "prior_semantics": module.PRIOR_SEMANTICS, "candidate_scale": ACTION_SCALE,
        "utility_oracle": "model.verifier.sudoku_score",
        "search_contract": {"rollouts": 2, "max_depth": 1, "c_puct": 1.5},
        "policy_fit_depths": [0],
    }
    ckpt = out/"action_policy.pt"
    save_m09_action_checkpoint(
        module, ckpt, core=core, optimizer=optimizer, trained_steps=POLICY_STEPS,
        fitted_version=VERSION, provenance=provenance,
    )
    loaded = load_m09_action_checkpoint(ckpt, core)

    # First access to untouched test inputs for action evaluation occurs here.
    test_inputs = torch.from_numpy(datasets["test"].inputs[:TEST_N]).long()
    test_comp = comparison_suite(core.model, test_inputs, loaded.module, bank)
    save_comparison_rows(out/"comparisons"/"untouched_test.jsonl", test_comp)
    test_schedule = scheduling_audit(test_comp)
    test_pass, test_checks = benefit_gate(test_comp)
    test_checks = {**test_checks, "budget_aligned_schedule_realized": bool(
        test_schedule["trained_identity_then_one_challenger"] and test_schedule["unguided_identity_then_action1"]
    )}
    test_pass = bool(test_pass and test_checks["budget_aligned_schedule_realized"])
    test_report = {
        "pass": test_pass, "checks": test_checks,
        "baseline_means": {k:v["mean_score"] for k,v in test_comp["baselines"].items()},
        "paired_vs_unguided": test_comp["paired_vs_unguided"],
        "paired_vs_fixed_random": test_comp["paired_vs_fixed_random"],
        "paired_vs_identity": test_comp["paired_vs_identity"],
        "work_totals": {k:v["work_totals"] for k,v in test_comp["baselines"].items()},
        "scheduling_audit": test_schedule, "test_examples": TEST_N,
        "reference_target_used": False,
    }
    write_json(out/"untouched_test.json", test_report)

    after = tensor_state_sha256(core.model)
    freeze = {
        "tensor_hash_before": before, "tensor_hash_after": after, "unchanged": before == after,
        "all_parameters_requires_grad_false": not any(p.requires_grad for p in core.model.parameters()),
        "reasoner_parameter_in_action_optimizer": bool(
            {id(p) for p in core.model.parameters()} & {id(p) for g in optimizer.param_groups for p in g["params"]}
        ),
    }
    write_json(out/"freeze_audit.json", freeze)
    if not freeze["unchanged"] or freeze["reasoner_parameter_in_action_optimizer"]:
        raise RuntimeError("reasoner freeze/optimizer ownership violated")

    commands = [
        "python scripts/train_action_policy_v3.py --out outputs/m09_trained_actions_v3",
        "python -m pytest tests/test_m09_trained_actions.py -q",
        "python -m pytest -m 'not slow' -ra",
    ]
    write_json(out/"exact_commands.json", commands)
    summary = {
        "status": "complete" if test_pass else "incomplete_heldout_benefit_not_confirmed",
        "acceptance_pass": test_pass, "selected_version": VERSION,
        "prior_failed_run": FIRST_FAILED_RUN,
        "prior_failures": {"v1_mean_delta": -0.000850951402551598, "v2_mean_delta": 0.002762063251187404},
        "reasoner": reasoner_report, "reasoner_freeze_audit": freeze,
        "target_generation": target_report, "selected_training": training,
        "development": dev_report, "untouched_test": test_report,
        "checkpoint": str(ckpt), "checkpoint_sha256": loaded.sha256,
        "checkpoint_format": loaded.payload["format"], "checkpoint_version": loaded.payload["version"],
        "test_inputs_evaluated_after_checkpoint_freeze": True,
        "reference_target_used": False, "exact_commands": commands,
        "transition_effect": transition_effect(core.model, loaded.module, root_fit_rows[0]),
    }
    write_json(out/"summary.json", summary)
    (out/"SUMMARY.md").write_text(
        "# M09 v3 Summary\n\n"
        f"- status: **{summary['status']}**\n"
        f"- development delta vs unguided: `{dev_report['paired_vs_unguided']['mean_delta']}`\n"
        f"- test delta vs unguided: `{test_report['paired_vs_unguided']['mean_delta']}`\n"
        f"- test wins/ties/losses: `{test_report['paired_vs_unguided']['wins']}/"
        f"{test_report['paired_vs_unguided']['ties']}/{test_report['paired_vs_unguided']['losses']}`\n"
        f"- checkpoint: `{loaded.sha256}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if test_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
