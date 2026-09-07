#!/usr/bin/env python3
"""M14 Attempt 4: front-loaded anytime supervision + semantic early exit.

Protocol: docs/M14_ATTEMPT4_PROTOCOL.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.measurement_env import measurement_environment
from eval.edge_energy import energy_counter_inventory
from model.single_stream_trm import InputConditionedSingleStreamTRM
import scripts.m14_attempt2_single_stream as a2
import scripts.m14_attempt3_semantic_early_exit as a3
import scripts.m14_attempt3_run as a3runtime  # installs exactly-once semantic timing into a3
from scripts.m14_primary_experiment import (
    CONFIRMATION_COST_N,
    CONFIRM_DATA_SEED,
    DEVELOPMENT_COST_N,
    DEV_N,
    LATENCY_REPEATS,
    MODEL_SEEDS,
    PRIMARY_BASELINE,
    RESERVE_CONFIRM_DATA_SEED,
    TRAIN_N,
    TRAIN_STEPS,
    VAL_N,
    write_json,
    sha256_file,
)

ATTEMPT4_PROTOCOL_COMMIT = "7b2ec6fb0c3264307b9623603431072c9c62b769"
FRONT_LOADED_WEIGHTS = [0.4, 0.3, 0.2, 0.1]
ATTEMPT3_RUN = 34169488811
ATTEMPT3_HEAD = "d1e491f412494c0756753bfa4b78b8002133d39c"
ATTEMPT3_ARTIFACT_ID = 10035348365
ATTEMPT3_ARTIFACT_SHA256 = "fb653a307d918f8e010d6648694ef2bdeb08654bee9faaf093f1f71ff07eaaac"


def frontloaded_candidate_training_loss(
    model: InputConditionedSingleStreamTRM,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Fixed reverse-of-historical anytime CE schedule; not a searched weight set."""
    _, steps = model(x, height=4, width=4)
    if len(steps) != 4:
        raise RuntimeError(f"Attempt 4 requires four supervision outputs, got {len(steps)}")
    return sum(
        float(w) * a2.blank_ce(step["logits"], x, y)
        for w, step in zip(FRONT_LOADED_WEIGHTS, steps)
    )


# Patch the Attempt-2 training helper before any training object is constructed.
# train_all/train_one resolve candidate_training_loss from the a2 module globals.
a2.candidate_training_loss = frontloaded_candidate_training_loss


def freeze_candidate(candidate: str, out: Path) -> dict:
    files = []
    for seed in MODEL_SEEDS:
        cp = out / "checkpoints" / f"{a2.CANDIDATE_KIND}_seed{seed}.pt"
        bp = out / "checkpoints" / f"single_pass_seed{seed}.pt"
        files += [
            {"role": "candidate", "seed": seed, "path": str(cp), "sha256": sha256_file(cp)},
            {"role": "primary_baseline", "seed": seed, "path": str(bp), "sha256": sha256_file(bp)},
        ]
    rec = {
        "attempt": 4,
        "candidate_config_id": candidate,
        "architecture_semantics": InputConditionedSingleStreamTRM.SEMANTICS,
        "candidate_training_weights": FRONT_LOADED_WEIGHTS,
        "candidate_training_objective": "front_loaded_anytime_blank_cell_cross_entropy_v1",
        "termination_rule": "stop_if_sudoku_semantic_validity_true_else_continue_to_frozen_max_budget",
        "learned_halter_used": False,
        "files": files,
        "frozen_before_confirmation": True,
        "confirmation_seed": CONFIRM_DATA_SEED,
        "git_sha": git_sha(),
        "timestamp_unix": time.time(),
    }
    write_json(out / "confirmation_freeze_manifest.json", rec)
    return rec


def git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".m14a4/experiment")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    write_json(out / "prior_attempts.json", {
        "adaptive_development_attempts_before_attempt4": 3,
        "attempt1": {
            "run": a2.ATTEMPT1_RUN, "head": a2.ATTEMPT1_HEAD,
            "artifact_id": a2.ATTEMPT1_ARTIFACT_ID,
            "artifact_zip_sha256": a2.ATTEMPT1_ARTIFACT_SHA256,
            "status": "INCOMPLETE", "confirmation_opened": False,
        },
        "attempt2": {
            "run": a3.ATTEMPT2_RUN, "head": a3.ATTEMPT2_HEAD,
            "artifact_id": a3.ATTEMPT2_ARTIFACT_ID,
            "artifact_zip_sha256": a3.ATTEMPT2_ARTIFACT_SHA256,
            "status": "INCOMPLETE", "confirmation_opened": False,
            "development_quality_difference": -0.640625,
            "development_latency_ratio": 0.8557505964071943,
        },
        "attempt3": {
            "run": ATTEMPT3_RUN, "head": ATTEMPT3_HEAD,
            "artifact_id": ATTEMPT3_ARTIFACT_ID,
            "artifact_zip_sha256": ATTEMPT3_ARTIFACT_SHA256,
            "status": "INCOMPLETE", "confirmation_opened": False,
            "selected_candidate": "semantic_exit_k1",
            "development_quality_difference": -0.637890625,
            "development_quality_ci95": [-0.6625, -0.612890625],
            "development_latency_ratio": 0.8127430466852386,
            "development_latency_ci95": [0.806199172466283, 0.822727757910161],
        },
    })
    write_json(out / "training_intervention.json", {
        "attempt": 4,
        "candidate_objective": "front_loaded_anytime_blank_cell_cross_entropy_v1",
        "historical_weights": [0.1, 0.2, 0.3, 0.4],
        "attempt4_weights": FRONT_LOADED_WEIGHTS,
        "weight_sum": sum(FRONT_LOADED_WEIGHTS),
        "selection_method": "exact_reverse_of_historical_schedule_not_a_weight_sweep",
        "baseline_objective_changed": False,
        "architecture_changed_vs_attempt3": False,
        "inference_rule_changed_vs_attempt3": False,
    })
    write_json(out / "environment.json", measurement_environment(
        backend="pytorch_cpu_frontloaded_anytime_semantic_early_exit",
        compiler_flags=None,
        extra={
            "milestone": 14, "attempt": 4, "git_sha": git_sha(),
            "protocol": "docs/M14_ATTEMPT4_PROTOCOL.md",
            "primary_cost": "complete_solve_latency",
        },
    ))
    write_json(out / "energy_inventory.json", energy_counter_inventory())

    train_ds, val_ds, dev_ds, dev_manifest = a3.build_development_data(out)
    training = a2.train_all(train_ds, val_ds, out)
    # Make the objective identity explicit beside every aggregate training record.
    write_json(out / "training_objective_audit.json", {
        "candidate_kind": a2.CANDIDATE_KIND,
        "weights": FRONT_LOADED_WEIGHTS,
        "function": "scripts.m14_attempt4_anytime_supervision.frontloaded_candidate_training_loss",
        "candidate_runs": [
            {"seed": r["seed"], "checkpoint": r["checkpoint"], "checkpoint_sha256": r["checkpoint_sha256"]}
            for r in training if r["kind"] == a2.CANDIDATE_KIND
        ],
        "baseline_runs": [
            {"seed": r["seed"], "checkpoint": r["checkpoint"], "checkpoint_sha256": r["checkpoint_sha256"]}
            for r in training if r["kind"] == "single_pass"
        ],
    })

    _, selection = a3.run_validation(val_ds, out)
    candidate = str(selection["selected_config_id"])
    _, dev_effect, dev_gate = a3.evaluate_surface(
        candidate, dev_ds, "development", out,
        cost_n=DEVELOPMENT_COST_N, repeats=LATENCY_REPEATS,
    )
    a3.representative_energy(candidate, dev_ds, out, "development")

    confirmation_opened = False
    confirmation_effect = None
    confirmation_gate = None
    confirmation_audit = None
    if dev_gate["pass"]:
        freeze_candidate(candidate, out)
        conf_ds, _, confirmation_audit = a3.build_confirmation_data(out, dev_manifest)
        confirmation_opened = True
        _, confirmation_effect, confirmation_gate = a3.evaluate_surface(
            candidate, conf_ds, "confirmation", out,
            cost_n=CONFIRMATION_COST_N, repeats=LATENCY_REPEATS,
        )
        a3.representative_energy(candidate, conf_ds, out, "confirmation")

    status = "COMPLETE" if confirmation_gate is not None and confirmation_gate["pass"] else "INCOMPLETE"
    summary = {
        "milestone": 14,
        "attempt": 4,
        "status": status,
        "claim_confirmed": bool(status == "COMPLETE"),
        "protocol_committed_before_attempt4_results": True,
        "protocol_commit": ATTEMPT4_PROTOCOL_COMMIT,
        "adaptive_development_attempts_before_confirmation": 4,
        "prior_attempt_count": 3,
        "primary_metric": "strict_sudoku_semantic_validity",
        "primary_baseline": PRIMARY_BASELINE,
        "candidate_family": "single_stream_frontloaded_anytime_with_symbolic_semantic_early_exit",
        "candidate_training_weights": FRONT_LOADED_WEIGHTS,
        "selected_candidate": candidate,
        "development_gate": dev_gate,
        "development_effect": dev_effect,
        "confirmation_opened": confirmation_opened,
        "confirmation_gate": confirmation_gate,
        "confirmation_effect": confirmation_effect,
        "confirmation_seed": CONFIRM_DATA_SEED if confirmation_opened else None,
        "confirmation_overlap_audit": confirmation_audit,
        "reserve_confirmation_seed_unopened": RESERVE_CONFIRM_DATA_SEED,
        "n_training_seeds": len(MODEL_SEEDS),
        "training_seeds": MODEL_SEEDS,
        "training_runs": len(training),
        "train_examples": TRAIN_N,
        "validation_examples": VAL_N,
        "development_examples": DEV_N,
        "confirmation_examples": 1024 if confirmation_opened else 0,
        "training_steps_per_model": TRAIN_STEPS,
        "semantic_stop_reference_target_used": False,
        "learned_halter_used": False,
        "task_specific_termination_bias": True,
        "thresholds_changed_after_prior_attempts": False,
        "baseline_weakened": False,
        "favorable_seed_filtering": False,
        "hidden_width_sweep": False,
        "learning_rate_sweep": False,
        "loss_weight_sweep": False,
        "loss_weight_schedule_fixed_before_attempt4_results": True,
        "architecture_changed_vs_attempt3": False,
        "inference_rule_changed_vs_attempt3": False,
        "scaling_law_claim": False,
        "energy_scope": "cpu_package_if_valid_otherwise_unavailable_not_gpu_not_whole_system",
    }
    write_json(out / "summary.json", summary)
    return 0 if status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
