#!/usr/bin/env python3
"""M14 Attempt 5: semantic early exit on the original dual-stream FP TRM.

Protocol: docs/M14_ATTEMPT5_PROTOCOL.md

This attempt returns to the strongest prior SPECTRA family (dim-64 FP dual-stream
TRM) and changes only complete-solve execution: a full supervision step is run,
the candidate is independently validated, and later supervision steps are skipped
only after a genuine valid Sudoku completion.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.measurement_env import measurement_environment
from eval.edge_energy import energy_counter_inventory, measure_energy_record
from model.system1_student import System1Student
from model.trm import TRM
from model.verifier import sudoku_correct
import scripts.m14_primary_experiment as m14

ATTEMPT5_PROTOCOL_COMMIT = "abbf24fc4166eab71be3d590e9754dd1b54612b7"
CANDIDATE_KIND = "fp_recursive_dim64"
CANDIDATE_FAMILY = "dual_stream_fp64_semantic_early_exit"
MAX_BUDGETS = [1, 2, 3, 4]

PRIOR_ATTEMPTS = {
    "attempt1": {
        "run": 34159474745,
        "head": "66acbc62f48b3e5b1e81080897b89562edfd8bd8",
        "artifact_id": 10032762057,
        "artifact_zip_sha256": "04a5b810d697227c4ccf88a365cf529647734ea0d95deec363f84fcaae885522",
        "status": "INCOMPLETE",
        "confirmation_opened": False,
        "development_quality_difference": -0.005078125,
        "development_latency_ratio": 1.0943352911334046,
    },
    "attempt2": {
        "run": 34168430915,
        "head": "0ed43b00fdba28678ced5fb144ec375276c995cd",
        "artifact_id": 10035110496,
        "artifact_zip_sha256": "ec21fc7f107bde497f0f1f7980bdaaa6cd3203173c050abee467ca8b4961b94b",
        "status": "INCOMPLETE",
        "confirmation_opened": False,
        "development_quality_difference": -0.640625,
        "development_latency_ratio": 0.8557505964071943,
    },
    "attempt3": {
        "run": 34169488811,
        "head": "d1e491f412494c0756753bfa4b78b8002133d39c",
        "artifact_id": 10035348365,
        "artifact_zip_sha256": "fb653a307d918f8e010d6648694ef2bdeb08654bee9faaf093f1f71ff07eaaac",
        "status": "INCOMPLETE",
        "confirmation_opened": False,
        "development_quality_difference": -0.637890625,
        "development_latency_ratio": 0.8127430466852386,
    },
    "attempt4": {
        "run": 34170096945,
        "head": "d8480dccd089a235ee8823161e1ed87745c28c22",
        "artifact_id": 10035628899,
        "artifact_zip_sha256": "1d7a346a47f82b360d25d97b6650a555216385b8ce523a0867302a98002438ee",
        "status": "INCOMPLETE",
        "confirmation_opened": False,
        "development_quality_difference": -0.680078125,
        "development_latency_ratio": 0.7752021795958917,
        "non_scientific_test_bug": "exact float equality for 0.4+0.3+0.2+0.1; fixed before Attempt 5 preregistration",
    },
}


class Attempt5Stop(RuntimeError):
    pass


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


@torch.inference_mode()
def dual_stream_semantic_exit_solve(
    model: TRM,
    x: torch.Tensor,
    max_steps: int,
    *,
    validator: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] = sudoku_correct,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Execute complete original dual-stream supervision steps online.

    No target/reference solution is accepted.  B=1 is deliberate because M14's
    primary latency unit is one complete solve.
    """
    if x.ndim != 2 or x.shape != (1, 16):
        raise ValueError("Attempt-5 semantic exit requires x[1,16]")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 4:
        raise ValueError("max_steps must be an integer in [1,4]")
    if model.ternary or model.act8 or model.dim != 64 or model.n != 1 or model.T != 1:
        raise ValueError("Attempt-5 runtime requires the frozen dim64 FP dual-stream TRM contract")
    model.eval()
    x_emb = model.token_embed(x) + model.encode_positions(x, 4, 4)
    y = torch.zeros_like(x_emb)
    z = torch.zeros_like(x_emb)
    answer: torch.Tensor | None = None
    final_semantic = False
    executed_steps = 0
    block_applications = 0
    output_head_calls = 0
    semantic_checks = 0
    stop_reason = "budget_exhausted"

    for _ in range(max_steps):
        # This is the production TRM dual-stream recurrence, not a surrogate.
        y, z = model.recursive_cycle(x_emb, y, z)
        executed_steps += 1
        block_applications += int(model.T * (model.n + 1) * len(model.blocks))
        answer = m14.clamp_givens(x, model.out_head(y).argmax(dim=-1))
        output_head_calls += 1
        decision = validator(x, answer, 2).bool()
        semantic_checks += 1
        if decision.numel() != 1:
            raise RuntimeError("semantic validator must produce one B=1 decision")
        final_semantic = bool(decision.item())
        if final_semantic:
            stop_reason = "semantic_valid"
            break

    assert answer is not None
    return answer, {
        "executed_steps": executed_steps,
        "block_applications": block_applications,
        "recursive_cycles": executed_steps,
        "output_head_calls": output_head_calls,
        "semantic_checks": semantic_checks,
        "max_steps": int(max_steps),
        "stopped_on_valid": final_semantic,
        "final_semantic": final_semantic,
        "stop_reason": stop_reason,
        "halt_head_calls": 0,
        "target_used": False,
        "architecture": "dual_stream_fp64_trm_n1_t1",
    }


def train_attempt5(train_ds, val_ds, out: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for kind in [CANDIDATE_KIND, "single_pass"]:
        for seed in m14.MODEL_SEEDS:
            records.append(m14.train_one(kind, seed, train_ds, val_ds, out))
    cand = {r["architecture"]["trainable_params"] for r in records if r["kind"] == CANDIDATE_KIND}
    base = {r["architecture"]["trainable_params"] for r in records if r["kind"] == "single_pass"}
    if len(cand) != 1 or len(base) != 1 or not (next(iter(cand)) < next(iter(base))):
        raise Attempt5Stop(f"candidate must be smaller than primary baseline: candidate={cand}, baseline={base}")
    m14.write_json(out / "parameter_audit.json", {
        "candidate_kind": CANDIDATE_KIND,
        "candidate_trainable_params": next(iter(cand)),
        "baseline_trainable_params": next(iter(base)),
        "candidate_smaller": True,
    })
    m14.write_json(out / "training_objective_audit.json", {
        "candidate_kind": CANDIDATE_KIND,
        "candidate_objective": "four_step_weighted_blank_cell_cross_entropy",
        "candidate_weights": m14.LATER_WEIGHTS,
        "baseline_objective": "blank_cell_cross_entropy",
        "candidate_architecture": "original_dual_stream_fp64_trm",
        "training_steps": m14.TRAIN_STEPS,
        "batch_size": m14.BATCH_SIZE,
        "lr": m14.LR,
        "weight_decay": m14.WEIGHT_DECAY,
        "candidate_checkpoints": [
            {"seed": r["seed"], "path": r["checkpoint"], "sha256": r["checkpoint_sha256"]}
            for r in records if r["kind"] == CANDIDATE_KIND
        ],
        "baseline_checkpoints": [
            {"seed": r["seed"], "path": r["checkpoint"], "sha256": r["checkpoint_sha256"]}
            for r in records if r["kind"] == "single_pass"
        ],
    })
    return records


def candidate_bundle(seed: int, max_steps: int, out: Path) -> m14.SolverBundle:
    ck = out / "checkpoints" / f"{CANDIDATE_KIND}_seed{seed}.pt"
    model, kind, payload = m14.load_checkpoint(ck)
    if kind != CANDIDATE_KIND or not isinstance(model, TRM):
        raise Attempt5Stop("candidate checkpoint did not restore the dim64 FP TRM")
    model.eval()
    last_work: dict[str, Any] = {}

    def solve(x: torch.Tensor) -> torch.Tensor:
        answer, work = dual_stream_semantic_exit_solve(model, x, int(max_steps))
        last_work.clear(); last_work.update(work)
        return answer

    return m14.SolverBundle(
        config_id=f"dual_stream_exit_k{max_steps}",
        family=CANDIDATE_FAMILY,
        seed=seed,
        solve=solve,
        work=lambda: dict(last_work),
        setup={
            "checkpoint": str(ck),
            "checkpoint_sha256": m14.sha256_file(ck),
            "max_steps": int(max_steps),
            "termination": "sudoku_semantic_validity_reference_free",
            "learned_halter_used": False,
            "architecture": payload["family"],
        },
    )


def baseline_bundle(
    seed: int,
    out: Path,
    *,
    int8: bool = False,
    audit_sink: Path | None = None,
) -> m14.SolverBundle | None:
    ck = out / "checkpoints" / f"single_pass_seed{seed}.pt"
    model, kind, _ = m14.load_checkpoint(ck)
    if kind != "single_pass" or not isinstance(model, System1Student):
        raise Attempt5Stop("baseline checkpoint did not restore System1Student")
    audit = {"seed": seed, "source_checkpoint": str(ck), "source_sha256": m14.sha256_file(ck)}
    if int8:
        q, qa = m14.dynamic_int8_baseline(model); audit.update(qa)
        if audit_sink is not None:
            m14.append_jsonl(audit_sink, audit)
        if q is None:
            return None
        model = q
    cid = "single_pass_int8" if int8 else m14.PRIMARY_BASELINE

    def solve(x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            logits, _ = model(x, height=4, width=4)
        return m14.clamp_givens(x, logits.argmax(dim=-1))

    return m14.SolverBundle(
        cid, cid, seed, solve,
        work=lambda: {"block_applications": 2, "supervision_steps": 1},
        setup=audit,
    )


def evaluate_bundle(
    bundle: m14.SolverBundle,
    ds,
    split: str,
    *,
    cost_n: int,
    repeats: int,
    out_rows: Path,
) -> list[dict[str, Any]]:
    xs = torch.from_numpy(ds.inputs).long(); ys = torch.from_numpy(ds.targets).long()
    n_cost = min(int(cost_n), len(ds))

    def checked_solve(xi: torch.Tensor):
        answer = bundle.solve(xi)
        work = bundle.work() if bundle.work is not None else {}
        if work.get("final_semantic") is not None:
            semantic = bool(work["final_semantic"])
        else:
            semantic = bool(sudoku_correct(xi, answer, 2).bool().item())
        return answer, semantic, dict(work)

    for i in range(min(4, len(ds))):
        checked_solve(xs[i:i+1])

    rows: list[dict[str, Any]] = []
    for i in range(len(ds)):
        xi = xs[i:i+1]
        latency = None; answer = None; semantic = None; first_work: dict[str, Any] = {}
        if i < n_cost:
            times: list[float] = []
            for _ in range(max(1, int(repeats))):
                t0 = time.perf_counter_ns()
                a, ok, work = checked_solve(xi)
                times.append((time.perf_counter_ns() - t0) / 1e6)
                if answer is None:
                    answer = a.detach().cpu(); semantic = ok; first_work = work
            latency = float(statistics.median(times))
        else:
            a, ok, work = checked_solve(xi)
            answer = a.detach().cpu(); semantic = ok; first_work = work
        assert answer is not None and semantic is not None
        target = ys[i:i+1]; blank = xi.cpu() == 0
        row = {
            "attempt": 5,
            "split": split,
            "config_id": bundle.config_id,
            "family": bundle.family,
            "seed": bundle.seed,
            "example_index": i,
            "example_id": ds.ids[i],
            "semantic_success": int(bool(semantic)),
            "exact_reference_match": int(bool((answer == target).all().item())),
            "blank_cell_accuracy": float((answer[blank] == target[blank]).float().mean()),
            "latency_ms": latency,
            "latency_repeats": repeats if i < n_cost else 0,
            "complete_solve_timing_includes_semantic_check": True,
            "target_used_inside_solver": False,
            "work_executed_steps": first_work.get("executed_steps"),
            "work_block_applications": first_work.get("block_applications"),
            "work_recursive_cycles": first_work.get("recursive_cycles"),
            "work_semantic_checks_internal": first_work.get("semantic_checks"),
            "work_stopped_on_valid": first_work.get("stopped_on_valid"),
            "work_stop_reason": first_work.get("stop_reason"),
            "work_output_head_calls": first_work.get("output_head_calls"),
            "work_halt_head_calls": first_work.get("halt_head_calls"),
            "final_semantic_reused_without_duplicate_check": first_work.get("final_semantic") is not None,
        }
        m14.append_jsonl(out_rows, row); rows.append(row)
    return rows


def aggregate_with_work(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggs = m14.all_aggregates(rows)
    for agg in aggs:
        cid = agg["config_id"]
        rr = [r for r in rows if r["config_id"] == cid and r["seed"] != "symbolic"]
        steps = [float(r["work_executed_steps"]) for r in rr if r.get("work_executed_steps") is not None]
        blocks = [float(r["work_block_applications"]) for r in rr if r.get("work_block_applications") is not None]
        stops = [bool(r["work_stopped_on_valid"]) for r in rr if r.get("work_stopped_on_valid") is not None]
        if steps:
            agg["executed_steps_mean"] = float(np.mean(steps)); agg["executed_steps_median"] = float(np.median(steps))
        if blocks:
            agg["block_applications_mean"] = float(np.mean(blocks)); agg["block_applications_median"] = float(np.median(blocks))
        if stops:
            agg["semantic_early_exit_fraction"] = float(np.mean(stops))
    return aggs


def choose_candidate(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["config_id"]: r for r in aggregates}
    base = by.get(m14.PRIMARY_BASELINE)
    if base is None or base.get("latency_median_ms") is None:
        raise Attempt5Stop("primary baseline validation aggregate missing")
    base_q = float(base["semantic_validity"]); base_lat = float(base["latency_median_ms"])
    candidates: list[dict[str, Any]] = []
    for k in MAX_BUDGETS:
        cid = f"dual_stream_exit_k{k}"; raw = by.get(cid)
        if not raw or raw.get("latency_median_ms") is None:
            continue
        row = dict(raw)
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
        raise Attempt5Stop("no dual-stream semantic-exit validation candidates")
    b_pool = [r for r in candidates if r["path_b_point_eligible"]]
    a_pool = [r for r in candidates if r["path_a_point_eligible"]]
    if b_pool:
        pool = b_pool; rule = "path_b_point_eligible_priority"
    elif a_pool:
        pool = a_pool; rule = "path_a_point_eligible_fallback"
    else:
        matched = [r for r in candidates if r["validation_latency_ratio_vs_baseline"] <= 1.15]
        pool = matched if matched else candidates
        rule = "diagnostic_fallback_no_gate_relaxation"
    chosen = sorted(pool, key=lambda r: (
        -float(r["semantic_validity"]),
        float(r["latency_median_ms"]),
        int(str(r["config_id"]).rsplit("k", 1)[1]),
    ))[0]
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


def run_validation(val_ds, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = out / "validation_rows.jsonl"
    if path.exists(): path.unlink()
    rows: list[dict[str, Any]] = []
    for seed in m14.MODEL_SEEDS:
        for k in MAX_BUDGETS:
            rows.extend(evaluate_bundle(candidate_bundle(seed, k, out), val_ds, "validation",
                cost_n=m14.VALIDATION_COST_N, repeats=m14.LATENCY_REPEATS, out_rows=path))
        fp = baseline_bundle(seed, out, int8=False); assert fp is not None
        rows.extend(evaluate_bundle(fp, val_ds, "validation", cost_n=m14.VALIDATION_COST_N,
            repeats=m14.LATENCY_REPEATS, out_rows=path))
        q = baseline_bundle(seed, out, int8=True, audit_sink=out / "int8_conversion_audit.jsonl")
        if q is not None:
            rows.extend(evaluate_bundle(q, val_ds, "validation", cost_n=m14.VALIDATION_COST_N,
                repeats=m14.LATENCY_REPEATS, out_rows=path))
    rows.extend(evaluate_bundle(m14.symbolic_bundle(), val_ds, "validation",
        cost_n=m14.VALIDATION_COST_N, repeats=m14.LATENCY_REPEATS, out_rows=path))
    aggs = aggregate_with_work(rows); selection = choose_candidate(aggs)
    m14.write_json(out / "validation_operating_points.json", aggs)
    m14.write_csv(out / "validation_operating_points.csv", aggs)
    m14.write_json(out / "candidate_selection.json", selection)
    return rows, selection


def evaluate_surface(candidate: str, ds, split: str, out: Path, *, cost_n: int, repeats: int):
    path = out / f"{split}_rows.jsonl"
    if path.exists(): path.unlink()
    k = int(candidate.rsplit("k", 1)[1]); rows: list[dict[str, Any]] = []
    for seed in m14.MODEL_SEEDS:
        rows.extend(evaluate_bundle(candidate_bundle(seed, k, out), ds, split,
            cost_n=cost_n, repeats=repeats, out_rows=path))
        fp = baseline_bundle(seed, out, int8=False); assert fp is not None
        rows.extend(evaluate_bundle(fp, ds, split, cost_n=cost_n, repeats=repeats, out_rows=path))
        q = baseline_bundle(seed, out, int8=True, audit_sink=out / "int8_conversion_audit.jsonl")
        if q is not None:
            rows.extend(evaluate_bundle(q, ds, split, cost_n=cost_n, repeats=repeats, out_rows=path))
    rows.extend(evaluate_bundle(m14.symbolic_bundle(), ds, split, cost_n=cost_n, repeats=repeats, out_rows=path))
    cand = [r for r in rows if r["config_id"] == candidate]
    base = [r for r in rows if r["config_id"] == m14.PRIMARY_BASELINE]
    effect = m14.paired_effect(cand, base, seed=(m14.BOOTSTRAP_SEED + 1 if split == "confirmation" else m14.BOOTSTRAP_SEED))
    gate = m14.practical_gate(effect)
    m14.write_json(out / f"{split}_operating_points.json", aggregate_with_work(rows))
    m14.write_json(out / f"{split}_effect.json", effect)
    m14.write_json(out / f"{split}_decision.json", gate)
    return rows, effect, gate


def representative_energy(candidate: str, ds, out: Path, split: str) -> dict[str, Any]:
    k = int(candidate.rsplit("k", 1)[1]); xs = torch.from_numpy(ds.inputs[:32]).long()
    bundles = {
        candidate: candidate_bundle(m14.MODEL_SEEDS[0], k, out),
        m14.PRIMARY_BASELINE: baseline_bundle(m14.MODEL_SEEDS[0], out, int8=False),
    }
    result = {"scope": "cpu_package_rapl_not_gpu_not_whole_system", "split": split, "configs": {}}
    for cid, bundle in bundles.items():
        assert bundle is not None
        def run(bundle=bundle):
            for i in range(len(xs)):
                # checked once: candidate's final decision is internal; baseline gets final validator.
                a = bundle.solve(xs[i:i+1])
                work = bundle.work() if bundle.work is not None else {}
                if work.get("final_semantic") is None:
                    _ = sudoku_correct(xs[i:i+1], a, 2)
        rec = measure_energy_record(run, n_runs=1)
        result["configs"][cid] = {
            **rec,
            "problems_per_window": len(xs),
            "joules_per_complete_solve": (
                float(rec["energy_joules"]) / len(xs)
                if rec.get("available") and rec.get("energy_joules") is not None else None
            ),
        }
    m14.write_json(out / f"{split}_energy.json", result)
    return result


def freeze_candidate(candidate: str, out: Path) -> dict[str, Any]:
    files = []
    for seed in m14.MODEL_SEEDS:
        cp = out / "checkpoints" / f"{CANDIDATE_KIND}_seed{seed}.pt"
        bp = out / "checkpoints" / f"single_pass_seed{seed}.pt"
        files += [
            {"role": "candidate", "seed": seed, "path": str(cp), "sha256": m14.sha256_file(cp)},
            {"role": "primary_baseline", "seed": seed, "path": str(bp), "sha256": m14.sha256_file(bp)},
        ]
    rec = {
        "attempt": 5,
        "candidate_config_id": candidate,
        "candidate_kind": CANDIDATE_KIND,
        "candidate_architecture": "original_dual_stream_fp64_trm_n1_t1",
        "candidate_training_weights": m14.LATER_WEIGHTS,
        "termination_rule": "stop_if_sudoku_semantic_validity_true_else_continue_to_frozen_max_budget",
        "learned_halter_used": False,
        "files": files,
        "frozen_before_confirmation": True,
        "confirmation_seed": m14.CONFIRM_DATA_SEED,
        "git_sha": git_sha(),
        "timestamp_unix": time.time(),
    }
    m14.write_json(out / "confirmation_freeze_manifest.json", rec)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=".m14a5/experiment")
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass

    m14.write_json(out / "prior_attempts.json", {
        "adaptive_development_attempts_before_attempt5": 4,
        **PRIOR_ATTEMPTS,
    })
    m14.write_json(out / "environment.json", measurement_environment(
        backend="pytorch_cpu_dual_stream_fp64_semantic_early_exit_complete_solve",
        compiler_flags=None,
        extra={
            "milestone": 14, "attempt": 5, "git_sha": git_sha(),
            "protocol": "docs/M14_ATTEMPT5_PROTOCOL.md",
            "primary_cost": "complete_solve_latency",
        },
    ))
    m14.write_json(out / "energy_inventory.json", energy_counter_inventory())

    train_ds, val_ds, dev_ds, dev_manifest = m14.build_development_data(out)
    training = train_attempt5(train_ds, val_ds, out)
    _, selection = run_validation(val_ds, out)
    candidate = str(selection["selected_config_id"])
    _, dev_effect, dev_gate = evaluate_surface(candidate, dev_ds, "development", out,
        cost_n=m14.DEVELOPMENT_COST_N, repeats=m14.LATENCY_REPEATS)
    representative_energy(candidate, dev_ds, out, "development")

    confirmation_opened = False; confirmation_effect = None; confirmation_gate = None; confirmation_audit = None
    if dev_gate["pass"]:
        freeze_candidate(candidate, out)
        conf_ds, _, confirmation_audit = m14.build_confirmation_data(out, dev_manifest)
        confirmation_opened = True
        _, confirmation_effect, confirmation_gate = evaluate_surface(candidate, conf_ds, "confirmation", out,
            cost_n=m14.CONFIRMATION_COST_N, repeats=m14.LATENCY_REPEATS)
        representative_energy(candidate, conf_ds, out, "confirmation")

    status = "COMPLETE" if confirmation_gate is not None and confirmation_gate["pass"] else "INCOMPLETE"
    summary = {
        "milestone": 14,
        "attempt": 5,
        "status": status,
        "claim_confirmed": bool(status == "COMPLETE"),
        "protocol_committed_before_attempt5_results": True,
        "protocol_commit": ATTEMPT5_PROTOCOL_COMMIT,
        "adaptive_development_attempts_before_confirmation": 5,
        "prior_attempt_count": 4,
        "primary_metric": "strict_sudoku_semantic_validity",
        "primary_baseline": m14.PRIMARY_BASELINE,
        "candidate_family": CANDIDATE_FAMILY,
        "candidate_kind": CANDIDATE_KIND,
        "candidate_training_weights": m14.LATER_WEIGHTS,
        "selected_candidate": candidate,
        "development_gate": dev_gate,
        "development_effect": dev_effect,
        "confirmation_opened": confirmation_opened,
        "confirmation_gate": confirmation_gate,
        "confirmation_effect": confirmation_effect,
        "confirmation_seed": m14.CONFIRM_DATA_SEED if confirmation_opened else None,
        "confirmation_overlap_audit": confirmation_audit,
        "reserve_confirmation_seed_unopened": m14.RESERVE_CONFIRM_DATA_SEED,
        "n_training_seeds": len(m14.MODEL_SEEDS),
        "training_seeds": m14.MODEL_SEEDS,
        "training_runs": len(training),
        "train_examples": m14.TRAIN_N,
        "validation_examples": m14.VAL_N,
        "development_examples": m14.DEV_N,
        "confirmation_examples": m14.CONFIRM_N if confirmation_opened else 0,
        "training_steps_per_model": m14.TRAIN_STEPS,
        "semantic_stop_reference_target_used": False,
        "learned_halter_used": False,
        "task_specific_termination_bias": True,
        "thresholds_changed_after_prior_attempts": False,
        "baseline_weakened": False,
        "favorable_seed_filtering": False,
        "hidden_width_sweep": False,
        "learning_rate_sweep": False,
        "loss_weight_sweep": False,
        "architecture_sweep": False,
        "scaling_law_claim": False,
        "energy_scope": "cpu_package_if_valid_otherwise_unavailable_not_gpu_not_whole_system",
    }
    m14.write_json(out / "summary.json", summary)
    return 0 if status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
