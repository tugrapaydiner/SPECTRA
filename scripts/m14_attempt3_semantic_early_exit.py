#!/usr/bin/env python3
"""M14 Attempt 3: semantic early exit over the trained single-stream recurrence.

Protocol: docs/M14_ATTEMPT3_PROTOCOL.md

Attempts 1 and 2 remain immutable development evidence. Confirmation seed
2026091402 is generated only after the unchanged M14 development gate passes.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.measurement_env import measurement_environment
from data.datasets import GridDataset
from eval.edge_energy import energy_counter_inventory, measure_energy_record
from model.single_stream_trm import InputConditionedSingleStreamTRM
from model.system1_student import System1Student
from model.verifier import sudoku_correct
from scripts.m14_attempt2_single_stream import (
    ATTEMPT1_ARTIFACT_ID,
    ATTEMPT1_ARTIFACT_SHA256,
    ATTEMPT1_HEAD,
    ATTEMPT1_RUN,
    CANDIDATE_KIND,
    baseline_bundle,
    load_model,
    train_all,
)
from scripts.m14_primary_experiment import (
    BOOTSTRAP_SEED,
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
    VALIDATION_COST_N,
    all_aggregates,
    append_jsonl,
    build_confirmation_data,
    build_development_data,
    clamp_givens,
    paired_effect,
    practical_gate,
    sha256_file,
    symbolic_bundle,
    write_csv,
    write_json,
    SolverBundle,
)

ATTEMPT2_RUN = 34168430915
ATTEMPT2_HEAD = "0ed43b00fdba28678ced5fb144ec375276c995cd"
ATTEMPT2_ARTIFACT_ID = 10035110496
ATTEMPT2_ARTIFACT_SHA256 = "ec21fc7f107bde497f0f1f7980bdaaa6cd3203173c050abee467ca8b4961b94b"
ATTEMPT3_PROTOCOL_COMMIT = "41e23c69fd6a0664ce1f5e6cd4712f95898294d8"
MAX_BUDGETS = [1, 2, 3, 4]


class Attempt3Stop(RuntimeError):
    pass


@torch.inference_mode()
def semantic_early_exit_solve(
    model: InputConditionedSingleStreamTRM,
    x: torch.Tensor,
    max_steps: int,
    *,
    validator: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] = sudoku_correct,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run online recurrence and stop only on independently valid Sudoku output.

    Batch size one is deliberate: M14's primary cost unit is a complete B=1 solve.
    No target solution is accepted by this function.
    """
    if x.ndim != 2 or x.shape[0] != 1 or x.shape[1] != 16:
        raise ValueError("semantic early exit requires x[1,16]")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 4:
        raise ValueError("max_steps must be an integer in [1,4]")
    model.eval()
    x_emb = model.token_embed(x) + model.encode_positions(x, 4, 4)
    y = torch.zeros_like(x_emb)
    z = torch.zeros_like(x_emb)
    answer: torch.Tensor | None = None
    semantic_checks = 0
    block_applications = 0
    executed_steps = 0
    stopped_on_valid = False
    stop_reason = "budget_exhausted"

    for step in range(max_steps):
        y, z = model.recursive_cycle(x_emb, y, z)
        executed_steps += 1
        block_applications += int(model.T * len(model.blocks))
        logits = model.out_head(y)
        answer = clamp_givens(x, logits.argmax(dim=-1))
        valid = validator(x, answer, 2).bool()
        semantic_checks += 1
        if valid.numel() != 1:
            raise RuntimeError("semantic validator must return one decision for B=1")
        if bool(valid.item()):
            stopped_on_valid = True
            stop_reason = "semantic_valid"
            break

    assert answer is not None
    return answer, {
        "executed_steps": executed_steps,
        "block_applications": block_applications,
        "semantic_checks": semantic_checks,
        "max_steps": int(max_steps),
        "stopped_on_valid": stopped_on_valid,
        "stop_reason": stop_reason,
        "output_head_calls": executed_steps,
        "halt_head_calls": 0,
        "target_used": False,
        "semantics": model.SEMANTICS,
    }


def candidate_bundle(seed: int, max_steps: int, out: Path) -> SolverBundle:
    ck = out / "checkpoints" / f"{CANDIDATE_KIND}_seed{seed}.pt"
    model, payload = load_model(ck)
    if not isinstance(model, InputConditionedSingleStreamTRM):
        raise Attempt3Stop("candidate checkpoint did not restore single-stream model")
    model.eval()
    last_work: dict[str, Any] = {}

    def solve(x: torch.Tensor) -> torch.Tensor:
        answer, work = semantic_early_exit_solve(model, x, int(max_steps))
        last_work.clear(); last_work.update(work)
        return answer

    return SolverBundle(
        config_id=f"semantic_exit_k{max_steps}",
        family="single_stream_semantic_early_exit",
        seed=seed,
        solve=solve,
        work=lambda: dict(last_work),
        setup={
            "checkpoint": str(ck),
            "checkpoint_sha256": sha256_file(ck),
            "max_steps": int(max_steps),
            "termination": "sudoku_semantic_validity_reference_free",
            "semantics": payload["architecture"]["semantics"],
            "learned_halter_used": False,
        },
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
        latency = None; answer = None; semantic = None; first_work: dict[str, Any] = {}
        if i < n_cost:
            times: list[float] = []
            for _r in range(max(1, int(repeats))):
                t0 = time.perf_counter_ns()
                a = bundle.solve(xi)
                ok = sudoku_correct(xi, a, 2).bool()
                times.append((time.perf_counter_ns() - t0) / 1e6)
                if answer is None:
                    answer = a.detach().cpu(); semantic = bool(ok.item())
                    first_work = bundle.work() if bundle.work is not None else {}
            latency = float(statistics.median(times))
        else:
            answer = bundle.solve(xi).detach().cpu()
            semantic = bool(sudoku_correct(xi, answer, 2).item())
            first_work = bundle.work() if bundle.work is not None else {}
        assert answer is not None and semantic is not None
        target = ys[i:i+1]; blank = xi.cpu() == 0
        row = {
            "attempt": 3,
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
            "work_executed_steps": first_work.get("executed_steps"),
            "work_block_applications": first_work.get("block_applications"),
            "work_semantic_checks_internal": first_work.get("semantic_checks"),
            "work_stopped_on_valid": first_work.get("stopped_on_valid"),
            "work_stop_reason": first_work.get("stop_reason"),
            "work_output_head_calls": first_work.get("output_head_calls"),
            "work_halt_head_calls": first_work.get("halt_head_calls"),
        }
        append_jsonl(out_rows, row); rows.append(row)
    return rows


def aggregate_with_work(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggs = all_aggregates(rows)
    by = {r["config_id"]: r for r in aggs}
    for cid, agg in by.items():
        rr = [r for r in rows if r["config_id"] == cid and r["seed"] != "symbolic"]
        steps = [float(r["work_executed_steps"]) for r in rr if r.get("work_executed_steps") is not None]
        blocks = [float(r["work_block_applications"]) for r in rr if r.get("work_block_applications") is not None]
        stops = [bool(r["work_stopped_on_valid"]) for r in rr if r.get("work_stopped_on_valid") is not None]
        if steps:
            agg["executed_steps_mean"] = float(np.mean(steps))
            agg["executed_steps_median"] = float(np.median(steps))
        if blocks:
            agg["block_applications_mean"] = float(np.mean(blocks))
            agg["block_applications_median"] = float(np.median(blocks))
        if stops:
            agg["semantic_early_exit_fraction"] = float(np.mean(stops))
    return aggs


def choose_candidate(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["config_id"]: r for r in aggregates}
    base = by.get(PRIMARY_BASELINE)
    if base is None or base.get("latency_median_ms") is None:
        raise Attempt3Stop("primary baseline validation row missing")
    base_q = float(base["semantic_validity"]); base_lat = float(base["latency_median_ms"])
    candidates: list[dict[str, Any]] = []
    for k in MAX_BUDGETS:
        cid = f"semantic_exit_k{k}"; raw = by.get(cid)
        if not raw or not raw.get("n") or raw.get("latency_median_ms") is None:
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
        raise Attempt3Stop("no semantic-exit validation candidates")
    b_pool = [r for r in candidates if r["path_b_point_eligible"]]
    a_pool = [r for r in candidates if r["path_a_point_eligible"]]
    if b_pool:
        pool=b_pool; rule="path_b_point_eligible_priority"
    elif a_pool:
        pool=a_pool; rule="path_a_point_eligible_fallback"
    else:
        matched=[r for r in candidates if r["validation_latency_ratio_vs_baseline"] <= 1.15]
        pool=matched if matched else candidates
        rule="diagnostic_fallback_no_gate_relaxation"
    chosen = sorted(
        pool,
        key=lambda r: (
            -float(r["semantic_validity"]),
            float(r["latency_median_ms"]),
            int(str(r["config_id"]).rsplit("k",1)[1]),
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
    path=out/"validation_rows.jsonl"
    if path.exists(): path.unlink()
    rows: list[dict[str, Any]]=[]
    for seed in MODEL_SEEDS:
        for k in MAX_BUDGETS:
            rows.extend(evaluate_bundle(
                candidate_bundle(seed,k,out),val_ds,"validation",
                cost_n=VALIDATION_COST_N,repeats=LATENCY_REPEATS,out_rows=path,
            ))
        fp=baseline_bundle(seed,out,int8=False)
        assert fp is not None
        rows.extend(evaluate_bundle(fp,val_ds,"validation",cost_n=VALIDATION_COST_N,repeats=LATENCY_REPEATS,out_rows=path))
        q=baseline_bundle(seed,out,int8=True,audit_sink=out/"int8_conversion_audit.jsonl")
        if q is not None:
            rows.extend(evaluate_bundle(q,val_ds,"validation",cost_n=VALIDATION_COST_N,repeats=LATENCY_REPEATS,out_rows=path))
    rows.extend(evaluate_bundle(symbolic_bundle(),val_ds,"validation",cost_n=VALIDATION_COST_N,repeats=LATENCY_REPEATS,out_rows=path))
    aggs=aggregate_with_work(rows); selection=choose_candidate(aggs)
    write_json(out/"validation_operating_points.json",aggs); write_csv(out/"validation_operating_points.csv",aggs)
    write_json(out/"candidate_selection.json",selection)
    return rows,selection


def evaluate_surface(
    candidate: str, ds: GridDataset, split: str, out: Path, *, cost_n: int, repeats: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    path=out/f"{split}_rows.jsonl"
    if path.exists(): path.unlink()
    k=int(candidate.rsplit("k",1)[1]); rows=[]
    for seed in MODEL_SEEDS:
        rows.extend(evaluate_bundle(candidate_bundle(seed,k,out),ds,split,cost_n=cost_n,repeats=repeats,out_rows=path))
        fp=baseline_bundle(seed,out,int8=False); assert fp is not None
        rows.extend(evaluate_bundle(fp,ds,split,cost_n=cost_n,repeats=repeats,out_rows=path))
        q=baseline_bundle(seed,out,int8=True,audit_sink=out/"int8_conversion_audit.jsonl")
        if q is not None:
            rows.extend(evaluate_bundle(q,ds,split,cost_n=cost_n,repeats=repeats,out_rows=path))
    rows.extend(evaluate_bundle(symbolic_bundle(),ds,split,cost_n=cost_n,repeats=repeats,out_rows=path))
    cand=[r for r in rows if r["config_id"]==candidate]; base=[r for r in rows if r["config_id"]==PRIMARY_BASELINE]
    effect=paired_effect(cand,base,seed=(BOOTSTRAP_SEED+1 if split=="confirmation" else BOOTSTRAP_SEED))
    gate=practical_gate(effect)
    write_json(out/f"{split}_operating_points.json",aggregate_with_work(rows))
    write_json(out/f"{split}_effect.json",effect); write_json(out/f"{split}_decision.json",gate)
    return rows,effect,gate


def representative_energy(candidate: str, ds: GridDataset, out: Path, split: str) -> dict[str, Any]:
    k=int(candidate.rsplit("k",1)[1]); xs=torch.from_numpy(ds.inputs[:32]).long()
    bundles={candidate:candidate_bundle(MODEL_SEEDS[0],k,out),PRIMARY_BASELINE:baseline_bundle(MODEL_SEEDS[0],out,int8=False)}
    records={"scope":"cpu_package_rapl_not_gpu_not_whole_system","split":split,"configs":{}}
    for cid,bundle in bundles.items():
        assert bundle is not None
        def run(bundle=bundle):
            for i in range(len(xs)):
                a=bundle.solve(xs[i:i+1]); _=sudoku_correct(xs[i:i+1],a,2)
        rec=measure_energy_record(run,n_runs=1)
        records["configs"][cid]={**rec,"problems_per_window":len(xs),"joules_per_complete_solve":(
            float(rec["energy_joules"])/len(xs) if rec.get("available") and rec.get("energy_joules") is not None else None
        )}
    write_json(out/f"{split}_energy.json",records); return records


def freeze_candidate(candidate: str, out: Path) -> dict[str, Any]:
    files=[]
    for seed in MODEL_SEEDS:
        cp=out/"checkpoints"/f"{CANDIDATE_KIND}_seed{seed}.pt"; bp=out/"checkpoints"/f"single_pass_seed{seed}.pt"
        files += [
            {"role":"candidate","seed":seed,"path":str(cp),"sha256":sha256_file(cp)},
            {"role":"primary_baseline","seed":seed,"path":str(bp),"sha256":sha256_file(bp)},
        ]
    rec={
        "attempt":3,"candidate_config_id":candidate,"architecture_semantics":InputConditionedSingleStreamTRM.SEMANTICS,
        "termination_rule":"stop_if_sudoku_semantic_validity_true_else_continue_to_frozen_max_budget",
        "learned_halter_used":False,"files":files,"frozen_before_confirmation":True,
        "confirmation_seed":CONFIRM_DATA_SEED,"git_sha":git_sha(),"timestamp_unix":time.time(),
    }
    write_json(out/"confirmation_freeze_manifest.json",rec); return rec


def git_sha() -> str:
    import subprocess
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: return "unknown"


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=".m14a3/experiment"); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    write_json(out/"prior_attempts.json",{
        "adaptive_development_attempts_before_attempt3":2,
        "attempt1":{"run":ATTEMPT1_RUN,"head":ATTEMPT1_HEAD,"artifact_id":ATTEMPT1_ARTIFACT_ID,"artifact_zip_sha256":ATTEMPT1_ARTIFACT_SHA256,"status":"INCOMPLETE","confirmation_opened":False},
        "attempt2":{"run":ATTEMPT2_RUN,"head":ATTEMPT2_HEAD,"artifact_id":ATTEMPT2_ARTIFACT_ID,"artifact_zip_sha256":ATTEMPT2_ARTIFACT_SHA256,"status":"INCOMPLETE","confirmation_opened":False,
                    "development_quality_difference":-0.640625,"development_quality_ci95":[-0.669921875,-0.6109375],
                    "development_latency_ratio":0.8557505964071943,"development_latency_ci95":[0.8525784873955143,0.8585636511619983]},
    })
    write_json(out/"environment.json",measurement_environment(
        backend="pytorch_cpu_semantic_early_exit_complete_solve",compiler_flags=None,
        extra={"milestone":14,"attempt":3,"git_sha":git_sha(),"protocol":"docs/M14_ATTEMPT3_PROTOCOL.md","primary_cost":"complete_solve_latency"},
    ))
    write_json(out/"energy_inventory.json",energy_counter_inventory())
    train_ds,val_ds,dev_ds,dev_manifest=build_development_data(out)
    training=train_all(train_ds,val_ds,out)
    _,selection=run_validation(val_ds,out); candidate=str(selection["selected_config_id"])
    _,dev_effect,dev_gate=evaluate_surface(candidate,dev_ds,"development",out,cost_n=DEVELOPMENT_COST_N,repeats=LATENCY_REPEATS)
    representative_energy(candidate,dev_ds,out,"development")
    confirmation_opened=False; confirmation_effect=None; confirmation_gate=None; confirmation_audit=None
    if dev_gate["pass"]:
        freeze_candidate(candidate,out)
        conf_ds,_,confirmation_audit=build_confirmation_data(out,dev_manifest)
        confirmation_opened=True
        _,confirmation_effect,confirmation_gate=evaluate_surface(candidate,conf_ds,"confirmation",out,cost_n=CONFIRMATION_COST_N,repeats=LATENCY_REPEATS)
        representative_energy(candidate,conf_ds,out,"confirmation")
    status="COMPLETE" if confirmation_gate is not None and confirmation_gate["pass"] else "INCOMPLETE"
    result={
        "milestone":14,"attempt":3,"status":status,"claim_confirmed":bool(status=="COMPLETE"),
        "protocol_committed_before_attempt3_results":True,"protocol_commit":ATTEMPT3_PROTOCOL_COMMIT,
        "adaptive_development_attempts_before_confirmation":3,"prior_attempt_count":2,
        "primary_metric":"strict_sudoku_semantic_validity","primary_baseline":PRIMARY_BASELINE,
        "candidate_family":"single_stream_recursive_with_symbolic_semantic_early_exit","selected_candidate":candidate,
        "development_gate":dev_gate,"development_effect":dev_effect,
        "confirmation_opened":confirmation_opened,"confirmation_gate":confirmation_gate,"confirmation_effect":confirmation_effect,
        "confirmation_seed":CONFIRM_DATA_SEED if confirmation_opened else None,"confirmation_overlap_audit":confirmation_audit,
        "reserve_confirmation_seed_unopened":RESERVE_CONFIRM_DATA_SEED,
        "n_training_seeds":len(MODEL_SEEDS),"training_seeds":MODEL_SEEDS,"training_runs":len(training),
        "train_examples":TRAIN_N,"validation_examples":VAL_N,"development_examples":DEV_N,
        "confirmation_examples":1024 if confirmation_opened else 0,"training_steps_per_model":TRAIN_STEPS,
        "semantic_stop_reference_target_used":False,"learned_halter_used":False,"task_specific_termination_bias":True,
        "thresholds_changed_after_prior_attempts":False,"baseline_weakened":False,"favorable_seed_filtering":False,
        "hidden_width_sweep":False,"learning_rate_sweep":False,"loss_weight_sweep":False,"scaling_law_claim":False,
        "energy_scope":"cpu_package_if_valid_otherwise_unavailable_not_gpu_not_whole_system",
    }
    write_json(out/"summary.json",result); return 0 if status=="COMPLETE" else 2


if __name__=="__main__": raise SystemExit(main())
