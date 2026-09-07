#!/usr/bin/env python3
"""M14 Attempt 2: preregistered early-exit credit allocation intervention."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import scripts.m14_primary_experiment as m14
from common.measurement_env import measurement_environment
from eval.edge_energy import energy_counter_inventory, measure_energy_record
from model.verifier import sudoku_correct


ATTEMPT = 2
ATTEMPT1_DATA_SEED = 2026091401
DATA_SEED = 2026091411
CONFIRM_DATA_SEED = 2026091402
RESERVE_CONFIRM_DATA_SEED = 2026091403
MODEL_SEEDS = [1411, 2412, 3413, 4414, 5415]
TRAIN_N = 4096
VAL_N = 512
DEV_N = 512
CONFIRM_N = 1024
EARLY_WEIGHTS = [0.70, 0.10, 0.10, 0.10]
CANDIDATE_BUDGETS = [1, 2, 3, 4]
VALIDATION_COST_N = 64
DEVELOPMENT_COST_N = 128
CONFIRMATION_COST_N = 256
LATENCY_REPEATS = 2
DEV_BOOTSTRAP_SEED = 2026091519
CONFIRM_BOOTSTRAP_SEED = 2026091520


def write_json(path: Path, payload: Any) -> None:
    m14.write_json(path, payload)


def manifest_set(manifest: dict[str, Any]) -> set[str]:
    return m14.manifest_fingerprints(manifest)


def build_hierarchy(out: Path):
    # Regenerate Attempt 1 only for exact fingerprint contamination audit.
    cfg1 = m14.make_cfg(ATTEMPT1_DATA_SEED)
    _, old_manifest = m14.build_data_splits(
        cfg1, TRAIN_N, VAL_N, DEV_N, seed=ATTEMPT1_DATA_SEED,
        manifest_path=out / "manifests" / "attempt1_regenerated_for_overlap_audit.json",
    )
    cfg2 = m14.make_cfg(DATA_SEED)
    ds, manifest = m14.build_data_splits(
        cfg2, TRAIN_N, VAL_N, DEV_N, seed=DATA_SEED,
        manifest_path=out / "manifests" / "train_validation_development.json",
    )
    old = manifest_set(old_manifest); new = manifest_set(manifest)
    overlap = sorted(old & new)
    audit = {
        "attempt1_seed": ATTEMPT1_DATA_SEED,
        "attempt2_seed": DATA_SEED,
        "attempt1_fingerprints": len(old),
        "attempt2_fingerprints": len(new),
        "exact_overlap_count": len(overlap),
        "overlap_examples": overlap[:20],
    }
    write_json(out / "manifests" / "attempt1_attempt2_overlap_audit.json", audit)
    if overlap:
        raise m14.ExperimentStop(f"Attempt 2 hierarchy overlaps Attempt 1 by {len(overlap)} fingerprints")
    return ds["train"], ds["validation"], ds["test"], manifest, old


def train_models(train_ds, val_ds, out: Path) -> list[dict[str, Any]]:
    # The intervention is exactly the preregistered first-heavy supervision vector.
    m14.LATER_WEIGHTS[:] = list(EARLY_WEIGHTS)
    runs=[]
    contract={
        "attempt": ATTEMPT,
        "candidate_kind": "fp_recursive_dim64",
        "candidate_deep_supervision_weights": EARLY_WEIGHTS,
        "baseline_kind": "single_pass",
        "steps": m14.TRAIN_STEPS,
        "batch_size": m14.BATCH_SIZE,
        "lr": m14.LR,
        "weight_decay": m14.WEIGHT_DECAY,
        "grad_clip": m14.CLIP_NORM,
        "training_seeds": MODEL_SEEDS,
        "thresholds_inherited_unchanged_from": "docs/M14_PROTOCOL.md",
    }
    write_json(out / "training_contract.json", contract)
    for seed in MODEL_SEEDS:
        r=m14.train_one("fp_recursive_dim64", seed, train_ds, val_ds, out)
        r["attempt2_supervision_weights"] = list(EARLY_WEIGHTS)
        runs.append(r)
    for seed in MODEL_SEEDS:
        runs.append(m14.train_one("single_pass", seed, train_ds, val_ds, out))
    return runs


def candidate_bundle(seed: int, budget: int, out: Path) -> m14.SolverBundle:
    b=m14.recursive_bundle("fp_recursive_dim64", seed, budget, out, reserve=True)
    b.config_id=f"early64_n{budget}"
    b.family="early_exit_fp64"
    assert b.setup is not None
    b.setup["attempt2_deep_supervision_weights"] = list(EARLY_WEIGHTS)
    return b


def baseline_bundle(seed: int, out: Path, *, int8: bool=False):
    return m14.single_pass_bundle(seed, out, int8=int8, audit_sink=out/"int8_conversion_audit.jsonl")


def evaluate_validation(val_ds, out: Path):
    rows_path=out/"validation_rows.jsonl"
    if rows_path.exists(): rows_path.unlink()
    rows=[]
    for seed in MODEL_SEEDS:
        for n in CANDIDATE_BUDGETS:
            rows += m14.evaluate_bundle(candidate_bundle(seed,n,out),val_ds,"validation",cost_n=VALIDATION_COST_N,repeats=LATENCY_REPEATS,out_rows=rows_path)
        rows += m14.evaluate_bundle(baseline_bundle(seed,out,int8=False),val_ds,"validation",cost_n=VALIDATION_COST_N,repeats=LATENCY_REPEATS,out_rows=rows_path)
        qb=baseline_bundle(seed,out,int8=True)
        if qb is not None:
            rows += m14.evaluate_bundle(qb,val_ds,"validation",cost_n=VALIDATION_COST_N,repeats=LATENCY_REPEATS,out_rows=rows_path)
    rows += m14.evaluate_bundle(m14.symbolic_bundle(),val_ds,"validation",cost_n=VALIDATION_COST_N,repeats=LATENCY_REPEATS,out_rows=rows_path)
    agg=m14.all_aggregates(rows); by={r["config_id"]:r for r in agg}
    base=by["single_pass_fp"]; base_lat=float(base["latency_median_ms"])
    candidates=[]
    for n in CANDIDATE_BUDGETS:
        r=copy.deepcopy(by[f"early64_n{n}"])
        ratio=float(r["latency_median_ms"])/base_lat
        r["validation_latency_ratio_vs_baseline"]=ratio
        r["latency_match_status"]="matched" if .85 <= ratio <= 1.15 else "unmatched"
        candidates.append(r)
    eligible=[r for r in candidates if r["validation_latency_ratio_vs_baseline"] <= 1.15]
    pool=eligible if eligible else candidates
    chosen=sorted(pool,key=lambda r:(-float(r["semantic_validity"]),float(r["latency_median_ms"]),str(r["config_id"])))[0]
    selection={
        "attempt":ATTEMPT,
        "selected_config_id":chosen["config_id"],
        "selection_pool":"latency_le_1.15x_baseline" if eligible else "unmatched_fallback",
        "selected_validation":chosen,
        "primary_baseline_validation":base,
        "all_candidate_validation":candidates,
        "selection_used_development":False,
        "selection_used_confirmation":False,
    }
    write_json(out/"validation_operating_points.json",agg)
    m14.write_csv(out/"validation_operating_points.csv",agg)
    write_json(out/"candidate_selection.json",selection)
    return rows,agg,selection


def bundle_for_config(cid: str, seed: int, out: Path):
    if cid.startswith("early64_n"):
        return candidate_bundle(seed,int(cid.rsplit("n",1)[1]),out)
    if cid=="single_pass_fp": return baseline_bundle(seed,out,int8=False)
    if cid=="single_pass_int8": return baseline_bundle(seed,out,int8=True)
    if cid=="symbolic_exact": return m14.symbolic_bundle()
    raise ValueError(cid)


def evaluate_surface(candidate: str, ds, split: str, out: Path, *, cost_n: int, bootstrap_seed: int):
    path=out/f"{split}_rows.jsonl"
    if path.exists(): path.unlink()
    rows=[]
    context=[candidate,"early64_n4","single_pass_fp","single_pass_int8","symbolic_exact"]
    for cid in dict.fromkeys(context):
        if cid=="symbolic_exact":
            rows += m14.evaluate_bundle(m14.symbolic_bundle(),ds,split,cost_n=cost_n,repeats=LATENCY_REPEATS,out_rows=path)
            continue
        for seed in MODEL_SEEDS:
            b=bundle_for_config(cid,seed,out)
            if b is not None:
                rows += m14.evaluate_bundle(b,ds,split,cost_n=cost_n,repeats=LATENCY_REPEATS,out_rows=path)
    cand=[r for r in rows if r["config_id"]==candidate]
    base=[r for r in rows if r["config_id"]=="single_pass_fp"]
    effect=m14.paired_effect(cand,base,bootstraps=m14.BOOTSTRAPS,seed=bootstrap_seed)
    gate=m14.practical_gate(effect)
    write_json(out/f"{split}_effect.json",effect)
    write_json(out/f"{split}_decision.json",gate)
    write_json(out/f"{split}_operating_points.json",m14.all_aggregates(rows))
    return rows,effect,gate


def energy_record(candidate: str, ds, out: Path, split: str):
    xs=torch.from_numpy(ds.inputs[:32]).long()
    payload={"scope":"cpu_package_rapl_not_gpu_not_whole_system","split":split,"configs":{}}
    for cid in [candidate,"single_pass_fp"]:
        b=bundle_for_config(cid,MODEL_SEEDS[0],out)
        def run(bundle=b):
            for i in range(len(xs)):
                a=bundle.solve(xs[i:i+1]); _=sudoku_correct(xs[i:i+1],a,2)
        rec=measure_energy_record(run,n_runs=1)
        payload["configs"][cid]={**rec,"problems_per_window":len(xs),"joules_per_complete_solve":(float(rec["energy_joules"])/len(xs) if rec.get("available") and rec.get("energy_joules") is not None else None)}
    write_json(out/f"{split}_energy.json",payload)


def freeze(candidate: str, out: Path):
    files=[]
    for seed in MODEL_SEEDS:
        cp=out/"checkpoints"/f"fp_recursive_dim64_seed{seed}.pt"
        bp=out/"checkpoints"/f"single_pass_seed{seed}.pt"
        files += [
            {"role":"candidate","seed":seed,"path":str(cp),"sha256":m14.sha256_file(cp)},
            {"role":"primary_baseline","seed":seed,"path":str(bp),"sha256":m14.sha256_file(bp)},
        ]
    rec={
        "attempt":ATTEMPT,"candidate_config_id":candidate,"selected_n_sup":int(candidate.rsplit("n",1)[1]),
        "candidate_deep_supervision_weights":EARLY_WEIGHTS,"files":files,
        "frozen_before_confirmation":True,"git_sha":m14.git_sha(),"timestamp_unix":time.time(),
    }
    write_json(out/"confirmation_freeze_manifest.json",rec)
    return rec


def build_confirmation(out: Path, attempt1_fps: set[str], attempt2_manifest: dict[str, Any]):
    cfg=m14.make_cfg(CONFIRM_DATA_SEED)
    ds,manifest=m14.build_data_splits(cfg,0,0,CONFIRM_N,seed=CONFIRM_DATA_SEED,manifest_path=out/"manifests"/"confirmation_1.json")
    conf=manifest_set(manifest); a2=manifest_set(attempt2_manifest)
    ov1=sorted(conf & attempt1_fps); ov2=sorted(conf & a2)
    audit={
        "confirmation_seed":CONFIRM_DATA_SEED,
        "confirmation_fingerprints":len(conf),
        "attempt1_overlap_count":len(ov1),"attempt2_overlap_count":len(ov2),
        "attempt1_overlap_examples":ov1[:20],"attempt2_overlap_examples":ov2[:20],
        "reserve_confirmation_seed_unopened":RESERVE_CONFIRM_DATA_SEED,
    }
    write_json(out/"manifests"/"confirmation_overlap_audit.json",audit)
    if ov1 or ov2:
        raise m14.ExperimentStop("Attempt 2 confirmation overlap audit failed")
    return ds["test"],manifest,audit


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=".m14a2/experiment"); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    env=measurement_environment(backend="pytorch_cpu_complete_solve",compiler_flags=None,extra={
        "milestone":14,"attempt":ATTEMPT,"git_sha":m14.git_sha(),"protocol":"docs/M14_ATTEMPT2_PROTOCOL.md",
        "primary_cost":"complete_solve_latency","candidate_supervision_weights":EARLY_WEIGHTS,
    })
    write_json(out/"environment.json",env); write_json(out/"energy_inventory.json",energy_counter_inventory())
    train_ds,val_ds,dev_ds,manifest,attempt1_fps=build_hierarchy(out)
    training=train_models(train_ds,val_ds,out)
    _,val_agg,selection=evaluate_validation(val_ds,out)
    candidate=str(selection["selected_config_id"])
    _,dev_effect,dev_gate=evaluate_surface(candidate,dev_ds,"development",out,cost_n=DEVELOPMENT_COST_N,bootstrap_seed=DEV_BOOTSTRAP_SEED)
    energy_record(candidate,dev_ds,out,"development")

    confirmation_opened=False; confirmation_effect=None; confirmation_gate=None
    if dev_gate["pass"]:
        freeze(candidate,out)
        conf_ds,_,_=build_confirmation(out,attempt1_fps,manifest)
        confirmation_opened=True
        _,confirmation_effect,confirmation_gate=evaluate_surface(candidate,conf_ds,"confirmation",out,cost_n=CONFIRMATION_COST_N,bootstrap_seed=CONFIRM_BOOTSTRAP_SEED)
        energy_record(candidate,conf_ds,out,"confirmation")

    status="COMPLETE" if confirmation_gate is not None and confirmation_gate["pass"] else "INCOMPLETE"
    summary={
        "milestone":14,"attempt":ATTEMPT,"status":status,"claim_confirmed":status=="COMPLETE",
        "protocol":"docs/M14_ATTEMPT2_PROTOCOL.md","protocol_committed_before_results":True,
        "attempt1_is_development_evidence":True,
        "intervention":"dim64 FP recursive with early-exit-weighted deep supervision [0.70,0.10,0.10,0.10]",
        "candidate_supervision_weights":EARLY_WEIGHTS,"selected_candidate":candidate,
        "primary_baseline":"single_pass_fp","primary_metric":"strict_sudoku_semantic_validity",
        "training_seeds":MODEL_SEEDS,"n_training_seeds":len(MODEL_SEEDS),
        "data_seed":DATA_SEED,"train_examples":TRAIN_N,"validation_examples":VAL_N,"development_examples":DEV_N,
        "training_steps_per_model":m14.TRAIN_STEPS,"all_training_runs":len(training),
        "development_effect":dev_effect,"development_gate":dev_gate,
        "confirmation_opened":confirmation_opened,"confirmation_effect":confirmation_effect,"confirmation_gate":confirmation_gate,
        "confirmation_seed":CONFIRM_DATA_SEED if confirmation_opened else None,
        "confirmation_examples":CONFIRM_N if confirmation_opened else 0,
        "reserve_confirmation_seed_unopened":RESERVE_CONFIRM_DATA_SEED,
        "thresholds_changed_after_attempt1":False,"baseline_weakened":False,"favorable_seed_filtering":False,
        "scaling_law_claim":False,"reserve_intervention_used":False,
        "confirmation_manifest_opened_only_after_development_pass":confirmation_opened,
        "latency_match_tolerance":"validation median within +/-15%; unmatched points are not iso-budget",
        "development_adaptivity_note":"Attempt 1 informed the preregistered early-exit-loss intervention; no confirmation data informed Attempt 2.",
    }
    write_json(out/"summary.json",summary)
    return 0 if status=="COMPLETE" else 2


if __name__=="__main__": raise SystemExit(main())
