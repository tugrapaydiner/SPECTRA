#!/usr/bin/env python3
"""M14 Development Amendment B: reference-free validator-gated adaptive depth."""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import scripts.m14_primary_experiment as m
from scripts._common import build_data_splits

AMENDMENT_PREREG_SHA="e6cbb77ee09b73500bcd6f23c15e7ad2a0bf5e47"
AMENDMENT_A_DEV_SEED=2026091404
NEW_DEVELOPMENT_SEED=2026091405
WIDTHS=(32,48,64)
MAX_STEPS=(2,3)
PRIMARY_BASELINE=m.PRIMARY_BASELINE


def candidate_kind(width:int)->str:
    if width not in WIDTHS: raise ValueError(width)
    return f"fp_adaptive_w{width}"

def candidate_id(width:int,max_steps:int)->str:
    if max_steps not in MAX_STEPS: raise ValueError(max_steps)
    return f"fp{width}_semstop{max_steps}"

def parse_candidate(cid:str)->tuple[int,int]:
    for w in WIDTHS:
        for d in MAX_STEPS:
            if cid==candidate_id(w,d): return w,d
    raise ValueError(cid)


def make_recursive_model(width:int,seed:int)->m.TRM:
    m.set_seed(int(seed),deterministic=True)
    return m.TRM(dim=int(width),num_tokens=5,seq_len=16,n_layers=1,n=1,T=1,
                 N_sup=m.RECURSIVE_TRAIN_NSUP,heads=4,alpha_y=.1,alpha_z=.1,
                 max_grid_size=8,ternary=False,act8=False).cpu()

def ckpt_path(out:Path,width:int,seed:int)->Path:
    return out/"checkpoints"/f"{candidate_kind(width)}_seed{seed}.pt"


def train_recursive(width:int,seed:int,train_ds:m.GridDataset,val_ds:m.GridDataset,out:Path)->dict[str,Any]:
    kind=candidate_kind(width); model=make_recursive_model(width,seed); arch=m.family_record(kind,model)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=m.LR,weight_decay=m.WEIGHT_DECAY)
    x=torch.from_numpy(train_ds.inputs).long(); y=torch.from_numpy(train_ds.targets).long()
    rng=np.random.default_rng(int(seed)+141500+width)
    curve=out/"curves"/f"{kind}_seed{seed}.jsonl"; curve.parent.mkdir(parents=True,exist_ok=True)
    if curve.exists(): curve.unlink()
    snapshots={1,m.TRAIN_STEPS//2,3*m.TRAIN_STEPS//4,m.TRAIN_STEPS}
    losses=[]; max_grad=0.; train_seconds=0.; model.train()
    for step in range(1,m.TRAIN_STEPS+1):
        idx=torch.from_numpy(rng.integers(0,len(train_ds),size=m.BATCH_SIZE,dtype=np.int64))
        t0=time.perf_counter(); loss=m.training_loss(model,kind,x[idx],y[idx]); opt.zero_grad(set_to_none=True)
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(params,m.CLIP_NORM); opt.step(); train_seconds+=time.perf_counter()-t0
        lv=float(loss.detach()); gv=float(torch.as_tensor(grad).detach())
        if not math.isfinite(lv) or not math.isfinite(gv): raise m.ExperimentStop(f"nonfinite {kind}/seed{seed}/step{step}")
        losses.append(lv); max_grad=max(max_grad,gv)
        if step==1 or step%100==0 or step in snapshots:
            val=None
            if step in snapshots:
                model.eval(); val=m.quick_metrics(model,kind,val_ds); model.train()
            m.append_jsonl(curve,{"kind":kind,"width":width,"seed":seed,"step":step,
                                  "train_blank_ce":lv,"grad_norm_preclip":gv,"validation":val,
                                  "cumulative_train_seconds":train_seconds})
    model.eval(); window=min(50,len(losses)); first=float(np.mean(losses[:window])); last=float(np.mean(losses[-window:]))
    rec={"kind":kind,"width":width,"seed":seed,"steps":m.TRAIN_STEPS,"batch_size":m.BATCH_SIZE,
         "examples_sampled":m.TRAIN_STEPS*m.BATCH_SIZE,"train_seconds":train_seconds,
         "optimizer":{"name":"AdamW","lr":m.LR,"weight_decay":m.WEIGHT_DECAY},
         "clip_grad_norm":m.CLIP_NORM,"objective":"blank_cell_cross_entropy",
         "initial_loss_mean":first,"final_loss_mean":last,
         "relative_loss_improvement":(first-last)/max(abs(first),1e-12),"max_grad_norm_preclip":max_grad,
         "architecture":arch,"training_block_applications":m.TRAIN_STEPS*m.BATCH_SIZE*arch["block_apps_per_training_example"],
         "validation_final":m.quick_metrics(model,kind,val_ds),"tensor_state_sha256":m.tensor_state_sha256(model)}
    cp=ckpt_path(out,width,seed); m.save_checkpoint(cp,model,kind,seed,rec); rec["checkpoint"]=str(cp); rec["checkpoint_sha256"]=m.sha256_file(cp)
    m.append_jsonl(out/"training_runs.jsonl",rec); return rec


def load_recursive(path:Path,width:int)->tuple[m.TRM,dict[str,Any]]:
    p=torch.load(path,map_location="cpu",weights_only=False)
    if p.get("format")!="spectra.m14_model" or p.get("version")!=1: raise m.ExperimentStop(f"invalid checkpoint {path}")
    model=make_recursive_model(width,int(p["seed"])); model.load_state_dict(p["model_state"],strict=True); model.eval(); return model,p


def run_adaptive_solve(model:m.TRM,x:torch.Tensor,max_steps:int,
                       validator:Callable[[torch.Tensor,torch.Tensor,int],torch.Tensor]|None=None)->tuple[torch.Tensor,dict[str,Any]]:
    if max_steps not in MAX_STEPS: raise ValueError(max_steps)
    validator = m.sudoku_correct if validator is None else validator
    old=int(model.N_sup); model.N_sup=int(max_steps)
    try:
        state=model.init_execution_state(x,height=4,width=4)
        pred=None; validator_calls=0; realized=0; stopped=False; last_cum={}
        for _ in range(max_steps):
            out=model.run_execution_step(state); realized+=1; last_cum=dict(out["work_cumulative"])
            pred=m.clamp_givens(x,out["logits"].argmax(-1)); validator_calls+=1
            ok=validator(x,pred,2).bool()
            if bool(ok.all()): stopped=True; break
        assert pred is not None
        work={"realized_steps":realized,"max_steps":max_steps,"stopped_on_semantic_validity":stopped,
              "semantic_validator_calls_inside_solver":validator_calls,
              "recursive_cycle_calls":int(last_cum.get("recursive_cycles",0)),
              "block_applications":int(last_cum.get("block_applications",0)),
              "output_head_vectors":int(last_cum.get("output_head_vectors",0)),
              "verifier_calls":0,"router_calls":0,"action_calls":0,"conversion_calls":0}
        return pred,work
    finally:
        model.N_sup=old


def adaptive_bundle(width:int,seed:int,max_steps:int,out:Path)->m.SolverBundle:
    cp=ckpt_path(out,width,seed); model,payload=load_recursive(cp,width); last={}
    def solve(x:torch.Tensor)->torch.Tensor:
        nonlocal last
        with torch.inference_mode(): pred,last=run_adaptive_solve(model,x,max_steps)
        return pred
    def work()->dict[str,Any]: return dict(last)
    return m.SolverBundle(candidate_id(width,max_steps),"fp_adaptive_depth",seed,solve,work=work,
                          setup={"checkpoint":str(cp),"checkpoint_sha256":m.sha256_file(cp),"width":width,
                                 "n":1,"T":1,"training_N_sup":m.RECURSIVE_TRAIN_NSUP,"max_steps":max_steps,
                                 "stopping_rule":"reference_free_sudoku_semantic_validity",
                                 "trainable_params":int(payload["family"]["trainable_params"])})


def wrap_baseline(seed:int,out:Path,int8:bool=False)->m.SolverBundle|None:
    b=m.single_pass_bundle(seed,out,int8=int8,audit_sink=(out/"int8_conversion_audit.jsonl") if int8 else None)
    if b is None: return None
    last={"realized_steps":1,"block_applications":2,"semantic_validator_calls_inside_solver":0,
          "recursive_cycle_calls":0,"verifier_calls":0,"router_calls":0,"action_calls":0,"conversion_calls":1 if int8 else 0}
    return m.SolverBundle(b.config_id,b.family,b.seed,b.solve,work=lambda:dict(last),setup=b.setup)


def evaluate_bundle(bundle:m.SolverBundle,ds:m.GridDataset,split:str,*,cost_n:int,repeats:int,out_rows:Path)->list[dict[str,Any]]:
    xs=torch.from_numpy(ds.inputs).long(); ys=torch.from_numpy(ds.targets).long(); n_cost=min(cost_n,len(ds))
    for i in range(min(4,len(ds))):
        ans=bundle.solve(xs[i:i+1]); _=m.sudoku_correct(xs[i:i+1],ans,2)
    rows=[]
    for i in range(len(ds)):
        xi=xs[i:i+1]; answer=None; semantic=None; latency=None
        if i<n_cost:
            times=[]
            for _r in range(max(1,repeats)):
                t0=time.perf_counter_ns(); a=bundle.solve(xi); ok=m.sudoku_correct(xi,a,2).bool(); dt=(time.perf_counter_ns()-t0)/1e6
                times.append(dt)
                if answer is None: answer=a.detach().cpu(); semantic=bool(ok.item())
            latency=float(statistics.median(times))
        else:
            answer=bundle.solve(xi).detach().cpu(); semantic=bool(m.sudoku_correct(xi,answer,2).item())
        target=ys[i:i+1]; blank=xi.cpu()==0; work=bundle.work() if bundle.work else {}
        row={"split":split,"config_id":bundle.config_id,"family":bundle.family,"seed":bundle.seed,
             "example_index":i,"example_id":ds.ids[i],"semantic_success":int(semantic),
             "exact_reference_match":int(bool((answer==target).all().item())),
             "blank_cell_accuracy":float((answer[blank]==target[blank]).float().mean()),
             "latency_ms":latency,"latency_repeats":repeats if i<n_cost else 0,
             "complete_solve_timing_includes_semantic_check":True,"target_used_inside_solver":False,
             "work_realized_steps":work.get("realized_steps"),"work_block_applications":work.get("block_applications"),
             "work_recursive_cycle_calls":work.get("recursive_cycle_calls"),
             "work_semantic_validator_calls_inside_solver":work.get("semantic_validator_calls_inside_solver"),
             "work_total_semantic_validator_calls_including_final":(int(work.get("semantic_validator_calls_inside_solver",0))+1),
             "work_verifier_calls":work.get("verifier_calls",0),"work_router_calls":work.get("router_calls",0),
             "work_action_calls":work.get("action_calls",0),"work_conversion_calls":work.get("conversion_calls",0)}
        m.append_jsonl(out_rows,row); rows.append(row)
    return rows


def aggregate(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    base=m.all_aggregates(rows); by={r["config_id"]:r for r in base}
    for cid,r in by.items():
        rr=[x for x in rows if x["config_id"]==cid and x["seed"]!="symbolic"]
        steps=[int(x["work_realized_steps"]) for x in rr if x.get("work_realized_steps") is not None]
        blocks=[int(x["work_block_applications"]) for x in rr if x.get("work_block_applications") is not None]
        if steps:
            r["mean_realized_steps"]=float(np.mean(steps)); r["median_realized_steps"]=float(np.median(steps)); r["fraction_step1_stop"]=float(np.mean(np.asarray(steps)==1))
        if blocks: r["mean_block_applications"]=float(np.mean(blocks))
    return base


def manifest_union(*manifests:dict[str,Any])->set[str]:
    out=set()
    for man in manifests: out |= m.manifest_fingerprints(man)
    return out

def build_consumed_amendment_a_manifest(out:Path)->dict[str,Any]:
    cfg=m.make_cfg(AMENDMENT_A_DEV_SEED); _,man=build_data_splits(cfg,0,0,m.DEV_N,seed=AMENDMENT_A_DEV_SEED,
        manifest_path=out/"manifests"/"consumed_amendment_a_development.json"); return man

def build_fresh_development(out:Path,original:dict[str,Any],a_man:dict[str,Any])->tuple[m.GridDataset,dict[str,Any],dict[str,Any]]:
    cfg=m.make_cfg(NEW_DEVELOPMENT_SEED); ds,man=build_data_splits(cfg,0,0,m.DEV_N,seed=NEW_DEVELOPMENT_SEED,
        manifest_path=out/"manifests"/"amendment_b_development_reserve.json")
    used=manifest_union(original,a_man); cur=m.manifest_fingerprints(man); overlap=sorted(used&cur)
    audit={"prior_fingerprints":len(used),"later_fingerprints":len(cur),"overlap_count":len(overlap),"overlap_examples":overlap[:10],"later_seed":NEW_DEVELOPMENT_SEED}
    m.write_json(out/"manifests"/"amendment_b_development_overlap_audit.json",audit)
    if overlap: raise m.ExperimentStop("Amendment-B development overlaps prior evidence")
    return ds["test"],man,audit

def build_confirmation(out:Path,original:dict[str,Any],a_man:dict[str,Any],b_man:dict[str,Any])->tuple[m.GridDataset,dict[str,Any],dict[str,Any]]:
    cfg=m.make_cfg(m.CONFIRM_DATA_SEED); ds,man=build_data_splits(cfg,0,0,m.CONFIRM_N,seed=m.CONFIRM_DATA_SEED,
        manifest_path=out/"manifests"/"confirmation_1.json")
    used=manifest_union(original,a_man,b_man); cur=m.manifest_fingerprints(man); overlap=sorted(used&cur)
    audit={"prior_development_fingerprints":len(used),"confirmation_fingerprints":len(cur),"overlap_count":len(overlap),"overlap_examples":overlap[:10],
           "confirmation_seed":m.CONFIRM_DATA_SEED,"reserve_confirmation_seed_unopened":m.RESERVE_CONFIRM_DATA_SEED}
    m.write_json(out/"manifests"/"confirmation_overlap_audit.json",audit)
    if overlap: raise m.ExperimentStop("Amendment-B confirmation overlaps development evidence")
    return ds["test"],man,audit


def clear(path:Path)->None:
    if path.exists(): path.unlink()

def train_all(train_ds:m.GridDataset,val_ds:m.GridDataset,out:Path)->list[dict[str,Any]]:
    clear(out/"training_runs.jsonl"); runs=[]
    for seed in m.MODEL_SEEDS: runs.append(m.train_one("single_pass",seed,train_ds,val_ds,out))
    for w in WIDTHS:
        for seed in m.MODEL_SEEDS: runs.append(train_recursive(w,seed,train_ds,val_ds,out))
    return runs


def validation_candidate_rows(val_ds:m.GridDataset,out:Path)->list[dict[str,Any]]:
    path=out/"validation_rows.jsonl"; clear(path); rows=[]
    for seed in m.MODEL_SEEDS:
        for w in WIDTHS:
            for d in MAX_STEPS:
                rows.extend(evaluate_bundle(adaptive_bundle(w,seed,d,out),val_ds,"validation_amendment_b",cost_n=m.VALIDATION_COST_N,repeats=m.LATENCY_REPEATS,out_rows=path))
        rows.extend(evaluate_bundle(wrap_baseline(seed,out,False),val_ds,"validation_amendment_b",cost_n=m.VALIDATION_COST_N,repeats=m.LATENCY_REPEATS,out_rows=path))
    return rows

def select_candidate(aggs:list[dict[str,Any]],training:list[dict[str,Any]])->dict[str,Any]:
    by={r["config_id"]:dict(r) for r in aggs}; base=by[PRIMARY_BASELINE]; bq=float(base["semantic_validity"]); blat=float(base["latency_median_ms"])
    params={str(r["kind"]):int(r["architecture"]["trainable_params"]) for r in training}; bp=params["single_pass"]
    candidates=[]
    for w in WIDTHS:
        kp=params[candidate_kind(w)]
        if kp>=bp: raise m.ExperimentStop(f"candidate width {w} is not smaller than baseline")
        for d in MAX_STEPS:
            cid=candidate_id(w,d); r=by[cid]; q=float(r["semantic_validity"])-bq; ratio=float(r["latency_median_ms"])/blat
            r.update({"validation_quality_difference_vs_baseline":q,"validation_latency_ratio_vs_baseline":ratio,"trainable_params":kp,"parameter_ratio_vs_baseline":kp/bp,
                      "path_a_point_eligible":q>=.03 and ratio<=1.15,"path_b_point_eligible":q>=-.02 and ratio<=.65,"width":w,"max_steps":d})
            av=max(0.,.03-q)/.03 + max(0.,ratio-1.15)/.15; bv=max(0.,-.02-q)/.02 + max(0.,ratio-.65)/.35
            r["validation_violation_score"]=min(av,bv); candidates.append(r)
    eligible=[r for r in candidates if r["path_a_point_eligible"] or r["path_b_point_eligible"]]
    if eligible:
        chosen=sorted(eligible,key=lambda r:(-float(r["semantic_validity"]),float(r["validation_latency_ratio_vs_baseline"]),int(r["width"]),int(r["max_steps"])))[0]; pool="primary_point_threshold_eligible"
    else:
        chosen=sorted(candidates,key=lambda r:(float(r["validation_violation_score"]),-float(r["semantic_validity"]),float(r["latency_median_ms"]),int(r["width"]),int(r["max_steps"])))[0]; pool="preregistered_minimum_violation_fallback"
    return {"selected_config_id":chosen["config_id"],"selection_pool":pool,"selected_validation":chosen,"primary_baseline_validation":base,"all_candidate_validation":candidates,
            "selection_used_development":False,"selection_used_confirmation":False,"selection_rule":"docs/M14_DEVELOPMENT_AMENDMENT_B.md#5-validation-only-selection"}


def bundle_for(cid:str,seed:int,out:Path)->m.SolverBundle|None:
    if cid.startswith("fp") and "_semstop" in cid:
        w,d=parse_candidate(cid); return adaptive_bundle(w,seed,d,out)
    if cid==PRIMARY_BASELINE: return wrap_baseline(seed,out,False)
    if cid=="single_pass_int8": return wrap_baseline(seed,out,True)
    if cid=="symbolic_exact": return m.symbolic_bundle()
    raise ValueError(cid)

def eval_surface(candidate:str,ds:m.GridDataset,out:Path,*,split:str,filename:str,cost_n:int,bootstrap_seed:int)->tuple[list[dict[str,Any]],dict[str,Any],dict[str,Any]]:
    path=out/filename; clear(path); rows=[]
    for seed in m.MODEL_SEEDS:
        for cid in (candidate,PRIMARY_BASELINE,"single_pass_int8"):
            b=bundle_for(cid,seed,out)
            if b is not None: rows.extend(evaluate_bundle(b,ds,split,cost_n=cost_n,repeats=m.LATENCY_REPEATS,out_rows=path))
    rows.extend(evaluate_bundle(m.symbolic_bundle(),ds,split,cost_n=cost_n,repeats=m.LATENCY_REPEATS,out_rows=path))
    cand=[r for r in rows if r["config_id"]==candidate]; base=[r for r in rows if r["config_id"]==PRIMARY_BASELINE]
    eff=m.paired_effect(cand,base,seed=bootstrap_seed); gate=m.practical_gate(eff); return rows,eff,gate


def freeze(candidate:str,out:Path)->dict[str,Any]:
    w,d=parse_candidate(candidate); files=[]
    for seed in m.MODEL_SEEDS:
        cp=ckpt_path(out,w,seed); bp=out/"checkpoints"/f"single_pass_seed{seed}.pt"
        files += [{"role":"candidate_reasoner","seed":seed,"path":str(cp),"sha256":m.sha256_file(cp)},
                  {"role":"primary_baseline","seed":seed,"path":str(bp),"sha256":m.sha256_file(bp)}]
    rec={"candidate_config_id":candidate,"width":w,"max_steps":d,"stopping_rule":"reference_free_sudoku_semantic_validity",
         "files":files,"frozen_before_confirmation":True,"git_sha":m.git_sha(),"amendment_prereg_sha":AMENDMENT_PREREG_SHA,"timestamp_unix":time.time()}
    m.write_json(out/"confirmation_freeze_manifest.json",rec); return rec

def energy(candidate:str,ds:m.GridDataset,out:Path,split:str)->dict[str,Any]:
    xs=torch.from_numpy(ds.inputs[:32]).long(); rec={"scope":"cpu_package_rapl_not_gpu_not_whole_system","split":split,"configs":{}}
    for cid in (candidate,PRIMARY_BASELINE):
        b=bundle_for(cid,m.MODEL_SEEDS[0],out)
        def run():
            for i in range(len(xs)):
                ans=b.solve(xs[i:i+1]); _=m.sudoku_correct(xs[i:i+1],ans,2)
        e=m.measure_energy_record(run,n_runs=1); rec["configs"][cid]={**e,"problems_per_window":len(xs),"joules_per_complete_solve":float(e["energy_joules"])/len(xs) if e.get("available") and e.get("energy_joules") is not None else None}
    m.write_json(out/f"{split}_energy.json",rec); return rec


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=".m14b/experiment"); args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    m.write_json(out/"environment.json",m.measurement_environment(backend="pytorch_cpu_m11_explicit_step",compiler_flags=None,extra={"milestone":14,"development_amendment":"B","git_sha":m.git_sha(),"amendment_prereg_sha":AMENDMENT_PREREG_SHA,"primary_cost":"complete_solve_latency"}))
    m.write_json(out/"energy_inventory.json",m.energy_counter_inventory())
    train_ds,val_ds,_dev1,original=m.build_development_data(out); a_man=build_consumed_amendment_a_manifest(out)
    training=train_all(train_ds,val_ds,out)
    val_rows=validation_candidate_rows(val_ds,out); val_aggs=aggregate(val_rows); selection=select_candidate(val_aggs,training); candidate=str(selection["selected_config_id"])
    m.write_json(out/"validation_operating_points.json",val_aggs); m.write_csv(out/"validation_operating_points.csv",val_aggs); m.write_json(out/"candidate_selection.json",selection)
    dev_ds,b_man,dev_audit=build_fresh_development(out,original,a_man)
    dev_rows,dev_eff,dev_gate=eval_surface(candidate,dev_ds,out,split="development_amendment_b",filename="development_rows_amendment_b.jsonl",cost_n=m.DEVELOPMENT_COST_N,bootstrap_seed=m.BOOTSTRAP_SEED)
    m.write_json(out/"development_operating_points_amendment_b.json",aggregate(dev_rows)); m.write_json(out/"development_effect_amendment_b.json",dev_eff); m.write_json(out/"development_decision_amendment_b.json",dev_gate); energy(candidate,dev_ds,out,"development_amendment_b")
    confirmation_opened=False; conf_eff=conf_gate=conf_audit=None
    if bool(dev_gate["pass"]):
        freeze(candidate,out); conf_ds,_conf_man,conf_audit=build_confirmation(out,original,a_man,b_man); confirmation_opened=True
        conf_rows,conf_eff,conf_gate=eval_surface(candidate,conf_ds,out,split="confirmation",filename="confirmation_rows.jsonl",cost_n=m.CONFIRMATION_COST_N,bootstrap_seed=m.BOOTSTRAP_SEED+1)
        m.write_json(out/"confirmation_operating_points.json",aggregate(conf_rows)); m.write_json(out/"confirmation_effect.json",conf_eff); m.write_json(out/"confirmation_decision.json",conf_gate); energy(candidate,conf_ds,out,"confirmation")
    status="COMPLETE" if confirmation_opened and conf_gate is not None and bool(conf_gate["pass"]) else "INCOMPLETE"
    summary={"milestone":14,"development_amendment":"B","status":status,"claim_confirmed":status=="COMPLETE",
             "amendment_preregistered_before_results":True,"amendment_prereg_sha":AMENDMENT_PREREG_SHA,
             "prior_attempts":[{"name":"attempt1","run":34159474745,"status":"INCOMPLETE","confirmation_opened":False},{"name":"amendment_a","run":34164642690,"status":"INCOMPLETE","confirmation_opened":False}],
             "primary_metric":"strict_sudoku_semantic_validity","primary_baseline":PRIMARY_BASELINE,
             "candidate_widths":list(WIDTHS),"candidate_max_steps":list(MAX_STEPS),"selected_candidate":candidate,"selection_used_validation_only":True,
             "validation_operating_points":val_aggs,"development_seed":NEW_DEVELOPMENT_SEED,"development_examples":m.DEV_N,"development_overlap_audit":dev_audit,
             "development_effect":dev_eff,"development_gate":dev_gate,"confirmation_opened":confirmation_opened,
             "confirmation_seed":m.CONFIRM_DATA_SEED if confirmation_opened else None,"confirmation_examples":m.CONFIRM_N if confirmation_opened else 0,
             "confirmation_overlap_audit":conf_audit,"confirmation_effect":conf_eff,"confirmation_gate":conf_gate,
             "reserve_confirmation_seed_unopened":m.RESERVE_CONFIRM_DATA_SEED,
             "confirmation_becomes_development_evidence_if_future_changes_follow":bool(confirmation_opened and status!="COMPLETE"),
             "n_training_seeds":len(m.MODEL_SEEDS),"training_seeds":list(m.MODEL_SEEDS),"training_steps":m.TRAIN_STEPS,"train_examples":m.TRAIN_N,"validation_examples":m.VAL_N,"batch_size":m.BATCH_SIZE,"training_runs":len(training),
             "thresholds_changed_after_results":False,"baseline_weakened":False,"favorable_seed_filtering":False,"scaling_law_claim":False,"primary_gate_unchanged_from_m14":True,
             "stopping_rule":"reference_free_sudoku_semantic_validity","claim_scope_if_passed":"tasks_with_exact_reference_free_semantic_validator",
             "energy_scope":"cpu_package_if_valid_otherwise_unavailable_not_gpu_not_whole_system",
             "sequential_selection_note":"Attempt 1, Amendment A, and their development rows are disclosed selection history. Amendment B uses fresh development seed 2026091405; independent confirmation remains mandatory."}
    m.write_json(out/"summary.json",summary); return 0 if status=="COMPLETE" else 2

if __name__=="__main__": raise SystemExit(main())