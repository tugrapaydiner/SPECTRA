"""Matched evaluator targets and first-hit distributions on frozen FP32 pools.

The first-hit head estimates time to a first exact-valid decode under one named
identity continuation. It is not an optimal-search value or calibrated guarantee.
Categorical first-hit mass makes cumulative horizon probabilities monotone by
construction; this algebraic property does not establish prediction accuracy.
"""
from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.grounded_targets import assert_frozen_reasoner, tensor_state_sha256
from model.grounded_verifier import GroundedStateVerifier
from model.verifier import sudoku_correct, sudoku_score
from .experiment import CORE_TENSORS, POOL_PATHS, RECIPE, Dataset
from .identity import canonical_json, file_sha256, positive_int, strict_json, write_json
from .semantics import SemanticMismatch

TARGETS=('improvement','validity','quality','first_hit')
HORIZON=3
POLICY=RECIPE['first_hit_policy']


def clamp_decode(core,x,y):
    return torch.where(x!=0,x,core.out_head(y).argmax(dim=-1))


@dataclass
class Pool:
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    answers: torch.Tensor
    validity: torch.Tensor
    quality: torch.Tensor
    improvement: torch.Tensor
    first_hit: torch.Tensor
    ids: tuple[str,...]
    work: dict[str,Any]

    @property
    def state_count(self):return self.x.shape[0]

    @property
    def puzzle_count(self):return len(self.ids)


@torch.inference_mode()
def build_pool(core,codebook,dataset: Dataset,*,batch_size:int=64) -> Pool:
    assert_frozen_reasoner(core);positive_int(batch_size,'batch_size')
    values={key:[] for key in ('x','y','z','answers','validity','quality','improvement','first_hit')}
    prefixes=sorted({p[:i] for p in POOL_PATHS for i in range(1,len(p)+1)},key=lambda p:(len(p),p))
    calls=0;label_calls=0;started=time.perf_counter();cpu=time.process_time()
    for start in range(0,len(dataset.inputs),batch_size):
        x=torch.from_numpy(dataset.inputs[start:start+batch_size]).long();emb=core.token_embed(x)+core.encode_positions(x,4,4)
        states={(): (torch.zeros_like(emb),torch.zeros_like(emb))}
        for path in prefixes:
            y,z=states[path[:-1]];z_action=codebook.apply_action(z,path[-1])
            states[path]=core.recursive_cycle(emb,y,z_action);calls+=1
        y=torch.stack([states[p][0] for p in POOL_PATHS],dim=1).reshape(-1,16,64)
        z=torch.stack([states[p][1] for p in POOL_PATHS],dim=1).reshape(-1,16,64)
        xx=x.repeat_interleave(len(POOL_PATHS),dim=0)
        ans=clamp_decode(core,xx,y);ok=sudoku_correct(xx,ans,2).bool();quality=sudoku_score(xx,ans,box=2)
        first=torch.where(ok,0,HORIZON+1).long();yy,zz=y,z;before=quality
        improvement=None
        # Offline labels: all states are continued equally, regardless of solve status.
        xxemb=core.token_embed(xx)+core.encode_positions(xx,4,4)
        for horizon in range(1,HORIZON+1):
            yy,zz=core.recursive_cycle(xxemb,yy,zz);label_calls+=1
            aa=clamp_decode(core,xx,yy);now=sudoku_correct(xx,aa,2).bool()
            if horizon==1:improvement=(sudoku_score(xx,aa,box=2)>before+1e-6).float()
            first=torch.where((first==HORIZON+1)&now,horizon,first)
        for key,value in dict(x=xx,y=y,z=z,answers=ans,validity=ok.float(),quality=quality,
                              improvement=improvement,first_hit=first).items():
            # Clone outside inference mode below before these tensors enter autograd.
            values[key].append(value.detach().cpu())
    combined={k:torch.cat(v,dim=0) for k,v in values.items()}
    work={"pool_paths":[list(p) for p in POOL_PATHS],"distinct_prefixes":[list(p) for p in prefixes],
          "puzzle_count":len(dataset.inputs),"state_count":len(dataset.inputs)*len(POOL_PATHS),
          "batched_pool_cycle_calls":calls,"pool_transition_equivalents":len(dataset.inputs)*len(prefixes),
          "batched_label_cycle_calls":label_calls,"offline_label_transition_equivalents":len(dataset.inputs)*len(POOL_PATHS)*HORIZON,
          "wall_seconds":time.perf_counter()-started,"process_cpu_seconds":time.process_time()-cpu,
          "states":"fp32","decoder":RECIPE['decoder'],"targets_used_for_transitions":False,
          "reuse_is_offline_analysis_not_runtime_speedup":True}
    return Pool(**combined,ids=dataset.ids,work=work)


class TargetStateEvaluator(nn.Module):
    def __init__(self,target:str,core_sha256:str,*,seed:int):
        super().__init__()
        if target not in TARGETS:raise SemanticMismatch(f"unknown target {target}")
        self.target=target;self.core_sha256=core_sha256
        torch.manual_seed(seed)
        self.backbone=GroundedStateVerifier(num_tokens=5,dim=64,n_layers=1,heads=4,max_grid_size=8,act_bits=8,include_y=True)
        if target=='first_hit':
            # Shared backbone and a different output cardinality, explicitly disclosed.
            torch.manual_seed(seed+1)
            self.backbone.head=nn.Linear(64,HORIZON+2)

    def logits(self,x,y,z,width:int=4):
        return self.backbone.forward_logits(x,y,z,width)

    def forward(self,*args,**kwargs):
        raise SemanticMismatch("use current_value or within_horizon with explicit target semantics")

    def current_value(self,x,y,z,*,target:str):
        if target!=self.target or target=='first_hit':raise SemanticMismatch("current-value target mismatch")
        return torch.sigmoid(self.logits(x,y,z))

    def first_hit_probabilities(self,x,y,z,*,continuation_policy:str):
        if self.target!='first_hit' or continuation_policy!=POLICY:
            raise SemanticMismatch("first-hit target/continuation policy mismatch")
        return torch.softmax(self.logits(x,y,z),dim=-1)

    def within_horizon(self,x,y,z,horizon:int,*,continuation_policy:str):
        horizon=positive_int(horizon,'horizon',allow_zero=True)
        if horizon>HORIZON:raise SemanticMismatch("requested horizon exceeds trained support")
        return self.first_hit_probabilities(x,y,z,continuation_policy=continuation_policy)[...,:horizon+1].sum(dim=-1).clamp(0.0,1.0)

    def contract(self):
        return {"format":"spectra.target_evaluator.v1","target":self.target,"core_tensor_sha256":self.core_sha256,
                "state":"fp32_xyz_with_internal_verifier_z_fakequant","decoder":RECIPE['decoder'],"checker":RECIPE['checker'],
                "continuation_policy":POLICY if self.target in ('first_hit','improvement') else None,
                "max_supported_horizon":HORIZON if self.target=='first_hit' else (1 if self.target=='improvement' else None),
                "first_hit_classes":[0,1,2,3,'not_within_3'] if self.target=='first_hit' else None,
                "calibration_or_coverage_guarantee":False}


def backbone_hash(model: TargetStateEvaluator) -> str:
    h=hashlib.sha256()
    for name,tensor in sorted(model.backbone.state_dict().items()):
        if name.startswith('head.'):continue
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fit_targets(core,seed:int,pool:Pool,output:Path,fit_manifest_sha:str) -> dict[str,Any]:
    assert_frozen_reasoner(core)
    before_core=tensor_state_sha256(core)
    if before_core!=CORE_TENSORS[seed]:raise ValueError("wrong frozen source model")
    aux=output/'auxiliary';aux.mkdir(parents=True,exist_ok=True)
    audit_path=aux/f'fit_seed{seed}.json'
    if audit_path.exists():raise ValueError("auxiliary fit already exists; no silent retraining")
    # Materialize normal tensors; inference-created tensors cannot be saved by autograd.
    data={key:getattr(pool,key).clone() for key in ('x','y','z','validity','quality','improvement','first_hit')}
    spec=RECIPE['auxiliary'];rng=np.random.default_rng(seed+spec['batch_seed_offset'])
    schedule=rng.integers(0,pool.state_count,size=(spec['steps'],spec['batch_size']),dtype=np.int64)
    schedule_sha=hashlib.sha256(schedule.tobytes()).hexdigest();records=[];initial_backbones=[]
    for target in TARGETS:
        model=TargetStateEvaluator(target,CORE_TENSORS[seed],seed=seed+spec['backbone_init_seed_offset'])
        initial_backbones.append(backbone_hash(model));model.train()
        params=list(model.parameters());opt=torch.optim.AdamW(params,lr=spec['lr'],weight_decay=spec['weight_decay'])
        if any(id(p) in {id(q) for q in core.parameters()} for p in params):raise ValueError("auxiliary optimizer owns frozen core")
        curve=[];start=time.perf_counter();cpu=time.process_time()
        for step,index in enumerate(schedule,1):
            ix=torch.from_numpy(index);logits=model.logits(data['x'][ix],data['y'][ix],data['z'][ix])
            if target=='first_hit':loss=F.cross_entropy(logits,data[target][ix])
            else:loss=F.binary_cross_entropy_with_logits(logits,data[target][ix])
            if not bool(torch.isfinite(loss)):raise FloatingPointError("non-finite auxiliary loss")
            opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,spec['clip'])
            if not bool(torch.isfinite(grad)):raise FloatingPointError("non-finite auxiliary gradient")
            opt.step();curve.append({"step":step,"loss":float(loss.detach()),"preclip_gradient_norm":float(grad)})
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        path=aux/f'{target}_seed{seed}.pt'
        with path.open('xb') as f:
            torch.save({"format":"spectra.m16_evaluator","version":1,"seed":seed,"target":target,
                "contract":model.contract(),"state":model.state_dict(),"fit_manifest_sha256":fit_manifest_sha,
                "schedule_sha256":schedule_sha,"recipe_sha256":hashlib.sha256(canonical_json(RECIPE)).hexdigest()},f)
        curve_path=aux/f'{target}_seed{seed}_curve.json';write_json(curve_path,curve,exclusive=True)
        records.append({"target":target,"checkpoint":path.name,"sha256":file_sha256(path),
            "tensor_sha256":tensor_state_sha256(model),"parameter_count":sum(p.numel() for p in model.parameters()),
            "initial_backbone_sha256":initial_backbones[-1],"final_backbone_sha256":backbone_hash(model),
            "wall_seconds":time.perf_counter()-start,"process_cpu_seconds":time.process_time()-cpu,
            "initial_loss":curve[0]['loss'],"final_loss":curve[-1]['loss'],"schedule_sha256":schedule_sha})
    if len(set(initial_backbones))!=1:raise ValueError("matched targets did not share the same backbone initialization")
    if tensor_state_sha256(core)!=before_core:raise ValueError("frozen core moved during auxiliary fitting")
    result={"seed":seed,"frozen_core_tensor_sha256":before_core,"frozen_core_unchanged":True,
            "fit_manifest_sha256":fit_manifest_sha,"recipe":spec,"models":records,
            "training_states":pool.state_count,"training_puzzles":pool.puzzle_count,
            "pool_work":pool.work,"labels":{"validity_mean":float(pool.validity.mean()),
                "improvement_mean":float(pool.improvement.mean()),"quality_mean":float(pool.quality.mean()),
                "first_hit_counts":torch.bincount(pool.first_hit,minlength=HORIZON+2).tolist()}}
    write_json(audit_path,result,exclusive=True)
    return result


def load_targets(output:Path,seed:int) -> dict[str,TargetStateEvaluator]:
    audit=strict_json((output/'auxiliary'/f'fit_seed{seed}.json').read_bytes());models={}
    for row in audit['models']:
        path=output/'auxiliary'/row['checkpoint']
        if file_sha256(path)!=row['sha256']:raise ValueError("auxiliary checkpoint hash mismatch")
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['format']!='spectra.m16_evaluator' or payload['version']!=1 or payload['seed']!=seed or payload['target']!=row['target']:
            raise ValueError("auxiliary checkpoint identity mismatch")
        model=TargetStateEvaluator(row['target'],CORE_TENSORS[seed],seed=seed+RECIPE['auxiliary']['backbone_init_seed_offset'])
        if payload['contract']!=model.contract():raise SemanticMismatch("checkpoint target contract mismatch")
        if payload['fit_manifest_sha256']!=audit['fit_manifest_sha256'] or payload['fit_manifest_sha256']!=file_sha256(output/'data'/'fit.json'):
            raise ValueError('auxiliary fit-data binding mismatch')
        if payload['schedule_sha256']!=row['schedule_sha256'] or payload['recipe_sha256']!=hashlib.sha256(canonical_json(RECIPE)).hexdigest():
            raise ValueError('auxiliary recipe/schedule binding mismatch')
        model.load_state_dict(payload['state'],strict=True);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        if tensor_state_sha256(model)!=row['tensor_sha256']:raise ValueError("auxiliary tensor hash mismatch")
        models[row['target']]=model
    if set(models)!=set(TARGETS):raise ValueError("missing matched target")
    return models


@torch.inference_mode()
def predict_pool(models,legacy,pool:Pool,*,batch_size:int=128) -> dict[str,np.ndarray]:
    chunks={key:[] for key in ('legacy_improvement','improvement','validity','quality','first_hit_h0','first_hit_h1','first_hit_h2','first_hit_h3')}
    for start in range(0,pool.state_count,batch_size):
        x,y,z=pool.x[start:start+batch_size],pool.y[start:start+batch_size],pool.z[start:start+batch_size]
        chunks['legacy_improvement'].append(legacy.value_state(x,y,z,width=4).cpu().numpy())
        for target in ('improvement','validity','quality'):
            chunks[target].append(models[target].current_value(x,y,z,target=target).cpu().numpy())
        cdf=models['first_hit'].first_hit_probabilities(x,y,z,continuation_policy=POLICY).cumsum(dim=-1).clamp(0.0,1.0)
        for horizon in range(HORIZON+1):chunks[f'first_hit_h{horizon}'].append(cdf[:,horizon].cpu().numpy())
    result={key:np.concatenate(value) for key,value in chunks.items()}
    if any(not np.isfinite(value).all() or (value<0).any() or (value>1+1e-6).any() for value in result.values()):
        raise ValueError("invalid prediction probabilities")
    return result
