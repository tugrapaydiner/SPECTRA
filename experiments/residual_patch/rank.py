"""Adaptive follow-up: learn within-state choices rather than absolute success.

All original training observations are retained. Tied labels contribute no
pairwise ordering; their states are not quietly relabeled as positive evidence.
Models are NOT calibrated success probabilities. Runtime stays unchanged.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .runtime import F,H,W,native_scores
from .data import write_json


def pairs_from_snapshots(targets,snapshots):
    left=[];right=[];weight=[];informative=0
    for s in snapshots:
        first=s['candidate_start'];count=s['candidate_count'];edges=[]
        for i in range(first,first+count):
            for j in range(first,first+count):
                if targets[i]>targets[j]:edges.append((i,j,float(targets[i]-targets[j])))
        if not edges:continue
        informative+=1;total=sum(d for _,_,d in edges)
        for i,j,d in edges:left.append(i);right.append(j);weight.append(d/total)
    if not left:raise ValueError('no within-state outcome ordering to learn')
    return np.asarray(left),np.asarray(right),np.asarray(weight),informative

def train(config,data_dir,out):
    out=Path(out);out.mkdir(exist_ok=False,parents=True);data_dir=Path(data_dir)
    with np.load(data_dir/'labels.npz',allow_pickle=False) as d:
        raw=d['features'];targets=d['targets']
    with gzip.open(data_dir/'snapshots.jsonl.gz','rt') as f: snapshots=[json.loads(line) for line in f]
    left,right,weights,informative=pairs_from_snapshots(targets,snapshots)
    prob=torch.tensor(weights/weights.sum(),dtype=torch.float64)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reports=[]
    for mode in ['coarse','residual']:
        mask=np.ones(F) if mode=='residual' else np.isin(np.arange(F),config['coarse_indices']).astype(float)
        mean=raw.mean(0);std=raw.std(0);std=np.where(std<1e-8,1,std)
        x=torch.tensor((raw-mean)/std*mask,dtype=torch.float64)
        for seed in config['model_seeds']:
            start=time.perf_counter_ns();torch.manual_seed(seed)
            model=torch.nn.Sequential(torch.nn.Linear(F,H,dtype=torch.float64),torch.nn.ReLU(),torch.nn.Linear(H,1,dtype=torch.float64))
            opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            generator=torch.Generator().manual_seed(seed+20000);trace=[]
            for step in range(config['training_steps']):
                idx=torch.multinomial(prob,config['batch_size'],replacement=True,generator=generator)
                li=torch.as_tensor(left[idx.numpy()]);ri=torch.as_tensor(right[idx.numpy()])
                margin=model(x[li])-model(x[ri])
                loss=torch.nn.functional.softplus(-margin).mean()
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if step%100==0 or step==config['training_steps']-1:trace.append({'step':step+1,'pairwise_loss':float(loss.detach())})
            with torch.no_grad():
                a=model[0].weight.numpy()*mask[None,:]/std[None,:];b=model[0].bias.numpy()-a@mean
                w=np.concatenate([a.ravel(),b,model[2].weight.numpy().ravel(),model[2].bias.numpy()])
                scores=native_scores(raw,w);error=float(np.max(np.abs(scores-model(x).numpy().ravel())))
                if error>1e-10:raise AssertionError('native rank export differs')
            record={'schema':'spectra.residual_patch.model.v1','mode':mode,'seed':seed,'parameters':W,'features':F,'hidden':H,
               'weights':w.tolist(),'target':'within-snapshot ordering of empirical 512-move success; not calibrated probability',
               'teacher_repeats':config['teacher_repeats'],'training_labels_sha256':hashlib.sha256((data_dir/'labels.npz').read_bytes()).hexdigest(),
               'training_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'training_steps':config['training_steps'],'training_ns':time.perf_counter_ns()-start,'native_max_abs_error':error,
               'informative_training_states':informative,'ordered_pairs':len(left),
               'train_order_accuracy':float(np.average(scores[left]>scores[right],weights=weights)),
               'loss_trace':trace,'coarse_boundary':'global residual and static candidate geometry only; same residual-built pool'}
            write_json(out/f'{mode}_{seed}.json',record)
            reports.append({k:v for k,v in record.items() if k not in ('weights','loss_trace')})
            print('rank trained',mode,seed,record['train_order_accuracy'],flush=True)
    write_json(out/'summary.json',reports);return reports
