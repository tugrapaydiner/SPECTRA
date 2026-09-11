"""Train small conditional patch scorers; export complete scalar native weights.

This is not a new recursive neural architecture. Recurrence is in the observed
search state; the learned component is a 289-parameter feed-forward scorer.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from .runtime import F,H,W,native_scores
from .data import write_json


def train(config,data_dir,out):
    out=Path(out);out.mkdir(exist_ok=False,parents=True)
    labels_path=Path(data_dir)/'labels.npz'
    with np.load(labels_path,allow_pickle=False) as d:
        raw=d['features'];targets=d['targets'];ids=d['case_index']
    if raw.ndim!=2 or raw.shape[1]!=F or not len(raw) or not np.isfinite(raw).all() or not np.isin(targets,[0,0.5,1]).all():
        raise ValueError('invalid native counterfactual labels')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    target=torch.tensor(targets[:,None],dtype=torch.float64)
    counts=np.bincount(ids)
    probs=torch.tensor(1/counts[ids],dtype=torch.float64)
    probs/=probs.sum()
    reports=[]
    for mode in ['coarse','residual']:
        mask=np.ones(F) if mode=='residual' else np.isin(np.arange(F),config['coarse_indices']).astype(float)
        mean=raw.mean(0);std=raw.std(0);std=np.where(std<1e-8,1,std)
        x=torch.tensor((raw-mean)/std*mask,dtype=torch.float64)
        for seed in config['model_seeds']:
            start=time.perf_counter_ns();torch.manual_seed(seed)
            model=torch.nn.Sequential(torch.nn.Linear(F,H,dtype=torch.float64),torch.nn.ReLU(),torch.nn.Linear(H,1,dtype=torch.float64))
            optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            generator=torch.Generator().manual_seed(seed+10000)
            trace=[]
            for step in range(config['training_steps']):
                idx=torch.multinomial(probs,config['batch_size'],replacement=True,generator=generator)
                logits=model(x[idx]);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target[idx])
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                if step%100==0 or step==config['training_steps']-1:trace.append({'step':step+1,'bce':float(loss.detach())})
            with torch.no_grad():
                a=model[0].weight.numpy()*mask[None,:]/std[None,:]
                b=model[0].bias.numpy()-a@mean
                w=np.concatenate([a.ravel(),b,model[2].weight.numpy().ravel(),model[2].bias.numpy()])
                torch_scores=model(x).numpy().ravel()
                scores=native_scores(raw,w)
                error=float(np.max(np.abs(scores-torch_scores)))
                if error>1e-10:raise AssertionError('native export differs from trained model')
                pred=torch.sigmoid(torch.tensor(scores)).numpy()
            record={'schema':'spectra.residual_patch.model.v1','mode':mode,'seed':seed,'parameters':W,
               'features':F,'hidden':H,'weights':w.tolist(),'target':f"success within {config['teacher_moves']} probSAT-style moves after the proposed patch",
               'teacher_repeats':config['teacher_repeats'],'training_labels_sha256':hashlib.sha256(labels_path.read_bytes()).hexdigest(),
               'training_steps':config['training_steps'],'training_ns':time.perf_counter_ns()-start,'native_max_abs_error':error,
               'train_brier':float(np.mean((pred-targets)**2)),'loss_trace':trace,
               'coarse_boundary':'coarse sees global unsatisfied count and input/patch-size/degree features; BOTH pools use exact residuals'}
            write_json(out/f'{mode}_{seed}.json',record)
            reports.append({k:v for k,v in record.items() if k not in ('weights','loss_trace')})
            print('trained',mode,seed,record['train_brier'],record['training_ns']/1e9,flush=True)
    write_json(out/'summary.json',reports);return reports
