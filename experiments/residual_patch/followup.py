"""Controlled ranking/frequency follow-up; retain the original pilot unchanged."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from . import rank
from .data import cases,write_json
from .experiment import sha,sources,model_bank,solve_one,seal,environment
from .runtime import parameters

VARIANTS={'probsat':('probsat',128,'bce'), 'walksat':('walksat',128,'bce'),
    'random128':('random_patch',128,'bce'),'greedy128':('greedy_patch',128,'bce'),
    'bce32':('learned_patch',32,'bce'),'bce128':('learned_patch',128,'bce'),
    'rank32':('learned_patch',32,'rank'),'rank128':('learned_patch',128,'rank'),
    'coarse_rank128':('coarse_patch',128,'rank')}
CONTROL=tuple(VARIANTS)[:4]
HERE=Path(__file__).resolve().parent

def own_sources():
    return {**sources(),**{n:sha((HERE/n).read_bytes()) for n in ('rank.py','followup.py','FOLLOWUP.md')}}

def check_binding(original,out):
    from .experiment import assert_source
    assert_source(original)
    lock=json.loads((out/'source.json').read_text())
    if lock!=own_sources():raise ValueError('follow-up source changed')
    cfg=json.loads((original/'config.json').read_text())
    if json.loads((out/'config.json').read_text())!=cfg:raise ValueError('configuration changed')
    bound=json.loads((out/'training_origin.json').read_text())
    if bound['labels_sha256']!=sha((original/'training/labels.npz').read_bytes()):raise ValueError('training data changed')
    return cfg

def freeze(original,out):
    out.mkdir(exist_ok=False,parents=True)
    cfg=json.loads((original/'config.json').read_text())
    write_json(out/'config.json',cfg);write_json(out/'source.json',own_sources());write_json(out/'environment.json',environment())
    write_json(out/'training_origin.json',{'labels_sha256':sha((original/'training/labels.npz').read_bytes()),
        'snapshots_sha256':sha((original/'training/snapshots.jsonl.gz').read_bytes()),'scope':'same original training data; no new teacher data'})
    with zipfile.ZipFile(out/'followup_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('rank.py','followup.py','FOLLOWUP.md'):z.write(HERE/name,name)
    return cfg

def banks(original,out,cfg):
    result={'bce':model_bank(original,cfg),'rank':{}}
    for mode in ('coarse','residual'):
        for seed in cfg['model_seeds']:
            p=json.loads((out/f'models/{mode}_{seed}.json').read_text())
            if p['seed']!=seed or p['mode']!=mode or p['training_source_sha256']!=sha((HERE/'rank.py').read_bytes()):raise ValueError('rank model mismatch')
            result['rank'][(mode,seed)]=parameters(p['weights'])
    return result

def run_variant(c,variant,ms,ss,budget,cfg,bank,work=False):
    arm,interval,model=VARIANTS[variant]
    local=dict(cfg,patch_interval=interval)
    r=solve_one(c,arm,ms,ss,budget,local,bank[model],work_only=work)
    r.update(variant=variant,patch_interval=interval,model_family=model)
    return r

def models_hash(original,out):
    return {**{'bce/'+p.name:sha(p.read_bytes()) for p in sorted((original/'models').glob('*.json'))},
            **{'rank/'+p.name:sha(p.read_bytes()) for p in sorted((out/'models').glob('*.json'))}}

def evaluate(original,out,split):
    cfg=check_binding(original,out);bank=banks(original,out,cfg);dest=out/split;dest.mkdir(exist_ok=False)
    inputs=cases(cfg,split);write_json(dest/'cases.json',inputs)
    if split=='development':
        selection=json.loads((out/'validation/selection.json').read_text())
        write_json(dest/'pre_execution_lock.json',{'sources':own_sources(),'models':models_hash(original,out),
             'selection':selection,'primary_candidate':'rank128','scope':'adaptive development, not confirmation'})
    events=[(r,i,ms,ss,b,v) for r in range(cfg['timing_rounds']) for i in range(len(inputs))
         for ms in cfg['model_seeds'] for ss in cfg['search_seeds'] for b in cfg['wall_budget_ms'] for v in VARIANTS]
    np.random.default_rng(64017 if split=='validation' else 74018).shuffle(events)
    write_json(dest/'schedule.json',events);rows=[];start=time.perf_counter_ns()
    with gzip.open(dest/'rows.jsonl.gz','wt') as f:
        for i,(r,ci,ms,ss,b,v) in enumerate(events):
            row=run_variant(inputs[ci],v,ms,ss,b,cfg,bank);row['round']=r
            f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');rows.append(row)
            if i%2000==0:f.flush();print('followup',split,i,len(events),flush=True)
    with gzip.open(dest/'work_rows.jsonl.gz','wt') as f:
        for c in inputs:
            for ms in cfg['model_seeds']:
                for ss in cfg['search_seeds']:
                    for v in VARIANTS:
                        row=run_variant(c,v,ms,ss,0,cfg,bank,True);row['round']=0
                        f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    report=aggregate(cfg,inputs,rows);report['execution_ns']=time.perf_counter_ns()-start
    write_json(dest/'summary.json',report)
    if split=='validation':
        selection={}
        for family in cfg['families']:
            for n in cfg['eval_sizes']:
                eligible=[s for s in report['strata'] if s['family']==family and s['nvars']==n and s['budget_ms']==cfg['primary_wall_budget_ms'] and s['variant'] in CONTROL]
                best=max(eligible,key=lambda s:(s['success_rate'],-s['mean_complete_ns'],-CONTROL.index(s['variant'])))
                selection[f'{family}:{n}']=best['variant']
        write_json(dest/'selection.json',selection)
    else:write_json(dest/'paired_analysis.json',paired(cfg,inputs,rows,selection))
    check_binding(original,out);seal(out);return report

def aggregate(cfg,inputs,rows):
    by_id={c['id']:c for c in inputs};out=[]
    for family in cfg['families']:
        for n in cfg['eval_sizes']:
            for b in cfg['wall_budget_ms']:
                for v in VARIANTS:
                    rs=[r for r in rows if by_id[r['case_id']]['family']==family and by_id[r['case_id']]['nvars']==n and r['budget_ms']==b and r['variant']==v]
                    out.append({'family':family,'nvars':n,'budget_ms':b,'variant':v,'rows':len(rs),
                        'successes':sum(r['within_budget'] for r in rs),'success_rate':float(np.mean([r['within_budget'] for r in rs])),
                        'mean_complete_ns':float(np.mean([r['complete_ns'] for r in rs])),
                        'p95_complete_ns':float(np.quantile([r['complete_ns'] for r in rs],.95)),
                        'late_valid':sum(r['valid'] and not r['within_budget'] for r in rs),
                        'mean_calls':float(np.mean([r['work']['model_calls'] for r in rs]))})
    return {'cases':len(inputs),'observations':len(rows),'strata':out,'scope':'full-cost adaptive validation/development; no independent confirmation'}

def paired(cfg,inputs,rows,selection):
    ms=cfg['model_seeds'];b=cfg['primary_wall_budget_ms']
    collected={(c['id'],m,v):[] for c in inputs for m in ms for v in VARIANTS}
    for r in rows:
        if r['budget_ms']==b:collected[(r['case_id'],r['model_seed'],r['variant'])].append(r['within_budget'])
    arrays={v:np.array([[np.mean(collected[(c['id'],m,v)]) for m in ms] for c in inputs]) for v in VARIANTS}
    base=np.array([arrays[selection[f"{c['family']}:{c['nvars']}"]][i] for i,c in enumerate(inputs)])
    groups=[[i for i,c in enumerate(inputs) if c['family']==family and c['nvars']==n] for family in cfg['families'] for n in cfg['eval_sizes']]
    rng=np.random.default_rng(270319);results={}
    for v in VARIANTS:
        delta=arrays[v]-base;boot=[]
        for _ in range(cfg['bootstrap_repeats']):
            ci=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);mi=rng.integers(0,len(ms),len(ms))
            boot.append(float(delta[np.ix_(ci,mi)].mean()))
        lo,hi=np.quantile(boot,[.025,.975])
        results[v]={'success_rate':float(arrays[v].mean()),'control_rate':float(base.mean()),'gain':float(delta.mean()),
            'crossed_95_interval':[float(lo),float(hi)],'per_model_seed_gain':list(map(float,delta.mean(0)))}
    main=results['rank128']
    return {'primary':'rank128','primary_budget_ms':b,'selected_controls':selection,'results':results,
        'rank128_minus_bce128':float((arrays['rank128']-arrays['bce128']).mean()),
        'rank128_minus_rank32':float((arrays['rank128']-arrays['rank32']).mean()),
        'rank128_minus_coarse_rank128':float((arrays['rank128']-arrays['coarse_rank128']).mean()),
        'quality_gate':bool(main['gain']>=cfg['minimum_quality_gain'] and main['crossed_95_interval'][0]>0),
        'qualification':'formula-stratified crossed bootstrap with three model seeds; repeated timing/search trials collapsed; adaptive pilot, no multiplicity adjustment'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train','validation','development']);p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='train':cfg=freeze(a.original,a.out);report=rank.train(cfg,a.original/'training',a.out/'models');seal(a.out)
    else:report=evaluate(a.original,a.out,a.command)
    print(json.dumps(report,sort_keys=True,indent=2,allow_nan=False))

if __name__=='__main__':main()
