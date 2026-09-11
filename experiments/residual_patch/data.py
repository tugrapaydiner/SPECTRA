"""Deterministic unfiltered 3-SAT inputs and training-only counterfactual data."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from data.cnf import CNF, SplitMix64, random_3sat
from .runtime import State


def write_json(path, value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')

def formula(n, seed, family, config):
    m=n*config['ratio_numerator']//config['ratio_denominator']
    if family=='uniform':return random_3sat(n,m,seed)[0]
    if family!='community':raise ValueError('undeclared generator family')
    groups=config['community_groups']
    if n%groups or n//groups<3:raise ValueError('invalid community geometry')
    rng=SplitMix64(seed);seen=set();clauses=[]
    while len(clauses)<m:
        local=rng.below(config['community_internal_denominator'])<config['community_internal_numerator']
        size=n//groups if local else n
        offset=size*rng.below(groups) if local else 0
        vs=[]
        while len(vs)<3:
            v=1+offset+rng.below(size)
            if v not in vs:vs.append(v)
        c=tuple(sorted((v if rng.below(2) else -v for v in vs),key=abs))
        if c not in seen:clauses.append(c);seen.add(c)
    return CNF(n,tuple(clauses))

def cases(config, split):
    if split not in ('train','validation','development'):raise ValueError('undeclared split')
    sizes=config['train_sizes'] if split=='train' else config['eval_sizes']
    out=[]
    for family in config['families']:
        for n in sizes:
            for i in range(config[split+'_per_stratum']):
                seed=config[split+'_seed_start']+len(out)
                p=formula(n,seed,family,config)
                out.append({'id':f'{split}:{family}:{n}:{i}', 'family':family,'nvars':n,'seed':seed,
                            'formula':p.record(),'ordered_sha256':p.sha256(),'normalized_sha256':p.sha256(normalize_order=True)})
    return out

def assignment(n,seed):
    rng=SplitMix64(seed)
    return tuple(bool(rng.below(2)) for _ in range(n))

def collect(config, out):
    """Full labels come from a fixed probSAT-style 512-move continuation.

    No reference/planted assignment is used. Preserve snapshots skipped because
    the source trajectory already solved. All resulting data are TRAINING ONLY.
    """
    out=Path(out);out.mkdir(exist_ok=False,parents=True)
    train=cases(config,'train');write_json(out/'cases.json',train)
    X=[];y=[];case_ids=[];states=[];teacher_rows=[];counterfactuals=[]
    t0=time.perf_counter_ns()
    for ci,case in enumerate(train):
        p=CNF.from_record(case['formula'])
        for depth in config['snapshot_moves']:
            initial_seed=case['seed']^0x6e624eb7
            with State(p,assignment(p.nvars,initial_seed)) as s:
                source=s.run(initial_seed+1,moves=depth,cb=config['break_exponent'])
                record={'case_index':ci,'depth':depth,'source':source,'candidate_start':len(X),'candidate_count':0}
                if source['native_status']==1:
                    states.append(record);continue
                pairs,features=s.pool(initial_seed+depth+2)
                record['pairs']=pairs.tolist();record['candidate_count']=len(pairs)
                for pair,feature in zip(pairs,features):
                    successes=0;candidate_id=len(X)
                    for rep in range(config['teacher_repeats']):
                        continuation_seed=initial_seed+depth+100+rep
                        with State(p,source['witness']) as t:
                            t.patch(tuple(int(v) for v in pair if v>=0))
                            result=t.run(continuation_seed,moves=config['teacher_moves'],cb=config['break_exponent'])
                            actual=p.satisfied(result['witness'])
                            if actual!=(result['native_status']==1):raise AssertionError('false teacher witness')
                            successes+=actual
                            counterfactuals.append({'candidate':candidate_id,'repeat':rep,'seed':continuation_seed,**result})
                    X.append(feature);y.append(successes/config['teacher_repeats']);case_ids.append(ci)
                states.append(record)
        if ci%24==0:print('collection',ci,len(train),'candidates',len(X),flush=True)
    np.savez_compressed(out/'labels.npz',features=np.asarray(X,dtype=np.float64),
                       targets=np.asarray(y,dtype=np.float64),case_index=np.asarray(case_ids,dtype=np.int32))
    # Lossless full continuations include failed witnesses, work and timings.
    import gzip
    for name,rows in [('snapshots',states),('counterfactuals',counterfactuals)]:
        with gzip.open(out/(name+'.jsonl.gz'),'wt') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    summary={'cases':len(train),'snapshots':len(states),'unsolved_snapshots':sum(s['candidate_count']>0 for s in states),
             'candidate_labels':len(X),'counterfactuals':len(counterfactuals),'positive_rate':float(np.mean(y)),
             'collection_ns':time.perf_counter_ns()-t0,'scope':'training counterfactuals only; no validation or development read'}
    write_json(out/'summary.json',summary);return summary
