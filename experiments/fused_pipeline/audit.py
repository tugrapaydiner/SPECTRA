"""Independent standard-library inventory and timing arithmetic audit.

Checks known original artifacts and the entire request schedule; does not import
benchmark/runtime implementations or claim outside-researcher replication.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

TASKS=('chess','penguins','titanic','wdbc','wine','zoo')
ARMS=('python','compiled','fused','numpy')
SEED=20260927

def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate key')
        out[k]=v
    return out

def read(path):return json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inside(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()) or path.is_symlink():raise ValueError('unsafe inventory path')
    return path

def audit(run,inputs,source):
    protocol=read(run/'protocol.json');fidelity=read(run/'fidelity.json');summary=read(run/'summary.json')
    models=[f'{t}-{s}' for t in TASKS for s in (101,202,303)]
    if protocol['models']!=models or tuple(protocol['arms'])!=ARMS or protocol['repeats']!=5 or protocol['seed']!=SEED:
        raise ValueError('protocol mismatch')
    for name,digest in protocol['source_sha256'].items():
        if sha(inside(source,name))!=digest:raise ValueError('changed source: '+name)
    original=read(inputs/'SHA256.json')
    records=[]
    for line in (run/'rows.jsonl').read_text().splitlines():
        records.append(json.loads(line,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x))))
    rng=random.Random(SEED);expected=[];total_labels=0;jobs={}
    for ordinal,model in enumerate(models):
        folder=inputs/'models'/model
        for filename in ('cases.json','preprocessing.json','model.srt'):
            digest=sha(folder/filename)
            if digest!=original[f'models/{model}/{filename}']['sha256'] or digest!=protocol['model_files'][f'{model}/{filename}']:
                raise ValueError('input hash mismatch')
        cases=read(folder/'cases.json');n=len(cases['rows'])
        if len(cases['expected'])!=n or fidelity[model]['rows']!=n:raise ValueError('input count mismatch')
        ordering=list(range(n));random.Random(SEED+ordinal).shuffle(ordering)
        requests=[];cursor=0;pattern=(1,1,8,32,128)
        while cursor<n:
            count=min(pattern[len(requests)%5],n-cursor)
            requests.append(ordering[cursor:cursor+count]);cursor+=count
        if fidelity[model]['requests']!=requests:raise ValueError('trace permutation mismatch')
        for repeat in range(5):
            arms=list(ARMS);rng.shuffle(arms)
            for arm in arms:
                jobs[model,repeat,arm]=0
                for request,indices in enumerate(requests):
                    expected.append((model,repeat,arm,request,len(indices)));total_labels+=len(indices)
    if len(expected)!=len(records):raise ValueError('timing inventory mismatch')
    for r,e in zip(records,expected):
        if set(r)!={'model','repeat','arm','request','rows','ns','matched'} or tuple(r[k] for k in ('model','repeat','arm','request','rows'))!=e:
            raise ValueError('timing order/schema mismatch')
        if any(type(r[k]) is not int for k in ('repeat','request','rows','ns')) or r['ns']<=0 or r['matched'] is not True:
            raise ValueError('invalid measurement or fidelity')
        jobs[r['model'],r['repeat'],r['arm']]+=r['ns']
    costs={m:{a:statistics.median(jobs[m,j,a] for j in range(5)) for a in ARMS} for m in models}
    if costs!=summary['models']:raise ValueError('aggregate mismatch')
    ratios=[]
    for task in TASKS:
        r=statistics.geometric_mean(costs[m]['fused']/costs[m]['compiled'] for m in models if m.startswith(task+'-'))
        if abs(r-summary['tasks'][task]['fused_over']['compiled'])>1e-12:raise ValueError('task ratio mismatch')
        ratios.append(r)
    panel=statistics.geometric_mean(ratios);gate=panel<=1/1.1 and max(ratios)<=1.1
    if abs(panel-summary['panel_fused_over_compiled'])>1e-12 or gate!=summary['primary_gate']:
        raise ValueError('gate mismatch')
    return {'status':'PASS','timing_records':len(records),'checked_predictions':total_labels,'model_files':len(protocol['model_files']),
            'source_files':len(protocol['source_sha256']),'panel_ratio':panel,'primary_gate':gate}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('run','inputs','source','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();result=audit(a.run,a.inputs,a.source)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
