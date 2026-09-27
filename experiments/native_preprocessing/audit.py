"""Independent standard-library audit; never imports the benchmark or runtime.

Checks bound inputs/sources, every feature byte, timing grid/order, medians and
point gates. Hashes identify files; they do not authenticate an entirely replaced
packet. Actual native execution is a separate recorded fidelity obligation.
"""
from __future__ import annotations
from array import array
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

TASKS=('wine','wdbc','chess','penguins','titanic','zoo')

def unique(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate JSON key')
        d[k]=v
    return d

def reject(v):raise ValueError('nonfinite JSON')
def load(p):return json.loads(Path(p).read_text(),object_pairs_hook=unique,parse_constant=reject)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe(root,name):
    p=(root/name).resolve()
    if Path(name).is_absolute() or not p.is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError('invalid inventory path')
    return p

def same(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if type(a) is float:return type(b) in (int,float) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return type(a) is type(b) and a==b

def check(run,inputs,source):
    run,inputs,source=map(Path,(run,inputs,source))
    p=load(run/'protocol.json');fr=load(inputs/'freeze.json')
    if p['freeze_sha256']!=sha(inputs/'freeze.json'):raise ValueError('freeze mismatch')
    expected_names={f'{t}-{s}' for t in TASKS for s in (101,202,303)}
    if set(p['models'])!=expected_names or len(p['models'])!=18:raise ValueError('model grid mismatch')
    if [m['name'] for m in fr['models']]!=p['models']:raise ValueError('model order mismatch')
    if p['repeats']!=31 or type(p['repeats']) is not int:raise ValueError('repetition mismatch')
    arms=('python_native','compiled_native','numpy_native','sklearn')
    if p['schema'].endswith('.v2'):
        arms=('python_native','compiled_materialized','compiled_native','numpy_native','sklearn')
        seed=2026092702
    elif p['schema'].endswith('.v1'):seed=20260927
    else:raise ValueError('unknown benchmark schema')
    if p['arms']!=list(arms) or p['seed']!=seed:raise ValueError('schedule mismatch')
    for name,digest in p['source_sha256'].items():
        if sha(safe(source,name))!=digest:raise ValueError('source mismatch')
    observed=load(run/'fidelity.json');counts={};pairs=features=0
    for model in fr['models']:
        folder=safe(inputs,'models/'+model['name']+'/cases.json').parent
        for name,digest in model['files'].items():
            if sha(safe(folder,name))!=digest:raise ValueError('input binding mismatch')
        doc=load(folder/'preprocessing.json');cases=load(folder/'cases.json')
        if doc['model_sha256']!=sha(folder/'model.srt'):raise ValueError('plan/model mismatch')
        output=array('d')
        for row in cases['rows']:
            vals=[]
            for op in doc['operations']:
                v=row[op['column']]
                if op['kind']=='numeric':
                    x=float.fromhex(op['fill']) if v is None else float(v)
                    if math.isnan(x):x=float.fromhex(op['fill'])
                    vals.append((x-float.fromhex(op['mean']))/float.fromhex(op['scale']))
                else:vals.extend(float(v==k) for k in op['categories'])
            if len(vals)!=doc['features'] or not all(map(math.isfinite,vals)):raise ValueError('feature inventory')
            output.extend(vals)
        if output.tobytes()!=(folder/'transformed.f64').read_bytes():raise ValueError('independent features differ')
        n=len(cases['rows']);pairs+=n;features+=len(output);counts[model['name']]=n
        entry={'count':n,'transformed_sha256':hashlib.sha256(output.tobytes()).hexdigest(),
               'predictions_sha256':hashlib.sha256(json.dumps(cases['expected'],separators=(',',':')).encode()).hexdigest()}
        if observed['models'][model['name']]!=entry:raise ValueError('fidelity receipt mismatch')
    if observed['pairs']!=pairs or observed['features']!=features:raise ValueError('fidelity totals')
    rows=[json.loads(x,object_pairs_hook=unique,parse_constant=reject) for x in (run/'rows.jsonl').read_text().splitlines()]
    schedule=[];rng=random.Random(seed)
    for m in p['models']:
        for size,n in [('one',1),('batch',min(128,counts[m]))]:
            for rep in range(31):
                order=list(arms);rng.shuffle(order)
                schedule.extend((m,size,n,rep,a) for a in order)
    if len(rows)!=len(schedule):raise ValueError('timing inventory mismatch')
    groups={}
    for r,key in zip(rows,schedule):
        if set(r)!={'model','size','rows','arm','repeat','ns','matched'}:raise ValueError('unexpected timing fields')
        if (r['model'],r['size'],r['rows'],r['repeat'],r['arm'])!=key:raise ValueError('timing order mismatch')
        if type(r['repeat']) is not int or type(r['rows']) is not int or type(r['ns']) is not int or r['ns']<=0 or r['matched'] is not True:
            raise ValueError('invalid timing or comparison field')
        groups.setdefault((r['model'],r['size'],r['arm']),[]).append(r['ns']/r['rows']/1000)
    med={'|'.join(k):statistics.median(v) for k,v in groups.items()};tasks={}
    for task in TASKS:
        models=sorted(n for n in expected_names if n.startswith(task+'-'));tasks[task]={}
        for size in ('one','batch'):
            lat={a:statistics.median([med[f'{m}|{size}|{a}'] for m in models]) for a in arms}
            ratios={a:statistics.geometric_mean([med[f'{m}|{size}|compiled_native']/med[f'{m}|{size}|{a}'] for m in models]) for a in arms if a!='compiled_native'}
            tasks[task][size]={'us_per_row':lat,'compiled_over':ratios}
    derived={'cells':len(rows),'model_medians_us':med,'tasks':tasks,
             'primary_gate':all(tasks[t]['batch']['compiled_over']['python_native']<=1/1.2 for t in TASKS)}
    if not same(derived,load(run/'summary.json')):raise ValueError('derived summary mismatch')
    return {'status':'PASS','timing_cells':len(rows),'feature_values':features,'model_input_pairs':pairs,
            'point_gate':derived['primary_gate'],'scope':'independent arithmetic/inventory/aggregation audit, not outside replication'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','inputs','source'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();print(json.dumps(check(args.run,args.inputs,args.source),indent=2))
