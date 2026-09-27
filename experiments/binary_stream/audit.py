"""Standard-library audit of both frozen binary-stream timing inventories.

Requires the unchanged previous SDK manifest as an identity anchor. Checks the
entire grid, output digests and aggregates without importing runtime/benchmark
code. A file receipt is not authentication or independent hardware replication.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import statistics

TASKS=('chess','penguins','titanic','wdbc','wine','zoo')
MODELS=tuple(f'{t}-{s}' for t in TASKS for s in (101,202,303))
BINARY=('chess','titanic','wdbc')
EXPECTED_ROWS={'chess':799,'penguins':86,'titanic':328,'wdbc':143,'wine':45,'zoo':26}
EXPECTED_FEATURES={'chess':73,'penguins':10,'titanic':13,'wdbc':30,'wine':13,'zoo':31}
SDK_MANIFEST_SHA='017f63aa30ee9f2db773ab86234b24b5eada3a9f2fcc97049d0802dbede90f4e'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def unique(pairs):
    out={}
    for key,value in pairs:
        if key in out: raise ValueError('duplicate JSON key')
        out[key]=value
    return out

def reject(value): raise ValueError('nonfinite JSON token')
def decode(text): return json.loads(text,object_pairs_hook=unique,parse_constant=reject)
def read(path): return decode(Path(path).read_text(encoding='utf-8'))
def digest(labels):
    return hashlib.sha256(json.dumps(labels,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

def inside(root,name):
    if type(name) is not str or '\\' in name: raise ValueError('invalid inventory name')
    rel=PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts: raise ValueError('escaping inventory')
    path=Path(root)
    for part in rel.parts:
        path=path/part
        if path.is_symlink(): raise ValueError('symlinked inventory')
    if not path.is_file(): raise ValueError('missing inventory member')
    return path

def same(actual,expected):
    if type(actual) is not type(expected): raise ValueError('aggregate type differs')
    if isinstance(expected,dict):
        if set(actual)!=set(expected): raise ValueError('aggregate inventory differs')
        for k,v in expected.items(): same(actual[k],v)
    elif isinstance(expected,list):
        if len(actual)!=len(expected): raise ValueError('aggregate size differs')
        for a,e in zip(actual,expected): same(a,e)
    elif isinstance(expected,float):
        if not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-13,abs_tol=1e-13):
            raise ValueError('aggregate differs')
    elif actual!=expected: raise ValueError('aggregate differs')

def audit(run,inputs,source,manifest,libraries,kind):
    run,inputs,source=map(Path,(run,inputs,source))
    if sha(manifest)!=SDK_MANIFEST_SHA: raise ValueError('prior identity anchor differs')
    anchor=read(manifest); p=read(run/'protocol.json')
    is_raw=kind=='raw'
    arms=('compiled_default','compiled_stream','fused_default','fused_stream') if is_raw else ('exhaustive','beretta_cert','binary_stream')
    seed=20260927 if is_raw else 2026092707
    repeats=5 if is_raw else 31
    schema='spectra.binary_stream.raw_trace.v1' if is_raw else 'spectra.binary_stream.benchmark.v1'
    if p['schema']!=schema or p['models']!=list(MODELS) or p['arms']!=list(arms) or type(p['seed']) is not int or p['seed']!=seed or type(p['repeats']) is not int or p['repeats']!=repeats or p['tables'] is not False:
        raise ValueError('fixed protocol differs')
    if is_raw:
        if p['pattern']!=[1,1,8,32,128]: raise ValueError('trace pattern differs')
    elif p['chunks']!=[1,4,32,128] or p['input_dtype']!='float64': raise ValueError('batch shape differs')
    sources={x.relative_to(source).as_posix() for x in (source/'spectra').glob('svm*.py')}
    sources|={x.relative_to(source).as_posix() for pattern in ('*.cpp','*.hpp') for x in (source/'spectra/_native/ovo').rglob(pattern)}
    sources|={'experiments/binary_stream/PROTOCOL.md','experiments/binary_stream/bench.py'}
    if is_raw: sources|={'experiments/binary_stream/bench_pipeline.py','experiments/fused_pipeline/bench.py'}
    if set(p['source_sha256'])!=sources: raise ValueError('source inventory differs')
    for name,value in p['source_sha256'].items():
        if sha(inside(source,name))!=value: raise ValueError('source bytes differ')
    actual_libs={Path(lib).name:sha(lib) for lib in libraries}
    expected_libs=p['libraries'] if is_raw else {p['library_name']:p['library_sha256']}
    if actual_libs!=expected_libs: raise ValueError('library identity differs')
    filenames=('model.srt','cases.json','preprocessing.json') if is_raw else ('model.srt','cases.json','transformed.f64','preprocessing.json')
    inventory=p['model_files'] if is_raw else p['inputs_sha256']
    if set(inventory)!={f'{m}/{f}' for m in MODELS for f in filenames}: raise ValueError('input inventory differs')
    expected={}; rng=random.Random(seed); schedule=[]
    for ordinal,model in enumerate(MODELS):
        for filename in filenames:
            name=f'{model}/{filename}'; path=inside(inputs,name); value=sha(path)
            if value!=inventory[name] or value!=anchor[f'models/{name}']['sha256'] or path.stat().st_size!=anchor[f'models/{name}']['bytes']:
                raise ValueError('input identity differs')
        case=read(inputs/model/'cases.json'); n=EXPECTED_ROWS[model.split('-')[0]]
        if len(case['rows'])!=n or len(case['expected'])!=n: raise ValueError('case inventory differs')
        expected[model]=case['expected']
        if is_raw:
            order=list(range(n));random.Random(seed+ordinal).shuffle(order)
            requests=[];start=0
            while start<n:
                count=min((1,1,8,32,128)[len(requests)%5],n-start)
                requests.append(order[start:start+count]);start+=count
            for repeat in range(repeats):
                run_arms=list(arms);rng.shuffle(run_arms)
                for arm in run_arms:
                    for request,indices in enumerate(requests):
                        schedule.append({'model':model,'repeat':repeat,'arm':arm,'request':request,'rows':len(indices),'output_sha256':digest([expected[model][i] for i in indices])})
        else:
            if (inputs/model/'transformed.f64').stat().st_size!=n*EXPECTED_FEATURES[model.split('-')[0]]*8: raise ValueError('feature size differs')
            for repeat in range(repeats):
                jobs=[(chunk,arm) for chunk in (1,4,32,128) for arm in arms];rng.shuffle(jobs)
                for chunk,arm in jobs:
                    schedule.append({'model':model,'repeat':repeat,'chunk':chunk,'arm':arm,'rows':n,'output_sha256':digest(expected[model])})
    records=[decode(line) for line in (run/'rows.jsonl').read_text().splitlines()]
    if len(records)!=len(schedule): raise ValueError('timing inventory differs')
    for r,e in zip(records,schedule):
        if set(r)!=set(e)|{'ns'}: raise ValueError('timing schema differs')
        for k,v in e.items():
            if type(r[k]) is not type(v) or r[k]!=v: raise ValueError('timing order/outcome differs')
        if type(r['ns']) is not int or r['ns']<=0: raise ValueError('invalid nanoseconds')
    tasks={}
    if is_raw:
        jobs={}
        for r in records:
            k=r['model'],r['repeat'],r['arm'];jobs[k]=jobs.get(k,0)+r['ns']
        costs={m:{a:statistics.median(jobs[m,j,a] for j in range(repeats)) for a in arms} for m in MODELS}
        for task in TASKS:
            models=[m for m in MODELS if m.startswith(task+'-')]
            tasks[task]={b+'_stream_over_default':statistics.geometric_mean(costs[m][b+'_stream']/costs[m][b+'_default'] for m in models) for b in ('compiled','fused')}
            tasks[task]['fused_regressions']=sum(costs[m]['fused_stream']>costs[m]['fused_default'] for m in models)
        result={'cells':len(records),'predictions':sum(r['rows'] for r in records),'costs':costs,'tasks':tasks,
                'pooled_fused_ratio':sum(v['fused_stream'] for v in costs.values())/sum(v['fused_default'] for v in costs.values()),
                'equal_task_fused_ratio':statistics.geometric_mean(v['fused_stream_over_default'] for v in tasks.values())}
    else:
        grouped={}
        for r in records: grouped.setdefault(f"{r['model']}|{r['chunk']}|{r['arm']}",[]).append(r['ns'])
        medians={k:statistics.median(v) for k,v in grouped.items()}
        for task in TASKS:
            models=[m for m in MODELS if m.startswith(task+'-')];tasks[task]={}
            for chunk in (1,4,32,128):
                ratios={a:statistics.geometric_mean(medians[f'{m}|{chunk}|binary_stream']/medians[f'{m}|{chunk}|{a}'] for m in models) for a in arms[:-1]}
                times={a:statistics.median(medians[f'{m}|{chunk}|{a}']/EXPECTED_ROWS[task]/1000 for m in models) for a in arms}
                tasks[task][str(chunk)]={'us_per_row':times,'stream_over':ratios,'regressions_vs_default':sum(medians[f'{m}|{chunk}|binary_stream']>medians[f'{m}|{chunk}|beretta_cert'] for m in models)}
        primary=statistics.geometric_mean(tasks[t]['32']['stream_over']['beretta_cert'] for t in BINARY)
        result={'schema':'spectra.binary_stream.summary.v1','cells':len(records),'predictions':sum(r['rows'] for r in records),'model_medians_ns':medians,'tasks':tasks,
                'binary_batch32_ratio':primary,'pooled_all_models_batch32_ratio':sum(medians[f'{m}|32|binary_stream'] for m in MODELS)/sum(medians[f'{m}|32|beretta_cert'] for m in MODELS),
                'primary_gate':primary<=1/1.2 and all(tasks[t]['32']['stream_over']['beretta_cert']<=1.10 for t in BINARY)}
    same(read(run/'summary.json'),result)
    return {'status':'PASS','kind':kind,'timing_cells':len(records),'checked_predictions':result['predictions'],'source_members':len(sources),'input_members':len(inventory),'primary_gate':result.get('primary_gate'),'scope':'recorded inventory and arithmetic, not external replication'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','inputs','source','manifest','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--library',type=Path,action='append',required=True)
    parser.add_argument('--kind',choices=['batch','raw'],required=True)
    args=parser.parse_args()
    result=audit(args.run,args.inputs,args.source,args.manifest,args.library,args.kind)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
