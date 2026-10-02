"""Independent standard-library audit of recorded total-tree evidence.

Does not import the candidate or execute a native library. Reconstructs the fixed
schedule, results, cost arithmetic, preserved prefixes and source-byte bindings.
Independent.py separately reconstructs the mathematical proof with Fraction.
Hashes do not authenticate clocks, outside replication or hostile wholesale edits.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import xml.etree.ElementTree as ET

TASKS=('letter','pendigits','satellite','optdigits')
ARMS=('official128','official1210','exported_cpp','full16','residual_adaptive','total_flat','total_interned','exact_flat','exact_interned')
RESOURCE_POLICIES=('flat','interned','residual','official')


def require(ok, message):
    if not ok:raise ValueError(message)


def decode(text):
    def unique(pairs):
        out={}
        for key,value in pairs:
            require(key not in out,'duplicate JSON key');out[key]=value
        return out
    def invalid(_):raise ValueError('nonfinite JSON token')
    return json.loads(text,object_pairs_hook=unique,parse_constant=invalid)


def read(path):return decode(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def member(root,name):
    p=Path(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name,'unsafe mapped path')
    result=Path(root)
    for part in p.parts:
        result/=part;require(not result.is_symlink(),'symlink inventory entry')
    require(result.is_file(),'missing evidence file '+name)
    return result


def compare(a,b):
    if isinstance(b,dict):
        for key,value in b.items():
            require(key in a,'missing aggregate');compare(a[key],value)
    elif type(b) is float:
        require(type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),'wrong numeric aggregate')
    else:require(type(a) is type(b) and a==b,'wrong aggregate')


def resource_schedule(resource, observed):
    """Require the whole recorded matrix before any zip can truncate validation."""
    require(type(resource) is dict and resource.get('isolated') is True,
            'resource run not isolated')
    jobs=resource.get('jobs')
    require(type(jobs) is list and len(jobs)==48 and len(observed)==48,
            'resource schedule or observations incomplete')
    for job in jobs:
        require(type(job) is list and len(job)==3,'invalid resource job')
        task,policy,repeat=job
        require(type(task) is str and task in TASKS and
                type(policy) is str and policy in RESOURCE_POLICIES and
                type(repeat) is int and repeat>=0,'invalid resource job identity')
    repeats={job[2] for job in jobs}
    require(len(repeats)==3,'resource matrix needs three distinct repeats')
    expected={(task,policy,repeat) for task in TASKS for policy in RESOURCE_POLICIES for repeat in repeats}
    require({tuple(job) for job in jobs}==expected,'resource matrix missing or duplicated jobs')
    return jobs


def resource_output(literal):
    require(type(literal) is dict and type(literal.get('returncode')) is int and
            literal['returncode']==0,'resource process failed')
    require(type(literal.get('stdout')) is str,'missing resource process output')
    output=decode(literal['stdout'])
    fields={'task','policy','setup_ns','memory_kib','runtime_info','model_bytes',
            'model_sha256','library_sha256','matched','numerical_frameworks','scope'}
    require(type(output) is dict and fields<=output.keys(),'incomplete resource process output')
    for name in ('setup_ns','model_bytes'):
        require(type(output[name]) is int and output[name]>0,'invalid resource measurement')
    memory=output['memory_kib']
    require(type(memory) is dict and all(type(memory.get(key)) is int and memory[key]>0
                                       for key in ('VmHWM','VmRSS')),'invalid resource memory')
    require(type(output['runtime_info']) is dict and type(output['scope']) is str and
            bool(output['scope']),'invalid resource context')
    for key in ('model_sha256','library_sha256'):
        value=output[key]
        require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),
                'invalid resource identity')
    return output


def audit(root):
    root=Path(root);results=root/'results';run=results/'benchmark';lock=read(run/'LOCK.json')
    paths=read(root/'PATHS.json');sources=read(root/'SOURCES.json')
    compare(lock,{'tasks':list(TASKS),'arms':list(ARMS),'chunks':[1,32,256],'repeats':7,'cycles':10,'seed':2026092967,
                  'gate':{'batch32_geometric_ratio_limit':1.10,'batch32_max_task_ratio_limit':1.25,'baseline':'full16'}})
    for path,digest in lock['files'].items():
        require(path in paths and sha(member(root,paths[path]))==digest,'artifact bytes changed')
    for path,digest in lock['source'].items():
        require(path in sources and sha(member(root,sources[path]))==digest,'timed source changed')
    jobs=[];rng=random.Random(2026092967)
    for task in TASKS:
        for repeat in range(7):
            sequence=[(n,arm) for n in (1,32,256) for arm in ARMS];rng.shuffle(sequence)
            jobs.extend([task,repeat,n,arm] for n,arm in sequence)
    require(lock['jobs']==jobs,'fixed schedule differs')
    records=[];ratios=[];summary=read(run/'SUMMARY.json');expected_by_task={}
    for task in TASKS:
        original=next(p for p in paths if p.endswith('/models/'+task+'/indices.i32') and 'baseline_sdk' in p)
        index_bytes=member(root,paths[original]).read_bytes()
        require(len(index_bytes)%4==0,'index bytes invalid')
        indices=struct.unpack('<'+'i'*(len(index_bytes)//4),index_bytes);expected_by_task[task]=indices
        observations=[decode(s) for s in (run/(task+'.jsonl')).read_text().splitlines()]
        require(len(observations)==189,'missing timing cells')
        expected=[j for j in jobs if j[0]==task]
        for r,j in zip(observations,expected):
            require([r[k] for k in ('task','repeat','chunk','arm')]==j,'timing reordered')
            require(all(type(r[k]) is int for k in ('repeat','chunk','rows','cycles','wall_ns','cpu_ns')),'invalid timing type')
            require(r['wall_ns']>0 and r['cpu_ns']>0 and r['rows']==len(indices) and r['cycles']==10,'invalid timed scope')
            require(r['indices_sha256']==hashlib.sha256(index_bytes).hexdigest(),'wrong timed result')
        records+=observations
        med={clock:{str(n):{arm:statistics.median(r[clock+'_ns'] for r in observations if r['chunk']==n and r['arm']==arm)/(len(indices)*10000) for arm in ARMS} for n in (1,32,256)} for clock in ('wall','cpu')}
        compare(summary['tasks'][task]['us_per_row'],med)
        ratio=med['wall']['32']['total_interned']/med['wall']['32']['full16'];ratios.append(ratio)
        paired=[]
        for repeat in range(7):
            cells={r['arm']:r['wall_ns'] for r in observations if r['repeat']==repeat and r['chunk']==32}
            paired.append(cells['total_interned']/cells['full16'])
        compare(summary['tasks'][task],{'batch32_total_over_prior_full16':ratio,'batch32_paired_ratios':paired,'batch32_paired_regressions':sum(v>1 for v in paired)})
    g=statistics.geometric_mean(ratios)
    compare(summary,{'cells':756,'geometric_ratio':g,'max_task_ratio':max(ratios),'performance_gate':g<=1.10 and max(ratios)<=1.25,'repeated_predictions':sum(r['rows']*10 for r in records)})
    amendment=read(run/'RESUMPTION.json')
    require(sha(run/'benchmark_initial.py')==amendment['original_sha256'],'old orchestration changed')
    require(sha(member(root,'source/experiments/tree_total/benchmark.py'))==amendment['resumed_sha256'],'resume code changed')
    for task,receipt,name in [('letter',amendment,'letter_prefix.jsonl'),('satellite',read(run/'satellite_resumption.json'),'satellite_prefix.jsonl')]:
        prefix=(run/name).read_bytes();full=(run/(task+'.jsonl')).read_bytes()
        count=receipt['preserved_prefix_cells'] if task=='letter' else receipt['prefix_cells']
        digest=receipt['preserved_prefix_sha256'] if task=='letter' else receipt['prefix_sha256']
        require(full.startswith(prefix) and len(prefix.splitlines())==count and hashlib.sha256(prefix).hexdigest()==digest,'completed prefix replaced')
    proof=read(results/'independent.json');require(proof['status']=='PASS' and len(proof['files'])==8,'incomplete proof reconstruction')
    replay=read(results/'replay/LOCK.json')
    for name,digest in replay['source_files'].items():require(sha(member(root,'source/experiments/tree_total/'+name))==digest,'replay source changed')
    total_scores=stress_rows=stress_exact=0
    for task in TASKS:
        r=read(results/'replay'/task/'result.json');n=len(expected_by_task[task]);c=r['classes']
        require(r['rows']==n and r['score_values']==n*c,'score inventory differs')
        data=(results/'replay'/task/'reference_scores.f64').read_bytes()
        require(len(data)==n*c*8,'missing source scores');values=struct.unpack('<'+'d'*(n*c),data)
        predicted=[max(range(c),key=lambda j:values[i*c+j]) for i in range(n)]
        require(predicted==list(expected_by_task[task]),'source score labels differ')
        total_scores+=n*c
        for layout in ('flat','interned'):
            artifact=results/'replay'/task/(layout+'.sctt');entry=next(x for x in proof['files'] if (x['task'],x['layout'])==(task,layout))
            require(sha(artifact)==entry['sha256']==r['models'][layout]['sha256'] and artifact.stat().st_size==entry['bytes'],'reconstructed proof differs')
            w=r['models'][layout]['work']['total']
            require(w['unresolved']==0 and w['coarse_certified']+w['exact_completed']==n,'coverage summary differs')
        for kind in ('uniform','boundary'):
            stress=r['stress'][kind];q=(results/'replay'/task/(kind+'.u8')).read_bytes();d=r['models']['interned']['info']['features']
            require(len(q)==stress['rows']*d,'stress input size')
            scores=(results/'replay'/task/(kind+'-scores.f64')).read_bytes();ids=(results/'replay'/task/(kind+'-indices.i32')).read_bytes()
            require(hashlib.sha256(scores).hexdigest()==stress['source_scores_sha256'] and hashlib.sha256(ids).hexdigest()==stress['indices_sha256'],'stress observation differs')
            require(stress['work']['unresolved']==0,'stress unresolved')
            stress_rows+=stress['rows'];stress_exact+=stress['work']['exact_completed'];total_scores+=len(scores)//8
    resource=read(results/'resources/LOCK.json');observed=[decode(l) for l in (results/'resources/rows.jsonl').read_text().splitlines()]
    jobs=resource_schedule(resource,observed)
    for i,(r,job) in enumerate(zip(observed,jobs)):
        require(type(r) is dict and type(r.get('repeat')) is int and
                [r.get(k) for k in ('task','policy','repeat')]==job and
                r.get('matched') is True and r.get('numerical_frameworks')==[],'resource context differs')
        literal=read(results/'resources'/f'process-{i}.json')
        compare(r,resource_output(literal))
    return {'status':'PASS','timing_cells':len(records),'repeated_predictions':sum(r['rows']*10 for r in records),
            'retained_rows':sum(map(len,expected_by_task.values())),'stress_rows':stress_rows,'stress_exact_fallbacks':stress_exact,
            'recorded_source_score_values':total_scores,'independent_reconstructed_files':8,'isolated_resource_processes':len(observed),
            'geometric_ratio':g,'performance_gate':g<=1.10 and max(ratios)<=1.25,
            'scope':'recorded bytes, scopes, outcomes, preserved interruptions and arithmetic; not authenticated clocks or universal backend equivalence'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=audit(a.root)
    with a.out.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r,indent=2))
