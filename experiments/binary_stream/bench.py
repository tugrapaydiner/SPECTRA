"""Frozen retained-model whole-job batches; no fitting or runtime tuning."""
from __future__ import annotations
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from spectra.svm_shared import PreparedModel

TASKS=('chess','penguins','titanic','wdbc','wine','zoo')
ARMS=('exhaustive','beretta_cert','binary_stream')
CHUNKS=(1,4,32,128)
SEED=2026092707
REPEATS=31


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def output_hash(labels):return hashlib.sha256(json.dumps(labels,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)

def summarize(records):
    groups={}
    for r in records:groups.setdefault((r['model'],r['chunk'],r['arm']),[]).append(r['ns'])
    medians={'|'.join(map(str,k)):statistics.median(v) for k,v in groups.items()}
    tasks={}
    for task in TASKS:
        models=sorted({r['model'] for r in records if r['model'].startswith(task+'-')})
        rows={m:next(r['rows'] for r in records if r['model']==m) for m in models}
        tasks[task]={}
        for chunk in CHUNKS:
            ratios={ref:statistics.geometric_mean(medians[f'{m}|{chunk}|binary_stream']/medians[f'{m}|{chunk}|{ref}'] for m in models) for ref in ARMS[:-1]}
            us={a:statistics.median(medians[f'{m}|{chunk}|{a}']/rows[m]/1000 for m in models) for a in ARMS}
            tasks[task][str(chunk)]={'us_per_row':us,'stream_over':ratios,
                'regressions_vs_default':sum(medians[f'{m}|{chunk}|binary_stream']>medians[f'{m}|{chunk}|beretta_cert'] for m in models)}
    primary=statistics.geometric_mean(tasks[t]['32']['stream_over']['beretta_cert'] for t in ('chess','titanic','wdbc'))
    pooled=sum(medians[f'{m}|32|binary_stream'] for m in rows_all(records))/sum(medians[f'{m}|32|beretta_cert'] for m in rows_all(records))
    return {'schema':'spectra.binary_stream.summary.v1','cells':len(records),'predictions':sum(r['rows'] for r in records),
       'model_medians_ns':medians,'tasks':tasks,'binary_batch32_ratio':primary,'pooled_all_models_batch32_ratio':pooled,
       'primary_gate':primary<=1/1.2 and all(tasks[t]['32']['stream_over']['beretta_cert']<=1.10 for t in ('chess','titanic','wdbc'))}

def rows_all(records):return sorted({r['model'] for r in records})

def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    models=[f'{t}-{s}' for t in TASKS for s in (101,202,303)]
    sources=sorted((ROOT/'spectra').glob('svm*.py'))+sorted((ROOT/'spectra/_native/ovo').rglob('*.cpp'))+sorted((ROOT/'spectra/_native/ovo').rglob('*.hpp'))+[Path(__file__),Path(__file__).with_name('PROTOCOL.md')]
    files={f'{m}/{f}':sha(a.inputs/m/f) for m in models for f in ('model.srt','cases.json','transformed.f64','preprocessing.json')}
    protocol={'schema':'spectra.binary_stream.benchmark.v1','seed':SEED,'repeats':REPEATS,'arms':ARMS,'chunks':CHUNKS,'models':models,
       'inputs_sha256':files,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
       'library_sha256':sha(a.library),'library_name':a.library.name,'platform':platform.platform(),'python':sys.version,
       'affinity':sorted(os.sched_getaffinity(0)),
       'cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
       'tables':False,'input_dtype':'float64','scope':'complete all-row jobs using prepared checked buffers and fresh label lists; loading/build/preprocessing/JSON outside timer; each output checked; no production trace'}
    write(a.out/'protocol.json',protocol)
    records=[];rng=random.Random(SEED)
    with (a.out/'rows.jsonl').open('x') as log:
        for name in models:
            folder=a.inputs/name;cases=json.loads((folder/'cases.json').read_text());expected=cases['expected']
            data=array('d');data.frombytes((folder/'transformed.f64').read_bytes())
            with PreparedModel(folder/'model.srt',a.library,input_dtype='float64') as model,model.session() as worker:
                d=model.features;n=len(expected);assert len(data)==n*d
                batches={c:[data[s*d:min(s+c,n)*d] for s in range(0,n,c)] for c in CHUNKS}
                def invoke(arm,chunk):
                    output=[]
                    for batch in batches[chunk]:output.extend(worker.predict_buffer(batch,schedule=arm))
                    return output
                for arm in ARMS:
                    for chunk in CHUNKS:assert invoke(arm,chunk)==expected
                digest=output_hash(expected)
                jobs=[(chunk,arm) for chunk in CHUNKS for arm in ARMS]
                for repeat in range(REPEATS):
                    order=list(jobs);rng.shuffle(order)
                    for chunk,arm in order:
                        begin=time.perf_counter_ns();result=invoke(arm,chunk);ns=time.perf_counter_ns()-begin
                        if result!=expected:raise AssertionError((name,repeat,chunk,arm))
                        record={'model':name,'repeat':repeat,'chunk':chunk,'arm':arm,'rows':n,'ns':ns,'output_sha256':output_hash(result)}
                        assert record['output_sha256']==digest
                        records.append(record);log.write(json.dumps(record,separators=(',',':'))+'\n')
            print(name,'complete',flush=True)
    result=summarize(records);write(a.out/'summary.json',result)
    print(json.dumps(result['tasks'],indent=2));print('primary gate:',result['primary_gate'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('inputs','library','out'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
