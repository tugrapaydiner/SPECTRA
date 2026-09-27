"""Deterministic mixed-request trace; no fitting or timing-based model selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from spectra.svm_pipeline import PreparedPipeline
from experiments.native_preprocessing.bench import NumpyPlan

ARMS=('python','compiled','fused','numpy')
TASKS=('chess','penguins','titanic','wdbc','wine','zoo')
SEED=20260927

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)

def trace(n,seed):
    order=list(range(n));random.Random(seed).shuffle(order)
    result=[];start=0;pattern=(1,1,8,32,128)
    while start<n:
        count=min(pattern[len(result)%5],n-start)
        result.append(order[start:start+count]);start+=count
    return result

def summarize(records):
    jobs={};ones={}
    for r in records:
        k=(r['model'],r['repeat'],r['arm'])
        jobs[k]=jobs.get(k,0)+r['ns']
        if r['rows']==1:ones.setdefault((r['model'].split('-')[0],r['arm']),[]).append(r['ns'])
    model_cost={}
    for model in sorted({r['model'] for r in records}):
        model_cost[model]={a:statistics.median([jobs[model,i,a] for i in range(5)]) for a in ARMS}
    tasks={}
    for task in TASKS:
        models=[m for m in model_cost if m.startswith(task+'-')]
        ratios={a:statistics.geometric_mean(model_cost[m]['fused']/model_cost[m][a] for m in models) for a in ARMS if a!='fused'}
        total_rows=sum(r['rows'] for r in records if r['model'].startswith(task+'-') and r['repeat']==0 and r['arm']=='fused')
        per_row={a:sum(model_cost[m][a] for m in models)/total_rows/1000 for a in ARMS}
        tasks[task]={'fused_over':ratios,'trace_cost_us_per_row':per_row,
                     'single_request_median_us':{a:statistics.median(ones[task,a])/1000 for a in ARMS},
                     'single_request_p95_us':{a:sorted(ones[task,a])[int(.95*(len(ones[task,a])-1))]/1000 for a in ARMS},
                     'matched_model_regressions':sum(model_cost[m]['fused']>model_cost[m]['compiled'] for m in models)}
    ratio=statistics.geometric_mean(v['fused_over']['compiled'] for v in tasks.values())
    return {'timing_cells':len(records),'predictions_checked':sum(r['rows'] for r in records),
            'models':model_cost,'tasks':tasks,'panel_fused_over_compiled':ratio,
            'primary_gate':ratio<=1/1.10 and all(v['fused_over']['compiled']<=1.10 for v in tasks.values())}

def main(args):
    import numpy as np
    from threadpoolctl import threadpool_limits
    args.out.mkdir(parents=True,exist_ok=False)
    models=[f'{t}-{s}' for t in TASKS for s in (101,202,303)]
    files={}
    for name in models:
        folder=args.inputs/'models'/name
        for f in ('model.srt','preprocessing.json','cases.json'):
            files[f'{name}/{f}']=sha(folder/f)
    source=[ROOT/'experiments/fused_pipeline/PROTOCOL.md',Path(__file__),ROOT/'experiments/native_preprocessing/bench.py']
    source+=sorted((ROOT/'spectra').glob('svm*.py'))+sorted((ROOT/'spectra/_native/ovo').rglob('*.cpp'))+sorted((ROOT/'spectra/_native/ovo').rglob('*.hpp'))
    protocol={'schema':'spectra.fused_pipeline.trace.v1','seed':SEED,'repeats':5,'models':models,'arms':ARMS,
              'pattern':[1,1,8,32,128],'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source},
              'model_files':files,'libraries':{str(p):sha(p) for p in (args.library,args.preprocessor)},
              'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
              'cpu':next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
              'affinity':sorted(os.sched_getaffinity(0)),
              'scope':'all raw-input preprocessing, input construction, SVM execution and fresh labels; file parsing and setup/build excluded; deterministic replay not live production'}
    write(args.out/'protocol.json',protocol)
    records=[];fidelity={};randomizer=random.Random(SEED)
    with threadpool_limits(1), (args.out/'rows.jsonl').open('x') as output:
        for ordinal,name in enumerate(models):
            folder=args.inputs/'models'/name;data=json.loads((folder/'cases.json').read_text())
            raw=data['rows'];labels=data['expected'];requests=trace(len(raw),SEED+ordinal)
            reference=NumpyPlan(folder/'preprocessing.json')
            with PreparedPipeline(folder,args.library) as p, PreparedPipeline(folder,args.library,preprocessor_library=args.preprocessor) as c,p.session() as pw,c.session() as cw:
                full=c.preprocessor.transform(raw);assert full.tobytes()==p.preprocessor.transform(raw).tobytes()==reference.transform(raw).tobytes()
                assert cw.predict_many(raw)==cw.predict_fused(raw)==pw.predict_many(raw)==labels
                fidelity[name]={'rows':len(raw),'features':len(full),'feature_sha256':hashlib.sha256(full.tobytes()).hexdigest(),
                                'model_sha256':sha(folder/'model.srt'),'requests':requests}
                batches=[([raw[i] for i in idx],[labels[i] for i in idx]) for idx in requests]
                def invoke(arm,rows):
                    if arm=='python':return pw.predict_many(rows)
                    if arm=='compiled':return cw.predict_many(rows)
                    if arm=='fused':return cw.predict_fused(rows)
                    return pw._worker.predict_buffer(reference.transform(rows))
                for arm in ARMS:
                    for rows,wanted in batches[:5]:assert invoke(arm,rows)==wanted
                for repeat in range(5):
                    order=list(ARMS);randomizer.shuffle(order)
                    for arm in order:
                        for index,(rows,wanted) in enumerate(batches):
                            start=time.perf_counter_ns();result=invoke(arm,rows);elapsed=time.perf_counter_ns()-start
                            assert result==wanted,(name,repeat,arm,index)
                            record={'model':name,'repeat':repeat,'arm':arm,'request':index,'rows':len(rows),'ns':elapsed,'matched':True}
                            records.append(record);output.write(json.dumps(record,separators=(',',':'))+'\n')
            print(name,'complete',len(requests),'requests',flush=True)
    write(args.out/'fidelity.json',fidelity)
    report=summarize(records);write(args.out/'summary.json',report)
    print(json.dumps(report['tasks'],indent=2));print('gate',report['primary_gate'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','library','preprocessor','out'):parser.add_argument('--'+n,type=Path,required=True)
    main(parser.parse_args())
