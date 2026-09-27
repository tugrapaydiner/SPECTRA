"""Full raw-row mixed traces: paired existing/new profiles, same native library."""
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
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from spectra.svm_pipeline import PreparedPipeline
from experiments.fused_pipeline.bench import trace
from experiments.binary_stream.bench import TASKS,write,sha,output_hash
ARMS=('compiled_default','compiled_stream','fused_default','fused_stream')
SEED=20260927

def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    models=[f'{t}-{s}' for t in TASKS for s in (101,202,303)]
    files={f'{m}/{f}':sha(a.inputs/m/f) for m in models for f in ('model.srt','preprocessing.json','cases.json')}
    sources=sorted((ROOT/'spectra').glob('svm*.py'))+sorted((ROOT/'spectra/_native/ovo').rglob('*.cpp'))+sorted((ROOT/'spectra/_native/ovo').rglob('*.hpp'))+[Path(__file__),Path(__file__).with_name('PROTOCOL.md'),ROOT/'experiments/fused_pipeline/bench.py',Path(__file__).with_name('bench.py')]
    write(a.out/'protocol.json',{'schema':'spectra.binary_stream.raw_trace.v1','models':models,'arms':ARMS,'repeats':5,'seed':SEED,
        'pattern':[1,1,8,32,128],'tables':False,'model_files':files,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'libraries':{p.name:sha(p) for p in (a.library,a.preprocessor)},'affinity':sorted(os.sched_getaffinity(0)),
        'python':sys.version,'host':platform.platform(),'scope':'raw rows to fresh complete labels; all inference/preprocessing included; setup/build excluded; deterministic retained replay'})
    records=[];rng=random.Random(SEED)
    with (a.out/'rows.jsonl').open('x') as f:
        for ordinal,name in enumerate(models):
            folder=a.inputs/name;cases=json.loads((folder/'cases.json').read_text());rows=cases['rows'];expected=cases['expected']
            requests=trace(len(rows),SEED+ordinal)
            batches=[([rows[i] for i in ind],[expected[i] for i in ind]) for ind in requests]
            with PreparedPipeline(folder,a.library,preprocessor_library=a.preprocessor) as m,m.session() as w:
                def call(arm,x):
                    mode='binary_stream' if arm.endswith('stream') else 'beretta_cert'
                    return w.predict_fused(x,schedule=mode) if arm.startswith('fused') else w.predict_many(x,schedule=mode)
                for arm in ARMS:assert call(arm,rows)==expected
                for repeat in range(5):
                    order=list(ARMS);rng.shuffle(order)
                    for arm in order:
                        for request,(x,wanted) in enumerate(batches):
                            start=time.perf_counter_ns();out=call(arm,x);ns=time.perf_counter_ns()-start
                            assert out==wanted
                            record={'model':name,'repeat':repeat,'arm':arm,'request':request,'rows':len(x),'ns':ns,'output_sha256':output_hash(out)}
                            records.append(record);f.write(json.dumps(record,separators=(',',':'))+'\n')
    jobs={}
    for r in records:
        k=r['model'],r['repeat'],r['arm'];jobs[k]=jobs.get(k,0)+r['ns']
    costs={m:{a:statistics.median(jobs[m,i,a] for i in range(5)) for a in ARMS} for m in models}
    tasks={}
    for t in TASKS:
        ms=[m for m in models if m.startswith(t+'-')]
        tasks[t]={base+'_stream_over_default':statistics.geometric_mean(costs[m][base+'_stream']/costs[m][base+'_default'] for m in ms) for base in ('compiled','fused')}
        tasks[t]['fused_regressions']=sum(costs[m]['fused_stream']>costs[m]['fused_default'] for m in ms)
    pooled=sum(v['fused_stream'] for v in costs.values())/sum(v['fused_default'] for v in costs.values())
    summary={'cells':len(records),'predictions':sum(r['rows'] for r in records),'costs':costs,'tasks':tasks,'pooled_fused_ratio':pooled,
        'equal_task_fused_ratio':statistics.geometric_mean(v['fused_stream_over_default'] for v in tasks.values())}
    write(a.out/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','library','preprocessor','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
