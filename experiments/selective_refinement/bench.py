"""Complete input-to-label jobs; fixed source, policies, models and output checks."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,random,statistics,struct,sys,time
from pathlib import Path
from contextlib import ExitStack
import numpy as np
from experiments.budgeted_prototypes.float_control import FloatSession
from .session import RefinementSession
from .study import TASKS,sha,write,source_snapshot
ARMS=('fast','strong','primary','strict','loose','blind','mlp32')
CHUNKS=(1,32,256);REPEATS=7;SEED=2026093037

def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    outcome=json.loads((a.evaluation/'RESULTS.json').read_text())
    policies=json.loads((a.calibration/'POLICIES.json').read_text())
    lock=json.loads((a.models/'FINAL_MODELS.json').read_text())
    for name,h in lock['files'].items():
        if sha(a.models/name)!=h:raise ValueError('timed model bytes changed')
    write(a.out/'LOCK.json',{'source':source_snapshot(a.out/'source'),
        'models_sha256':sha(a.models/'FINAL_MODELS.json'),'policies_sha256':sha(a.calibration/'POLICIES.json'),
        'evaluation_sha256':sha(a.evaluation/'RESULTS.json'),'library_sha256':sha(a.library),'mlp_sha256':sha(a.mlplib),
        'tasks':TASKS,'arms':ARMS,'chunks':CHUNKS,'repeats':REPEATS,'seed':SEED,
        'cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
        'platform':platform.platform(),'python':sys.version,'affinity':sorted(os.sched_getaffinity(0)),
        'scope':'whole-dataset uint8-to-fresh-label jobs, routing and scaling included, preparation/training excluded'})
    observations=[];rng=random.Random(SEED)
    with (a.out/'timings.jsonl').open('x') as log:
        for task in TASKS:
            with np.load(a.evaluation/task/'input.npz') as f:q=f['q']
            with np.load(a.evaluation/task/'predictions.npz') as f:expected={name:f[name].tolist() for name in ARMS}
            with ExitStack() as stack:
                sessions={name:stack.enter_context(RefinementSession(a.models/task/'fast/model.spp',a.models/task/'strong/model.srt',a.library,
                    threshold=policies[task][key]['threshold'],accept_fraction=policies[task][key]['accept_fraction']))
                    for name,key in (('primary','0.01'),('strict','0.005'),('loose','0.02'))}
                mlp=stack.enter_context(FloatSession(a.models/task/'mlp/model.sfn',a.mlplib))
                def call(name,chunk):
                    out=[]
                    for start in range(0,len(q),chunk):
                        x=q[start:start+chunk]
                        if name=='mlp32':pred=mlp.predict_buffer(x)
                        elif name in sessions:pred=sessions[name].predict_buffer(x)
                        else:pred=sessions['primary'].predict_buffer(x,mode=name)
                        out.extend(pred)
                    return out
                for name in ARMS:
                    if call(name,256)!=expected[name]:raise ValueError('warm output mismatch')
                for repeat in range(REPEATS):
                    schedule=[(chunk,name) for chunk in CHUNKS for name in ARMS];rng.shuffle(schedule)
                    for chunk,name in schedule:
                        begin=time.perf_counter_ns();pred=call(name,chunk);duration=time.perf_counter_ns()-begin
                        if pred!=expected[name]:raise ValueError('timed output mismatch')
                        row={'task':task,'repeat':repeat,'chunk':chunk,'arm':name,'rows':len(q),'ns':duration,
                             'prediction_sha256':hashlib.sha256(struct.pack('<'+str(len(pred))+'q',*pred)).hexdigest()}
                        observations.append(row);log.write(json.dumps(row)+'\n');log.flush()
                print(task,'complete timing',flush=True)
    results={}
    for task in TASKS:
        qual=outcome['tasks'][task]['quality'];n=qual['fast']['rows']
        median={str(c):{name:statistics.median(r['ns'] for r in observations if r['task']==task and r['chunk']==c and r['arm']==name)/n/1000
                       for name in ARMS} for c in CHUNKS}
        ratio=median['32']['primary']/median['32']['strong']
        cf,cp,cs=(qual[name]['correct'] for name in ('fast','primary','strong'))
        gate=(200*(cp-cf)>=n and ratio<=.5 and 100*(cs-cp)<=n)
        results[task]={'us_per_row':median,'primary_over_strong':ratio,'gain_over_fast_points':100*(cp-cf)/n,
                       'loss_against_strong_points':100*(cs-cp)/n,'gate':gate}
    write(a.out/'RESULTS.json',{'tasks':results,'timing_cells':len(observations),'repeated_predictions':sum(r['rows'] for r in observations),
                              'two_task_gate':sum(r['gate'] for r in results.values())>=2})
    print(json.dumps(results,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('models','calibration','evaluation','library','mlplib','out'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
