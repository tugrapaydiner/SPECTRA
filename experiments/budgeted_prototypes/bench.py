"""Raw integer features to fresh labels: matched native complete-job experiment."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,random,statistics,struct,sys,time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.budgeted_prototypes.session import PrototypeSession
from experiments.budgeted_prototypes.controls import ControlSession
from experiments.budgeted_prototypes.strong_controls import BlasSession
from experiments.budgeted_prototypes.study import TASKS,sha,write,source_hashes
LAYOUTS=('original','packet','register')
FAMILIES=('fixed','centers','local')
ARMS=tuple(f'{f}_{l}' for l in LAYOUTS for f in FAMILIES)+('local_scalar','local_direct_exp','svc','mlp','linear','mlp_blas','svc_finite')
CHUNKS=(1,32,256);REPEATS=7;SEED=2026092912

def main(a):
    a.out.mkdir(parents=True,exist_ok=False);quality=json.loads((a.evaluation/'quality.json').read_text());lock=json.loads((a.models/'FINAL_LOCK.json').read_text())
    for name,h in lock['files'].items():
        if sha(a.models/name)!=h:raise ValueError('final model changed')
    cpu=next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name'))
    write(a.out/'protocol.json',{'arms':ARMS,'chunks':CHUNKS,'repeats':REPEATS,'seed':SEED,'tasks':TASKS,
        'source':source_hashes(),'final_lock_sha256':sha(a.models/'FINAL_LOCK.json'),'quality_sha256':sha(a.evaluation/'quality.json'),
        'libraries':{str(p):sha(p) for p in (a.original,a.packet,a.register,a.controls,a.blas,a.finite)},'cpu':cpu,'platform':platform.platform(),
        'python':sys.version,'affinity':sorted(os.sched_getaffinity(0)),
        'scope':'complete all-row jobs, original uint8 features to fresh labels; native normalization/scaling included; loading/fitting/table creation excluded'})
    records=[];randomizer=random.Random(SEED);direct_differences={}
    with (a.out/'timings.jsonl').open('x') as log:
        for task in TASKS:
            with ExitStack() as stack:
                engines={f'{f}_{l}':stack.enter_context(PrototypeSession(a.models/f'{task}-{f}'/'model.spp',getattr(a,l))) for l in LAYOUTS for f in FAMILIES}
                fit=json.loads((a.models/f'{task}-local'/'fit.json').read_text());d=fit['features'];D=fit['maximum']
                q=np.fromfile(a.evaluation/task/'input.u8',dtype=np.uint8).reshape(-1,d)
                controls={arm:stack.enter_context(ControlSession(a.models/f'{task}-{arm}'/('model.srt' if arm=='svc' else 'model.snn'),a.controls,maximum=D)) for arm in ('svc','mlp','linear')}
                controls['mlp_blas']=stack.enter_context(BlasSession(a.models/f'{task}-mlp'/'model.snn',a.blas))
                controls['svc_finite']=stack.enter_context(ControlSession(a.models/f'{task}-svc'/'model.srt',a.finite,maximum=D))
                expected={arm:np.load(a.evaluation/task/f'{arm}-predictions.npz')['prediction'].tolist() for arm in ('fixed','centers','local','svc','mlp','linear')}
                expected['mlp_blas']=expected['mlp'];expected['svc_finite']=expected['svc']
                for family in FAMILIES:
                    for layout in LAYOUTS:expected[f'{family}_{layout}']=expected[family]
                expected['local_scalar']=expected['local']
                expected['local_direct_exp']=engines['local_register'].predict_buffer(q,mode='direct_exp')
                direct_differences[task]=sum(x!=y for x,y in zip(expected['local'],expected['local_direct_exp']))
                def call(arm,chunk):
                    out=[]
                    for start in range(0,len(q),chunk):
                        rows=q[start:start+chunk]
                        if arm in engines:out.extend(engines[arm].predict_buffer(rows))
                        elif arm=='local_scalar':out.extend(engines['local_register'].predict_buffer(rows,mode='scalar'))
                        elif arm=='local_direct_exp':out.extend(engines['local_register'].predict_buffer(rows,mode='direct_exp'))
                        else:out.extend(controls[arm].predict_buffer(rows))
                    return out
                for arm in ARMS:
                    if call(arm,256)!=expected[arm]:raise ValueError('warmup changed prediction')
                for repeat in range(REPEATS):
                    jobs=[(chunk,arm) for chunk in CHUNKS for arm in ARMS];randomizer.shuffle(jobs)
                    for chunk,arm in jobs:
                        begin=time.perf_counter_ns();prediction=call(arm,chunk);ns=time.perf_counter_ns()-begin
                        if prediction!=expected[arm]:raise ValueError('timed output disagreement')
                        digest=hashlib.sha256(struct.pack('<'+'q'*len(prediction),*prediction)).hexdigest()
                        row={'task':task,'repeat':repeat,'chunk':chunk,'arm':arm,'rows':len(q),'ns':ns,'prediction_sha256':digest}
                        records.append(row);log.write(json.dumps(row)+'\n');log.flush()
                    print(task,'repeat',repeat,'complete',flush=True)
    summaries={}
    for task in TASKS:
        n=quality[task]['arms']['local']['rows']
        med={str(c):{arm:statistics.median(r['ns'] for r in records if r['task']==task and r['chunk']==c and r['arm']==arm)/n/1000 for arm in ARMS} for c in CHUNKS}
        ratio=med['32']['local_register']/med['32']['fixed_register'];gain=quality[task]['local_vs_fixed']['gain_points']
        summaries[task]={'us_per_row':med,'local_over_fixed':ratio,'local_gain_points':gain,'joint_gate':gain>=.5 and ratio<=1.25,
                         'direct_exp_prediction_disagreements':direct_differences[task],
                         'layout_ratios':{f:med['32'][f+'_register']/med['32'][f+'_original'] for f in FAMILIES},
                         'local_over_mlp':med['32']['local_register']/med['32']['mlp'],
                         'local_over_svc':med['32']['local_register']/med['32']['svc'],
                         'local_over_mlp_blas':med['32']['local_register']/med['32']['mlp_blas'],
                         'local_over_svc_finite':med['32']['local_register']/med['32']['svc_finite']}
    ratios=[v for task in summaries.values() for v in task['layout_ratios'].values()]
    layout_ratio=statistics.geometric_mean(ratios)
    write(a.out/'summary.json',{'layout_ratio':layout_ratio,'layout_gate':layout_ratio<=1/1.10 and max(ratios)<=1.10,'cells':len(records),'checked_predictions':sum(r['rows'] for r in records),
             'tasks':summaries,'primary_two_task_gate':sum(r['joint_gate'] for r in summaries.values())>=2})
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('models','evaluation','original','packet','register','controls','blas','finite','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
