"""Replay each complete published timing panel on one newly frozen model set.

This generates NEW measurements; it never reconstructs lost timing samples.
Stage names preserve the original comparator sets/seeds for audit compatibility.
No model fitting, parameter selection or accuracy-based dispatch is performed.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, statistics, struct, sys, time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.budgeted_prototypes.session import PrototypeSession
from experiments.budgeted_prototypes.controls import ControlSession
from experiments.budgeted_prototypes.strong_controls import BlasSession
from experiments.budgeted_prototypes.float_control import FloatSession
from experiments.budgeted_prototypes.study import TASKS,sha,write,source_hashes

PANELS={'original':('benchmark',2026092909),'strong':('benchmark_strong',2026092912),'final':('benchmark_final',2026092917)}
FAMILIES=('fixed','centers','local');LAYOUTS=('original','packet','register')


def main(a):
    run,seed=PANELS[a.panel];a.out.mkdir(parents=True,exist_ok=False)
    quality=json.loads((a.evaluation/'quality.json').read_text())
    lock=json.loads((a.models/'FINAL_LOCK.json').read_text())
    for name,h in lock['files'].items():
        if sha(a.models/name)!=h:raise ValueError('model changed after final lock')
    arms=tuple(f'{f}_{l}' for l in LAYOUTS for f in FAMILIES)+('local_scalar','local_direct_exp','svc','mlp','linear')
    libraries=[a.original,a.packet,a.register,a.controls]
    if a.panel!='original':arms+=('mlp_blas','svc_finite');libraries += [a.blas,a.finite]
    if a.panel=='final':arms+=('mlp_float32',);libraries+=[a.floatlib]
    protocol={'arms':arms,'tasks':TASKS,'chunks':(1,32,256),'repeats':7,'seed':seed,'source':source_hashes(),
        'final_lock_sha256':sha(a.models/'FINAL_LOCK.json'),'quality_sha256':sha(a.evaluation/'quality.json'),
        'libraries':{str(p):sha(p) for p in libraries},'cpu':next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),
        'platform':platform.platform(),'python':sys.version,'affinity':sorted(os.sched_getaffinity(0)),
        'reproduction':True,'scope':'New complete-job observations from uint8 codes to fresh labels; preparation/fitting excluded; no historical sample recovery.'}
    if a.panel=='final':protocol['fp32_lock_sha256']=sha(a.fp32/'LOCK.json')
    write(a.out/'protocol.json',protocol)
    records=[];rng=random.Random(seed);direct={}
    with (a.out/'timings.jsonl').open('x') as log:
        for task in TASKS:
            fit=json.loads((a.models/f'{task}-local/fit.json').read_text());d=fit['features'];D=fit['maximum']
            q=np.fromfile(a.evaluation/task/'input.u8',dtype=np.uint8).reshape(-1,d)
            expected={arm:np.load(a.evaluation/task/f'{arm}-predictions.npz')['prediction'].tolist() for arm in (*FAMILIES,'svc','mlp','linear')}
            with ExitStack() as stack:
                proto={f'{f}_{l}':stack.enter_context(PrototypeSession(a.models/f'{task}-{f}/model.spp',getattr(a,l))) for l in LAYOUTS for f in FAMILIES}
                ctl={arm:stack.enter_context(ControlSession(a.models/f'{task}-{arm}'/('model.srt' if arm=='svc' else 'model.snn'),a.controls,maximum=D)) for arm in ('svc','mlp','linear')}
                if a.panel!='original':
                    ctl['mlp_blas']=stack.enter_context(BlasSession(a.models/f'{task}-mlp/model.snn',a.blas))
                    ctl['svc_finite']=stack.enter_context(ControlSession(a.models/f'{task}-svc/model.srt',a.finite,maximum=D))
                    expected['mlp_blas']=expected['mlp'];expected['svc_finite']=expected['svc']
                if a.panel=='final':
                    ctl['mlp_float32']=stack.enter_context(FloatSession(a.fp32/f'{task}.sfn',a.floatlib))
                    expected['mlp_float32']=np.load(a.fp32/f'{task}-predictions.npz')['prediction'].tolist()
                for key in proto:expected[key]=expected[key.split('_')[0]]
                expected['local_scalar']=expected['local']
                expected['local_direct_exp']=proto['local_register'].predict_buffer(q,mode='direct_exp')
                direct[task]=sum(x!=y for x,y in zip(expected['local_direct_exp'],expected['local']))
                def call(arm,chunk):
                    out=[]
                    for start in range(0,len(q),chunk):
                        block=q[start:start+chunk]
                        if arm in proto:out.extend(proto[arm].predict_buffer(block))
                        elif arm in ('local_scalar','local_direct_exp'):out.extend(proto['local_register'].predict_buffer(block,mode=arm[len('local_'):]))
                        else:out.extend(ctl[arm].predict_buffer(block))
                    return out
                for arm in arms:
                    if call(arm,256)!=expected[arm]:raise ValueError('warmup mismatch: '+arm)
                for repeat in range(7):
                    jobs=[(c,arm) for c in (1,32,256) for arm in arms];rng.shuffle(jobs)
                    for chunk,arm in jobs:
                        start=time.perf_counter_ns();p=call(arm,chunk);elapsed=time.perf_counter_ns()-start
                        if p!=expected[arm]:raise ValueError('timed prediction changed: '+arm)
                        record={'task':task,'repeat':repeat,'chunk':chunk,'arm':arm,'rows':len(q),'ns':elapsed,
                                'prediction_sha256':hashlib.sha256(struct.pack('<'+str(len(p))+'q',*p)).hexdigest()}
                        records.append(record);log.write(json.dumps(record)+'\n');log.flush()
                    print(a.panel,task,'repeat',repeat,'complete',flush=True)
    tasks={}
    for task in TASKS:
        n=quality[task]['arms']['local']['rows'];med={str(c):{arm:statistics.median(r['ns'] for r in records if r['task']==task and r['chunk']==c and r['arm']==arm)/n/1000 for arm in arms} for c in (1,32,256)}
        v=med['32'];ratio=v['local_register']/v['fixed_register'];gain=quality[task]['local_vs_fixed']['gain_points']
        tasks[task]={'us_per_row':med,'local_over_fixed':ratio,'local_gain_points':gain,'joint_gate':gain>=.5 and ratio<=1.25,
            'direct_exp_prediction_disagreements':direct[task],'layout_ratios':{f:v[f+'_register']/v[f+'_original'] for f in FAMILIES},
            'local_over_mlp':v['local_register']/v['mlp'],'local_over_svc':v['local_register']/v['svc']}
        if a.panel!='original':tasks[task].update(local_over_mlp_blas=v['local_register']/v['mlp_blas'],local_over_svc_finite=v['local_register']/v['svc_finite'])
        if a.panel=='final':tasks[task]['local_over_mlp_float32']=v['local_register']/v['mlp_float32']
    ratios=[v for t in tasks.values() for v in t['layout_ratios'].values()];g=statistics.geometric_mean(ratios)
    result={'tasks':tasks,'cells':len(records),'checked_predictions':sum(r['rows'] for r in records),'layout_ratio':g,
        'layout_gate':g<=1/1.10 and max(ratios)<=1.10,'primary_two_task_gate':sum(t['joint_gate'] for t in tasks.values())>=2}
    write(a.out/'summary.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',choices=tuple(PANELS),required=True)
    for name in ('models','evaluation','original','packet','register','controls','blas','finite','floatlib','fp32','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
