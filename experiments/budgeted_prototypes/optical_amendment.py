"""Prospectively broaden all optical prototype widths after a boundary optimum.

Keeps the entire original selection, adds the locked36 cells, and never reads a
new test input. This is a control correction during development, not a new test.
"""
from __future__ import annotations
import argparse,json,random,time,warnings,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.budgeted_prototypes.learning import train,features
from experiments.budgeted_prototypes.study import write,sha,source_hashes,settings

def initialize(a):
    a.out.mkdir(parents=True,exist_ok=False)
    jobs=[{'seed':s,'arm':arm,'choice':{'prototypes':p,'gamma':g}} for s in (611,977) for arm in ('fixed','centers','local') for p in (256,512,1024) for g in (.125,.5)]
    random.Random(2026092910).shuffle(jobs)
    for i,j in enumerate(jobs):j['job']=i
    write(a.out/'LOCK.json',{'jobs':jobs,'source':source_hashes(),'training_sha256':sha(a.new/'optdigits-train.npz'),
         'original_selection_sha256':sha(a.selection/'LOCK.json'),'amendment_commit':'3ef41698ed8cce5105b912939e093670156d1e53',
         'scope':'no test parsing or prediction; equal additional widths for all prototype families'})
def worker(a):
    lock=json.loads((a.out/'LOCK.json').read_text())
    for name,h in lock['source'].items():
        if sha(Path(__file__).parent/name)!=h:raise ValueError('amendment source changed')
    with np.load(a.new/'optdigits-train.npz') as f:q=f['q'];y=f['y'];D=int(f['maximum'])
    if sha(a.new/'optdigits-train.npz')!=lock['training_sha256']:raise ValueError('training changed')
    with threadpool_limits(1):
        for job in lock['jobs']:
            if job['job']%a.workers!=a.worker:continue
            folder=a.out/f'job-{job["job"]:03d}';folder.mkdir(exist_ok=False)
            with np.load(a.selection/f'optdigits-{job["seed"]}-split.npz') as f:fit=f['fit'];val=f['validation']
            s=settings(job['arm'],job['choice'],job['seed']);begin=time.process_time()
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always');arrays,report=train(q[fit],y[fit],D,s,validation=(q[val],y[val]))
            np.savez_compressed(folder/'model.npz',**arrays);write(folder/'training.json',report)
            prediction=arrays['classes'][(features(q[val],arrays,D,s)@arrays['head']+arrays['bias']).argmax(1)]
            np.savez_compressed(folder/'validation.npz',prediction=prediction,expected=y[val])
            record={**job,'task':'optdigits','correct':int(np.sum(prediction==y[val])),'validation_rows':len(val),'fit_rows':len(fit),
                    'cpu_seconds':time.process_time()-begin,'warnings':[str(w.message) for w in ws],
                    'files':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}}
            write(folder/'record.json',record);print(json.dumps(record),flush=True)
    write(a.out/f'worker-{a.worker}-complete.json',{'status':'COMPLETE'})
def select(a):
    chosen=json.loads((a.selection/'selected.json').read_text());records=[]
    for folder in (a.selection,a.out):
        for path in folder.glob('job-*/record.json'):
            r=json.loads(path.read_text())
            if r['task']=='optdigits' and r['arm'] in ('fixed','centers','local'):records.append(r)
    if len(records)!=72:raise ValueError('not all optical prototype choices are present')
    for arm in ('fixed','centers','local'):
        rank=[]
        for p in (256,512,1024):
            for index,g in enumerate((2.,8.,.125,.5)):
                selected=[r for r in records if r['arm']==arm and r['choice']=={'prototypes':p,'gamma':g}]
                if len(selected)!=2:raise ValueError('missing choice or duplicate repetition')
                score=sum(r['correct'] for r in selected)/sum(r['validation_rows'] for r in selected)
                rank.append((score,-p,-index,p,g))
        best=max(rank);chosen['optdigits'][arm]={'choice':{'prototypes':best[3],'gamma':best[4]},'validation_accuracy':best[0],
            'source':'complete initial-plus-width-amendment selection'}
    a.final.mkdir(parents=True,exist_ok=False);write(a.final/'selected.json',chosen)
    write(a.final/'LOCK.json',{'original_lock_sha256':sha(a.selection/'LOCK.json'),'amendment_lock_sha256':sha(a.out/'LOCK.json'),
              'all_selection_cells':284,'selection_only':True,'selected_sha256':sha(a.final/'selected.json')})
    print(json.dumps(chosen,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['initialize','worker','select'])
    for name in ('selection','new','out','final'):p.add_argument('--'+name,type=Path)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--worker',type=int,default=0);a=p.parse_args();globals()[a.operation](a)
