"""Frozen training-only selection and refitting for a fixed inference budget.

Every fitted candidate is saved with its predictions, timing and warnings. The
new OptDigits official test is not parsed by this program. Pickles are our own
training artifacts only, never a deployment format or untrusted-input interface.
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, json, os, pickle, random, sys, time, warnings
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.budgeted_prototypes.learning import Settings,train,features,grouped_split
TASKS=('letter','pendigits','satellite','optdigits')
SEEDS=(611,977)
ARMS=('fixed','centers','local','svc','mlp')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)
def source_hashes():
    folder=Path(__file__).parent
    return {p.name:sha(p) for p in sorted(folder.iterdir()) if p.suffix in ('.py','.cpp','.md')}
def train_path(prior,new,task):
    if task=='optdigits':return new/'optdigits-train.npz'
    return prior/('satellite/data/train.npz' if task=='satellite' else f'datasets/{task}-train.npz')
def choices(task,arm):
    if arm in ('fixed','centers','local'):
        return [{'prototypes':p,'gamma':g} for p in (256,512,1024) for g in ((8.,32.) if task=='satellite' else (2.,8.))]
    if arm=='svc':return [{'C':c,'gamma':g} for c in (1.,10.,100.) for g in ((.125,.5,2.) if task=='optdigits' else (.5,2.,8.))]
    if arm=='mlp':return [{'width':w,'alpha':alpha} for w in (128,256) for alpha in (1e-5,1e-3)]
    raise ValueError('unknown family')
def settings(arm,choice,seed):
    return Settings(**choice,seed=seed,arm=arm,epochs=200,reg=1e-6,quarter=4,units=4,
                    affine=False,adaptive_gamma=False,normalized=False,average_tail=False)
def control(arm,choice,seed):
    if arm=='svc':return SVC(**choice,kernel='rbf',cache_size=128)
    if arm=='mlp':
        return make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(choice['width'],)*2,
            alpha=choice['alpha'],activation='relu',solver='adam',batch_size=128,learning_rate_init=.002,
            max_iter=400,early_stopping=True,validation_fraction=.1,n_iter_no_change=30,random_state=seed))
    if arm=='linear':return LinearSVC(C=10.,dual='auto',max_iter=5000,random_state=seed)
    raise ValueError('unknown control')
def initialize(a):
    a.out.mkdir(parents=True,exist_ok=False);jobs=[]
    for task in TASKS:
        path=train_path(a.prior,a.new,task)
        with np.load(path) as f:q=f['q'];y=f['y']
        for seed in SEEDS:
            fit,val=grouped_split(q,y,seed)
            np.savez_compressed(a.out/f'{task}-{seed}-split.npz',fit=fit,validation=val)
            for arm in ARMS:
                for i,ch in enumerate(choices(task,arm)):
                    jobs.append({'task':task,'seed':seed,'arm':arm,'choice_index':i,'choice':ch})
    random.Random(20260929).shuffle(jobs)
    for i,j in enumerate(jobs):j['job']=i
    write(a.out/'LOCK.json',{'source':source_hashes(),'inputs':{t:sha(train_path(a.prior,a.new,t)) for t in TASKS},
         'jobs':jobs,'split_seed_inventory':SEEDS,'maximum_numerical_threads_per_process':1,
         'development_role':'original training partitions only; no official test predictions',
         'protocol_commit':'622241a499ca01b95bf6446402df9277bbd5212c'})
    print('locked jobs',len(jobs),flush=True)
def worker(a):
    lock=json.loads((a.out/'LOCK.json').read_text())
    for name,h in lock['source'].items():
        if sha(Path(__file__).parent/name)!=h:raise ValueError('selection source changed: '+name)
    with threadpool_limits(1):
        for job in lock['jobs']:
            if job['job']%a.workers!=a.worker:continue
            dest=a.out/f'job-{job["job"]:03d}';dest.mkdir(exist_ok=False)
            task,seed,arm,ch=job['task'],job['seed'],job['arm'],job['choice']
            path=train_path(a.prior,a.new,task)
            if sha(path)!=lock['inputs'][task]:raise ValueError('training data changed')
            with np.load(path) as f:q=f['q'];y=f['y'];D=int(f['maximum'])
            with np.load(a.out/f'{task}-{seed}-split.npz') as f:fit=f['fit'];val=f['validation']
            cpu=time.process_time();wall=time.perf_counter()
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                if arm in ('fixed','centers','local'):
                    s=settings(arm,ch,seed);arrays,report=train(q[fit],y[fit],D,s,validation=(q[val],y[val]))
                    np.savez_compressed(dest/'model.npz',**arrays)
                    score=features(q[val],arrays,D,s)@arrays['head']+arrays['bias'];pred=arrays['classes'][score.argmax(1)]
                    write(dest/'training.json',report)
                else:
                    model=control(arm,ch,seed);model.fit(q[fit].astype(float)/D,y[fit]);pred=model.predict(q[val].astype(float)/D)
                    (dest/'model.pkl').write_bytes(pickle.dumps(model,protocol=4))
                    params={'supports':len(model.support_)} if arm=='svc' else {'iterations':int(model[-1].n_iter_),'validation_scores':model[-1].validation_scores_}
                    write(dest/'training.json',params)
            np.savez_compressed(dest/'validation.npz',prediction=pred,expected=y[val])
            rec={**job,'correct':int(np.sum(pred==y[val])),'validation_rows':len(val),'fit_rows':len(fit),
                 'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall,
                 'warnings':[str(w.message) for w in ws],
                 'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}}
            write(dest/'record.json',rec);print(json.dumps(rec),flush=True)
    write(a.out/f'worker-{a.worker}-complete.json',{'worker':a.worker,'jobs':sum(j['job']%a.workers==a.worker for j in lock['jobs']),'status':'COMPLETE'})
def select(a):
    lock=json.loads((a.out/'LOCK.json').read_text());records=[]
    for job in lock['jobs']:
        rec=json.loads((a.out/f'job-{job["job"]:03d}'/'record.json').read_text())
        if any(rec[k]!=v for k,v in job.items()):raise ValueError('job inventory changed')
        records.append(rec)
    chosen={}
    for task in TASKS:
        chosen[task]={}
        for arm in ARMS:
            ranking=[]
            for i,ch in enumerate(choices(task,arm)):
                rr=[r for r in records if r['task']==task and r['arm']==arm and r['choice_index']==i]
                if len(rr)!=len(SEEDS):raise ValueError('incomplete selection grid')
                score=sum(r['correct'] for r in rr)/sum(r['validation_rows'] for r in rr)
                ranking.append((score,-i,i))
            score,_,i=max(ranking);chosen[task][arm]={'choice':choices(task,arm)[i],'choice_index':i,'validation_accuracy':score}
    write(a.out/'selected.json',chosen)
    write(a.out/'selection_summary.json',{'fits':len(records),'cpu_seconds':sum(r['cpu_seconds'] for r in records),'chosen':chosen})
    print(json.dumps(chosen,indent=2))
def refit(a):
    chosen=json.loads((a.selection/'selected.json').read_text())
    if a.worker==0:
        a.out.mkdir(parents=True,exist_ok=False)
        write(a.out/'FIT_LOCK.json',{'selection_sha256':sha(a.selection/'selected.json'),'source':source_hashes(),
             'inputs':{t:sha(train_path(a.prior,a.new,t)) for t in TASKS},'seed':20260929,
             'scope':'all final models frozen before test predictions'})
    else:
        if not (a.out/'FIT_LOCK.json').is_file():raise ValueError('start final worker0 first')
    jobs=[(t,arm) for t in TASKS for arm in (*ARMS,'linear')]
    with threadpool_limits(1):
        for i,(task,arm) in enumerate(jobs):
            if i%a.workers!=a.worker:continue
            folder=a.out/(task+'-'+arm);folder.mkdir(exist_ok=False)
            with np.load(train_path(a.prior,a.new,task)) as f:q=f['q'];y=f['y'];D=int(f['maximum'])
            ch={} if arm=='linear' else chosen[task][arm]['choice'];cpu=time.process_time();wall=time.perf_counter()
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                if arm in ('fixed','centers','local'):
                    s=settings(arm,ch,20260929);arrays,report=train(q,y,D,s)
                    np.savez_compressed(folder/'model.npz',**arrays);write(folder/'settings.json',dataclasses.asdict(s));write(folder/'training.json',report)
                    from experiments.budgeted_prototypes.export import encode
                    (folder/'model.spp').write_bytes(encode(arrays,D,s))
                else:
                    model=control(arm,ch,20260929);model.fit(q.astype(float)/D,y)
                    (folder/'model.pkl').write_bytes(pickle.dumps(model,protocol=4))
                    if arm=='svc':
                        from spectra.svm_export import export_prepared_svc
                        export_prepared_svc(model,folder/'model.srt')
                        write(folder/'training.json',{'supports':len(model.support_),'classes':model.classes_.tolist()})
                    elif arm=='mlp':
                        scale,net=model.steps[0][1],model.steps[1][1]
                        np.savez_compressed(folder/'weights.npz',**{f'w{k}':v for k,v in enumerate(net.coefs_)},
                             **{f'b{k}':v for k,v in enumerate(net.intercepts_)},mean=scale.mean_,scale=scale.scale_,classes=net.classes_,maximum=D)
                        write(folder/'training.json',{'iterations':int(net.n_iter_),'validation_scores':net.validation_scores_})
                    else:
                        np.savez_compressed(folder/'weights.npz',w0=model.coef_.T,b0=model.intercept_,
                             mean=np.zeros(q.shape[1]),scale=np.ones(q.shape[1]),classes=model.classes_,maximum=D)
            write(folder/'fit.json',{'task':task,'arm':arm,'choice':ch,'training_rows':len(q),'features':q.shape[1],'maximum':D,
                   'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall,'warnings':[str(w.message) for w in ws],
                   'files':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}})
            print(task,arm,'COMPLETE',flush=True)
    write(a.out/f'worker-{a.worker}-complete.json',{'status':'COMPLETE'})
def finish(a):
    records=[json.loads((a.out/f'{t}-{arm}'/'fit.json').read_text()) for t in TASKS for arm in (*ARMS,'linear')]
    write(a.out/'FINAL_LOCK.json',{'source':source_hashes(),'files':{p.relative_to(a.out).as_posix():sha(p) for p in sorted(a.out.rglob('*')) if p.is_file()},
        'models':24,'fits':records,'cpu_seconds':sum(r['cpu_seconds'] for r in records),'scope':'final artifact lock before official evaluation'})
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['initialize','worker','select','refit','finish'])
    for k in ('prior','new','out','selection'):p.add_argument('--'+k,type=Path)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--worker',type=int,default=0);a=p.parse_args()
    globals()[a.operation](a)
