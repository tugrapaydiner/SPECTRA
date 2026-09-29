"""Two explicitly secondary local-covariance arms. Original four-arm grid retained."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.integer_interactions.learning import fit_map,project,make_kernel,gram,split
from experiments.integer_interactions.study import path,grid,write,sha
ARMS=('local_supervised','local_unsupervised')

def main(a):
    a.out.mkdir(parents=True,exist_ok=False);records=[];t=time.process_time()
    write(a.out/'LOCK.json',{'pilot':a.pilot,'source_sha256':sha(Path(__file__).with_name('learning.py')),'script_sha256':sha(__file__),
        'input_sha256':{k:sha(path(a.data,k)) for k in ('letter','pendigits','satellite')},'arms':ARMS,'scope':'secondary training-only metric diagnosis, no test predictions'})
    with threadpool_limits(1),(a.out/'rows.jsonl').open('x') as log:
        for task in ('letter','pendigits','satellite'):
            data=np.load(path(a.data,task));q=data['q'];y=data['y'];D=int(data['maximum'])
            for seed in ((611,) if a.pilot else (611,977,1543)):
                fit,val=split(q,y,seed,4000 if a.pilot else 6000,1500 if a.pilot else 2000)
                np.savez_compressed(a.out/f'{task}-{seed}-split.npz',fit=fit,validation=val)
                for arm in ARMS:
                    A,rec=fit_map(q[fit],y[fit],D,arm,seed=seed);write(a.out/f'{task}-{seed}-{arm}-metric.json',rec)
                    zt=project(q[fit],A,D);zv=project(q[val],A,D)
                    gammas=((8.,32.) if task=='satellite' else (2.,8.)) if a.pilot else sorted({g for c,g in grid(task)})
                    for g in gammas:
                        k=make_kernel(A,D,g);scratch=a.out/'gram.scratch';start=time.process_time();kt=gram(zt,zt,k,path=scratch);kv=gram(zv,zt,k);kcpu=time.process_time()-start
                        for c in ((10.,) if a.pilot else (1.,10.,100.)):
                            start=time.process_time();m=SVC(C=c,kernel='precomputed',cache_size=128).fit(kt,y[fit]);p=m.predict(kv)
                            r={'task':task,'seed':seed,'arm':arm,'C':c,'gamma':g,'correct':int(np.sum(p==y[val])),
                               'validation_rows':len(val),'fit_rows':len(fit),'supports':len(m.support_),'fit_predict_cpu_seconds':time.process_time()-start,'kernel_cpu_seconds_per_gamma':kcpu}
                            records.append(r);log.write(json.dumps(r)+'\n');log.flush();print(r,flush=True)
                            np.savez_compressed(a.out/f'{task}-{seed}-{arm}-C{c:g}-g{g:g}.npz',prediction=p,expected=y[val])
                        del kt,kv,m;scratch.unlink()
    write(a.out/'cost.json',{'fits':len(records),'cpu_seconds':time.process_time()-t})
    if not a.pilot:
        choices={}
        for task in ('letter','pendigits','satellite'):
            choices[task]={}
            for arm in ARMS:
                candidates=[]
                for i,(c,g) in enumerate(grid(task)):
                    rr=[r for r in records if r['task']==task and r['arm']==arm and r['C']==c and r['gamma']==g];assert len(rr)==3
                    candidates.append((sum(r['correct'] for r in rr)/sum(r['validation_rows'] for r in rr),-sum(r['supports'] for r in rr),-i,c,g))
                best=max(candidates);choices[task][arm]={'C':best[3],'gamma':best[4],'validation_accuracy':best[0]}
        write(a.out/'selected.json',choices)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
