"""Training-partition-only diagnostic. Never reads an official test file."""
from __future__ import annotations
import argparse,hashlib,json,sys,time,os
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.integer_interactions.learning import fit_map,project,make_kernel,gram,split

def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    paths={'letter':a.data/'datasets/letter-train.npz','pendigits':a.data/'datasets/pendigits-train.npz','satellite':a.data/'satellite/data/train.npz'}
    write(a.out/'LOCK.json',{'data_sha256':{t:hashlib.sha256(p.read_bytes()).hexdigest() for t,p in paths.items()},
        'source_sha256':hashlib.sha256(Path(__file__).with_name('learning.py').read_bytes()).hexdigest(),
        'fit_cap':4000,'val_cap':1500,'seed':611,'arms':['uniform','diagonal','full','whitening'],'C':10,'scope':'development-only pilot; no test reads'})
    total=time.process_time();records=[]
    with threadpool_limits(1),(a.out/'rows.jsonl').open('x') as f:
        for task,path in paths.items():
            dat=np.load(path);q=dat['q'];y=dat['y'];D=int(dat['maximum']);fit,val=split(q,y,611,4000,1500)
            np.savez_compressed(a.out/(task+'-split.npz'),fit=fit,validation=val)
            for arm in ('uniform','diagonal','full','whitening'):
                A,rec=fit_map(q[fit],y[fit],D,arm,seed=611)
                write(a.out/f'{task}-{arm}-metric.json',rec)
                zt=project(q[fit],A,D);zv=project(q[val],A,D)
                for g in ((8.,32.) if task=='satellite' else (2.,8.)):
                    k=make_kernel(A,D,g);start=time.process_time();kt=gram(zt,zt,k);kv=gram(zv,zt,k)
                    model=SVC(C=10,kernel='precomputed',cache_size=128).fit(kt,y[fit]);pred=model.predict(kv)
                    r={'task':task,'arm':arm,'gamma':g,'C':10,'correct':int(np.sum(pred==y[val])),'rows':len(val),'supports':len(model.support_),
                       'cpu_seconds':time.process_time()-start,'metric_cpu_seconds':rec['cpu_seconds'],'table_entries':len(k.high)+len(k.low),'converged':rec.get('success')}
                    records.append(r);f.write(json.dumps(r)+'\n');f.flush();print(r,flush=True)
                    np.savez_compressed(a.out/f'{task}-{arm}-g{g:g}.npz',prediction=pred,expected=y[val])
                    del model,kt,kv
    write(a.out/'cost.json',{'cpu_seconds':time.process_time()-total,'fits':len(records)})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
