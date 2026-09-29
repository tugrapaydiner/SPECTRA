"""Locked matched selection and refitting: never reads an official test partition."""
from __future__ import annotations
import argparse,hashlib,json,sys,time,pickle,struct,zlib,warnings
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.integer_interactions.learning import fit_map,project,make_kernel,gram,split
ARMS=('uniform','diagonal','full','whitening');SEEDS=(611,977,1543)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def path(data,task): return data/('satellite/data/train.npz' if task=='satellite' else 'datasets/'+task+'-train.npz')
def grid(task): return [(c,g) for c in (1.,10.,100.) for g in ((2.,8.,32.,128.) if task=='satellite' else (.5,2.,8.,32.))]
def sources():return {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}

def select(a):
    a.out.mkdir(parents=True,exist_ok=False);records=[];start=time.process_time()
    write(a.out/'LOCK.json',{'source':sources(),'inputs':{t:sha(path(a.data,t)) for t in ('letter','pendigits','satellite')},
                           'arms':ARMS,'seeds':SEEDS,'grids':{t:grid(t) for t in ('letter','pendigits','satellite')},'scope':'training-only model selection'})
    with threadpool_limits(1),(a.out/'rows.jsonl').open('x') as log:
        for task in ('letter','pendigits','satellite'):
            dat=np.load(path(a.data,task));q=dat['q'];y=dat['y'];D=int(dat['maximum'])
            for seed in SEEDS:
                fit,val=split(q,y,seed);np.savez_compressed(a.out/f'{task}-{seed}-split.npz',fit=fit,validation=val)
                for arm in ARMS:
                    A,rec=fit_map(q[fit],y[fit],D,arm,seed=seed)
                    write(a.out/f'{task}-{seed}-{arm}-metric.json',rec)
                    zt=project(q[fit],A,D);zv=project(q[val],A,D)
                    for g in sorted({g for c,g in grid(task)}):
                        k=make_kernel(A,D,g);scratch=a.out/'gram.scratch';ks=time.process_time()
                        kt=gram(zt,zt,k,path=scratch);kv=gram(zv,zt,k);kernelcost=time.process_time()-ks
                        for c in (1.,10.,100.):
                            clock=time.process_time()
                            with warnings.catch_warnings(record=True) as ws:
                                warnings.simplefilter('always');model=SVC(C=c,kernel='precomputed',cache_size=128).fit(kt,y[fit])
                            pred=model.predict(kv)
                            r={'task':task,'seed':seed,'arm':arm,'C':c,'gamma':g,'correct':int(np.sum(pred==y[val])),
                               'validation_rows':len(val),'fit_rows':len(fit),'supports':len(model.support_),'fit_predict_cpu_seconds':time.process_time()-clock,
                               'kernel_cpu_seconds_per_gamma':kernelcost,'warnings':[str(w.message) for w in ws]}
                            np.savez_compressed(a.out/f'{task}-{seed}-{arm}-C{c:g}-g{g:g}.npz',prediction=pred,expected=y[val])
                            records.append(r);log.write(json.dumps(r)+'\n');log.flush();print(r,flush=True)
                        del model,kt,kv;scratch.unlink()
    choices={}
    for task in ('letter','pendigits','satellite'):
        choices[task]={}
        for arm in ARMS:
            ranked=[]
            for index,(c,g) in enumerate(grid(task)):
                rr=[r for r in records if r['task']==task and r['arm']==arm and r['C']==c and r['gamma']==g]
                assert len(rr)==3
                ranked.append((sum(r['correct'] for r in rr)/sum(r['validation_rows'] for r in rr),-sum(r['supports'] for r in rr),-index,c,g))
            z=max(ranked);choices[task][arm]={'C':z[3],'gamma':z[4],'validation_accuracy':z[0]}
    write(a.out/'selected.json',choices);write(a.out/'cost.json',{'cpu_seconds':time.process_time()-start,'fits':len(records)})

def encode(model,q,A,D,k):
    """Bounded dedicated format with transform + raw support bytes + SVM values."""
    A=np.asarray(A,dtype='<i2');q=np.asarray(q[model.support_],dtype=np.uint8)
    labels=model.classes_.tolist();meta=json.dumps({'labels':labels},separators=(',',':')).encode()
    payload=struct.pack('<d',k.coefficient)+A.tobytes()+q.tobytes()+np.asarray(model.n_support_,dtype='<u4').tobytes()
    payload+=np.asarray(model.dual_coef_,dtype='<f8').tobytes()+np.asarray(model.intercept_,dtype='<f8').tobytes()+meta
    # Fixed header: magic,d,rank,maximum,classes,nsv,bits,meta_bytes,payload_bytes,crc.
    header=struct.pack('<8sIIIIIIIII',b'SPINT001',A.shape[1],len(A),D,len(labels),len(q),k.bits,len(meta),len(payload),zlib.crc32(payload))
    return header+payload

def refit(a):
    a.out.mkdir(parents=True,exist_ok=False);selected=json.loads((a.selection/'selected.json').read_text());records=[];start=time.process_time()
    write(a.out/'FIT_LOCK.json',{'selection_sha256':sha(a.selection/'selected.json'),'source':sources(),'inputs':{t:sha(path(a.data,t)) for t in ('letter','pendigits','satellite')},'scope':'all final fits without official test access'})
    with threadpool_limits(1):
        for task in ('letter','pendigits','satellite'):
            dat=np.load(path(a.data,task));q=dat['q'];y=dat['y'];D=int(dat['maximum'])
            for arm in ARMS:
                folder=a.out/(task+'-'+arm);folder.mkdir()
                A,rec=fit_map(q,y,D,arm);write(folder/'metric.json',rec)
                z=project(q,A,D);choice=selected[task][arm];k=make_kernel(A,D,choice['gamma']);scratch=a.out/'gram.scratch'
                clock=time.process_time();kt=gram(z,z,k,path=scratch);kcost=time.process_time()-clock
                clock=time.process_time();model=SVC(C=choice['C'],kernel='precomputed',cache_size=128).fit(kt,y);fcost=time.process_time()-clock
                raw=encode(model,q,A,D,k);(folder/'model.sik').write_bytes(raw)
                (folder/'reference.pkl').write_bytes(pickle.dumps(model,protocol=4))
                np.savez_compressed(folder/'reference.npz',A=A,raw_supports=q[model.support_],support_index=model.support_,coefficients=model.dual_coef_,
                      intercepts=model.intercept_,counts=model.n_support_,classes=model.classes_,maximum=D,coefficient=k.coefficient,bits=k.bits,high=k.high,low=k.low,bound=k.bound)
                r={'task':task,'arm':arm,**choice,'metric_cpu_seconds':rec['cpu_seconds'],'kernel_cpu_seconds':kcost,'fit_cpu_seconds':fcost,
                   'training_rows':len(q),'supports':len(model.support_),'features':q.shape[1],'rank':len(A),'model_bytes':len(raw),
                   'table_entries':len(k.high)+len(k.low),'full_table_entries':k.bound+1,'signature_bound':k.bound,'model_sha256':sha(folder/'model.sik')}
                write(folder/'fit.json',r);records.append(r);print(r,flush=True)
                del model,kt,z;scratch.unlink()
    write(a.out/'FINAL_LOCK.json',{'source':sources(),'files':{p.relative_to(a.out).as_posix():sha(p) for p in a.out.rglob('*') if p.is_file()},'fits':records,'cpu_seconds':time.process_time()-start,'scope':'all12 final models fixed before any official test prediction'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['select','refit']);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--selection',type=Path);a=p.parse_args();(select if a.mode=='select' else refit)(a)
