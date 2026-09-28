"""Deliberately aligned 16-way parity diagnostic, separate from natural tasks."""
from __future__ import annotations
import argparse,json,pickle,time,sys
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.learned_finite.learning import distances,table,align_tables,spectral_tables,GAMMA_MULTIPLIERS,CS
from experiments.learned_finite.model import encode
from experiments.learned_finite.train import fit_one
from experiments.learned_finite.data import sha,write_json

def train(out):
    out.mkdir(parents=True,exist_ok=False);start=time.process_time();d=16
    x=((np.arange(65536,dtype=np.uint32)[:,None]>>np.arange(d))&1).astype(np.uint8);y=(x.sum(1)%2).astype(int)
    chosen=np.random.default_rng(20260928).choice(65536,192,replace=False)
    tr=chosen[:128];va=chosen[128:];te=np.setdiff1d(np.arange(65536),chosen)
    np.savez_compressed(out/'indices.npz',fit=tr,validation=va,test=te)
    D=distances(x[tr],x[tr],np.ones(d));VD=distances(x[va],x[tr],np.ones(d));scales=GAMMA_MULTIPLIERS/np.median(D[D>0])
    orders=[1,2,3,4,6,8,12,16];radial=np.stack([table([g],[1.],d) for g in scales])
    lut,info=align_tables(np.concatenate((spectral_tables(d,orders),radial)),D,y[tr]);info['orders']=orders
    options={'rbf':[(t,{'gamma':float(g)}) for t,g in zip(radial,scales)],'spectral':[(lut,info)]}
    result=[]
    with threadpool_limits(1):
      for family,choices in options.items():
        best=None;attempts=[]
        for lut,parameters in choices:
          for C in CS:
            model=SVC(kernel='precomputed',C=C);cpu,log=fit_one(model,np.ascontiguousarray(lut[D]),y[tr])
            correct=int((model.predict(np.ascontiguousarray(lut[VD]))==y[va]).sum())
            attempts.append({'parameters':{**parameters,'C':C},'validation_correct':correct,'supports':len(model.support_),'cpu_seconds':cpu,'warnings':log})
            rank=correct,-len(model.support_)
            if best is None or rank>best[0]:best=rank,model,lut,parameters,C
        _,model,lut,parameters,C=best
        raw=encode(x[tr][model.support_],[0,1],np.ones(d,np.uint8),lut,model.n_support_,model.dual_coef_,model.intercept_)
        (out/(family+'.lfk')).write_bytes(raw)
        (out/(family+'.pkl')).write_bytes(pickle.dumps({'model':model,'fit_indices':tr,'table':lut},protocol=4))
        result.append({'family':family,'selected':{**parameters,'C':C},'attempts':attempts,'model_sha256':sha(out/(family+'.lfk')),'bytes':len(raw)})
    write_json(out/'FROZEN.json',{'models':result,'cpu_seconds':time.process_time()-start,
        'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'scope':'128 fit +64 validation vertices only; all other65344 vertices reserved'})
    print(json.dumps({'selection':[(r['family'],r['selected']) for r in result],'cpu_seconds':time.process_time()-start},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();train(a.out)
