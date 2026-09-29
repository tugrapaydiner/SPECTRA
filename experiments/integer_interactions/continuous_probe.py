"""Retained pilot diagnostic: did integer projection cause full-NCA degradation?

Reuse the already learned continuous factor; no new metric fit or test access.
Fit the two fixed C10 pilot RBFs and report every validation result.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.integer_interactions.study import path,write,sha
p=argparse.ArgumentParser()
for name in ('data','pilot','out'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
write(a.out/'LOCK.json',{'pilot_lock':sha(a.pilot/'LOCK.json'),'source':sha(__file__),'scope':'post-pilot training/validation diagnostic, no official test or revised metric'})
records=[];cpu=time.process_time()
with threadpool_limits(1):
 for task in ('letter','pendigits','satellite'):
  data=np.load(path(a.data,task));q=data['q'];y=data['y'];D=int(data['maximum'])
  split=np.load(a.pilot/(task+'-split.npz'));fit=split['fit'];val=split['validation']
  for arm in ('diagonal','full'):
   rec=json.loads((a.pilot/f'{task}-{arm}-metric.json').read_text());B=np.array(rec['continuous_factor']);A=np.r_[8*B,np.eye(B.shape[1])]
   z=q.astype(float)@A.T;norm=float(np.sum(A*A))/q.shape[1]
   dt=cdist(z[fit],z[fit],'sqeuclidean');dv=cdist(z[val],z[fit],'sqeuclidean')
   for g in ((8.,32.) if task=='satellite' else (2.,8.)):
    alpha=g/(D*D*norm);kt=np.exp(-alpha*dt);kv=np.exp(-alpha*dv)
    m=SVC(C=10,kernel='precomputed',cache_size=128).fit(kt,y[fit]);pred=m.predict(kv)
    r={'task':task,'arm':arm,'gamma':g,'correct':int(np.sum(pred==y[val])),'rows':len(val),'supports':len(m.support_)}
    records.append(r);np.savez_compressed(a.out/f'{task}-{arm}-g{g:g}.npz',prediction=pred,expected=y[val]);print(r,flush=True)
write(a.out/'results.json',{'records':records,'cpu_seconds':time.process_time()-cpu,'fits':len(records)})
