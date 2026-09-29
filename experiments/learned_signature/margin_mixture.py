"""Training-only simplex kernel learning using the summed binary SVC dual objective.

A projected-gradient experiment using existing SVC solves, not a claim of a new
optimizer or proof of global convergence under finite numerical tolerances.
"""
from __future__ import annotations
import argparse,json,sys,time,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from experiments.learned_signature.learning import *


def project_simplex(v):
 v=np.asarray(v,dtype=np.float64)
 if v.ndim!=1 or not np.isfinite(v).all() or not len(v):raise ValueError('finite vector required')
 u=np.sort(v)[::-1];t=(np.cumsum(u)-1.)/np.arange(1,len(v)+1)
 rho=np.flatnonzero(u>t)[-1];w=np.maximum(v-t[rho],0.);return w/w.sum()


def dual_vectors(model):
 """Actual pairwise signed coefficients, located in the training matrix."""
 starts=np.cumsum(np.r_[0,model.n_support_]);out=[]
 for i in range(len(model.classes_)):
  for j in range(i+1,len(model.classes_)):
   pos=np.r_[np.arange(starts[i],starts[i+1]),np.arange(starts[j],starts[j+1])]
   coeff=np.r_[model.dual_coef_[j-1,starts[i]:starts[i+1]],model.dual_coef_[i,starts[j]:starts[j+1]]]
   mask=coeff!=0;out.append((model.support_[pos[mask]],coeff[mask]))
 return out


def fixed_dual_objective(kernels,weights,pairs):
 norms=np.zeros(len(kernels),dtype=np.float64);linear=0.
 for indices,coefficients in pairs:
  linear+=float(np.abs(coefficients).sum())
  for m,K in enumerate(kernels):norms[m]+=float(coefficients@K[np.ix_(indices,indices)]@coefficients)
 return float(linear-.5*np.dot(weights,norms)),-.5*norms


def learn_margin_mixture(q,y,cap,gamma,C,seed):
 start=time.perf_counter();cpu=time.process_time();idx=stratified_subset(y,2048,seed+71)
 q,y=q[idx],y[idx];w=np.ones(q.shape[1],dtype=np.uint32);sig=signatures(q,q,w)
 kernels=[table_profile(cap,w,gamma*s)[sig] for s in RADIAL_SCALES]
 weights=np.ones(len(kernels))/len(kernels);traces=[];fits=0;fit_seconds=0.
 def solve(a):
  nonlocal fits,fit_seconds
  K=np.zeros_like(kernels[0])
  for v,base in zip(a,kernels):K+=v*base
  t=time.process_time()
  with warnings.catch_warnings(record=True) as warnings_seen:
   warnings.simplefilter('always');model=SVC(C=C,kernel='precomputed',cache_size=64,tol=1e-3).fit(K,y)
  elapsed=time.process_time()-t;fit_seconds+=elapsed;fits+=1
  objective,gradient=fixed_dual_objective(kernels,a,dual_vectors(model))
  return objective,gradient,len(model.support_),[str(v.message) for v in warnings_seen]
 objective,gradient,ns,warn=solve(weights)
 traces.append({'iteration':-1,'objective':objective,'weights':weights.tolist(),'supports':ns,'warnings':warn,'accepted':True})
 for iteration in range(8):
  scale=max(float(np.max(np.abs(gradient))),1e-20);accepted=False
  for backtrack in range(4):
   step=.5/(2**backtrack);proposed=project_simplex(weights-step*gradient/scale)
   if np.max(np.abs(proposed-weights))<1e-10:break
   obj,grad,ns,warn=solve(proposed)
   ok=obj<=objective+1e-8*max(1.,abs(objective))
   traces.append({'iteration':iteration,'backtrack':backtrack,'step':step,'objective':obj,'weights':proposed.tolist(),'supports':ns,'warnings':warn,'accepted':bool(ok)})
   if ok:weights,objective,gradient=proposed,obj,grad;accepted=True;break
  if not accepted:break
 return weights,{'rows':len(y),'seed':seed,'fits':fits,'fit_cpu':fit_seconds,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start,'trace':traces}


def develop(a):
 a.out.mkdir(parents=True,exist_ok=False);q,y,cap=load_part(a.data,a.task);fit,valid=split_indices(y,a.seed)
 qf,yf,qv,yv=q[fit],y[fit],q[valid],y[valid];gamma=base_gamma(qf,cap);w=np.ones(q.shape[1],dtype=np.uint32)
 sig=signatures(qf,qf,w);vsig=signatures(qv,qf,w);rows=[]
 write_json(a.out/'source.json',{'seed':a.seed,'task':a.task,'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'data_sha256':sha(a.data/(a.task+'.zip'))})
 with (a.out/'rows.jsonl').open('x') as log:
  for gi,mult in enumerate(MULTIPLIERS):
   for ci,C in enumerate(C_VALUES):
    coef,learning=learn_margin_mixture(qf,yf,cap,gamma*mult,C,a.seed)
    table=table_profile(cap,w,gamma*mult,coef,RADIAL_SCALES);K=table[sig];V=table[vsig]
    start=time.perf_counter();cpu=time.process_time()
    with warnings.catch_warnings(record=True) as warn:warnings.simplefilter('always');model=SVC(C=C,kernel='precomputed',cache_size=128).fit(K,yf)
    cost=time.process_time()-cpu;wall=time.perf_counter()-start;pred=model.predict(V);correct=int(np.sum(pred==yv))
    row={'task':a.task,'family':'margin','seed':a.seed,'C':C,'multiplier':mult,'grid_index':gi*3+ci,'correct':correct,'rows':len(yv),'accuracy':correct/len(yv),'supports':len(model.support_),'weights':w.tolist(),'mixture':coef.tolist(),'fit_cpu':cost,'fit_wall':wall,'kernel_learning_cpu':learning['cpu_seconds'],'inner_fits':learning['fits'],'warning':[str(v.message) for v in warn]}
    rows.append(row);log.write(json.dumps(row)+'\n');log.flush();write_json(a.out/f'learning-{gi}-{ci}.json',learning)
    print(a.task,a.seed,gi,ci,correct,len(yv),'mixture',coef.round(3),'learningCPU',round(learning['cpu_seconds'],2),flush=True)
    del model,K,V
 write_json(a.out/'summary.json',max(rows,key=lambda r:(r['correct'],-r['supports'],-r['grid_index'])))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--task',choices=['letter','pendigits'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 with threadpool_limits(1):develop(a)
