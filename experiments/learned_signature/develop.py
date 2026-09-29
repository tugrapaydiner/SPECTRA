"""Complete fixed training-only selection; no evaluation-part access."""
from __future__ import annotations
import argparse,gc,json,time,warnings,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from experiments.learned_signature.learning import *

def main(a):
 a.out.mkdir(parents=True,exist_ok=False);q,y,cap=load_part(a.data,a.task);fit,valid=split_indices(y,a.seed)
 qf,yf,qv,yv=q[fit],y[fit],q[valid],y[valid];np.savez(a.out/'split.npz',fit=fit,validation=valid)
 gamma=base_gamma(qf,cap);weights,metric=learn_metric(qf,yf,cap,a.seed);write_json(a.out/'metric.json',metric)
 families=['rbf','metric','alignment','uniform','metric_alignment']
 write_json(a.out/'source.json',{'task':a.task,'seed':a.seed,'development_rows':len(q),'fit_rows':len(fit),'validation_rows':len(valid),'cap':cap,'gamma_base':gamma,'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'data_sha256':sha(a.data/(a.task+'.zip')),'families':families})
 all_rows=[]
 with (a.out/'rows.jsonl').open('x') as log:
  for family in families:
   w=weights if family.startswith('metric') else np.ones(q.shape[1],dtype=np.uint32)
   sig=signatures(qf,qf,w);vsig=signatures(qv,qf,w)
   for index,mult in enumerate(MULTIPLIERS):
    t=time.process_time()
    if family in ('alignment','metric_alignment'):coeff,diag=learn_mixture(qf,yf,cap,w,gamma*mult,a.seed);scales=RADIAL_SCALES
    elif family=='uniform':coeff=np.ones(len(RADIAL_SCALES))/len(RADIAL_SCALES);scales=RADIAL_SCALES
    else:coeff=np.array([1.]);scales=(1.,)
    setup=time.process_time()-t;table=table_profile(cap,w,gamma*mult,coeff,scales);K=table[sig];Kv=table[vsig]
    for ci,C in enumerate(C_VALUES):
     start,cpu=time.perf_counter(),time.process_time();model=SVC(C=C,kernel='precomputed',cache_size=128.,tol=1e-3,shrinking=True)
     with warnings.catch_warnings(record=True) as warn:warnings.simplefilter('always');model.fit(K,yf)
     wall,cost=time.perf_counter()-start,time.process_time()-cpu;pred=model.predict(Kv);correct=int(np.sum(pred==yv));ns=len(model.support_)
     row={'task':a.task,'seed':a.seed,'family':family,'C':C,'multiplier':mult,'grid_index':index*3+ci,'correct':correct,'rows':len(yv),'accuracy':correct/len(yv),'supports':ns,'fit_cpu':cost,'fit_wall':wall,'kernel_learning_cpu':setup if ci==0 else 0.,'mixture':coeff.tolist(),'weights':w.tolist(),'warning':[str(v.message) for v in warn]}
     all_rows.append(row);log.write(json.dumps(row)+'\n');log.flush();print(f'{a.task} {a.seed} {family} C={C:g} g={mult:g} {correct}/{len(yv)} sv={ns} fit={cost:.2f}s',flush=True);del model,pred
    del table,K,Kv;gc.collect()
   del sig,vsig
  xf=qf.astype(np.float64)/cap;xv=qv.astype(np.float64)/cap
  for C in (.1,1.,10.):
   model=LinearSVC(C=C,max_iter=20000,random_state=a.seed);cpu=time.process_time();start=time.perf_counter()
   with warnings.catch_warnings(record=True) as warn:warnings.simplefilter('always');model.fit(xf,yf)
   cost=time.process_time()-cpu;wall=time.perf_counter()-start;correct=int(np.sum(model.predict(xv)==yv))
   row={'task':a.task,'seed':a.seed,'family':'linear','C':C,'correct':correct,'rows':len(yv),'accuracy':correct/len(yv),'fit_cpu':cost,'fit_wall':wall,'warning':[str(v.message) for v in warn]};all_rows.append(row);log.write(json.dumps(row)+'\n');log.flush();print(row,flush=True)
  model=MLPClassifier(hidden_layer_sizes=(64,64),max_iter=300,early_stopping=True,validation_fraction=.15,n_iter_no_change=25,learning_rate_init=.003,alpha=1e-4,batch_size=128,random_state=a.seed);cpu=time.process_time();start=time.perf_counter()
  with warnings.catch_warnings(record=True) as warn:warnings.simplefilter('always');model.fit(xf,yf)
  cost=time.process_time()-cpu;wall=time.perf_counter()-start;correct=int(np.sum(model.predict(xv)==yv))
  row={'task':a.task,'seed':a.seed,'family':'mlp','correct':correct,'rows':len(yv),'accuracy':correct/len(yv),'fit_cpu':cost,'fit_wall':wall,'epochs':int(model.n_iter_),'warning':[str(v.message) for v in warn]};all_rows.append(row);log.write(json.dumps(row)+'\n');log.flush();print(row,flush=True)
 summary={f:max([r for r in all_rows if r['family']==f],key=lambda r:(r['correct'],-r.get('supports',0),-r.get('grid_index',0))) for f in families+['linear','mlp']};write_json(a.out/'summary.json',summary)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--task',choices=['letter','pendigits'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 with threadpool_limits(1):main(a)
