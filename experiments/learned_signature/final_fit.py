"""One final fit per frozen family/task; deliberately does not load test rows."""
from __future__ import annotations
import argparse,gc,json,pickle,sys,time,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from experiments.learned_signature.learning import *
from experiments.learned_signature.model import from_precomputed
from experiments.learned_signature.pair_experts import fit_pair_experts


def fit(a):
 a.out.mkdir(parents=True,exist_ok=False)
 selected=json.loads(a.selection.read_text())['tasks'][a.task]
 q,y,cap=load_part(a.data,a.task);weights,metric=learn_metric(q,y,cap,1401);gamma=base_gamma(q,cap)
 write_json(a.out/'metric.json',metric)
 write_json(a.out/'source.json',{'task':a.task,'selection_sha256':sha(a.selection),'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'data_sha256':sha(a.data/(a.task+'.zip')),'rows':len(q),'gamma_base':gamma})
 reports=[]
 for family in ('rbf','metric','alignment','uniform','metric_alignment'):
  folder=a.out/family;folder.mkdir();choice=selected[family]['chosen'];start=time.perf_counter();cpu=time.process_time()
  w=weights if family.startswith('metric') else np.ones(q.shape[1],dtype=np.uint32)
  if family in ('alignment','metric_alignment'):coef,diag=learn_mixture(q,y,cap,w,gamma*choice['multiplier'],1401);scales=RADIAL_SCALES
  elif family=='uniform':coef=np.ones(len(RADIAL_SCALES))/len(RADIAL_SCALES);diag={};scales=RADIAL_SCALES
  else:coef=np.array([1.]);diag={};scales=(1.,)
  table=table_profile(cap,w,gamma*choice['multiplier'],coef,scales)
  # One C-contiguous binary64 training matrix on disk; no full-size signature copy.
  kernel_file=folder/'training-kernel.tmp'
  K=np.memmap(kernel_file,mode='w+',dtype=np.float64,shape=(len(q),len(q)))
  for first in range(0,len(q),128):K[first:first+128]=table[signatures(q[first:first+128],q,w)]
  K.flush();setup_cpu=time.process_time()-cpu
  model=SVC(C=choice['C'],kernel='precomputed',cache_size=128,tol=1e-3,shrinking=True)
  start_fit=time.process_time()
  with warnings.catch_warnings(record=True) as captured:warnings.simplefilter('always');model.fit(K,y)
  fit_cpu=time.process_time()-start_fit
  metadata={'family':family,'task':a.task,'domain_max':cap,'C':choice['C'],'gamma':gamma*choice['multiplier'],'gamma_multiplier':choice['multiplier'],'mixture':coef.tolist(),'scales':list(scales),'definition':'authoritative finite binary64 radial table','selection_sha256':sha(a.selection)}
  info=from_precomputed(model,q,w,table,folder/'model.lkt',metadata)
  (folder/'model.pkl').write_bytes(pickle.dumps(model,protocol=4)) # own trusted evaluation comparator, not deployment
  np.save(folder/'table.npy',table);np.save(folder/'weights.npy',w)
  row={'family':family,'task':a.task,'configuration':choice,'model':info,'setup_cpu':setup_cpu,'fit_cpu':fit_cpu,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start,'warnings':[str(v.message) for v in captured],'mixture_learning':diag}
  write_json(folder/'fit.json',row);reports.append(row);print(a.task,family,info,'fit_cpu',fit_cpu,flush=True)
  del model,K,table;gc.collect();kernel_file.unlink()
 # Exactly one full-data fit using the fixed nested rule, not chosen on final results.
 report=fit_pair_experts(q,y,cap,1401,a.out/'pair_experts');reports.append({'family':'pair_experts','task':a.task,**report})
 xf=q.astype(np.float64)/cap
 for family in ('linear','mlp'):
  folder=a.out/family;folder.mkdir()
  if family=='linear':model=LinearSVC(C=selected[family]['chosen']['C'],max_iter=20000,random_state=1401)
  else:model=MLPClassifier(hidden_layer_sizes=(64,64),max_iter=300,early_stopping=True,validation_fraction=.15,n_iter_no_change=25,learning_rate_init=.003,alpha=1e-4,batch_size=128,random_state=1401)
  cpu=time.process_time();start=time.perf_counter()
  with warnings.catch_warnings(record=True) as captured:warnings.simplefilter('always');model.fit(xf,y)
  raw=pickle.dumps(model,protocol=4);(folder/'model.pkl').write_bytes(raw)
  # Save an inert numerical representation too; no pickle needed by a native control.
  if family=='linear':np.savez(folder/'weights.npz',coefficients=model.coef_,intercept=model.intercept_,labels=model.classes_)
  else:np.savez(folder/'weights.npz',**{f'W{i}':w for i,w in enumerate(model.coefs_)},**{f'b{i}':b for i,b in enumerate(model.intercepts_)},labels=model.classes_)
  row={'family':family,'task':a.task,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start,'sha256':sha(folder/'model.pkl'),'pickle_bytes':len(raw),'epochs':int(getattr(model,'n_iter_',0)),'warnings':[str(v.message) for v in captured]}
  reports.append(row);write_json(folder/'fit.json',row);print(row,flush=True)
 write_json(a.out/'summary.json',reports)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('data','selection','out'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--task',choices=['letter','pendigits'],required=True);a=p.parse_args()
 with threadpool_limits(1):fit(a)
