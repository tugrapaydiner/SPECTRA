"""Select the additional margin family by three-split means, then fit it once."""
from __future__ import annotations
import argparse,gc,json,pickle,statistics,sys,time,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from experiments.learned_signature.learning import *
from experiments.learned_signature.margin_mixture import learn_margin_mixture
from experiments.learned_signature.model import from_precomputed
from experiments.learned_signature.dense import export


def finish(a):
 selection=json.loads(a.selection.read_text());selection['secondary_amendment']='MARGIN_AMENDMENT.md';sources={}
 for task in ('letter','pendigits'):
  grouped={};all_rows=[]
  for seed in (1401,2402,3403):
   path=a.results/f'{task}-{seed}-margin/rows.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
   if len(rows)!=21 or not (path.parent/'summary.json').exists():raise ValueError('margin development incomplete')
   sources[str(path.relative_to(a.results))]=sha(path)
   for row in rows:grouped.setdefault((row['C'],row['multiplier']),[]).append(row)
   all_rows.extend(rows)
  configs=[]
  for (C,mult),rs in grouped.items():
   if sorted(r['seed'] for r in rs)!=[1401,2402,3403]:raise ValueError('missing split')
   configs.append({'C':C,'multiplier':mult,'mean_accuracy':statistics.mean(r['accuracy'] for r in rs),'mean_supports':statistics.mean(r['supports'] for r in rs),'grid_index':rs[0]['grid_index'],'split_accuracy':{str(r['seed']):r['accuracy'] for r in rs}})
  chosen=max(configs,key=lambda v:(v['mean_accuracy'],-v['mean_supports'],-v['grid_index']))
  selection['tasks'][task]['margin']={'chosen':chosen,'all_configurations':configs,'total_fit_cpu':sum(r['fit_cpu'] for r in all_rows),'total_kernel_learning_cpu':sum(r['kernel_learning_cpu'] for r in all_rows),'inner_fits':sum(r['inner_fits'] for r in all_rows),'validation_gain_pp':100*(chosen['mean_accuracy']-selection['tasks'][task]['rbf']['chosen']['mean_accuracy'])}
 selection['secondary_sources']=sources
 write_json(a.out,selection)
 for task in ('letter','pendigits'):
  choice=selection['tasks'][task]['margin']['chosen'];folder=a.models/task/'margin';folder.mkdir(parents=True,exist_ok=False)
  q,y,cap=load_part(a.data,task);w=np.ones(q.shape[1],dtype=np.uint32);gamma=base_gamma(q,cap)*choice['multiplier'];cpu=time.process_time();wall=time.perf_counter()
  coef,learning=learn_margin_mixture(q,y,cap,gamma,choice['C'],1401);table=table_profile(cap,w,gamma,coef,RADIAL_SCALES)
  file=folder/'training-kernel.tmp';K=np.memmap(file,mode='w+',dtype=np.float64,shape=(len(q),len(q)))
  for first in range(0,len(q),128):K[first:first+128]=table[signatures(q[first:first+128],q,w)]
  K.flush();setup=time.process_time()-cpu;fit_start=time.process_time()
  with warnings.catch_warnings(record=True) as warn:warnings.simplefilter('always');model=SVC(C=choice['C'],kernel='precomputed',cache_size=128,tol=1e-3).fit(K,y)
  fit_cpu=time.process_time()-fit_start
  info=from_precomputed(model,q,w,table,folder/'model.lkt',{'family':'margin','task':task,'domain_max':cap,'gamma':gamma,'mixture':coef.tolist(),'scales':list(RADIAL_SCALES),'C':choice['C'],'selection_sha256':sha(a.out),'definition':'authoritative finite binary64 table'})
  (folder/'model.pkl').write_bytes(pickle.dumps(model,protocol=4));np.save(folder/'weights.npy',w);np.save(folder/'table.npy',table)
  row={'family':'margin','task':task,'model':info,'configuration':choice,'setup_cpu':setup,'fit_cpu':fit_cpu,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall,'kernel_learning':learning,'warnings':[str(v.message) for v in warn]}
  write_json(folder/'fit.json',row);print(task,choice,coef,info,flush=True);del model,K,table;gc.collect();file.unlink()
 # Export the already-fitted small controls without another training call.
 for task in ('letter','pendigits'):
  cap=15 if task=='letter' else 100
  for family in ('linear','mlp'):
   folder=a.models/task/family
   model=pickle.loads((folder/'model.pkl').read_bytes());print(task,family,export(model,folder/'control.json',cap),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('data','results','models','selection','out'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args()
 with threadpool_limits(1):finish(a)
