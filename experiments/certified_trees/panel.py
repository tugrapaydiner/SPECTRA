"""New fixed source models, not reconstruction of the lost tree pilot.
All four CatBoost exports precede evaluation; no test-based parameter selection.
"""
from __future__ import annotations
import argparse,hashlib,json,time,sys
from pathlib import Path
TASKS=('letter','pendigits','satellite','optdigits')
PARAMS=dict(iterations=256,depth=6,learning_rate=.08,l2_leaf_reg=3,bootstrap_type='No',
 random_strength=0,random_seed=20260929,thread_count=1,loss_function='MultiClass',
 allow_writing_files=False,task_type='CPU',verbose=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def fit(data,out):
 import numpy as np,catboost
 from catboost import CatBoostClassifier
 from threadpoolctl import threadpool_limits
 if catboost.__version__!='1.2.8':raise ValueError('pinned CatBoost1.2.8 required')
 out.mkdir(parents=True,exist_ok=False)
 write(out/'FIT_LOCK.json',dict(tasks=TASKS,params=PARAMS,script_sha256=sha(__file__),
  catboost=catboost.__version__,numpy=np.__version__,python=sys.version,
  training={t:sha(data/(t+'-train.npz')) for t in TASKS},scope='new compatibility panel, not old weights or fresh holdout'))
 records=[]
 with threadpool_limits(1):
  for task in TASKS:
   folder=out/task;folder.mkdir()
   with np.load(data/(task+'-train.npz')) as f:q,y,D=f['q'],f['y'],int(f['maximum'])
   if q.dtype!=np.uint8 or q.ndim!=2 or (q>D).any():raise ValueError('original uint8 codes required')
   m=CatBoostClassifier(**PARAMS);cpu,wall=time.process_time(),time.perf_counter();m.fit(q,y)
   fcpu,fwall=time.process_time()-cpu,time.perf_counter()-wall
   for fmt in ('cbm','json','cpp'):m.save_model(str(folder/('source.'+fmt)),format=fmt)
   rec=dict(task=task,train_rows=len(q),features=q.shape[1],maximum=D,classes=m.classes_.tolist(),
     cpu_seconds=fcpu,wall_seconds=fwall,tree_count=m.tree_count_,params=m.get_all_params(),
     files={p.name:sha(p) for p in folder.iterdir()})
   write(folder/'fit.json',rec);records.append(rec);print(task,'fitted',fcpu,flush=True)
 write(out/'FINAL_LOCK.json',dict(fits=records,files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()},scope='all four models fixed before test prediction'))
def evaluate(data,models,out):
 import numpy as np
 from catboost import CatBoostClassifier
 from sklearn.metrics import f1_score,confusion_matrix
 from threadpoolctl import threadpool_limits
 lock=json.loads((models/'FINAL_LOCK.json').read_text())
 for p,h in lock['files'].items():
  if sha(models/p)!=h:raise ValueError('frozen source changed')
 out.mkdir(parents=True,exist_ok=False);write(out/'OPENING.json',dict(lock_sha256=sha(models/'FINAL_LOCK.json'),tests={t:sha(data/(t+'-test.npz')) for t in TASKS},script_sha256=sha(__file__),scope='previously exposed benchmarks; fixed new models'))
 reports={}
 with threadpool_limits(1):
  for t in TASKS:
   with np.load(data/(t+'-test.npz')) as f:q,y=f['q'],f['y']
   m=CatBoostClassifier().load_model(str(models/t/'source.cbm'))
   scores=m.predict(q,prediction_type='RawFormulaVal',thread_count=1);idx=np.argmax(scores,axis=1);pred=m.classes_[idx]
   if not np.array_equal(pred,m.predict(q,thread_count=1).ravel()):raise ValueError('class convention mismatch')
   np.savez_compressed(out/(t+'.npz'),q=q,y=y,prediction=pred,indices=idx,scores=scores)
   reports[t]=dict(correct=int(np.sum(pred==y)),rows=len(y),accuracy=float(np.mean(pred==y)),macro_f1=float(f1_score(y,pred,average='macro')),confusion=confusion_matrix(y,pred).tolist(),source_sha256=sha(models/t/'source.json'),file_sha256=sha(out/(t+'.npz')))
   print(t,reports[t]['correct'],'/',len(y),flush=True)
 write(out/'quality.json',reports)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','evaluate'])
 for n in ('data','out','models'):p.add_argument('--'+n,type=Path,required=n!='models')
 a=p.parse_args()
 if a.mode=='fit':fit(a.data,a.out)
 else:evaluate(a.data,a.models,a.out)
