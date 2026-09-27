"""Locked, same-function evaluation on the original subject-separated HAR test set."""
from __future__ import annotations
from array import array
import argparse,hashlib,json,os,platform,random,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False)
def label_hash(labels):return hashlib.sha256(json.dumps(labels,separators=(',',':')).encode()).hexdigest()
def main(a):
 import numpy as np,joblib,sklearn
 from threadpoolctl import threadpool_limits
 from spectra.svm_shared import PreparedModel
 from spectra.svm_receipt import create_receipt,verify_receipt
 a.out.mkdir(parents=True,exist_ok=False)
 native_sources={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'spectra').rglob('*') if p.is_file() and p.suffix in ('.cpp','.hpp','.py')}
 lock={'protocol_commit':'9e8c16829b65e1c32a4c37d3dd08a6753f7480e6','model_freeze_sha256':sha(a.models/'FROZEN_MODELS.json'),
       'source_sha256':sha(__file__),'native_source':native_sources,'library_sha256':sha(a.library),
       'test_source':{n:sha(a.data/n) for n in ('test/X_test.txt','test/y_test.txt','test/subject_test.txt','train/subject_train.txt')},
       'models':{n:sha(a.models/n/'model.joblib') for n in ('rbf_c1','rbf_c10','linear_c1')},
       'selection':'C10 primary, C1 robustness, C1 linear control; no test-selected settings; packets rejected on training probe',
       'repeats':7,'batch_sizes':[1,32],'timing_subset':'256 evenly spaced original test indices','seed':2026092719,
       'python':sys.version,'sklearn':sklearn.__version__,'numpy':np.__version__,'platform':platform.platform(),
       'cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
       'affinity':sorted(os.sched_getaffinity(0)),'scope':'extracted UCI features, NOT raw-sensor feature extraction; warm callable and fresh labels'}
 write(a.out/'EVALUATION_LOCK.json',lock) # Written before any test feature/label loading.
 x=np.loadtxt(a.data/'test/X_test.txt',dtype=np.float64);y=np.loadtxt(a.data/'test/y_test.txt',dtype=np.int64)
 subjects=np.loadtxt(a.data/'test/subject_test.txt',dtype=np.int64);train_sub=np.loadtxt(a.data/'train/subject_train.txt',dtype=np.int64)
 assert x.shape==(2947,561) and y.shape==subjects.shape==(2947,) and np.isfinite(x).all()
 assert set(subjects).isdisjoint(set(train_sub))
 selected=np.linspace(0,len(x)-1,256,dtype=int);probe=np.ascontiguousarray(x[selected]);rng=random.Random(lock['seed'])
 write(a.out/'TEST_INVENTORY.json',{'test_subjects':sorted(map(int,set(subjects))),'training_subjects':sorted(map(int,set(train_sub))),
       'row_indices':selected.tolist(),'labels_sha256':label_hash(y.tolist()),'rows':len(y),'features':561})
 quality={};records=[];fidelity={};receipts=[]
 with threadpool_limits(1),(a.out/'timings.jsonl').open('x') as log:
  for name in ('rbf_c1','rbf_c10','linear_c1'):
   model=joblib.load(a.models/name/'model.joblib') # Trusted locally generated fitting artifact.
   predictions=model.predict(x).tolist()
   quality[name]={'correct':sum(int(t==p) for t,p in zip(y,predictions)),'total':len(y),
     'by_subject':{str(s):{'correct':sum(int(y[i]==predictions[i]) for i in range(len(y)) if subjects[i]==s),
                         'total':int(sum(subjects==s))} for s in sorted(map(int,set(subjects)))},'predictions_sha256':label_hash(predictions)}
   write(a.out/(name+'-predictions.json'),predictions)
   owner=worker=None;arms=['sklearn']
   if name.startswith('rbf'):
    owner=PreparedModel(a.models/name/'model.srt',a.library,input_dtype='float64');worker=owner.session()
    arms+=['native_exhaustive','native_lazy','native_beretta_cert']
   try:
    if worker:
     for mode in ('exhaustive','lazy','beretta_cert'):
      out=worker.predict_buffer(x,schedule=mode)
      if out!=predictions:raise ValueError(f'{name} {mode} fidelity mismatch')
     fidelity[name]={'rows':len(y),'matched_schedules':3,'owner_info':owner.info,'worker_info':worker.info}
     # Numerical replay only on a fixed small subset; do not call this all-row replay.
     for i in np.linspace(0,len(y)-1,8,dtype=int).tolist():
      receipt=create_receipt(worker,x[i].tolist())
      checked=verify_receipt(a.models/name/'model.srt',receipt,expected_input=x[i].tolist(),input_dtype='float64')
      if not checked['verified']:raise ValueError('independent receipt verification failed')
      receipts.append({'model':name,'index':i,'receipt':receipt,'verification':checked})
    wanted=[predictions[int(i)] for i in selected]
    def invoke(arm,batch):
     result=[]
     for start in range(0,len(probe),batch):
      data=probe[start:start+batch]
      if arm=='sklearn':result.extend(model.predict(data).tolist())
      else:result.extend(worker.predict_buffer(data,schedule=arm.removeprefix('native_')))
     return result
    for arm in arms:
     for batch in (1,32):assert invoke(arm,batch)==wanted
    jobs=[(arm,batch) for arm in arms for batch in (1,32)]
    for repeat in range(7):
     order=list(jobs);rng.shuffle(order)
     for arm,batch in order:
      cpu=time.process_time_ns();begin=time.perf_counter_ns();out=invoke(arm,batch);elapsed=time.perf_counter_ns()-begin;cpu=time.process_time_ns()-cpu
      if out!=wanted:raise ValueError('timed output mismatch')
      r={'model':name,'arm':arm,'batch':batch,'repeat':repeat,'rows':len(out),'wall_ns':elapsed,'cpu_ns':cpu,'output_sha256':label_hash(out)}
      records.append(r);log.write(json.dumps(r,separators=(',',':'))+'\n')
   finally:
    if worker:worker.close()
    if owner:owner.close()
   print(name,quality[name]['correct'],'/',len(y),flush=True)
 summary={}
 for name in quality:
  summary[name]={'accuracy':quality[name]['correct']/len(y),'correct':quality[name]['correct'],'total':len(y),'latency_us_per_row':{}}
  for batch in (1,32):
   summary[name]['latency_us_per_row'][str(batch)]={arm:statistics.median(r['wall_ns'] for r in records if r['model']==name and r['batch']==batch and r['arm']==arm)/256/1000 for arm in sorted({r['arm'] for r in records if r['model']==name})}
 write(a.out/'QUALITY.json',quality);write(a.out/'FIDELITY.json',fidelity);write(a.out/'RECEIPTS.json',receipts)
 write(a.out/'SUMMARY.json',{'models':summary,'cells':len(records),'timed_predictions':sum(r['rows'] for r in records),
 'distinct_test_rows':len(y),'native_fidelity_model_input_pairs':len(y)*2,'replayed_receipts':len(receipts),
 'note':'no new runtime promotion; two packing prototypes rejected on training data; same-task windows clustered by subject'})
 print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('data','models','library','out'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});main(a)
