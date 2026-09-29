"""One locked Satellite evaluation, complete original-margin check and timing.

No training in this script. Existing original partitions and every fixed control
are retained, without test selection. Spatial independence cannot be reconstructed.
"""
from __future__ import annotations
from contextlib import ExitStack
import argparse,ctypes as C,hashlib,json,os,pickle,platform,random,statistics,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.learned_metric.learning import kernel_matrix
from experiments.learned_metric.session import MetricSession
from experiments.learned_metric.controls import ControlSession,build as build_control
from experiments.learned_metric.qda_control import build as build_qda
from experiments.learned_metric.evaluate import Reference,expanded_model,summarize,differences,sha,write
ARMS=('uniform_compiled','variance_compiled','nca_compiled','nca_scalar_integer','nca_scalar_exp','nca_exhaustive','linear_native','mlp_native','qda_native','uniform_sklearn','nca_sklearn')

def main(a):
 a.out.mkdir(parents=True,exist_ok=False);lock=json.loads((a.models/'FINAL_LOCK.json').read_text())
 for f,h in lock['files'].items():
  if sha(a.models/f)!=h:raise ValueError('Satellite model modified')
 controls=a.out/'controls'
 for arm in ('linear','mlp'):build_control(a.models/arm,controls/arm)
 build_qda(a.models/'qda',controls/'qda')
 write(a.out/'TEST_OPENING.json',{'final_lock_sha256':sha(a.models/'FINAL_LOCK.json'),'test_sha256':sha(a.data/'test.npz'),
  'source':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'library_sha256':sha(a.library),'reference_library_sha256':sha(a.reference_library),
  'remote_preacquisition_protocol':'4d151102482ef8f90b8dc2fb09b2f61e936e7e84',
  'scope':'first model evaluation on this task in this continuation; historical public dataset, not independent geographic deployment'})
 train=np.load(a.data/'train.npz');test=np.load(a.data/'test.npz');q=test['q'];y=test['y'];expected={};fidelity=[];quality={}
 train_keys=set(map(bytes,train['q']));mask=np.asarray([bytes(row) not in train_keys for row in q])
 with threadpool_limits(1),ExitStack() as stack:
  engines={arm:stack.enter_context(MetricSession(a.models/arm/'model.sgm',a.library)) for arm in ('uniform','variance','nca')}
  sks={arm:pickle.loads((a.models/arm/'sklearn.pkl').read_bytes()) for arm in ('uniform','variance','nca')}
  refs={}
  for arm in ('uniform','variance','nca'):
   with np.load(a.models/arm/'reference.npz') as r:refs[arm]={k:r[k] for k in r.files}
   ref=refs[arm];pred=[]
   for start in range(0,len(q),128):pred.extend(sks[arm].predict(kernel_matrix(q[start:start+128],train['q'],ref['weights'],ref['table'])).tolist())
   expected[arm]=pred
   for mode in ('compiled','scalar_integer','scalar_exp','exhaustive'):assert engines[arm].predict_buffer(q,mode=mode)==pred,(arm,mode)
   expanded=expanded_model(ref);(a.out/(arm+'-reference.srt')).write_bytes(expanded);original=Reference(expanded,a.reference_library,len(ref['classes']),int(ref['weights'].sum()))
   hh=hashlib.sha256();count=0
   try:
    for start in range(0,len(q),64):
     old=original.probe(np.repeat(q[start:start+64],ref['weights'].astype(int),axis=1).astype(np.float64))
     new=engines[arm].probe(q[start:start+64]);assert new==old.tobytes(),(arm,start)
     hh.update(new);count+=old.size
   finally:original.close()
   _,work=engines[arm].inspect_buffer(q)
   fidelity.append({'arm':arm,'predictions':len(q),'pair_margins':count,'margin_sha256':hh.hexdigest(),'runtime_info':engines[arm].info,'work':work})
   quality[arm]={**summarize(np.asarray(pred),y),'nonoverlap':summarize(np.asarray(pred)[mask],y[mask])}
   np.savez_compressed(a.out/(arm+'-predictions.npz'),prediction=np.asarray(pred),expected=y,nonoverlap=mask)
   print(arm,quality[arm]['correct'],'/',len(y),'margin',count,flush=True)
  cn={arm:ControlSession(controls/arm) for arm in ('linear','mlp','qda')}
  for arm in cn:
   model=pickle.loads((a.models/arm/'model.pkl').read_bytes());pred=model.predict(q.astype(float)/255.)
   assert cn[arm].predict_buffer(q)==pred.tolist(),(arm,'native control disagrees')
   expected[arm]=pred.tolist();quality[arm]={**summarize(pred,y),'nonoverlap':summarize(pred[mask],y[mask])}
   np.savez_compressed(a.out/(arm+'-predictions.npz'),prediction=pred,expected=y,nonoverlap=mask)
   print(arm,quality[arm]['correct'],'/',len(y),flush=True)
  quality['difference']=differences(np.asarray(expected['nca']),np.asarray(expected['uniform']),y,q,20260928)
  quality['overlap_rows']=int((~mask).sum());quality['nonoverlap_rows']=int(mask.sum())
  write(a.out/'quality.json',quality);write(a.out/'fidelity.json',fidelity)
  # Timing source/settings fixed before every timed job. No fits or selections follow.
  write(a.out/'timing_protocol.json',{'arms':ARMS,'chunks':[1,32,256],'repeats':7,'seed':2026092811,'source_sha256':sha(__file__),
    'models':{arm:sha(a.models/arm/('model.sgm' if arm in engines else 'model.pkl')) for arm in expected},
    'controls':{arm:sha(controls/arm/'control.so') for arm in cn},'library':sha(a.library),'input':sha(a.data/'test.npz'),
    'cpu':next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
    'affinity':sorted(os.sched_getaffinity(0)),'scope':'whole jobs of all original test codes; preprocessing/model loading/table preparation excluded; fresh labels and actual fallback work included'})
  records=[];rng=random.Random(2026092811);training=np.ascontiguousarray(train['q'],dtype=float)
  def invoke(arm,chunk):
   out=[];key,mode=arm.split('_',1)
   for start in range(0,len(q),chunk):
    x=q[start:start+chunk]
    if mode=='native':out.extend(cn[key].predict_buffer(x))
    elif mode=='sklearn':
     distance=cdist(np.ascontiguousarray(x,dtype=float),training,'sqeuclidean',w=refs[key]['weights'].astype(float));s=distance.astype(np.int64)
     if not np.array_equal(s,distance):raise ValueError('not integer signature')
     out.extend(sks[key].predict(refs[key]['table'][s]).tolist())
    else:out.extend(engines[key].predict_buffer(x,mode=mode))
   return out
  for arm in ARMS:assert invoke(arm,256)==expected[arm.split('_')[0]]
  with (a.out/'timings.jsonl').open('x') as log:
   for repeat in range(7):
    jobs=[(c,arm) for c in (1,32,256) for arm in ARMS];rng.shuffle(jobs)
    for chunk,arm in jobs:
     start=time.perf_counter_ns();pred=invoke(arm,chunk);elapsed=time.perf_counter_ns()-start
     assert pred==expected[arm.split('_')[0]],(arm,repeat,chunk)
     r={'repeat':repeat,'chunk':chunk,'arm':arm,'ns':elapsed,'rows':len(q),'prediction_sha256':hashlib.sha256(np.asarray(pred,dtype='<i8').tobytes()).hexdigest()}
     records.append(r);log.write(json.dumps(r)+'\n');log.flush()
    print('repeat',repeat,'done',flush=True)
  med={str(c):{arm:statistics.median(r['ns'] for r in records if r['chunk']==c and r['arm']==arm)/len(q)/1000 for arm in ARMS} for c in (1,32,256)}
  ratio=med['32']['nca_compiled']/med['32']['uniform_compiled'];gain=quality['difference']['difference_points']
  write(a.out/'timing_summary.json',{'cells':len(records),'us_per_row':med,'nca_over_uniform':ratio,'accuracy_gain_points':gain,'joint_gate':gain>=.5 and ratio<=1.25})
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('models','data','library','reference-library','out'):p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
