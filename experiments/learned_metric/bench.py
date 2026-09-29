"""Frozen complete-call quality/cost comparison on the same raw integer inputs.

Old normalized-input model includes conversion; sklearn includes construction of
its requested precomputed kernel. No precomputed query kernels outside its timer.
All actual model-specific expected predictions are checked on every whole job.
"""
from __future__ import annotations
from contextlib import ExitStack
import argparse,hashlib,json,os,pickle,platform,random,statistics,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.learned_metric.session import MetricSession
from experiments.learned_metric.controls import ControlSession
from experiments.finite_kernel.session import FiniteSession
ARMS=('uniform_compiled','variance_compiled','nca_compiled','nca_scalar_integer','nca_exhaustive','nca_scalar_exp',
      'linear_native','mlp_native','uniform_sklearn','nca_sklearn','historical_finite')
CHUNKS=(1,32,256);REPEATS=7;SEED=2026092806

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(labels):return hashlib.sha256(np.asarray(labels,dtype='<i8').tobytes()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def summary(records,quality):
 med={}
 for t in ('letter','pendigits'):
  med[t]={str(c):{a:statistics.median(r['ns'] for r in records if r['task']==t and r['chunk']==c and r['arm']==a) for a in ARMS} for c in CHUNKS}
 results={}
 for task,costs in med.items():
  n=quality[task]['uniform']['rows'];main=costs['32'];ratio=main['nca_compiled']/main['uniform_compiled']
  paired=[]
  for rep in range(REPEATS):
   values={r['arm']:r['ns'] for r in records if r['task']==task and r['chunk']==32 and r['repeat']==rep}
   paired.append(values['nca_compiled']/values['uniform_compiled'])
  gain=quality[task]['difference']['difference_points']
  results[task]={'batch32_us_per_row':{a:v/n/1000 for a,v in main.items()},'nca_over_uniform':ratio,
    'quality_gain_points':gain,'joint_gate':gain>=.5 and ratio<=1.25,'paired_ratios':paired,
    'paired_regressions':sum(v>1 for v in paired),'batch_median_ns':costs}
 return {'cells':len(records),'checked_predictions':sum(r['rows'] for r in records),'tasks':results,
         'any_joint_gate':any(v['joint_gate'] for v in results.values())}

def main(a):
 a.out.mkdir(parents=True,exist_ok=False)
 lock=json.loads((a.models/'FINAL_LOCK.json').read_text())
 for f,h in lock['files'].items():
  if sha(a.models/f)!=h:raise ValueError('changed frozen model')
 quality=json.loads((a.evaluation/'quality.json').read_text())
 files={}
 for root,name in ((a.models,'models'),(a.data,'data'),(a.evaluation,'evaluation'),(a.controls,'controls')):
  for p in root.rglob('*'):
   if p.is_file() and p.suffix in ('.npz','.json','.pkl','.sgm','.so'):files[name+'/'+p.relative_to(root).as_posix()]=sha(p)
 sources=[p for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.cpp','.md')]+list((ROOT/'spectra/_native/ovo').glob('*'))
 protocol={'arms':ARMS,'chunks':CHUNKS,'repeats':REPEATS,'seed':SEED,'files':files,
    'libraries':{str(p):sha(p) for p in (a.library,a.old_library)},
    'historical_models':{t:sha(a.old_models/t/'model.srt') for t in ('letter','pendigits')},
    'source':{str(p.relative_to(ROOT)):sha(p) for p in sources if p.is_file()},
    'cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
    'platform':platform.platform(),'python':sys.version,'affinity':sorted(os.sched_getaffinity(0)),
    'scope':'complete all-row prediction jobs from prepared uint8 codes; model-specific predictions checked; loading/training/table-preparation excluded; no preprocessing before integer feature codes'}
 write(a.out/'protocol.json',protocol);randomizer=random.Random(SEED);records=[]
 with threadpool_limits(1),(a.out/'rows.jsonl').open('x') as log:
  for task in ('letter','pendigits'):
   train=np.load(a.data/(task+'-train.npz'));test=np.load(a.data/(task+'-test.npz'));q=test['q'];D=int(test['maximum'])
   expected={arm:np.load(a.evaluation/task/(arm+'-predictions.npz'))['prediction'].tolist() for arm in ('uniform','variance','nca','linear','mlp')}
   oldexpected=json.loads((a.old_models/task/'expected.json').read_text())
   # The exact old input convention is binary32 division widened to binary64.
   oldx=(q.astype(np.float32)/np.float32(D)).astype(np.float64)
   if oldx.tobytes()!=(a.old_models/task/'X.f64').read_bytes():raise ValueError('historical inputs differ')
   with ExitStack() as stack:
    engines={arm:stack.enter_context(MetricSession(a.models/(task+'-'+arm)/'model.sgm',a.library)) for arm in ('uniform','variance','nca')}
    controls={arm:ControlSession(a.controls/(task+'-'+arm)) for arm in ('linear','mlp')}
    old=stack.enter_context(FiniteSession(a.old_models/task/'model.srt',a.old_library))
    sks={arm:pickle.loads((a.models/(task+'-'+arm)/'sklearn.pkl').read_bytes()) for arm in ('uniform','nca')}
    refs={}
    for arm in ('uniform','nca'):
     with np.load(a.models/(task+'-'+arm)/'reference.npz') as stored:
      refs[arm]={'weights':stored['weights'].astype(float),'table':stored['table'].copy()}
    training=np.ascontiguousarray(train['q'],dtype=float)
    def call(arm,chunk):
     out=[]
     for start in range(0,len(q),chunk):
      x=q[start:start+chunk]
      if arm=='historical_finite':out.extend(old.predict_buffer((x.astype(np.float32)/np.float32(D)).astype(np.float64)))
      elif arm.endswith('_native'):out.extend(controls[arm.split('_')[0]].predict_buffer(x))
      elif arm.endswith('_sklearn'):
       key=arm.split('_')[0];r=refs[key]
       dist=cdist(np.ascontiguousarray(x,dtype=float),training,'sqeuclidean',w=r['weights'])
       code=dist.astype(np.int64)
       if not np.array_equal(code,dist) or (code<0).any() or code.max()>=len(r['table']):raise ValueError('invalid query signatures')
       out.extend(sks[key].predict(r['table'][code]).tolist())
      else:
       key,mode=arm.split('_',1);out.extend(engines[key].predict_buffer(x,mode=mode))
     return out
    def wanted(arm):return oldexpected if arm=='historical_finite' else expected[arm.split('_')[0]]
    for arm in ARMS:
     assert call(arm,256)==wanted(arm),(task,arm,'warm fidelity')
    jobs=[(chunk,arm) for chunk in CHUNKS for arm in ARMS]
    for repeat in range(REPEATS):
     order=list(jobs);randomizer.shuffle(order)
     for chunk,arm in order:
      start=time.perf_counter_ns();pred=call(arm,chunk);ns=time.perf_counter_ns()-start
      assert pred==wanted(arm),(task,chunk,repeat,arm)
      r={'task':task,'repeat':repeat,'chunk':chunk,'arm':arm,'rows':len(q),'ns':ns,'prediction_sha256':digest(pred)}
      log.write(json.dumps(r)+'\n');log.flush();records.append(r)
     print(task,'repeat',repeat,'complete',flush=True)
 write(a.out/'summary.json',summary(records,quality));print(json.dumps(summary(records,quality),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('models','data','evaluation','controls','library','old-library','old-models','out'):p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
