"""Complete execution-only rerun retaining v2 controls and all frozen models."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,random,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.learned_signature.native import Session
from experiments.learned_signature.dense import DenseSession
from experiments.learned_signature.learning import write_json,sha
from experiments.finite_kernel.session import FiniteSession
FAMILIES=('rbf','metric','alignment','uniform','metric_alignment','pair_experts','margin','rbf_v2','metric_v2','alignment_v2','uniform_v2','metric_alignment_v2','pair_experts_v2','margin_v2','linear','mlp','legacy')
CHUNKS=(1,32,256);REPEATS=7;SEED=202609282

def digest(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
def summarize(records):
 groups={}
 for r in records:groups.setdefault((r['task'],r['family'],r['mode'],r['chunk']),[]).append(r['ns'])
 results={}
 for (task,family,mode,chunk),ns in groups.items():
  n=next(r['rows'] for r in records if r['task']==task)
  results[f'{task}|{family}|{mode}|{chunk}']={'median_job_ns':statistics.median(ns),'median_us_per_row':statistics.median(ns)/n/1000,'observations':len(ns),'min_ns':min(ns),'max_ns':max(ns)}
 return {'cells':len(records),'checked_predictions':sum(r['rows'] for r in records),'results':results}

def run(a):
 a.out.mkdir(parents=True,exist_ok=False)
 protocol={'schema':'spectra.learned-signature.cost.v2','tasks':['letter','pendigits'],'families':list(FAMILIES),'chunks':list(CHUNKS),'repeats':REPEATS,'seed':SEED,'freeze_sha256':sha(a.freeze),'quality_sha256':sha(a.evaluation/'report.json'),'library_sha256':sha(a.library),'reference_library_sha256':sha(a.reference_library),'legacy_library_sha256':sha(a.legacy_library),'script_sha256':sha(__file__),'cpu':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'affinity':sorted(os.sched_getaffinity(0)),'scope':'complete prepared-native-input jobs, per-call validation, crossing, fresh labels, concatenation; excludes text parsing, model loading/build and conversion to each model native dtype; same trained classifier only within modes, not between families'}
 write_json(a.out/'protocol.json',protocol);records=[];rng=random.Random(SEED)
 with (a.out/'rows.jsonl').open('x') as log:
  for task in protocol['tasks']:
   with np.load(a.evaluation/f'{task}-cases.npz') as z:q=z['q'].copy()
   for family in FAMILIES:
    base_family=family[:-3] if family.endswith('_v2') else family;folder=a.models/task/base_family
    if family=='legacy':
     x=np.fromfile(a.legacy/task/'X.f64',dtype='<f8').reshape(q.shape);wanted=json.loads((a.legacy/task/'expected.json').read_text());worker=FiniteSession(a.legacy/task/'model.srt',a.legacy_library);modes=('compiled',)
    else:
     x=q;wanted=np.load(a.evaluation/f'{task}-{base_family}-pred.npy').tolist()
     if family in ('linear','mlp'):worker=DenseSession(folder/'control.json',a.library);modes=('dense',)
     else:worker=Session(folder/'model.lkt',a.reference_library if family.endswith('_v2') else a.library);modes=('selective',) if family.endswith('_v2') else ('selective','exhaustive')
    with worker:
     batches={chunk:[x[first:first+chunk] for first in range(0,len(x),chunk)] for chunk in CHUNKS}
     def call(mode,chunk):
      result=[]
      for batch in batches[chunk]:result.extend(worker.predict_buffer(batch) if mode=='dense' else worker.predict_buffer(batch,mode=mode))
      return result
     jobs=[(mode,chunk) for mode in modes for chunk in CHUNKS]
     for mode,chunk in jobs:
      if call(mode,chunk)!=wanted:raise ValueError('pre-timing mismatch')
     for repeat in range(REPEATS):
      order=jobs.copy();rng.shuffle(order)
      for mode,chunk in order:
       start=time.perf_counter_ns();got=call(mode,chunk);ns=time.perf_counter_ns()-start
       if got!=wanted:raise ValueError('timed mismatch')
       row={'task':task,'family':family,'mode':mode,'chunk':chunk,'repeat':repeat,'rows':len(x),'ns':ns,'output_sha256':digest(got)}
       records.append(row);log.write(json.dumps(row)+'\n');log.flush()
    print(task,family,'complete',flush=True)
 write_json(a.out/'summary.json',summarize(records))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('models','evaluation','freeze','library','reference-library','legacy','legacy-library','out'):p.add_argument('--'+n,type=Path,required=True)
 run(p.parse_args())
