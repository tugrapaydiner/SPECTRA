"""Fresh-process resource check of one already-frozen model, not a timing contest.

Use an extracted SDK with no numerical Python dependencies. Table preparation is
included; filesystem caches may be warm. Each process reads its OWN Linux VmHWM.
"""
from __future__ import annotations
import argparse,hashlib,json,platform,sys,time
from pathlib import Path


def run(a):
 sys.path.insert(0,str(a.sdk.resolve()))
 from experiments.learned_metric.session import MetricSession
 corpus=json.loads((a.sdk/'CORPUS.json').read_text());entry=corpus['models'][a.model]
 task=entry['task'];d=corpus['datasets'][task]['features']
 path=a.sdk/'models'/a.model/'model.sgm'
 if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['model_sha256']:raise ValueError('model identity mismatch')
 with (a.sdk/'inputs'/(task+'.u8')).open('rb') as f:row=bytearray(f.read(d))
 before=time.perf_counter_ns()
 with MetricSession(path,a.library) as w:
  prepared=time.perf_counter_ns()-before
  if w.predict_buffer(row)!=entry['expected'][:1]:raise ValueError('probe prediction mismatch')
  status=Path('/proc/self/status').read_text().splitlines()
  memory={key:int(next(line.split()[1] for line in status if line.startswith(key+':'))) for key in ('VmHWM','VmRSS')}
  if {'numpy','pandas','torch','scipy','sklearn'} & sys.modules.keys():raise ValueError('unexpected numerical framework')
  return {'model':a.model,'preparation_ns':prepared,'memory_kib':memory,'runtime_info':w.info,
   'model_sha256':entry['model_sha256'],'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),
   'matched':True,'python':sys.version,'platform':platform.platform(),
   'scope':'one frozen model and one validated prediction per fresh process; not cold-cache, energy or universal RAM evidence'}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sdk',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--model',required=True)
 print(json.dumps(run(p.parse_args()),sort_keys=True))
