"""Fresh-process model preparation and self high-water memory, no ML imports.

A one-row correctness check follows preparation. Stable filesystem caches are
allowed; this is not physical cold-storage startup or a production load test.
"""
from __future__ import annotations
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def run(a):
    record=json.loads(a.case.read_text());raw=bytearray(record['row'])
    if hashlib.sha256(a.model.read_bytes()).hexdigest()!=record['model_sha256']:raise ValueError('probe model identity')
    if a.backend=='interaction':
        from experiments.integer_interactions.session import InteractionSession as Session
    else:
        from experiments.learned_metric.session import MetricSession as Session
    start=time.perf_counter_ns()
    with Session(a.model,a.library) as session:
        prepared=time.perf_counter_ns()-start
        if session.predict_buffer(raw)!=[record['label']]:raise ValueError('probe output mismatch')
        status=Path('/proc/self/status').read_text().splitlines()
        memory={k:int(next(s.split()[1] for s in status if s.startswith(k+':'))) for k in ('VmRSS','VmHWM')}
        if {'numpy','scipy','pandas','sklearn','torch'}&sys.modules.keys():raise ValueError('numerical framework loaded')
        return {'backend':a.backend,'model':a.model.name,'prepared_ns':prepared,'memory_kib':memory,'runtime_info':session.info,
            'model_sha256':record['model_sha256'],'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'matched':True,
            'python':sys.version,'platform':platform.platform(),'scope':'one prepared model plus checked prediction; warm file caches; process memory not table bytes'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=['interaction','parent'],required=True)
    for n in ('model','library','case'):p.add_argument('--'+n,type=Path,required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True))
