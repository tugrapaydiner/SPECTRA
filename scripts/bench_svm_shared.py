"""Fixed retained-model comparison: sharing, public API costs and cost-aware pilot.

No model training/selection. Raw native batches reuse buffers; public API returns
fresh outputs. Timing records retain all repetitions; do not pool scope types.
"""
from array import array
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectra.svm import Session
from spectra.svm_shared import PreparedModel


def run(inputs,library,out):
    out.mkdir(parents=True,exist_ok=False)
    arms=['public_scalar_legacy','public_scalar_shared','public_batch_legacy','public_batch_shared',
          'native_batch_legacy','native_batch_shared','native_batch_cost']
    metadata={'schema':'spectra.shared_svm.v1','repeats':19,'rows':512,'seed':20260926,'arms':arms,
        'host':platform.platform(),'python':sys.version,'affinity':sorted(os.sched_getaffinity(0)),
        'scope':'complete 512-row jobs; includes conversion for public arms; native buffers reused; no load/preprocessing',
        'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'spectra/svm.py',ROOT/'spectra/svm_shared.py',Path(__file__).resolve()]+
            sorted((ROOT/'spectra/_native/ovo').rglob('*.cpp'))+sorted((ROOT/'spectra/_native/ovo').rglob('*.hpp'))},
        'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'input_hashes':{}}
    summaries={};rng=random.Random(metadata['seed'])
    with (out/'rows.jsonl').open('x') as stream:
        for task in ('pendigits','letter'):
            path=inputs/f'{task}.srt';rowfile=inputs/f'{task}.json';rows=json.loads(rowfile.read_text())
            if len(rows)!=512:raise ValueError('512 rows required')
            metadata['input_hashes'][task]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,rowfile)}
            tables=task=='letter'
            with Session(path,library,tables=tables) as old,PreparedModel(path,library,tables=tables) as model,model.session() as worker:
                expected=old.predict_many(rows,schedule='exhaustive')
                x32=array('f',[float(v) for row in rows for v in row])
                x64=array('d',iter(x32));p32=(C.c_float*len(x32)).from_buffer(x32);p64=(C.c_double*len(x64)).from_buffer(x64)
                oldout=(C.c_int*512)();newout=(C.c_int*512)();stats=(C.c_uint64*5)()
                def call(arm):
                    if arm=='public_scalar_legacy':return [old.predict(row) for row in rows]
                    if arm=='public_scalar_shared':return [worker.predict(row) for row in rows]
                    if arm=='public_batch_legacy':return old.predict_many(rows)
                    if arm=='public_batch_shared':return worker.predict_many(rows)
                    if arm=='native_batch_legacy':
                        code=old._lib.sp_svm_batch(old._handle,p32,512,16,5,-1,oldout,512)
                        if code:raise RuntimeError('native legacy error')
                        return oldout
                    mode=6 if arm=='native_batch_cost' else 5
                    code=worker._lib.sp_worker_run(worker._handle,p64,512,16,mode,-1,newout,512,stats,5,0)
                    if code:raise RuntimeError('native shared error')
                    return newout
                for arm in arms:
                    assert list(call(arm))==expected
                obs=[]
                for repeat in range(19):
                    order=list(arms);rng.shuffle(order)
                    for arm in order:
                        t=time.perf_counter_ns();output=call(arm);ns=time.perf_counter_ns()-t
                        if list(output)!=expected:raise AssertionError((task,arm,'decision mismatch'))
                        record={'task':task,'arm':arm,'repeat':repeat,'ns':ns,'matched':True}
                        if arm in ('native_batch_shared','native_batch_cost'):
                            record['work']=dict(zip(('kernels','pairs','terms','cert_checks','cost_scan_terms'),map(int,stats)))
                        stream.write(json.dumps(record)+'\n');stream.flush();obs.append(record)
                med={arm:statistics.median(r['ns'] for r in obs if r['arm']==arm)/512/1000 for arm in arms}
                workers=[model.session() for _ in range(32)]
                assert len({w.info['model_id'] for w in workers})==1
                stored={}
                for n in (1,2,8,32):
                    stored[str(n)]={'shared':model.info['shared_prepared_bytes']+sum(w.info['worker_scratch_bytes'] for w in workers[:n]),
                        'legacy_independent':n*(old.info['prepared_bytes']+old.info['scratch_bytes'])}
                for w in workers:w.close()
                summaries[task]={'median_us_per_row':med,'storage_counts':stored,'model_info':model.info,'worker_info':worker.info,
                    'cost_speedup':med['native_batch_shared']/med['native_batch_cost'],
                    'shared_native_vs_legacy_ratio':med['native_batch_shared']/med['native_batch_legacy']}
    metadata['interpretation']='Retained fixed models/inputs; not independent task confirmation, RSS or external reproduction.'
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    summaries['cost_promotion']=all(summaries[t]['cost_speedup']>=1.1 for t in ('pendigits','letter'))
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    return summaries


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('inputs','library','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.inputs,a.library,a.out),indent=2))
