"""Fixed panel evaluation. Source and all selected models must be sealed first."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import random
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from panel_data import TASKS,SEED,sha
from panel_runtime import ARMS,Arm,Timer,Libsvm,DP,IP,ROOT


def check_freeze(panel):
    record=json.loads((panel/'TEST_OPENING.json').read_text())
    for name,digest in record['files'].items():
        if sha(panel/name)!=digest:raise ValueError('frozen file changed: '+name)
    for name,digest in record['source'].items():
        if sha(ROOT/name)!=digest:raise ValueError('frozen source changed: '+name)
    return record


def run(panel:Path,build:Path,out:Path,task:str):
    check_freeze(panel)
    out.mkdir(parents=True,exist_ok=False)
    folder=panel/task;m=folder/'models';data=np.load(folder/'original.npz',allow_pickle=False)
    split=json.loads((folder/'split.json').read_text());fit=json.loads((m/'fitting.json').read_text())
    scaler=np.load(m/'scaler.npz');raw=data['x'][split['test']];y=data['y'][split['test']]
    mean,scale=scaler['mean'],scaler['scale'];x=np.ascontiguousarray((raw-mean)/scale)
    models={name:pickle.loads((m/(name+'.pkl')).read_bytes()) for name in ('selected','logistic','mlp')}
    labels=models['selected'].classes_.tolist();expected_label=models['selected'].predict(x)
    expected=np.asarray([labels.index(v) for v in expected_label],dtype=np.int32)
    np.savez_compressed(out/'test_inputs.npz',raw=raw,x=x,y=y,expected=expected)
    meta=dict(task=task,host=platform.platform(),cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
              affinity=sorted(os.sched_getaffinity(0)),python=sys.version,rows=len(x),dimensions=x.shape[1],labels=labels,
              seed=SEED,repeats=11,timing_positions=split['timing_test_positions'],
              hashes={'model':sha(m/'selected.spc'),'libsvm_model':sha(m/'selected.libsvm'),
                      'libsvm_library':sha(build/'libpanel.so'),'spectra_library':sha(build/'spectra/libspectra_svm.so')})
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    timer=Timer(build/'libpanel.so')
    arms={name:Arm(m/'selected.spc',build/'spectra/libspectra_svm.so',name) for name in ARMS}
    external=Libsvm(timer,m/'selected.libsvm',x.shape[1],labels)
    quality={name:dict(correct=int(np.sum(model.predict(x)==y)),rows=len(y)) for name,model in models.items()}
    fidelity={}
    predictions={}
    for name,arm in {**arms,'libsvm':external}.items():
        got=arm.batch(x);predictions[name]=got
        fidelity[name]=dict(mismatches=int(np.sum(got!=expected)),rows=len(x))
    np.savez_compressed(out/'predictions.npz',**predictions)
    # Certificates and work are outside performance timers and cover every test row.
    work=[]
    for row in x:
        cert=arms['cert_tables'].session.predict_with_certificate(row,schedule='beretta_cert')
        work.append(dict(class_index=cert.class_index,kernels=cert.evaluated_kernels,pairs=cert.evaluated_pairs,
                         terms=cert.evaluated_terms,certificate=list(cert.pair_outcomes)))
    (out/'work.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in work))
    storage={name:{'model':arm.model.info,'worker':arm.session.info} for name,arm in arms.items()}
    (out/'quality_fidelity.json').write_text(json.dumps(dict(quality=quality,fidelity=fidelity,storage=storage),indent=2)+'\n')
    # Do not silently drop an incorrect arm. Record predictions and reject timings.
    if any(item['mismatches'] for item in fidelity.values()):
        raise AssertionError(f'{task}: executor class disagreement; see retained predictions')
    positions=meta['timing_positions'];names=[*ARMS,'libsvm']
    # All numerical buffers are prepared outside native timer; only class output is required.
    pointers={i:x[i].ctypes.data_as(DP) for i in positions};result=C.c_int()
    for name in names:
        for i in positions[:8]:
            if name=='libsvm':external.batch(x[i:i+1])
            else:arms[name].batch(x[i:i+1])
    rng=random.Random(SEED)
    with (out/'native.jsonl').open('x') as f:
        for repeat in range(11):
            order=list(positions);rng.shuffle(order)
            for i in order:
                modes=list(names);rng.shuffle(modes)
                for name in modes:
                    if name=='libsvm':ns=timer.lib.panel_time_libsvm(external.handle,pointers[i],x.shape[1],C.byref(result))
                    else:
                        a=arms[name];ns=timer.lib.panel_time_shared(a.function,a.session._handle,pointers[i],x.shape[1],a.mode,C.byref(result))
                    if ns==2**64-1 or result.value!=int(expected[i]):raise AssertionError('timed native mismatch')
                    f.write(json.dumps(dict(repeat=repeat,case=i,arm=name,ns=int(ns),class_index=result.value))+'\n')
    floors=[int(timer.lib.panel_time_noop()) for _ in range(1000)]
    (out/'timer_floor.json').write_text(json.dumps(floors)+'\n')
    # Complete Python API includes raw-row conversion, scaler arithmetic and fresh labels.
    public=['spectra','libsvm','sklearn','logistic','mlp'];raw_rows=raw.tolist()
    def call(name,i):
        z=(np.asarray(raw_rows[i],dtype=np.float64)-mean)/scale
        if name=='spectra':return arms['cert_tables'].predict(z)
        if name=='libsvm':return external.predict(z)
        return models['selected' if name=='sklearn' else name].predict(z.reshape(1,-1))[0]
    expected_public={name:(expected_label if name in ('spectra','libsvm','sklearn') else models[name].predict(x)) for name in public}
    for name in public:
        for i in positions[:8]:assert call(name,i)==expected_public[name][i]
    rng=random.Random(SEED)
    with (out/'public.jsonl').open('x') as f:
        for repeat in range(11):
            order=list(positions);rng.shuffle(order)
            for i in order:
                modes=list(public);rng.shuffle(modes)
                for name in modes:
                    start=time.perf_counter_ns();value=call(name,i);ns=time.perf_counter_ns()-start
                    if value!=expected_public[name][i]:raise AssertionError('timed public mismatch')
                    f.write(json.dumps(dict(repeat=repeat,case=i,arm=name,ns=ns,label=value.item() if hasattr(value,'item') else value))+'\n')
    # Public batch64: complete selected-input job, same preprocessing and fresh labels.
    xr=raw[positions].tolist();target=expected_label[positions].tolist()
    with (out/'batch64.jsonl').open('x') as f:
        rng=random.Random(SEED)
        for repeat in range(11):
            modes=['spectra','libsvm','sklearn'];rng.shuffle(modes)
            for name in modes:
                start=time.perf_counter_ns();pred=[]
                for k in range(0,len(xr),64):
                    z=(np.asarray(xr[k:k+64],dtype=np.float64)-mean)/scale
                    if name=='spectra':pred.extend(arms['cert_tables'].session.predict_many(z))
                    elif name=='libsvm':pred.extend(labels[int(j)] for j in external.batch(z))
                    else:pred.extend(models['selected'].predict(z).tolist())
                elapsed=time.perf_counter_ns()-start
                if pred!=target:raise AssertionError('batch64 mismatch')
                f.write(json.dumps(dict(repeat=repeat,arm=name,ns=elapsed,rows=len(xr),matched=True))+'\n')
    for arm in arms.values():arm.close()
    external.close()
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='PASS',rows=len(x),quality=quality,fidelity=fidelity),indent=2)+'\n')
    print(task,'complete',quality,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--task',choices=TASKS,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.panel,a.build,a.out,a.task)
