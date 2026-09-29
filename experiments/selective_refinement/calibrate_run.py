"""Use only reserved calibration labels. Freeze all policies before testing."""
from __future__ import annotations
import argparse,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.conditioned_heads.pilot import training
from experiments.budgeted_prototypes.session import PrototypeSession
from experiments.budgeted_prototypes.learning import Settings,features
from experiments.budgeted_prototypes.float_control import FloatSession
from .session import RefinementSession
from .calibration import calibrate
from .study import TASKS,sha,write,source_snapshot

def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    models=json.loads((a.models/'FINAL_MODELS.json').read_text())
    for name,h in models['files'].items():
        if sha(a.models/name)!=h:raise ValueError('frozen model artifact changed')
    write(a.out/'OPENING.json',{'models_sha256':sha(a.models/'FINAL_MODELS.json'),'library_sha256':sha(a.library),
                              'source':source_snapshot(a.out/'source'),'uses_only_reserved_calibration':True})
    begin=time.process_time();policies={}
    with threadpool_limits(1):
        for task in TASKS:
            q,y,D=training(a.data,task)
            with np.load(a.models/task/'roles.npz') as f:cal=f['calibration'];dev=f['development']
            assert not set(map(bytes,q[cal]))&set(map(bytes,q[dev]))
            qcal=np.ascontiguousarray(q[cal]);folder=a.out/task;folder.mkdir()
            with PrototypeSession(a.models/task/'fast/model.spp',a.library) as fast:
                score=np.frombuffer(fast.scores(qcal),dtype=np.float64).reshape(len(qcal),-1).copy()
                fp=np.asarray(fast.labels)[score.argmax(1)]
                assert fast.predict_buffer(qcal)==fp.tolist()
            with RefinementSession(a.models/task/'fast/model.spp',a.models/task/'strong/model.srt',a.library,threshold=None) as engine:
                sp=np.asarray(engine.predict_buffer(qcal,mode='strong'))
            # Independent training frameworks check classifier identity, not score bit equality.
            with np.load(a.models/task/'fast/model.npz') as f:arrays={k:f[k] for k in f.files}
            settings=Settings(**json.loads((a.models/task/'fast/settings.json').read_text()))
            ref=arrays['classes'][(features(qcal,arrays,D,settings)@arrays['head']+arrays['bias']).argmax(1)]
            if not np.array_equal(fp,ref):raise ValueError('native fast labels differ from saved training model')
            svm=pickle.loads((a.models/task/'strong/model.pkl').read_bytes())
            if not np.array_equal(sp,svm.predict(qcal.astype(float)/D)):raise ValueError('native strong labels differ from saved model')
            ordered=np.sort(score,axis=1);gap=ordered[:,-1]-ordered[:,-2]
            report=calibrate(gap,fp,sp,y[cal]);policies[task]=report['policies']
            np.savez_compressed(folder/'observations.npz',indices=cal,fast=fp,strong=sp,truth=y[cal],scores=score,gap=gap)
            write(folder/'calibration.json',report)
            print(task,'calibration',len(cal),'fast',int((fp==y[cal]).sum()),'strong',int((sp==y[cal]).sum()),
                  'primary',report['policies']['0.01'],flush=True)
    write(a.out/'POLICIES.json',policies)
    write(a.out/'CALIBRATION_LOCK.json',{'policies_sha256':sha(a.out/'POLICIES.json'),'models_sha256':sha(a.models/'FINAL_MODELS.json'),
        'files':{str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.is_file() and 'source' not in p.relative_to(a.out).parts},
        'cpu_seconds':time.process_time()-begin,'official_tests_not_opened':True})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('models','data','library','out'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
