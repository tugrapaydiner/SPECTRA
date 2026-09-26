"""Reproduce the frozen five-domain panel without a private working directory."""
from __future__ import annotations
import argparse,json,os,pickle,shutil,time
import numpy as np
from pathlib import Path
from threadpoolctl import threadpool_limits
from panel_data import prepare,TASKS
from panel_runtime import build,Timer,Libsvm,Arm
from fit_models import fit_task,seal
from evaluate_panel import run
from replay_controls import replay
from audit_panel import audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acquisition',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(a.acquisition,a.out/'acquisition')
    try:os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    except AttributeError:raise SystemExit('the recorded timing protocol requires Linux CPU affinity')
    start=time.process_time()
    with threadpool_limits(limits=1):
        prepare(a.out/'acquisition/data',a.out/'panel')
        for task in TASKS:fit_task(a.out/'panel'/task)
        build(a.out/'build',a.out/'acquisition/libsvm')
        # Refuse to open test predictions when a selected export fails validation.
        timer=Timer(a.out/'build/libpanel.so');validation={}
        for task in TASKS:
            folder=a.out/'panel'/task;models=folder/'models'
            split=json.loads((folder/'split.json').read_text())
            original=np.load(folder/'original.npz',allow_pickle=False)
            scaler=np.load(models/'scaler.npz',allow_pickle=False)
            x=(original['x'][split['validation']]-scaler['mean'])/scaler['scale']
            fitted=pickle.loads((models/'selected.pkl').read_bytes())
            labels=fitted.classes_.tolist();expected=[labels.index(v) for v in fitted.predict(x)]
            native=Arm(models/'selected.spc',a.out/'build/spectra/libspectra_svm.so','cert_tables')
            external=Libsvm(timer,models/'selected.libsvm',x.shape[1],labels)
            try:
                validation[task]={name:int(np.sum(arm.batch(x)!=expected)) for name,arm in [('spectra',native),('libsvm',external)]}
                if any(validation[task].values()):raise AssertionError('validation export mismatch')
            finally:native.close();external.close()
        (a.out/'validation-fidelity.json').write_text(json.dumps(validation,indent=2)+'\n')
        seal(a.out/'panel',Path(__file__).parent)
        for task in TASKS:run(a.out/'panel',a.out/'build',a.out/'formal'/task,task)
        replay(a.out)
    report=audit(a.out,Path(__file__).resolve().parents[2])
    with (a.out/'AUDIT_FINAL.json').open('x') as f:json.dump(report,f,indent=2)
    (a.out/'runner.json').write_text(json.dumps({'status':'COMPLETE','cpu_seconds':time.process_time()-start,
        'primary_gate':report['primary_gate'],'note':'CI success means completed fidelity/audit, not the scientific speed gate'},indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tasks'},indent=2))


if __name__=='__main__':main()
