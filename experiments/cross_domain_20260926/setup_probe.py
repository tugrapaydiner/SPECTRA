"""Separate warm-cache preparation and preprocessing costs for frozen models.

This secondary receipt does not change models, the primary timing matrix or gates.
It is not cold program startup, process RSS or a throughput experiment.
"""
from pathlib import Path
import argparse,json,random,time
import numpy as np
from panel_data import TASKS,SEED
from panel_runtime import Arm,Timer,Libsvm
from evaluate_panel import check_freeze

def probe(root,out):
    check_freeze(root/'panel');timer=Timer(root/'build/libpanel.so');records=[]
    for task in TASKS:
        model=root/'panel'/task/'models';data=np.load(root/'formal'/task/'test_inputs.npz')
        scaler=np.load(model/'scaler.npz');labels=json.loads((root/'formal'/task/'metadata.json').read_text())['labels']
        rng=random.Random(SEED)
        for repeat in range(5):
            order=['spectra','libsvm'];rng.shuffle(order)
            for name in order:
                start=time.perf_counter_ns()
                if name=='spectra':obj=Arm(model/'selected.spc',root/'build/spectra/libspectra_svm.so','cert_tables')
                else:obj=Libsvm(timer,model/'selected.libsvm',data['x'].shape[1],labels)
                elapsed=time.perf_counter_ns()-start;obj.close()
                records.append(dict(task=task,repeat=repeat,arm=name,ns=elapsed))
        row=data['raw'][0].tolist();checksum=0.;mean=scaler['mean'];scale=scaler['scale']
        for repeat in range(11):
            start=time.perf_counter_ns()
            for _ in range(1000):z=(np.asarray(row,dtype=np.float64)-mean)/scale
            elapsed=time.perf_counter_ns()-start;checksum+=float(z.sum())
            records.append(dict(task=task,repeat=repeat,arm='preprocess',ns=elapsed,iterations=1000))
        assert np.isfinite(checksum)
    with out.open('x') as f:json.dump(dict(scope='warm-cache model construction (close excluded); separate repeated raw-row preprocessing; not cold process startup',rows=records),f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();probe(a.evidence,a.out)
