"""Secondary fixed-model execution comparison; no retraining or test-based selection."""
from array import array
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import struct
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,sort_keys=True)

def main(a):
    import joblib,numpy as np
    from threadpoolctl import threadpool_limits
    from spectra.linear import CompiledLinear
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    a.out.mkdir(parents=True,exist_ok=False)
    files=[a.model,a.library,a.bundle/'model.json',a.bundle/'parameters.f64',a.bundle/'model.cpp',a.data/'test/X_test.txt',a.data/'test/y_test.txt',ROOT/'spectra/linear.py',Path(__file__),Path(__file__).with_name('LINEAR_AMENDMENT.md')]
    write(a.out/'LINEAR_LOCK.json',{'files':{str(p):sha(p) for p in files},'seed':927032,'repeats':7,'indices':'linspace(0,n-1,256,dtype=int)','batch_sizes':[1,32],
        'scope':'post-test engineering of already frozen linear model; ordered binary64 arithmetic; no refitting; warm public calls with fresh labels'})
    x=np.loadtxt(a.data/'test/X_test.txt',dtype=np.float64);truth=np.loadtxt(a.data/'test/y_test.txt',dtype=np.int64)
    model=joblib.load(a.model);native=CompiledLinear(a.bundle,a.library)
    with threadpool_limits(1):
        expected=model.predict(x).tolist();pred=native.predict_buffer(x)
        if pred!=expected:raise ValueError('native/BLAS class disagreement')
        meta=json.loads((a.bundle/'model.json').read_text());o=meta['outputs'];d=meta['features']
        raw=(a.bundle/'parameters.f64').read_bytes();values=[v[0] for v in struct.iter_unpack('<d',raw)]
        weights=[values[c*d:(c+1)*d] for c in range(o)];bias=values[d*o:]
        ordered=[]
        for row in x.tolist():
            for c in range(o):
                value=0.
                for f in range(d):value+=row[f]*weights[c][f]
                ordered.append(value+bias[c])
        actual=native.scores_buffer(x)
        if array('d',actual).tobytes()!=array('d',ordered).tobytes():raise ValueError('ordered score bytes differ')
        blas=model.decision_function(x).ravel()
        write(a.out/'FIDELITY.json',{'model_input_pairs':len(x),'ordered_score_values':len(actual),'ordered_score_sha256':hashlib.sha256(array('d',actual).tobytes()).hexdigest(),
           'same_labels':True,'bitwise_ordered_scores':True,'maximum_abs_blas_difference':float(np.max(np.abs(blas-actual))),
           'correct':int(np.sum(np.asarray(pred)==truth)),'rows':len(x),'scope':'not bitwise equality to arbitrary BLAS or exact-real sums'})
        write(a.out/'predictions.json',pred)
        indices=np.linspace(0,len(x)-1,256,dtype=int);probe=np.ascontiguousarray(x[indices]);wanted=[expected[i] for i in indices]
        arms=['native','sklearn','numpy_specialized'];records=[];rng=random.Random(927032)
        def call(arm,chunk):
            out=[]
            for first in range(0,len(probe),chunk):
                block=probe[first:first+chunk]
                if arm=='native':out.extend(native.predict_buffer(block))
                elif arm=='sklearn':out.extend(model.predict(block).tolist())
                else:
                    # Same finite input validation and fresh original labels.
                    if not np.isfinite(block).all():raise ValueError('nonfinite')
                    scores=block @ model.coef_.T+model.intercept_
                    out.extend(model.classes_[scores.argmax(axis=1)].tolist())
            return out
        for arm in arms:
            for chunk in (1,32):assert call(arm,chunk)==wanted
        with (a.out/'timings.jsonl').open('x') as f:
            for repeat in range(7):
                schedule=[(b,arm) for b in (1,32) for arm in arms];rng.shuffle(schedule)
                for batch,arm in schedule:
                    start=time.perf_counter_ns();observed=call(arm,batch);elapsed=time.perf_counter_ns()-start
                    assert observed==wanted
                    record={'repeat':repeat,'batch':batch,'arm':arm,'rows':len(probe),'ns':elapsed,'labels_sha256':hashlib.sha256(json.dumps(observed).encode()).hexdigest()}
                    records.append(record);f.write(json.dumps(record)+'\n')
    summary={f'{arm}/batch{batch}':{'median_us_per_row':statistics.median(r['ns']/r['rows']/1000 for r in records if r['arm']==arm and r['batch']==batch)} for arm in arms for batch in (1,32)}
    write(a.out/'SUMMARY.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('data','model','bundle','library','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
