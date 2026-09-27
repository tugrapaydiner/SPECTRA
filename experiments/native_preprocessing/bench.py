"""Measure frozen raw pipelines; input models/pickles must be trusted.

No fitting or selection occurs. The NumPy control is copied from the previous
accepted raw-pipeline study. Existing SHA256 freeze is checked before unpickling.
Run with one pinned CPU and no concurrent benchmark/build jobs.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, platform, random, statistics, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
ARMS=('python_native','compiled_materialized','compiled_native','numpy_native','sklearn')
TASKS=('wine','wdbc','chess','penguins','titanic','zoo')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
class NumpyPlan:
    """Matched strong control: batched numeric operations and one-hot assignment."""
    def __init__(self,path):
        import numpy as np
        doc=read(path);self.features=doc['features'];self.numerical=[];self.categorical=[];offset=0
        for op in doc['operations']:
            if op['kind']=='numeric':self.numerical.append((op['column'],offset,*(float.fromhex(op[k]) for k in ('fill','mean','scale'))));offset+=1
            else:self.categorical.append((op['column'],{v:offset+i for i,v in enumerate(op['categories'])}));offset+=len(op['categories'])
        self.indices=[n[0] for n in self.numerical];self.output=[n[1] for n in self.numerical]
        self.fill=np.array([n[2] for n in self.numerical]);self.mean=np.array([n[3] for n in self.numerical]);self.scale=np.array([n[4] for n in self.numerical])
    def transform(self,rows):
        import numpy as np
        out=np.zeros((len(rows),self.features),dtype=np.float64)
        if self.numerical:
            x=np.asarray([[row[i] for i in self.indices] for row in rows],dtype=np.float64)
            np.copyto(x,np.broadcast_to(self.fill,x.shape),where=np.isnan(x));x-=self.mean;x/=self.scale
            out[:,self.output]=x
        for column,lookup in self.categorical:
            codes=np.fromiter((lookup.get(row[column],-1) for row in rows),dtype=np.int64,count=len(rows));valid=codes>=0
            out[np.arange(len(rows))[valid],codes[valid]]=1.
        return out

def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['model'],r['size'],r['arm']),[]).append(r['ns']/r['rows']/1000)
    med={'|'.join(k):statistics.median(v) for k,v in groups.items()}
    tasks={}
    for task in TASKS:
        models=sorted({r['model'] for r in rows if r['model'].startswith(task+'-')})
        tasks[task]={}
        for size in ('one','batch'):
            lat={a:statistics.median([med[f'{m}|{size}|{a}'] for m in models]) for a in ARMS}
            ratios={a:statistics.geometric_mean([med[f'{m}|{size}|compiled_native']/med[f'{m}|{size}|{a}'] for m in models]) for a in ARMS if a!='compiled_native'}
            tasks[task][size]={'us_per_row':lat,'compiled_over':ratios}
    return {'cells':len(rows),'model_medians_us':med,'tasks':tasks,
            'primary_gate':all(tasks[t]['batch']['compiled_over']['python_native']<=1/1.2 for t in TASKS)}

def main(a):
    import numpy as np,pandas as pd,sklearn
    from threadpoolctl import threadpool_limits
    from spectra.svm_pipeline import PreparedPipeline
    freeze=read(a.inputs/'freeze.json');models=freeze['models']
    assert len(models)==18 and {m['name'] for m in models}=={f'{t}-{s}' for t in TASKS for s in (101,202,303)}
    for m in models:
        for name,digest in m['files'].items():
            p=a.inputs/'models'/m['name']/name
            assert p.resolve().is_relative_to(a.inputs.resolve()) and sha(p)==digest
    a.out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'experiments/native_preprocessing/PROTOCOL.md',ROOT/'experiments/native_preprocessing/RAW_CONTAINER_AMENDMENT.md']
    sources+=sorted((ROOT/'spectra').glob('svm*.py'))+sorted((ROOT/'spectra/_native/ovo').rglob('*.cpp'))+sorted((ROOT/'spectra/_native/ovo').rglob('*.hpp'))
    protocol={'schema':'spectra.compiled_preprocessing.benchmark.v2','seed':2026092702,'repeats':31,'arms':ARMS,
        'models':[m['name'] for m in models],'freeze_sha256':sha(a.inputs/'freeze.json'),
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        'svm_library_sha256':sha(a.library),'preprocessor_sha256':sha(a.preprocessor),
        'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
        'platform':platform.platform(),'cpu':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        'affinity':sorted(os.sched_getaffinity(0)),
        'scope':'raw rows to fresh labels; all raw validation, preprocessing, buffer creation and inference included; setup/loading/file parsing excluded; retained data, no new accuracy'}
    write(a.out/'protocol.json',protocol)
    allrows=[];rng=random.Random(protocol['seed']);pairs=features=0;decisions={}
    with (a.out/'rows.jsonl').open('x') as stream,threadpool_limits(limits=1):
        for m in models:
            folder=a.inputs/'models'/m['name'];cases=read(folder/'cases.json');manual=NumpyPlan(folder/'preprocessing.json')
            with (folder/'baseline.pkl').open('rb') as f:pp,svc=pickle.load(f)
            with PreparedPipeline(folder,a.library) as py,PreparedPipeline(folder,a.library,preprocessor_library=a.preprocessor) as cc,py.session() as pw,cc.session() as cw:
                raw=cases['rows'];expected=cases['expected']
                xx=cc.preprocessor.transform(raw)
                assert xx.tobytes()==(folder/'transformed.f64').read_bytes()
                assert manual.transform(raw).tobytes()==xx.tobytes()==py.preprocessor.transform(raw).tobytes()
                assert cw.predict_many(raw)==expected==pw.predict_many(raw)
                decisions[m['name']]={'count':len(raw),'transformed_sha256':hashlib.sha256(xx.tobytes()).hexdigest(),
                                    'predictions_sha256':hashlib.sha256(json.dumps(expected,separators=(',',':')).encode()).hexdigest()}
                pairs+=len(raw);features+=len(xx)
                for size,count in [('one',1),('batch',min(128,len(raw)))]:
                    rows=raw[:count];wanted=expected[:count];columns=cases['columns']
                    def materialized_call():
                        checked=cc.preprocessor._reference._materialize(rows)
                        data=cc.preprocessor._module.transform(cc.preprocessor._plan,checked)
                        return cw._worker.predict_buffer(memoryview(data).cast('d'))
                    callbacks={'python_native':lambda:pw.predict_many(rows),
                        'compiled_materialized':materialized_call,
                        'compiled_native':lambda:cw.predict_many(rows),
                        'numpy_native':lambda:pw._worker.predict_buffer(manual.transform(rows)),
                        'sklearn':lambda:svc.predict(pp.transform(pd.DataFrame(rows,columns=columns))).tolist()}
                    for _ in range(3):
                        for call in callbacks.values():assert call()==wanted
                    for repeat in range(31):
                        order=list(ARMS);rng.shuffle(order)
                        for arm in order:
                            t=time.perf_counter_ns();answer=callbacks[arm]();dt=time.perf_counter_ns()-t
                            assert answer==wanted,(m['name'],arm,repeat)
                            r={'model':m['name'],'size':size,'rows':count,'arm':arm,'repeat':repeat,'ns':dt,'matched':True}
                            stream.write(json.dumps(r,separators=(',',':'))+'\n');allrows.append(r)
            print(m['name'],'completed',flush=True)
    write(a.out/'fidelity.json',{'pairs':pairs,'features':features,'models':decisions})
    summary=summarize(allrows);write(a.out/'summary.json',summary)
    print(json.dumps(summary['tasks'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','library','preprocessor','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
