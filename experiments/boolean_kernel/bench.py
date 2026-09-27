"""Frozen native comparisons; each complete job includes checked buffer API/output."""
from array import array
import argparse,hashlib,json,os,platform,random,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from spectra.svm_shared import PreparedModel
from experiments.native_baselines.runtime import NativeSession
MODELS=('wine-101','wdbc-101','chess-101','penguins-101','titanic-101','zoo-101','har')
CHUNKS=(1,32,256);REPEATS=11;SEED=20260928


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(labels):return hashlib.sha256(json.dumps(labels,separators=(',',':')).encode()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True)


def inventory():
    paths=sorted((ROOT/'spectra').rglob('*.py'))+sorted((ROOT/'spectra/_native').rglob('*.hpp'))+sorted((ROOT/'spectra/_native').rglob('*.cpp'))
    paths += [Path(__file__),Path(__file__).with_name('PROTOCOL.md'),ROOT/'experiments/native_baselines/runtime.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((a.models/'MODEL_FREEZE.json').read_text())['models'][a.model]
    corpus=json.loads((a.models.parent/'SHA256.json').read_text());folder=a.models/a.model
    bindings={f:sha(folder/f) for f in ('model.srt','X.f64','expected.json','libsvm.model')}
    assert all(v==corpus[f'models/{a.model}/{k}']['sha256'] for k,v in bindings.items())
    expected=json.loads((folder/'expected.json').read_text());d=frozen['features'];n=frozen['rows']
    x=array('d');x.frombytes((folder/'X.f64').read_bytes());assert len(x)==n*d
    controls={};resources=[]
    def reg(label,library,boolean='off',schedule='beretta_cert'):
        owner=PreparedModel(folder/'model.srt',library,boolean=boolean,input_dtype='float64');resources.append(owner)
        w=owner.session();resources.append(w)
        controls[label]=lambda values:w.predict_buffer(values,schedule=schedule)
    reg('parent_default',a.controls/'parent.so')
    reg('parent_stream',a.controls/'parent.so',schedule='binary_stream')
    for b in ('off','packed','lookup'):
        for s in ('beretta_cert','exhaustive'):
            reg(f'{b}_{s}',a.library,boolean=b,schedule=s)
    libsvm=NativeSession(a.controls/'libsvm.so',d,frozen['classes'],folder/'libsvm.model')
    resources.append(libsvm);controls['libsvm']=libsvm.predict_buffer
    if a.model!='har':
        gen=NativeSession(a.controls/(a.model+'.so'),d,frozen['classes'])
        resources.append(gen);controls['generated_c']=gen.predict_buffer
    else:
        # Comparator timed as its own DIFFERENT model; expected labels remain separate.
        linear=NativeSession(a.controls/'linear.so',d,frozen['classes']);resources.append(linear)
        controls['linear_model']=linear.predict_buffer
    protocol={'format':'spectra.boolean_kernel.bench.v1','model':a.model,'models':MODELS,'seed':SEED,'repeats':REPEATS,
        'chunks':CHUNKS,'arms':list(controls),'source_sha256':inventory(),'bindings':bindings,
        'base_commit':'c9f8bb6a2ca30d6840c69b463fd6dc1ea046ad6f',
        'libraries':{p.name:sha(p) for p in [a.library,*sorted(a.controls.glob('*.so'))] if not p.name.startswith('observe')},
        'cpu':next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),
        'python':sys.version,'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)),
        'scope':'one checked native call per prepared chunk and fresh labels; full-job concatenation included; load/build/preprocessing excluded'}
    proto=a.out/(a.model+'-protocol.json')
    if proto.exists():
        old=json.loads(proto.read_text());assert old['source_sha256']==protocol['source_sha256'] and old['libraries']==protocol['libraries']
    else:write(proto,protocol)
    batches={c:[memoryview(x)[i*d:min(i+c,n)*d] for i in range(0,n,c)] for c in CHUNKS}
    wanted={k:expected for k in controls}
    if a.model=='har':wanted['linear_model']=json.loads((a.models.parent/'runs/matched/validation/har-linear.json').read_text())
    def run(arm,chunk):
        result=[]
        for b in batches[chunk]:result.extend(controls[arm](b))
        return result
    try:
        for arm in controls:assert run(arm,256)==wanted[arm],('warmup',a.model,arm)
        for repeat in range(a.start,min(a.start+a.count,REPEATS)):
            jobs=[(chunk,arm) for chunk in CHUNKS for arm in controls]
            random.Random(SEED+100*MODELS.index(a.model)+repeat).shuffle(jobs)
            records=[]
            for chunk,arm in jobs:
                start=time.perf_counter_ns();result=run(arm,chunk);elapsed=time.perf_counter_ns()-start
                if result!=wanted[arm]:raise AssertionError((a.model,repeat,chunk,arm))
                records.append({'model':a.model,'repeat':repeat,'chunk':chunk,'arm':arm,'rows':n,'ns':elapsed,'output_sha256':digest(result)})
            write(a.out/f'{a.model}-{repeat:02d}.json',records)
            print(a.model,repeat,'complete',flush=True)
    finally:
        for r in reversed(resources):r.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('models','controls','library','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=REPEATS)
    main(p.parse_args())
