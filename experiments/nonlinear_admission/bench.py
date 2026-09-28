"""Same-model native comparisons and honest simpler-model controls.

Every row is timed and checked. No per-case fastest-profile selection. Results
report distinct numerical model quality, not an ensemble or universal winner.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pickle,platform,random,statistics,sys,time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.nonlinear_admission.study import TASKS,SEEDS,sha,save,load,verify_freeze
from experiments.nonlinear_admission.native import DenseSession
from experiments.native_baselines.runtime import NativeSession
from spectra.svm_shared import PreparedModel

SEED=20260928

def digest(values):return hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()

def main(a):
    from threadpoolctl import threadpool_limits
    verify_freeze(a.root)
    builds=json.loads((a.builds/'BUILD.json').read_text());paths={k:Path(v) for k,v in builds['libraries'].items()}
    for name,path in paths.items():
        if sha(path)!=builds['sha256'][name]:raise ValueError('native binary changed')
    for file,wanted in builds['model_files'].items():
        if sha(a.builds/file)!=wanted:raise ValueError('exported model or data changed')
    generated_selection=json.loads((a.builds/'GENERATED_SELECTION.json').read_text())
    a.out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    sources=sorted((ROOT/'experiments/nonlinear_admission').glob('*.py'))+[ROOT/'experiments/nonlinear_admission/dense.cpp']
    sources+=sorted((ROOT/'spectra').glob('svm*.py'))
    sources+=sorted((ROOT/'spectra/_native/ovo').rglob('*.cpp'))+sorted((ROOT/'spectra/_native/ovo').rglob('*.hpp'))
    sources += [ROOT/'experiments/native_baselines'/n for n in ('runtime.py','libsvm_bridge.cpp','prepare.py','build.py')]
    protocol={'seed':SEED,'repeats':7,'jobs':[('prepared',1),('prepared',32),('prepared',256),('with_scaling',32)],
      'generated_selection_sha256':sha(a.builds/'GENERATED_SELECTION.json'),
      'amendment_sha256':sha(Path(__file__).with_name('COMPARATOR_AMENDMENT.md')),
      'model_freeze_sha256':sha(a.root/'MODEL_FREEZE.json'),'build_sha256':sha(a.builds/'BUILD.json'),
      'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},'cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
      'affinity':sorted(os.sched_getaffinity(0)),'python':sys.version,'platform':platform.platform(),
      'scope':'complete all-row jobs; checked buffer to fresh labels, optional same fitted StandardScaler. Loading/build and original domain feature extraction excluded.'}
    save(a.out/'PROTOCOL.json',protocol);rows=[];inventory={};preparation={};rng=random.Random(SEED)
    with threadpool_limits(1),(a.out/'timings.jsonl').open('x') as log:
        for task in TASKS:
            x,y,g=load(a.root/'data'/task/'test.npz');info=builds['models'][task]['svm'];d=x.shape[1];labels=info['labels']
            with (a.root/'final'/task/'svm/model.pkl').open('rb') as f:svm=pickle.load(f)
            z=np.ascontiguousarray(svm['scaler'].transform(x));expected={};call={};transform={}
            with ExitStack() as stack:
                t=time.perf_counter();owner=stack.enter_context(PreparedModel(a.builds/task/'model.srt',paths['spectra'],input_dtype='float64'))
                w=stack.enter_context(owner.session());preparation[task]={'spectra_load_seconds':time.perf_counter()-t,'spectra_info':owner.info,'worker_info':w.info}
                t=time.perf_counter();lib=stack.enter_context(NativeSession(paths['libsvm'],d,labels,a.builds/task/'libsvm.model'))
                preparation[task]['libsvm_load_seconds']=time.perf_counter()-t
                expected_svm=np.load(a.root/'evaluation'/(task+'-svm.npy')).tolist()
                call['spectra']=lambda b:w.predict_buffer(b,schedule='beretta_cert')
                call['spectra_exhaustive']=lambda b:w.predict_buffer(b,schedule='exhaustive')
                call['libsvm']=lambda b:lib.predict_buffer(b)
                for name in call:expected[name]=expected_svm;transform[name]=svm['scaler']
                generated=generated_selection['models'][task]
                if 'binary' in generated:
                    if sha(generated['binary'])!=generated['sha256']:raise ValueError('generated binary changed')
                    c=stack.enter_context(NativeSession(Path(generated['binary']),d,labels))
                    call['generated_c']=lambda b:c.predict_buffer(b);expected['generated_c']=expected_svm;transform['generated_c']=svm['scaler']
                for name in ('linear','mlp-101','mlp-202','mlp-303'):
                    with (a.root/'final'/task/name/'model.pkl').open('rb') as f:model=pickle.load(f)
                    dense=stack.enter_context(DenseSession(paths['dense'],a.builds/task/(name+'.weights'),d,model['model'].classes_))
                    call[name]=dense.predict_buffer;transform[name]=model['scaler'];expected[name]=np.load(a.root/'evaluation'/(task+'-'+name+'.npy')).tolist()
                    call[name+'_sklearn']=(lambda b,m=model['model']:m.predict(b).tolist())
                    transform[name+'_sklearn']=model['scaler'];expected[name+'_sklearn']=expected[name]
                with (a.root/'final'/task/'tree/model.pkl').open('rb') as f:tree=pickle.load(f)
                call['tree_sklearn']=lambda b:tree['model'].predict(b).tolist();transform['tree_sklearn']=tree['scaler']
                expected['tree_sklearn']=np.load(a.root/'evaluation'/(task+'-tree.npy')).tolist()
                # One separately frozen model/scaler per family. Final scalers use
                # identical refit rows, and are checked rather than assumed equal.
                for name in call:
                    if not np.array_equal(transform[name].mean_,svm['scaler'].mean_) or not np.array_equal(transform[name].scale_,svm['scaler'].scale_):raise ValueError('different feature transform')
                inventory[task]={'arms':list(call),'rows':len(y),'features':d,'expected_sha256':{n:digest(p) for n,p in expected.items()},'model_bytes':(a.builds/task/'model.srt').stat().st_size,'generated_control':generated}
                def invoke(arm,scope,chunk):
                    out=[]
                    for first in range(0,len(x),chunk):
                        b=transform[arm].transform(x[first:first+chunk]) if scope=='with_scaling' else z[first:first+chunk]
                        out.extend(call[arm](b))
                    return out
                for arm in call:
                    out=invoke(arm,'prepared',256)
                    if out!=expected[arm]:
                        save(a.out/(task+'-'+arm+'-FAILURE.json'),{'differences':[i for i,(u,v) in enumerate(zip(out,expected[arm])) if u!=v]})
                        raise RuntimeError('class fidelity failure: '+task+'/'+arm)
                jobs=[(scope,chunk,arm) for scope,chunk in protocol['jobs'] for arm in call]
                for repeat in range(7):
                    order=list(jobs);rng.shuffle(order)
                    for scope,chunk,arm in order:
                        start=time.perf_counter_ns();output=invoke(arm,scope,chunk);elapsed=time.perf_counter_ns()-start
                        if output!=expected[arm]:raise RuntimeError('timed output differs')
                        row={'task':task,'repeat':repeat,'scope':scope,'batch':chunk,'arm':arm,'rows':len(y),'ns':elapsed,'predictions_sha256':digest(output)}
                        log.write(json.dumps(row)+'\n');log.flush();rows.append(row)
                    print(task,'repeat',repeat,'complete',flush=True)
    grouped={}
    for r in rows:grouped.setdefault((r['task'],r['scope'],r['batch'],r['arm']),[]).append(r['ns'])
    summary={}
    for task in TASKS:
        summary[task]={}
        for scope,chunk in protocol['jobs']:
            values={arm:statistics.median(grouped[task,scope,chunk,arm])/inventory[task]['rows']/1000 for arm in inventory[task]['arms']}
            summary[task][scope+'-'+str(chunk)]={'us_per_row':values,'libsvm_over_spectra':values['libsvm']/values['spectra'],
                'exhaustive_over_spectra':values['spectra_exhaustive']/values['spectra']}
    save(a.out/'INVENTORY.json',inventory);save(a.out/'PREPARATION.json',preparation)
    save(a.out/'SUMMARY.json',{'tasks':summary,'timing_records':len(rows),'repeated_predictions':sum(r['rows'] for r in rows)})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','builds','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
