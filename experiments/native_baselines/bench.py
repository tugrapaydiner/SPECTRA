"""Native competitor benchmark with unchanged frozen models and all test inputs.

Input-specific conversion, API checks and fresh labels are charged. Model loading,
compilation and original sensor feature extraction are outside these warm timers.
"""
from __future__ import annotations
import argparse,hashlib,json,os,platform,random,statistics,sys,time
from array import array
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from spectra.svm_shared import PreparedModel
from experiments.native_baselines.runtime import NativeSession
from experiments.native_baselines.prepare import sha,save
CHUNKS=(1,32,256);REPEATS=7;SEED=2026092709

def digest(values):return hashlib.sha256(json.dumps(values,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def summarize(records,validation,builds):
    groups={}
    for row in records:groups.setdefault((row['model'],row['arm'],row['chunk']),[]).append(row['ns'])
    out={}
    for name,info in validation['models'].items():
        arms=sorted({a for m,a,c in groups if m==name});d={}
        for chunk in CHUNKS:
            ns={a:statistics.median(groups[name,a,chunk]) for a in arms}
            d[str(chunk)]={'job_median_ns':ns,'us_per_row':{a:v/info['rows']/1000 for a,v in ns.items()}}
        out[name]={'rows':info['rows'],'features':info['features'],'supports':info['supports'],'batches':d}
    h=out['har']['batches']['32']['us_per_row'];native=h['libsvm'];s=h['spectra-avx2-beretta_cert']
    return {'format':'spectra.native_comparison.v1','cells':len(records),'repeated_predictions':sum(r['rows'] for r in records),
            'models':out,'har_default_avx2_over_libsvm':s/native,
            'har_linear_speedup_over_default_avx2':s/h['linear_native'],
            'har_generated_baseline_complete':builds['generated']['har']['status']=='PASS',
            'native_admission':'INCOMPLETE' if builds['generated']['har']['status']!='PASS' else ('PASS' if s<=min(native,h['m2cgen'])/1.2 else 'FAIL'),
            'task_admission':'FAIL' if validation['models']['har']['linear_accuracy']>=validation['models']['har']['svc_accuracy'] and h['linear_native']<s else 'UNDETERMINED',
            'scope':'internal native API complete jobs; no process-startup or raw-sensor preprocessing advantage'}

def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    freeze=json.loads((a.models/'MODEL_FREEZE.json').read_text());validation=json.loads((a.validation/'VALIDATION.json').read_text());builds=json.loads((a.builds/'builds.json').read_text())
    sources=[p for p in ROOT.rglob('*') if p.is_file() and (p.is_relative_to(ROOT/'spectra') or p.is_relative_to(Path(__file__).parent)) and p.suffix in ('.py','.hpp','.cpp','.md')]
    setup={'seed':SEED,'repeats':REPEATS,'chunks':CHUNKS,'models':list(freeze['models']),
           'freeze_sha256':sha(a.models/'MODEL_FREEZE.json'),'validation_sha256':sha(a.validation/'VALIDATION.json'),
           'builds_sha256':sha(a.builds/'builds.json'),'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
           'libraries_sha256':{p.relative_to(a.builds).as_posix():sha(p) for p in a.builds.rglob('*.so')},
           'inputs_sha256':{name:sha(a.models/name/'X.f64') for name in freeze['models']},
           'python':sys.version,'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)),
           'cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
           'scope':'one checked native call per chunk; LIBSVM dense-to-sparse conversion charged; fresh label list and full-job concatenation included; loading/compilation and feature extraction excluded'}
    save(a.out/'protocol.json',setup)
    rng=random.Random(SEED);records=[]
    with (a.out/'rows.jsonl').open('x') as log:
        for name,m in freeze['models'].items():
            folder=a.models/name;d=m['features'];x=array('d');x.frombytes((folder/'X.f64').read_bytes());assert len(x)==m['rows']*d
            calls={};resources=[];expected={}
            def register(arm,call,resource):
                resources.append(resource);calls[arm]=call
                expected[arm]=json.loads((a.validation/f'{name}-{arm}.json').read_text())
            try:
                lib=NativeSession(a.builds/'libsvm/libnative_svm.so',d,m['classes'],folder/'libsvm.model');register('libsvm',lib.predict_buffer,lib)
                if builds['generated'][name]['status']=='PASS':
                    gen=NativeSession(a.builds/f'm2cgen/{name}/model.so',d,m['classes']);register('m2cgen',gen.predict_buffer,gen)
                for target in ('portable','avx2'):
                    owner=PreparedModel(folder/'model.srt',a.builds/f'spectra-{target}/libspectra_svm.so',input_dtype='float64',tables=False)
                    resources.append(owner);w=owner.session();resources.append(w)
                    for schedule in ('exhaustive','beretta_cert','binary_stream'):
                        arm=f'spectra-{target}-{schedule}';calls[arm]=lambda values,w=w,schedule=schedule:w.predict_buffer(values,schedule=schedule)
                        expected[arm]=json.loads((a.validation/f'{name}-{arm}.json').read_text())
                if name=='har':
                    linear=NativeSession(a.builds/'linear/model.so',d,m['classes']);resources.append(linear);calls['linear_native']=linear.predict_buffer
                    expected['linear_native']=json.loads((a.validation/'har-linear.json').read_text())
                # Slice buffer views before timing, identically for every implementation.
                batches={c:[memoryview(x)[r*d:min(r+c,m['rows'])*d] for r in range(0,m['rows'],c)] for c in CHUNKS}
                def invoke(arm,chunk):
                    result=[]
                    for batch in batches[chunk]:result.extend(calls[arm](batch))
                    return result
                for arm in calls:
                    assert invoke(arm,256)==expected[arm],('warmup',name,arm)
                jobs=[(chunk,arm) for chunk in CHUNKS for arm in calls]
                for repeat in range(REPEATS):
                    order=list(jobs);rng.shuffle(order)
                    for chunk,arm in order:
                        t=time.perf_counter_ns();result=invoke(arm,chunk);ns=time.perf_counter_ns()-t
                        assert result==expected[arm],(name,repeat,chunk,arm)
                        row={'model':name,'repeat':repeat,'chunk':chunk,'arm':arm,'rows':len(result),'ns':ns,'output_sha256':digest(result)}
                        records.append(row);log.write(json.dumps(row,separators=(',',':'))+'\n');log.flush()
                print(name,'complete',len(jobs)*REPEATS,'cells',flush=True)
            finally:
                for resource in reversed(resources):resource.close()
    result=summarize(records,validation,builds);save(a.out/'summary.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('models','validation','builds','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
