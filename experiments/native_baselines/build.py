"""Build unmodified native competitors with recorded, bounded compilation.

Generated source and vendor source live outside the repository's runtime. Failure
is a measured missing comparator, never permission to silently change a model.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pickle,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bounded():
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
def execute(cmd,folder,name,timeout):
    started=time.perf_counter()
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,preexec_fn=bounded)
        record={'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except subprocess.TimeoutExpired as e:
        record={'command':cmd,'returncode':None,'exception':'TimeoutExpired',
                'stdout':(e.stdout or b'').decode(errors='replace') if isinstance(e.stdout,bytes) else e.stdout,
                'stderr':(e.stderr or b'').decode(errors='replace') if isinstance(e.stderr,bytes) else e.stderr}
    record['wall_seconds']=time.perf_counter()-started
    (folder/(name+'.json')).write_text(json.dumps(record,indent=2))
    return record

def generate(model,out):
    import m2cgen
    sys.setrecursionlimit(12000)
    with model.open('rb') as f:svc=pickle.load(f)
    text=m2cgen.export_to_c(svc)
    if len(text.encode())>64*1024**2:raise ValueError('generated source exceeds frozen 64 MiB limit')
    out.write_text(text)
    print(json.dumps({'generator':m2cgen.__version__,'model_sha256':sha(model),'source_sha256':sha(out),'source_bytes':out.stat().st_size}))

def wrapper(c,d):
    n=c*(c-1)//2
    call='v[0]=score(const_cast<double*>(x+r*d));' if c==2 else 'score(const_cast<double*>(x+r*d),v);'
    select='int winner=v[0]>=0?1:0;' if c==2 else f'''int votes[{c}]={{}};int p=0;for(int i=0;i<{c};++i)for(int j=i+1;j<{c};++j){{++votes[v[p++]>0?i:j];}}int winner=0;for(int i=1;i<{c};++i)if(votes[i]>votes[winner])winner=i;'''
    return f'''#include <cmath>
extern "C" {{
#include "model.c"
int nb_generated_run(const double* x,int rows,int d,int* out,double* scores){{
if(rows<0||rows>65536||d!={d}||1LL*rows*d>8000000||(rows&&(!x||!out)))return 1;
for(long long k=0;k<1LL*rows*d;++k)if(!std::isfinite(x[k]))return 2;
for(int r=0;r<rows;++r){{double v[{n}];{call}for(int j=0;j<{n};++j)if(!std::isfinite(v[j]))return 3;
{select}out[r]=winner;if(scores)for(int j=0;j<{n};++j)scores[r*{n}+j]=v[j];}}return 0;
}}
}}
'''

def main(a):
    from spectra.svm import build_runtime
    a.out.mkdir(parents=True,exist_ok=True)
    flags=['-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fPIC','-shared']
    existing=a.out/'builds.json'
    if existing.exists():
        results=json.loads(existing.read_text())
        assert results['libsvm']['sha256']==sha(a.out/'libsvm/libnative_svm.so')
        assert results['libsvm']['source']=={n:sha(a.vendor/n) for n in ('svm.cpp','svm.h','COPYRIGHT')}
    else:
        lib=a.out/'libsvm';lib.mkdir()
        command=['g++',*flags,'-I'+str(a.vendor),str(a.vendor/'svm.cpp'),str(Path(__file__).with_name('libsvm_bridge.cpp')),'-o',str(lib/'libnative_svm.so')]
        result=execute(command,lib,'build',180)
        if result['returncode']!=0:raise RuntimeError('native LIBSVM build failed')
        results={'libsvm':{'sha256':sha(lib/'libnative_svm.so'),'build':result,'source':{n:sha(a.vendor/n) for n in ('svm.cpp','svm.h','COPYRIGHT')}},'generated':{}}
        for target in ('portable','avx2'):
            p=build_runtime(a.out/('spectra-'+target),target=target)
            results['spectra-'+target]={'library':str(p),'sha256':sha(p)}
        existing.write_text(json.dumps(results,indent=2))
    if a.only=='core': return
    manifest=json.loads((a.models/'MODEL_FREEZE.json').read_text())
    # The new larger model is last; small controls are not selected from timings.
    for name in [n for n in manifest['models'] if n!='har']+['har']:
        if a.only and a.only!=name: continue
        if name in results['generated']: raise ValueError('existing build receipt; use a new build directory')
        info=manifest['models'][name];folder=a.out/'m2cgen'/name;folder.mkdir(parents=True)
        generated=execute([sys.executable,str(Path(__file__).resolve()),'--generate',str(a.models/name/'model.pkl'),str(folder/'model.c')],folder,'generate',120)
        r={'generate':generated,'status':'GENERATION_FAILED'}
        if generated['returncode']==0:
            (folder/'wrapper.cpp').write_text(wrapper(len(info['classes']),info['features']))
            cmd=['g++',*flags,str(folder/'wrapper.cpp'),'-o',str(folder/'model.so')]
            compiled=execute(cmd,folder,'compile',180);r.update(compile=compiled,status='COMPILE_FAILED',source_bytes=(folder/'model.c').stat().st_size,source_sha256=sha(folder/'model.c'))
            if compiled['returncode']==0:r.update(status='PASS',binary_bytes=(folder/'model.so').stat().st_size,binary_sha256=sha(folder/'model.so'))
        results['generated'][name]=r
        (a.out/'builds.json').write_text(json.dumps(results,indent=2));print(name,r['status'],flush=True)
    print('build inventory complete',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--generate':generate(Path(sys.argv[2]),Path(sys.argv[3]))
    else:
        p=argparse.ArgumentParser(description=__doc__)
        for n in ('vendor','models','out'):p.add_argument('--'+n,type=Path,required=True)
        p.add_argument('--only',choices=['core','wine-101','wdbc-101','chess-101','penguins-101','titanic-101','zoo-101','har'])
        main(p.parse_args())
