"""Open the frozen HAR test once; record all decisions, margins and subject results."""
from __future__ import annotations
import argparse,hashlib,json,pickle,sys,time,subprocess
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from spectra.svm_shared import PreparedModel
from experiments.native_baselines.runtime import NativeSession
from experiments.native_baselines.prepare import sha,save

def linear_source(model):
    c,d=model.coef_.shape
    coef=','.join(float(v).hex() for v in model.coef_.ravel());bias=','.join(float(v).hex() for v in model.intercept_)
    return f'''#include <cmath>
static const double coef[]={{{coef}}};
static const double bias[]={{{bias}}};
extern "C" int nb_generated_run(const double* x,int rows,int d,int* out,double* scores){{
if(rows<0||rows>65536||d!={d}||1LL*rows*d>8000000||(rows&&(!x||!out)))return 1;
for(long long k=0;k<1LL*rows*d;++k)if(!std::isfinite(x[k]))return 2;
for(int r=0;r<rows;++r){{double best=0;int winner=0;for(int c=0;c<{c};++c){{double s=0;for(int j=0;j<d;++j)s+=coef[c*d+j]*x[r*d+j];s+=bias[c];if(!std::isfinite(s))return 3;if(c==0||s>best){{best=s;winner=c;}}}}out[r]=winner;}}return 0;}}
'''

def main(a):
    a.out.mkdir(exist_ok=False,parents=True)
    freeze=json.loads((a.models/'MODEL_FREEZE.json').read_text());builds=json.loads((a.builds/'builds.json').read_text())
    report={'model_freeze_sha256':sha(a.models/'MODEL_FREEZE.json'),'builds_sha256':sha(a.builds/'builds.json'),'models':{},'scope':'fixed-model test predictions; one new HAR task with original subject-disjoint split'}
    with threadpool_limits(1):
        for name,m in freeze['models'].items():
            folder=a.models/name
            for n,h in m['files'].items():assert sha(folder/n)==h,(name,n)
            with (folder/'model.pkl').open('rb') as f:svc=pickle.load(f)
            if name=='har':
                x=np.loadtxt(a.har/'test/X_test.txt',dtype=np.float64)
                truth=np.loadtxt(a.har/'test/y_test.txt',dtype=np.int64)
                subjects=np.loadtxt(a.har/'test/subject_test.txt',dtype=np.int64)
                assert x.shape==(2947,561) and np.isfinite(x).all()
                x.tofile(folder/'X.f64');save(folder/'truth.json',truth.tolist());save(folder/'subjects.json',subjects.tolist())
            else:x=np.fromfile(folder/'X.f64',dtype=np.float64).reshape(m['rows'],m['features'])
            base=svc.predict(x).tolist();dec=np.asarray(svc._decision_function(x)).reshape(len(x),-1)
            if name!='har':assert base==json.loads((folder/'expected.json').read_text())
            else:save(folder/'expected.json',base)
            r={'rows':len(x),'features':m['features'],'supports':m['supports'],'classes':m['classes'],
               'model_bytes':(folder/'model.srt').stat().st_size,'libsvm_text_bytes':(folder/'libsvm.model').stat().st_size,
               'backends':{}}
            save(a.out/(name+'-sklearn.json'),base)
            with NativeSession(a.builds/'libsvm/libnative_svm.so',m['features'],m['classes'],folder/'libsvm.model') as w:
                p,b=w.predict_buffer(x,return_scores=True)
                values=np.frombuffer(b,dtype=np.float64).reshape(dec.shape)
                r['backends']['libsvm']={'disagreements':sum(a!=b for a,b in zip(p,base)),
                    'max_abs_margin_difference':float(np.max(np.abs(values-dec))), 'input_conversion':'dense-to-sparse nodes inside call'}
                save(a.out/(name+'-libsvm.json'),p)
            for target in ('portable','avx2'):
                lib=a.builds/f'spectra-{target}/libspectra_svm.so'
                with PreparedModel(folder/'model.srt',lib,input_dtype='float64',tables=False) as owner,owner.session() as w:
                    for schedule in ('exhaustive','beretta_cert','binary_stream'):
                        p=w.predict_buffer(x,schedule=schedule);arm=f'spectra-{target}-{schedule}'
                        r['backends'][arm]={'disagreements':sum(a!=b for a,b in zip(p,base))}
                        save(a.out/f'{name}-{arm}.json',p)
            result=builds['generated'].get(name,{'status':'NOT_FINISHED'})
            if result['status']=='PASS':
                with NativeSession(a.builds/f'm2cgen/{name}/model.so',m['features'],m['classes']) as w:
                    p,b=w.predict_buffer(x,return_scores=True);v=np.frombuffer(b,dtype=np.float64).reshape(dec.shape)
                    r['backends']['m2cgen']={'disagreements':sum(a!=b for a,b in zip(p,base)),
                        'max_abs_margin_difference':float(np.max(np.abs(v-dec))),'operation_order':'unmodified generated expression order'}
                    save(a.out/(name+'-m2cgen.json'),p)
            else:r['backends']['m2cgen']={'status':result['status'],'missing':True}
            if name=='har':
                r['svc_correct']=int(np.sum(np.array(base)==truth));r['svc_accuracy']=r['svc_correct']/len(x)
                with (folder/'linear.pkl').open('rb') as f:linear=pickle.load(f)
                lp=linear.predict(x);r['linear_correct']=int(np.sum(lp==truth));r['linear_accuracy']=r['linear_correct']/len(x)
                r['subjects']={str(s):{'rows':int(np.sum(subjects==s)),
                    'svc_correct':int(np.sum((np.array(base)==truth)&(subjects==s))),
                    'linear_correct':int(np.sum((lp==truth)&(subjects==s)))} for s in sorted(set(subjects))}
                save(a.out/'har-linear.json',lp.tolist())
                out=a.builds/'linear';out.mkdir(exist_ok=True)
                (out/'model.cpp').write_text(linear_source(linear))
                cmd=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',str(out/'model.cpp'),'-o',str(out/'model.so')]
                t=time.perf_counter();proc=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
                save(out/'build.json',{'command':cmd,'returncode':proc.returncode,'stderr':proc.stderr,'wall_seconds':time.perf_counter()-t})
                assert proc.returncode==0,proc.stderr
                with NativeSession(out/'model.so',m['features'],m['classes']) as w:
                    got=w.predict_buffer(x);assert got==lp.tolist()
                r['linear_native_matches']=True;r['linear_learned_parameters']=int(linear.coef_.size+linear.intercept_.size)
            report['models'][name]=r
            save(a.out/(name+'-checks.json'),r)
            print(name,json.dumps(r),flush=True)
    save(a.out/'VALIDATION.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('models','har','builds','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
