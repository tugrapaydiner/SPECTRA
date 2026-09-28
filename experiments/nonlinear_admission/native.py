"""Native comparison utilities. No modifications to SPECTRA's production engine."""
from __future__ import annotations
import argparse,ctypes as C,hashlib,json,os,pickle,resource,signal,struct,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.nonlinear_admission.study import sha,save,load,verify_freeze,TASKS
from experiments.native_baselines.prepare import export_libsvm
from spectra.svm_export import export_prepared_svc

FLAGS=['-std=c++17','-O3','-mavx2','-fno-fast-math','-ffp-contract=off','-fPIC','-shared']

def bounded():
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})

def execute(command,folder,seconds):
    folder.mkdir(parents=True,exist_ok=False)
    t=time.perf_counter();p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,preexec_fn=bounded,start_new_session=True)
    status='COMPLETE'
    try:out,err=p.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);out,err=p.communicate();status='TIMEOUT'
    if p.returncode and status=='COMPLETE':status='ERROR'
    (folder/'stdout.txt').write_bytes(out);(folder/'stderr.txt').write_bytes(err)
    rec={'command':command,'status':status,'returncode':p.returncode,'wall_seconds':time.perf_counter()-t,'cap_seconds':seconds}
    save(folder/'execution.json',rec);return rec

def export_dense(model,path):
    if hasattr(model,'coefs_'):
        if len(model.coefs_)!=2 or model.activation!='relu' or model.out_activation_!='softmax':raise ValueError('unsupported MLP graph')
        w1,w2=model.coefs_;b1,b2=model.intercepts_;d,h=w1.shape;c=w2.shape[1]
        arrays=(w1,b1,w2,b2)
    else:
        d=model.coef_.shape[1];h=0;c=len(model.classes_);arrays=(model.coef_.T,model.intercept_)
        if model.coef_.shape[0]!=c:raise ValueError('multiclass control required')
    if any(not np.isfinite(v).all() for v in arrays):raise ValueError('invalid model values')
    with Path(path).open('xb') as f:
        f.write(struct.pack('<8sIII',b'SPDENSE1',d,h,c))
        for arr in arrays:f.write(np.ascontiguousarray(arr,dtype='<f8').tobytes())
    return {'features':d,'hidden':h,'labels':model.classes_.tolist(),'classes':c,'parameters':sum(v.size for v in arrays),'sha256':sha(path),'bytes':Path(path).stat().st_size}

class DenseSession:
    def __init__(self,library,path,features,labels):
        self.lib=C.CDLL(str(library));self.d=features;self.labels=tuple(labels);self.ptr=None
        self.lib.dn_error.restype=C.c_char_p
        self.lib.dn_create.argtypes=[C.c_char_p];self.lib.dn_create.restype=C.c_void_p
        self.lib.dn_destroy.argtypes=[C.c_void_p];self.lib.dn_destroy.restype=None
        self.lib.dn_run.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_double)]
        self.ptr=self.lib.dn_create(str(path).encode())
        if not self.ptr:raise ValueError(self.lib.dn_error().decode())
    def predict_buffer(self,x,return_scores=False):
        if not self.ptr:raise ValueError('closed model')
        v=memoryview(x);data=None
        try:
            if v.format!='d' or v.readonly or not v.c_contiguous or v.ndim!=2 or v.shape[1]!=self.d:raise ValueError('contiguous binary64 matrix required')
            n=v.shape[0]
            if n>65536 or n*self.d>8000000:raise ValueError('batch limit')
            data=(C.c_double*(n*self.d)).from_buffer(v) if n else None
            if data is not None and C.addressof(data)%8:raise ValueError('alignment')
            out=(C.c_int*n)();scores=(C.c_double*(n*len(self.labels)))() if return_scores else None
            status=self.lib.dn_run(self.ptr,data,n,self.d,out,scores)
            if status:raise ValueError(self.lib.dn_error().decode())
            labels=[self.labels[i] for i in out]
            return (labels,bytes(scores)) if return_scores else labels
        finally:
            del data;v.release()
    def close(self):
        if self.ptr:self.lib.dn_destroy(self.ptr);self.ptr=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

def build(a):
    from spectra.svm import build_runtime
    verify_freeze(a.root);a.out.mkdir(exist_ok=False)
    started=time.perf_counter()
    paths={}
    paths['spectra']=str(build_runtime(a.out/'spectra',target='avx2'))
    commands={'libsvm':['g++',*FLAGS,'-I'+str(a.vendor),str(a.vendor/'svm.cpp'),str(ROOT/'experiments/native_baselines/libsvm_bridge.cpp'),'-o',str(a.out/'libsvm.so')],
              'dense':['g++',*FLAGS,str(Path(__file__).with_name('dense.cpp')),'-o',str(a.out/'dense.so')]}
    receipts={}
    for name,cmd in commands.items():
        receipt=execute(cmd,a.out/(name+'-compile'),180);receipts[name]=receipt
        if receipt['status']!='COMPLETE':raise RuntimeError(name+' compilation failed')
        paths[name]=str(a.out/(name+'.so'))
    models={}
    for task in TASKS:
        folder=a.out/task;folder.mkdir();entries={}
        for name in ('svm','linear','mlp-101','mlp-202','mlp-303'):
            with (a.root/'final'/task/name/'model.pkl').open('rb') as f:obj=pickle.load(f)
            m=obj['model'];entry={'labels':m.classes_.tolist(),'features':len(obj['scaler'].mean_), 'fit_pickle_sha256':sha(a.root/'final'/task/name/'model.pkl')}
            t=time.perf_counter()
            if name=='svm':
                entry.update(export_prepared_svc(m,folder/'model.srt'));export_libsvm(m,folder/'libsvm.model')
                np.savez(folder/'scaler.npz',mean=obj['scaler'].mean_,scale=obj['scaler'].scale_)
                entry['supports']=len(m.support_vectors_)
            else:entry.update(export_dense(m,folder/(name+'.weights')))
            entry['export_seconds']=time.perf_counter()-t;entries[name]=entry
        x,y,g=load(a.root/'data'/task/'test.npz')
        scaler=np.load(folder/'scaler.npz');z=np.ascontiguousarray((x-scaler['mean'])/scaler['scale'])
        z.tofile(folder/'X.f64');np.save(folder/'y.npy',y);models[task]=entries
    result={'libraries':paths,'sha256':{n:sha(p) for n,p in paths.items()},'models':models,'build_receipts':receipts,
      'all_build_export_seconds':time.perf_counter()-started,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('dense.cpp')]},
      'vendor_sha256':{n:sha(a.vendor/n) for n in ('svm.cpp','svm.h','COPYRIGHT')},
      'model_files':{p.relative_to(a.out).as_posix():sha(p) for t in TASKS for p in sorted((a.out/t).iterdir()) if p.is_file()}}
    save(a.out/'BUILD.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','out','vendor'):p.add_argument('--'+n,type=Path,required=True)
    build(p.parse_args())
