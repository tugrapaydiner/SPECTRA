"""Native, framework-free comparators with identical raw-code input boundaries."""
from __future__ import annotations
import ctypes as C,hashlib,json,struct,zlib,subprocess,time
from pathlib import Path
from spectra.svm_lifetime import _NativeOwner
from spectra.svm_shared import _decode
from .session import _unique
HEADER=struct.Struct('<8sIIIIIII')

def export_network(arrays):
    import numpy as np
    ws=[];bs=[];i=0
    while f'w{i}' in arrays:
        ws.append(np.asarray(arrays[f'w{i}'],dtype='<f8'));bs.append(np.asarray(arrays[f'b{i}'],dtype='<f8'));i+=1
    if not 1<=i<=4:raise ValueError('invalid network layer inventory')
    d=ws[0].shape[0];c=ws[-1].shape[1];D=int(arrays['maximum'])
    dims=[d]+[w.shape[1] for w in ws];mean=np.asarray(arrays['mean'],dtype='<f8');scale=np.asarray(arrays['scale'],dtype='<f8')
    if len(mean)!=d or len(scale)!=d or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('invalid scaler')
    for j,(w,b) in enumerate(zip(ws,bs)):
        if w.shape!=(dims[j],dims[j+1]) or b.shape!=(dims[j+1],) or not np.isfinite(w).all() or not np.isfinite(b).all():raise ValueError('invalid network geometry/weights')
    meta=json.dumps({'labels':np.asarray(arrays['classes']).tolist()},separators=(',',':'),allow_nan=False).encode()
    payload=struct.pack('<'+'I'*len(dims),*dims)+mean.tobytes()+scale.tobytes()
    for w,b in zip(ws,bs):payload+=w.tobytes()+b.tobytes()
    payload+=meta
    return HEADER.pack(b'SPNET001',d,c,D,i,len(meta),len(payload),zlib.crc32(payload))+payload

def build(out,target='avx2',sanitize=False):
    root=Path(__file__).resolve().parents[2];out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    src=Path(__file__).with_suffix('.cpp');base=root/'spectra/_native/ovo/runtime.cpp';lib=out/'controls.so'
    cmd=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fPIC','-shared']
    if target=='avx2':cmd+=['-mavx2']
    elif target!='portable':raise ValueError('unknown native target')
    if sanitize:cmd+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all']
    cmd+=[f'-DBP_BASE_RUNTIME="{base}"',str(src),'-o',str(lib)]
    start=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    paths=[src]+[x for x in base.parent.rglob('*') if x.suffix in ('.cpp','.hpp')]
    rec={'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'seconds':time.perf_counter()-start,
         'source_sha256':{x.relative_to(root).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}}
    if p.returncode==0:rec['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(rec,indent=2))
    if p.returncode:raise RuntimeError(p.stderr)
    return lib

class ControlSession(_NativeOwner):
    def __init__(self,model,library,maximum=None):
        self._init_lifetime('native baseline')
        with Path(model).open('rb') as f:raw=f.read(64*1024**2+1)
        if not 36<=len(raw)<=64*1024**2:raise ValueError('control model length')
        self.sha256=hashlib.sha256(raw).hexdigest();self.svm=raw[:8]==b'SPCSVM02'
        if self.svm:
            self.features,self.labels=_decode(raw)
            if type(maximum) is not int or not 1<=maximum<=255:raise ValueError('SVM needs explicit code range')
            self.maximum=maximum
        else:
            magic,d,c,D,L,nmeta,n,crc=HEADER.unpack_from(raw)
            if magic!=b'SPNET001' or not 1<=d<=256 or not 2<=c<=128 or not 1<=D<=255 or not 1<=L<=4 or not 1<=nmeta<=65536 or n+nmeta<4*(L+1) or len(raw)!=36+n or zlib.crc32(raw[36:])!=crc:raise ValueError('network header/CRC')
            meta=json.loads(raw[-nmeta:].decode('utf-8'),object_pairs_hook=_unique)
            from spectra.svm_shared import _labels_valid
            if type(meta) is not dict or set(meta)!={'labels'} or not _labels_valid(meta['labels'],c):raise ValueError('network label mapping')
            self.features=d;self.labels=tuple(meta['labels']);self.maximum=D
        self._lib=lib=C.CDLL(str(Path(library).resolve(strict=True)));lib.ct_abi.restype=C.c_int
        if lib.ct_abi()!=1:raise ValueError('control ABI')
        lib.et_error.restype=C.c_char_p
        prefix='ct_svm_' if self.svm else 'ct_nn_'
        create=getattr(lib,prefix+'create');create.argtypes=[C.c_void_p,C.c_uint64]+([C.c_int] if self.svm else []);create.restype=C.c_void_p
        destroy=getattr(lib,prefix+'destroy');destroy.argtypes=[C.c_void_p];destroy.restype=None
        info=getattr(lib,prefix+'info');info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];info.restype=C.c_int
        self._run=getattr(lib,prefix+'run');self._run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int)]+([] if self.svm else [C.POINTER(C.c_double),C.c_uint64]);self._run.restype=C.c_int
        data=C.create_string_buffer(raw,len(raw));h=create(data,len(raw),*([self.maximum] if self.svm else []))
        if not h:raise ValueError(lib.et_error().decode(errors='replace'))
        self._adopt(h,lib,prefix+'destroy')
        with self._operation() as handle:
            out=(C.c_uint64*5)();self._check(info(handle,out,5));self.info=dict(zip(('features','classes','supports_or_neural_parameters','prepared_bytes','worker_bytes'),out))
    def _check(self,status):
        if status:raise ValueError(self._lib.et_error().decode(errors='replace'))
    def _invoke(self,values,mode,scores):
        if mode not in ('compiled','scalar') or type(mode) is not str:raise ValueError('unsupported control mode')
        if scores and self.svm:raise ValueError('SVM observer not exposed by this adapter')
        try:v=memoryview(values)
        except TypeError as e:raise ValueError('uint8 buffer required') from e
        data=None
        try:
            if v.format!='B' or v.readonly or not v.c_contiguous or v.ndim not in (1,2) or (v.ndim==2 and v.shape[1]!=self.features):raise ValueError('writable contiguous uint8 codes required')
            n=v.nbytes//self.features
            if v.nbytes%self.features or v.nbytes>8000000 or n>65536:raise ValueError('control batch cap')
            data=(C.c_uint8*v.nbytes).from_buffer(v) if v.nbytes else None;out=(C.c_int*n)()
            with self._operation() as h:
                if self.svm:self._check(self._run(h,data,n,self.features,int(mode=='scalar'),out))
                else:
                    cells=n*len(self.labels) if scores else 0
                    if cells>8000000:raise ValueError('control score cap')
                    result=(C.c_double*cells)() if scores else None
                    self._check(self._run(h,data,n,self.features,int(mode=='scalar'),out,result,cells))
                    if scores:return bytes(result)
                return [self.labels[i] for i in out]
        finally:del data;v.release()
    def predict_buffer(self,values,*,mode='compiled'):return self._invoke(values,mode,False)
    def scores(self,values,*,mode='compiled'):return self._invoke(values,mode,True)
