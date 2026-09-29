"""Mechanical FP32 conversion, not fitting. Raw-code neural baseline interface."""
from pathlib import Path
import ctypes as C,hashlib,json,struct,zlib,subprocess,time
from .controls import ControlSession,HEADER
from .session import _unique
from spectra.svm_shared import _labels_valid

def export(arrays):
    import numpy as np
    ws=[];bs=[];i=0
    while f'w{i}' in arrays:ws.append(np.asarray(arrays[f'w{i}'],dtype='<f4'));bs.append(np.asarray(arrays[f'b{i}'],dtype='<f4'));i+=1
    if not 1<=i<=4:raise ValueError('FP32 layer count')
    d=ws[0].shape[0];c=ws[-1].shape[1];D=int(arrays['maximum']);dims=[d]+[w.shape[1] for w in ws]
    mean=np.asarray(arrays['mean'],dtype='<f4');scale=np.asarray(arrays['scale'],dtype='<f4')
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('FP32 invalid scaler')
    payload=struct.pack('<'+'I'*len(dims),*dims)+mean.tobytes()+scale.tobytes()
    for j,(w,b) in enumerate(zip(ws,bs)):
        if w.shape!=(dims[j],dims[j+1]) or b.shape!=(dims[j+1],) or not np.isfinite(w).all() or not np.isfinite(b).all():raise ValueError('FP32 conversion overflow/shape')
        payload+=w.tobytes()+b.tobytes()
    meta=json.dumps({'labels':np.asarray(arrays['classes']).tolist()},separators=(',',':'),allow_nan=False).encode();payload+=meta
    return HEADER.pack(b'SPNF0001',d,c,D,i,len(meta),len(payload),zlib.crc32(payload))+payload

def build(out,blas):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);src=Path(__file__).with_suffix('.cpp');root=src.parents[2];blas=Path(blas).resolve(strict=True);lib=out/'control.so'
    cmd=['g++','-std=c++17','-O3','-mavx2','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',f'-DBP_BASE_RUNTIME="{root}/spectra/_native/ovo/runtime.cpp"',str(src),str(blas),'-Wl,-rpath,'+str(blas.parent),'-o',str(lib)]
    start=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    record={'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'seconds':time.perf_counter()-start,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'blas_sha256':hashlib.sha256(blas.read_bytes()).hexdigest()}
    if p.returncode==0:record['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(record,indent=2))
    if p.returncode:raise RuntimeError(p.stderr)
    return lib

class FloatSession(ControlSession):
    def __init__(self,model,library):
        self._init_lifetime('FP32 native baseline');self.svm=False
        with Path(model).open('rb') as f:raw=f.read(64*1024**2+1)
        if not 36<=len(raw)<=64*1024**2:raise ValueError('FP32 model length')
        magic,d,c,D,L,meta,n,crc=HEADER.unpack_from(raw)
        if magic!=b'SPNF0001' or not 1<=d<=256 or not 2<=c<=128 or not 1<=D<=255 or not 1<=L<=4 or not 1<=meta<=65536 or len(raw)!=36+n or zlib.crc32(raw[36:])!=crc:raise ValueError('FP32 metadata/CRC')
        doc=json.loads(raw[-meta:].decode('utf-8'),object_pairs_hook=_unique)
        if type(doc) is not dict or set(doc)!={'labels'} or not _labels_valid(doc['labels'],c):raise ValueError('FP32 labels')
        self.features=d;self.maximum=D;self.labels=tuple(doc['labels']);self.sha256=hashlib.sha256(raw).hexdigest()
        self._lib=lib=C.CDLL(str(Path(library).resolve(strict=True)));lib.ct_abi.restype=C.c_int
        if lib.ct_abi()!=1:raise ValueError('FP32 ABI')
        lib.et_error.restype=C.c_char_p;lib.ct_blas_config.restype=C.c_char_p;self.blas_config=lib.ct_blas_config().decode()
        lib.ct_nn_create.argtypes=[C.c_void_p,C.c_uint64];lib.ct_nn_create.restype=C.c_void_p
        lib.ct_nn_destroy.argtypes=[C.c_void_p];lib.ct_nn_destroy.restype=None
        lib.ct_nn_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];lib.ct_nn_info.restype=C.c_int
        self._run=lib.ct_nn_run;self._run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_double),C.c_uint64];self._run.restype=C.c_int
        data=C.create_string_buffer(raw,len(raw));h=lib.ct_nn_create(data,len(raw))
        if not h:raise ValueError(lib.et_error().decode(errors='replace'))
        self._adopt(h,lib,'ct_nn_destroy')
        with self._operation() as h:
            out=(C.c_uint64*5)();self._check(lib.ct_nn_info(h,out,5));self.info=dict(zip(('features','classes','supports_or_neural_parameters','prepared_bytes','worker_bytes'),out))
