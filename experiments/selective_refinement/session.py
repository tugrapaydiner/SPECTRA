"""One native call per chunk; both retained models count toward memory."""
from __future__ import annotations
import ctypes as C,math
from pathlib import Path
from spectra.svm_lifetime import _NativeOwner
from spectra.svm_shared import _decode
from experiments.budgeted_prototypes.session import decode as fast_decode
MODES={'calibrated':0,'fast':1,'strong':2,'blind':3}
class RefinementSession(_NativeOwner):
    def __init__(self,fast,strong,library,*,threshold,accept_fraction=0.):
        self._init_lifetime('selective refinement')
        if threshold is None:threshold=math.inf
        if type(threshold) not in (float,int) or math.isnan(threshold) or threshold<0:raise ValueError('invalid threshold')
        if type(accept_fraction) not in (float,int) or not math.isfinite(accept_fraction) or not 0<=accept_fraction<=1:raise ValueError('invalid accept fraction')
        raw=[]
        for path in (fast,strong):
            with Path(path).open('rb') as f:data=f.read(64*1024**2+1)
            if len(data)>64*1024**2:raise ValueError('model cap')
            raw.append(data)
        d,p,D,Q,U,labels=fast_decode(raw[0]);ds,ls=_decode(raw[1])
        if d!=ds or labels!=ls:raise ValueError('model feature/class mappings differ')
        self.features=d;self.labels=labels;self.maximum=D
        lib=self._lib=C.CDLL(str(Path(library).resolve(strict=True)))
        lib.sr_abi.restype=C.c_int
        if lib.sr_abi()!=1:raise ValueError('ABI mismatch')
        lib.et_error.restype=C.c_char_p
        lib.sr_create.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_uint64,C.c_double,C.c_double];lib.sr_create.restype=C.c_void_p
        lib.sr_destroy.argtypes=[C.c_void_p];lib.sr_destroy.restype=None
        lib.sr_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];lib.sr_info.restype=C.c_int
        lib.sr_run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_uint64),C.c_int];lib.sr_run.restype=C.c_int
        a,b=(C.create_string_buffer(x,len(x)) for x in raw)
        h=lib.sr_create(a,len(raw[0]),b,len(raw[1]),threshold,accept_fraction)
        if not h:raise ValueError(lib.et_error().decode(errors='replace'))
        self._adopt(h,lib,'sr_destroy')
        with self._operation() as h:
            o=(C.c_uint64*5)();self._check(lib.sr_info(h,o,5));self.info=dict(zip(('features','classes','prototypes','support_vectors','counted_native_bytes'),o))
    def _check(self,rc):
        if rc:raise ValueError(self._lib.et_error().decode(errors='replace'))
    def _run(self,values,mode,stats):
        if type(mode) is not str or mode not in MODES:raise ValueError('unknown refinement mode')
        try:v=memoryview(values)
        except TypeError as e:raise ValueError('writable uint8 buffer required') from e
        data=None
        try:
            if v.format!='B' or v.readonly or not v.c_contiguous or v.ndim not in (1,2):raise ValueError('writable contiguous uint8 required')
            if v.ndim==2 and v.shape[1]!=self.features:raise ValueError('wrong feature width')
            rows=v.nbytes//self.features
            if v.nbytes%self.features or v.nbytes>8000000 or rows>65536:raise ValueError('input geometry or cap')
            data=(C.c_uint8*v.nbytes).from_buffer(v) if v.nbytes else None
            out=(C.c_int*rows)();work=(C.c_uint64*3)()
            with self._operation() as h:self._check(self._lib.sr_run(h,data,rows,self.features,MODES[mode],out,work,3))
            pred=[self.labels[i] for i in out]
            return (pred,dict(zip(('fast_evaluations','strong_evaluations','fast_accepted'),work))) if stats else pred
        finally:del data;v.release()
    def predict_buffer(self,values,*,mode='calibrated'):return self._run(values,mode,False)
    def inspect_buffer(self,values,*,mode='calibrated'):return self._run(values,mode,True)
