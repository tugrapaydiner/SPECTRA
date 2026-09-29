"""Framework-free trusted-model tree execution with explicit unresolved outputs.

Offline verify_binary binds a binary to its source and certificate policy. Runtime
requires its expected hash from that trusted receipt; a hash is not a signature.
Never interpret -1 (UNRESOLVED) as a predicted class or discard it from accuracy.
"""
from __future__ import annotations
from contextlib import contextmanager,ExitStack
import ctypes as C
import hashlib
from pathlib import Path
from spectra.svm_lifetime import _NativeOwner
from .packing import metadata

@contextmanager
def input_buffer(values,features):
    try:v=memoryview(values)
    except TypeError as e:raise ValueError('writable contiguous uint8 buffer required') from e
    data=None
    try:
        if v.format!='B' or v.readonly or not v.c_contiguous or v.ndim not in (1,2):raise ValueError('writable contiguous uint8 buffer required')
        if v.ndim==2 and v.shape[1]!=features:raise ValueError('feature width mismatch')
        rows=v.nbytes//features
        if v.nbytes%features or v.nbytes>8000000 or rows>65536:raise ValueError('input count/shape cap')
        data=(C.c_uint8*v.nbytes).from_buffer(v) if v.nbytes else None
        yield data,rows
    finally:del data;v.release()

def options(checkpoint,pairwise,scalar):
    if type(checkpoint) is not int or not 0<=checkpoint<=4096:raise ValueError('checkpoint must be an integer in0..4096')
    if type(pairwise) is not bool or type(scalar) is not bool:raise ValueError('boolean mode flags required')

class TreeSession(_NativeOwner):
    def __init__(self,model,library,*,expected_sha256):
        self._init_lifetime('certified tree model')
        with Path(model).open('rb') as f:raw=f.read(64*1024**2+1)
        self.sha256=hashlib.sha256(raw).hexdigest()
        if type(expected_sha256) is not str or len(expected_sha256)!=64 or self.sha256!=expected_sha256:raise ValueError('trusted binary identity mismatch')
        self.meta=metadata(raw);self.features=self.meta['features'];self.classes=self.meta['classes']
        self._lib=lib=C.CDLL(str(Path(library).resolve(strict=True)))
        lib.st_abi.restype=C.c_int
        if lib.st_abi()!=1:raise ValueError('unsupported tree ABI')
        lib.st_error.restype=C.c_char_p
        lib.st_create.argtypes=[C.c_void_p,C.c_uint64];lib.st_create.restype=C.c_void_p
        lib.st_destroy.argtypes=[C.c_void_p];lib.st_destroy.restype=None
        lib.st_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];lib.st_info.restype=C.c_int
        lib.st_run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_uint32),C.POINTER(C.c_int)];lib.st_run.restype=C.c_int
        lib.st_scores.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.POINTER(C.c_double),C.c_uint64];lib.st_scores.restype=C.c_int
        lib.st_hybrid.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_uint32),C.POINTER(C.c_uint8)];lib.st_hybrid.restype=C.c_int
        blob=C.create_string_buffer(raw,len(raw));h=lib.st_create(blob,len(raw))
        if not h:raise ValueError(lib.st_error().decode(errors='replace'))
        self._adopt(h,lib,'st_destroy')
        self._run_address=C.cast(lib.st_run,C.c_void_p).value
        if hasattr(lib,'st_layout'):lib.st_layout.restype=C.c_int;self.tile_rows=lib.st_layout()
        else:self.tile_rows=0
        with self._operation() as handle:
            out=(C.c_uint64*9)();self._check(lib.st_info(handle,out,9));self.info=dict(zip(('features','classes','trees','predicates','leaf_scalars','bits','binary_bytes','prepared_bytes','pairwise'),out))
    def _check(self,status):
        if status:raise ValueError(self._lib.st_error().decode(errors='replace'))
    def inspect_buffer(self,values,*,checkpoint=0,pairwise=True,scalar=False):
        options(checkpoint,pairwise,scalar)
        with input_buffer(values,self.features) as (data,n),self._operation() as h:
            out=(C.c_int*n)();steps=(C.c_uint32*n)();approx=(C.c_int*n)()
            self._check(self._lib.st_run(h,data,n,self.features,checkpoint,int(pairwise),int(scalar),out,steps,approx))
            return {'indices':list(out),'trees_evaluated':list(steps),'approximate_indices':list(approx)}
    def predict_buffer(self,values,*,checkpoint=0,pairwise=True,scalar=False):
        options(checkpoint,pairwise,scalar)
        with input_buffer(values,self.features) as (data,n),self._operation() as h:
            out=(C.c_int*n)();steps=(C.c_uint32*n)();approx=(C.c_int*n)()
            self._check(self._lib.st_run(h,data,n,self.features,checkpoint,int(pairwise),int(scalar),out,steps,approx))
            return list(out)
    def scores(self,values):
        with input_buffer(values,self.features) as (data,n),self._operation() as h:
            cells=n*self.classes
            if cells>8000000:raise ValueError('score count cap')
            out=(C.c_double*cells)();self._check(self._lib.st_scores(h,data,n,self.features,out,cells));return bytes(out)
    def hybrid(self,other,values,*,checkpoint=0,inspect=False):
        options(checkpoint,True,False)
        if type(other) is not TreeSession or other is self:raise ValueError('distinct stock TreeSession fallback required')
        # Runtime object ABI must originate in this exact loaded library. The C
        # check separately binds dimensions and original source-model identity.
        if self._lib._handle!=other._lib._handle:raise ValueError('hybrid libraries must be identical')
        with input_buffer(values,self.features) as (data,n),ExitStack() as stack:
            handles={id(owner):stack.enter_context(owner._operation()) for owner in sorted((self,other),key=id)}
            out=(C.c_int*n)();steps=(C.c_uint32*n)();fallback=(C.c_uint8*n)()
            self._check(self._lib.st_hybrid(handles[id(self)],handles[id(other)],data,n,self.features,checkpoint,out,steps,fallback))
            return {'indices':list(out),'trees_evaluated':list(steps),'fallback':list(fallback)} if inspect else list(out)
