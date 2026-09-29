"""One owning native call, explicit unresolved outcomes and source verification.

No fitting or numerical frameworks. Native code and verified model objects are
trusted. The proof concerns the original arithmetic contract, not true labels.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
import ctypes as C
from pathlib import Path
from spectra.svm_lifetime import _NativeOwner
from .packed import verify,metadata,MAX_BYTES

@dataclass(frozen=True,init=False)
class VerifiedCompact:
    raw: bytes
    info: object
    def __init__(self,source:bytes,raw:bytes,*,backend='dyadic'):
        receipt=verify(source,raw,backend=backend)
        object.__setattr__(self,'raw',bytes(raw))
        object.__setattr__(self,'info',MappingProxyType(receipt))
    @classmethod
    def from_files(cls,source,packed,*,backend='dyadic'):
        def read(p):
            with Path(p).open('rb') as f:v=f.read(MAX_BYTES+1)
            if len(v)>MAX_BYTES:raise ValueError('model file cap')
            return v
        return cls(read(source),read(packed),backend=backend)

FIELDS=('certified_first','certified_second','official_rows','unresolved_rows',
        'first_leaf_vectors','second_leaf_vectors','routed_trees','compact_prepared_bytes')

class TreeSession(_NativeOwner):
    def __init__(self,library,*,first=None,second=None,official_model=None,official_library=None,
                 features=None,maximum=None,classes=None):
        self._init_lifetime('tree certificate pipeline')
        for value in (first,second):
            if value is not None and type(value) is not VerifiedCompact:raise ValueError('source-verified compact object required')
        if first:
            self.features=first.info['features'];self.maximum=first.info['maximum'];self.classes=first.info['classes']
        else:self.features,self.maximum,self.classes=features,maximum,classes
        for value,lo,hi in ((self.features,1,256),(self.maximum,1,255),(self.classes,2,64)):
            if type(value) is not int or not lo<=value<=hi:raise ValueError('explicit bounded source shape required')
        if second and not first:raise ValueError('refinement needs first precision')
        if (official_model is None)!=(official_library is None):raise ValueError('official model and library must be supplied together')
        cbm=b''
        if official_model is not None:
            with Path(official_model).open('rb') as f:cbm=f.read(MAX_BYTES+1)
            if not 0<len(cbm)<=MAX_BYTES:raise ValueError('original model byte cap')
        self.has_second=second is not None;self.has_official=official_model is not None
        self._lib=lib=C.CDLL(str(Path(library).resolve(strict=True)))
        lib.tc_abi.restype=C.c_int
        if lib.tc_abi()!=2:raise ValueError('unsupported tree ABI')
        lib.tc_error.restype=C.c_char_p
        lib.tc_create.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_uint64,C.c_char_p,C.c_void_p,C.c_uint64,C.c_uint32,C.c_uint32,C.c_uint32]
        lib.tc_create.restype=C.c_void_p;lib.tc_destroy.argtypes=[C.c_void_p];lib.tc_destroy.restype=None
        lib.tc_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];lib.tc_info.restype=C.c_int
        lib.tc_run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int32),C.POINTER(C.c_uint32),C.POINTER(C.c_uint64),C.c_int]
        lib.tc_run.restype=C.c_int
        a=first.raw if first else b'';b=second.raw if second else b''
        arrays=[C.create_string_buffer(v,len(v)) if v else None for v in (a,b,cbm)]
        path=str(Path(official_library).resolve(strict=True)).encode('utf-8') if self.has_official else None
        h=lib.tc_create(arrays[0],len(a),arrays[1],len(b),path,arrays[2],len(cbm),self.features,self.maximum,self.classes)
        if not h:raise ValueError(lib.tc_error().decode(errors='replace'))
        self._adopt(h,lib,'tc_destroy')
        with self._operation() as ptr:
            info=(C.c_uint64*8)();self._check(lib.tc_info(ptr,info,8))
            self.info=dict(zip(('features','maximum','classes','trees','first_prepared_bytes','second_prepared_bytes','official_loaded','second_loaded'),info))
        self.info['official_serialized_bytes']=len(cbm)
    def _check(self,status):
        if status:raise ValueError(self._lib.tc_error().decode(errors='replace'))
    def _call(self,values,mode,checkpoint,refine,fallback,inspect):
        if type(mode) is not str or mode not in ('scalar','tiled'):raise ValueError('unknown traversal mode')
        if type(checkpoint) is not int or not 0<=checkpoint<=2**31-1:raise ValueError('invalid checkpoint')
        for flag in (refine,fallback):
            if type(flag) is not bool:raise ValueError('boolean policy required')
        if refine and not self.has_second:raise ValueError('second precision not loaded')
        if fallback and not self.has_official:raise ValueError('official fallback not loaded')
        try:view=memoryview(values)
        except TypeError as e:raise ValueError('uint8 buffer required') from e
        data=None
        try:
            if view.format!='B' or view.readonly or not view.c_contiguous or view.ndim not in (1,2):raise ValueError('writable contiguous uint8 buffer required')
            if view.ndim==2 and view.shape[1]!=self.features:raise ValueError('feature width mismatch')
            n=view.nbytes//self.features
            if view.nbytes%self.features or view.nbytes>8_000_000 or n>65536:raise ValueError('input cap/shape')
            data=(C.c_uint8*view.nbytes).from_buffer(view) if view.nbytes else None
            with self._operation() as h:
                out=(C.c_int32*n)();used=(C.c_uint32*n)();stats=(C.c_uint64*8)()
                self._check(self._lib.tc_run(h,data,n,self.features,int(mode=='tiled'),checkpoint,refine,fallback,out,used,stats,8))
                indices=list(out)
                if any(not -1<=i<self.classes for i in indices):raise ValueError('invalid native class')
                if inspect:return {'indices':indices,'trees_evaluated':list(used),'work':dict(zip(FIELDS,stats))}
                return indices
        finally:del data;view.release()
    def predict_buffer(self,values,*,mode='tiled',checkpoint=0,refine=False,fallback=False):
        return self._call(values,mode,checkpoint,refine,fallback,False)
    def inspect_buffer(self,values,*,mode='tiled',checkpoint=0,refine=False,fallback=False):
        return self._call(values,mode,checkpoint,refine,fallback,True)
