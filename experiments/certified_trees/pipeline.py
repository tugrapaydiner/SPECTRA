"""One owning lease for compact certification plus official source-model fallback.

The trusted exporter must bind CBM and JSON to the SAME source model. The compact
binary's bound JSON identity and both expected file hashes are mandatory here.
Full coverage keeps both models and the upstream native library; not compact-only
storage. Numeric source contracts are described separately from data accuracy.
"""
from __future__ import annotations
import ctypes as C,hashlib
from pathlib import Path
from spectra.svm_lifetime import _NativeOwner
from .packing import metadata
from .session import input_buffer,options

def read_checked(path,expected):
    with Path(path).open('rb') as f:raw=f.read(64*1024**2+1)
    if len(raw)>64*1024**2 or type(expected) is not str or len(expected)!=64 or hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('trusted file identity mismatch')
    return raw

class RefinementSession(_NativeOwner):
    def __init__(self,compact,original_cbm,library,*,compact_sha256,cbm_sha256,source_json_sha256):
        self._init_lifetime('owning certified refinement')
        q=read_checked(compact,compact_sha256);cb=read_checked(original_cbm,cbm_sha256);m=metadata(q)
        if m['magic']!='SPTCQ001' or type(source_json_sha256) is not str or m['source_sha256']!=source_json_sha256:raise ValueError('source JSON/CBM pairing required')
        self.features=m['features'];self.classes=m['classes'];self.maximum=m['maximum']
        self.compact_sha256=compact_sha256;self.cbm_sha256=cbm_sha256;self.source_json_sha256=source_json_sha256
        self._lib=lib=C.CDLL(str(Path(library).resolve(strict=True)));lib.cb_abi.restype=C.c_int
        if lib.cb_abi()!=1:raise ValueError('refinement ABI')
        lib.cb_error.restype=C.c_char_p
        lib.cb_pipeline_create.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_uint64,C.c_int];lib.cb_pipeline_create.restype=C.c_void_p
        lib.cb_pipeline_destroy.argtypes=[C.c_void_p];lib.cb_pipeline_destroy.restype=None
        lib.cb_pipeline_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];lib.cb_pipeline_info.restype=C.c_int
        lib.cb_pipeline_run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_uint32),C.POINTER(C.c_uint8)];lib.cb_pipeline_run.restype=C.c_int
        qb=C.create_string_buffer(q,len(q));cbm=C.create_string_buffer(cb,len(cb));handle=lib.cb_pipeline_create(qb,len(q),cbm,len(cb),self.maximum)
        if not handle:raise ValueError(lib.cb_error().decode(errors='replace'))
        self._adopt(handle,lib,'cb_pipeline_destroy')
        with self._operation() as h:
            info=(C.c_uint64*9)();self._check(lib.cb_pipeline_info(h,info,9));self.info={'compact_prepared_bytes':info[7],'compact_bytes':len(q),'cbm_bytes':len(cb),'classes':info[1],'trees':info[2]}
    def _check(self,rc):
        if rc:raise ValueError(self._lib.cb_error().decode(errors='replace'))
    def _invoke(self,values,checkpoint,inspect):
        options(checkpoint,True,False)
        with input_buffer(values,self.features) as (data,n),self._operation() as h:
            out=(C.c_int*n)();steps=(C.c_uint32*n)();fallback=(C.c_uint8*n)()
            self._check(self._lib.cb_pipeline_run(h,data,n,self.features,checkpoint,out,steps,fallback))
            return {'indices':list(out),'trees_evaluated':list(steps),'fallback':list(fallback)} if inspect else list(out)
    def predict_buffer(self,values,*,checkpoint=0):return self._invoke(values,checkpoint,False)
    def inspect_buffer(self,values,*,checkpoint=0):return self._invoke(values,checkpoint,True)
