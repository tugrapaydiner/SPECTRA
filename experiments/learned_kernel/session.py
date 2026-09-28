"""Explicit framework-free owner for one frozen learned radial-kernel model."""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import sys
from spectra.svm_lifetime import _NativeOwner
from .model import decode,MAX_BYTES

MODES={'compiled':0,'direct':1,'compiled_exhaustive':2,'direct_exhaustive':3}

class KernelSession(_NativeOwner):
    def __init__(self,model,library):
        self._init_lifetime('learned kernel')
        with Path(model).open('rb') as f:raw=f.read(MAX_BYTES+1)
        self.meta,body,self.features,self.labels,g,w=decode(raw)
        lib=self._lib=C.CDLL(str(Path(library).resolve(strict=True)))
        lib.mk_abi.restype=C.c_int
        if lib.mk_abi()!=1:raise ValueError('unsupported learned-kernel ABI')
        lib.et_error.restype=C.c_char_p
        lib.mk_create.argtypes=[C.c_void_p,C.c_uint64,C.c_int,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_double),C.c_int,C.POINTER(C.c_uint32),C.c_int];lib.mk_create.restype=C.c_void_p
        lib.mk_destroy.argtypes=[C.c_void_p];lib.mk_destroy.restype=None
        lib.mk_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];lib.mk_info.restype=C.c_int
        lib.mk_run.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_uint64),C.c_int];lib.mk_run.restype=C.c_int
        lib.mk_probe.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_int,C.c_int,C.POINTER(C.c_double),C.c_uint64];lib.mk_probe.restype=C.c_int
        buf=C.create_string_buffer(body,len(body));ga=(C.c_double*len(g))(*g);wa=(C.c_double*len(w))(*w)
        counts=(C.c_uint32*self.features)(*self.meta['feature_counts'])
        pointer=lib.mk_create(buf,len(body),self.meta['power'],self.meta['denominator'],ga,wa,len(g),counts,self.features)
        if not pointer:raise ValueError(lib.et_error().decode(errors='replace'))
        self._adopt(pointer,lib,'mk_destroy')
        with self._operation() as h:
            values=(C.c_uint64*9)();self._check(lib.mk_info(h,values,9))
            self.info=dict(zip(('features','classes','supports','denominator','power','components','table_entries','counted_bytes','admitted'),values))
    def _check(self,status):
        if status:raise ValueError(self._lib.et_error().decode(errors='replace'))
    def _invoke(self,values,mode,probe=False,inspect=False):
        if type(mode) is not str or mode not in MODES:raise ValueError('invalid mode')
        try:view=memoryview(values)
        except TypeError as e:raise ValueError('writable native binary64 buffer required') from e
        data=None
        try:
            endian='<d' if sys.byteorder=='little' else '>d'
            if view.format not in ('d','@d','=d',endian) or view.itemsize!=8 or view.readonly or not view.c_contiguous or view.ndim not in (1,2):raise ValueError('writable contiguous native binary64 required')
            if view.ndim==2 and view.shape[1]!=self.features:raise ValueError('feature width mismatch')
            size=view.nbytes//8
            if size%self.features or size>8_000_000:raise ValueError('element cap or partial row')
            rows=size//self.features
            if rows>65536:raise ValueError('row cap')
            data=(C.c_double*size).from_buffer(view) if size else None
            if data is not None and C.addressof(data)%8:raise ValueError('unaligned input')
            with self._operation() as h:
                if probe:
                    n=rows*len(self.labels)*(len(self.labels)-1)//2*3
                    if n>16_000_000:raise ValueError('probe cap')
                    out=(C.c_double*n)();self._check(self._lib.mk_probe(h,data,rows,self.features,out,n));return bytes(out)
                output=(C.c_int*rows)();stats=(C.c_uint64*4)()
                self._check(self._lib.mk_run(h,data,rows,self.features,MODES[mode],output,stats,4))
                labels=[self.labels[i] for i in output]
                return (labels,dict(zip(('kernel_values','query_exp_calls','pair_evaluations','domain_fallback_rows'),stats))) if inspect else labels
        finally:
            del data;view.release()
    def predict_buffer(self,values,*,mode='compiled'):return self._invoke(values,mode)
    def inspect_buffer(self,values,*,mode='compiled'):return self._invoke(values,mode,inspect=True)
    def probe(self,values):return self._invoke(values,'compiled',probe=True)
