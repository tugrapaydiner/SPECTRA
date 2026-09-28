"""Framework-free owning experimental session for a fitted finite-table model."""
from __future__ import annotations
import ctypes as C
from pathlib import Path
from .model import Model
from spectra.svm_lifetime import _NativeOwner
MODES={'selective':0,'exhaustive':1,'scalar':2}

class Session(_NativeOwner):
    def __init__(self,path,library):
        self._init_lifetime('learned-finite session');self.model=Model.load(path)
        lib=C.CDLL(str(Path(library).resolve(strict=True)));self._lib=lib
        lib.lf_abi.restype=C.c_int
        if lib.lf_abi()!=1:raise ValueError('unsupported finite ABI')
        lib.et_error.restype=C.c_char_p
        lib.lf_create.argtypes=[C.c_void_p,C.c_uint64];lib.lf_create.restype=C.c_void_p
        lib.lf_destroy.argtypes=[C.c_void_p];lib.lf_destroy.restype=None
        lib.lf_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];lib.lf_info.restype=C.c_int
        lib.lf_run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_uint64),C.c_int];lib.lf_run.restype=C.c_int
        lib.lf_margins.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.POINTER(C.c_double),C.c_uint64];lib.lf_margins.restype=C.c_int
        raw=C.create_string_buffer(self.model.raw,len(self.model.raw));handle=lib.lf_create(raw,len(self.model.raw))
        if not handle:raise ValueError(lib.et_error().decode())
        self._adopt(handle,lib,'lf_destroy')
        info=(C.c_uint64*6)()
        with self._operation() as handle:self._check(lib.lf_info(handle,info,6))
        self.info=dict(zip(('features','classes','supports','table_entries','native_bytes','weight_planes'),map(int,info)))
    def _check(self,status):
        if status:raise ValueError(self._lib.et_error().decode('utf-8',errors='replace'))
    def _view(self,values):
        try:view=memoryview(values)
        except TypeError:raise ValueError('contiguous uint8 buffer required') from None
        if view.format!='B' or not view.c_contiguous or view.ndim not in (1,2) or (view.ndim==2 and view.shape[1]!=self.model.d):
            view.release();raise ValueError('contiguous uint8 rows required')
        count=view.nbytes
        if count%self.model.d or count>8_000_000 or count//self.model.d>65536:
            view.release();raise ValueError('input geometry')
        data=((C.c_uint8*count).from_buffer_copy(view) if view.readonly else (C.c_uint8*count).from_buffer(view)) if count else None
        return view,data,count//self.model.d
    def predict_buffer(self,values,mode='selective'):
        if type(mode) is not str or mode not in MODES:raise ValueError('unknown mode')
        view,data,rows=self._view(values)
        try:
            result=(C.c_int*rows)();stats=(C.c_uint64*3)()
            with self._operation() as handle:
                self._check(self._lib.lf_run(handle,data,rows,self.model.d,MODES[mode],result,stats,3))
            return [self.model.labels[i] for i in result]
        finally:
            del data;view.release()
    def margins(self,values):
        view,data,rows=self._view(values)
        try:
            n=rows*len(self.model.pairs);out=(C.c_double*n)()
            with self._operation() as handle:self._check(self._lib.lf_margins(handle,data,rows,self.model.d,out,n))
            return bytes(out)
        finally:
            del data;view.release()
