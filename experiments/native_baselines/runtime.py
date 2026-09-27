"""Small standard-library benchmark adapters, not a new production SVM runtime."""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import threading

class NativeSession:
    def __init__(self,library,features,labels,model_file=None):
        self.features=features;self.labels=tuple(labels);self._lock=threading.Lock();self._ptr=None
        self.lib=C.CDLL(str(Path(library).resolve()))
        if model_file is not None:
            self.lib.nb_create.argtypes=[C.c_char_p,C.c_int];self.lib.nb_create.restype=C.c_void_p
            self.lib.nb_error.restype=C.c_char_p
            self.lib.nb_destroy.argtypes=[C.c_void_p]
            self.lib.nb_run.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_double)]
            self._ptr=self.lib.nb_create(str(Path(model_file).resolve()).encode(),features)
            if not self._ptr:raise ValueError(self.lib.nb_error().decode())
            self.run=self.lib.nb_run
        else:
            self.lib.nb_generated_run.argtypes=[C.POINTER(C.c_double),C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_double)]
            self.run=self.lib.nb_generated_run
        self._closed=False
    def predict_buffer(self,values,*,return_scores=False):
        view=memoryview(values)
        try:
            if view.format!='d' or view.readonly or not view.c_contiguous or view.ndim not in (1,2):raise ValueError('writable contiguous binary64 required')
            if view.ndim==2 and view.shape[1]!=self.features:raise ValueError('bad feature dimension')
            if view.nbytes% (8*self.features):raise ValueError('bad feature count')
            n=view.nbytes//(8*self.features)
            if n>65536 or view.nbytes//8>8000000:raise ValueError('batch limit')
            data=(C.c_double*(view.nbytes//8)).from_buffer(view) if view.nbytes else None
            if data is not None and C.addressof(data)%8:raise ValueError('unaligned input')
            count=len(self.labels)*(len(self.labels)-1)//2
            output=(C.c_int*n)();scores=(C.c_double*(n*count))() if return_scores else None
            with self._lock:
                if self._closed:raise ValueError('closed adapter')
                arguments=[data,n,self.features,output,scores]
                if self._ptr is not None:arguments.insert(0,self._ptr)
                code=self.run(*arguments)
                if code:raise ValueError(f'native comparison failure {code}')
            result=[self.labels[i] for i in output]
            return (result,bytes(scores)) if return_scores else result
        finally:
            if 'data' in locals():del data
            view.release()
    def close(self):
        with self._lock:
            if not self._closed and self._ptr:self.lib.nb_destroy(self._ptr);self._ptr=None
            self._closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
