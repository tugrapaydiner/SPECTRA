"""Inert native linear/MLP controls. No machine-learning dependency at inference."""
from __future__ import annotations
from array import array
import ctypes as C
import json,math
from pathlib import Path
from spectra.svm_lifetime import _NativeOwner

def export(model,path,cap):
 import numpy as np
 if hasattr(model,'coefs_'):weights=model.coefs_;bias=model.intercepts_
 else:weights=[model.coef_.T];bias=[model.intercept_]
 if any(w.ndim!=2 for w in weights):raise ValueError('matrix required')
 doc={'format':'spectra.dense-control.v1','widths':[weights[0].shape[0]]+[w.shape[1] for w in weights],
      'labels':model.classes_.tolist(),'cap':cap,'parameters':[]}
 for w,b in zip(weights,bias):doc['parameters'].extend(np.r_[w.ravel(),b].tolist())
 with Path(path).open('x') as f:json.dump(doc,f,separators=(',',':'),allow_nan=False)
 return {'parameters':len(doc['parameters']),'bytes':Path(path).stat().st_size}

class DenseSession(_NativeOwner):
 def __init__(self,path,library):
  self._init_lifetime('dense baseline')
  with Path(path).open('rb') as f:raw=f.read(64*1024*1024+1)
  if len(raw)>64*1024*1024:raise ValueError('control size')
  doc=json.loads(raw)
  if doc['format']!='spectra.dense-control.v1':raise ValueError('control format')
  self.widths=doc['widths'];self.labels=doc['labels'];self.features=self.widths[0]
  if len(self.labels)!=self.widths[-1] or not 2<=len(self.widths)<=5 or any(type(x)is not int or not 1<=x<=4096 for x in self.widths):raise ValueError('control geometry')
  params=array('d',doc['parameters']);self.parameter_count=len(params)
  self._lib=lib=C.CDLL(str(Path(library).resolve()));lib.et_error.restype=C.c_char_p
  lib.ldc_create.argtypes=[C.POINTER(C.c_double),C.c_uint64,C.POINTER(C.c_uint32),C.c_int,C.c_uint32];lib.ldc_create.restype=C.c_void_p
  lib.ldc_destroy.argtypes=[C.c_void_p];lib.ldc_destroy.restype=None
  lib.ldc_run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.POINTER(C.c_int)];lib.ldc_run.restype=C.c_int
  a=(C.c_double*len(params)).from_buffer(params);dims=(C.c_uint32*len(self.widths))(*self.widths)
  h=lib.ldc_create(a,len(params),dims,len(self.widths)-1,doc['cap'])
  if not h:raise ValueError(lib.et_error().decode(errors='replace'))
  self._adopt(h,lib,'ldc_destroy')
 def predict_buffer(self,q,**kwargs):
  view=memoryview(q);data=None
  try:
   if view.format!='B' or view.readonly or not view.c_contiguous or view.ndim not in (1,2) or view.nbytes%self.features or view.nbytes>8000000:raise ValueError('uint8 input')
   if view.ndim==2 and view.shape[1]!=self.features:raise ValueError('shape')
   n=view.nbytes//self.features
   if n>65536:raise ValueError('row cap')
   data=(C.c_uint8*view.nbytes).from_buffer(view) if n else None;out=(C.c_int*n)()
   with self._operation() as h:
    if self._lib.ldc_run(h,data,n,self.features,out):raise ValueError(self._lib.et_error().decode(errors='replace'))
    return [self.labels[i] for i in out]
  finally:del data;view.release()
