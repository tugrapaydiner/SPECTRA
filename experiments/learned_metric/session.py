"""Framework-free experimental learned integer-metric inference.

Inputs are original uint8 feature codes, never rounded/snapped float coordinates.
The library and model must be trusted. Normal class prediction, not true-label proof.
"""
from __future__ import annotations
import ctypes as C
import hashlib,struct,zlib
from pathlib import Path
from spectra.svm_shared import _decode,MAX_BYTES,MAX_ELEMENTS
from spectra.svm_lifetime import _NativeOwner
MODES={'compiled':1,'scalar_integer':2,'exhaustive':0,'scalar_exp':3}

def decode(raw):
 if not 28<=len(raw)<=MAX_BYTES:raise ValueError('invalid learned model length')
 magic,maximum,d,mass,inner,crc=struct.unpack_from('<8sIIIII',raw)
 if magic!=b'SPLMET01' or not 1<=d<=4096 or not 1<=maximum<=255 or not 0<mass or mass*maximum**2+1>4194304:
  raise ValueError('unsupported learned-metric geometry')
 if len(raw)!=28+2*d+inner or zlib.crc32(raw[28:])!=crc:raise ValueError('metric bytes/CRC mismatch')
 weights=struct.unpack_from('<'+'H'*d,raw,28)
 if sum(weights)!=mass or max(weights)>255:raise ValueError('metric weight inventory mismatch')
 features,labels=_decode(raw[28+2*d:])
 if features!=d:raise ValueError('inner/outer dimensions differ')
 return d,labels,maximum,weights

class MetricSession(_NativeOwner):
 def __init__(self,model,library):
  self._init_lifetime('learned metric session')
  with Path(model).open('rb') as f:raw=f.read(MAX_BYTES+1)
  self.features,self.labels,self.maximum,self.weights=decode(raw)
  self.sha256=hashlib.sha256(raw).hexdigest()
  self._lib=lib=C.CDLL(str(Path(library).resolve(strict=True)))
  lib.lm_abi.restype=C.c_int
  if lib.lm_abi()!=1:raise ValueError('unknown learned-metric ABI')
  lib.et_error.restype=C.c_char_p
  lib.lm_create.argtypes=[C.c_void_p,C.c_uint64];lib.lm_create.restype=C.c_void_p
  lib.lm_destroy.argtypes=[C.c_void_p];lib.lm_destroy.restype=None
  lib.lm_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];lib.lm_info.restype=C.c_int
  lib.lm_run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_uint64),C.c_int];lib.lm_run.restype=C.c_int
  lib.lm_probe.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.POINTER(C.c_double),C.c_uint64];lib.lm_probe.restype=C.c_int
  blob=C.create_string_buffer(raw,len(raw));ptr=lib.lm_create(blob,len(raw))
  if not ptr:raise ValueError(lib.et_error().decode(errors='replace'))
  self._adopt(ptr,lib,'lm_destroy')
  with self._operation() as h:
   out=(C.c_uint64*7)();self._check(lib.lm_info(h,out,7));self.info=dict(zip(('features','classes','support_vectors','maximum','table_entries','prepared_worker_bytes','common_weight'),out))
 def _check(self,status):
  if status:raise ValueError(self._lib.et_error().decode(errors='replace'))
 def _invoke(self,values,mode,probe,inspect):
  if type(mode) is not str or mode not in MODES:raise ValueError('invalid metric mode')
  try:view=memoryview(values)
  except TypeError as e:raise ValueError('uint8 feature buffer required') from e
  data=None
  try:
   if view.format!='B' or view.itemsize!=1 or view.readonly or not view.c_contiguous or view.ndim not in (1,2):raise ValueError('writable contiguous uint8 codes required')
   if view.ndim==2 and view.shape[1]!=self.features:raise ValueError('feature dimension mismatch')
   n=view.nbytes//self.features
   if view.nbytes%self.features or view.nbytes>MAX_ELEMENTS or n>65536:raise ValueError('metric batch bound/shape mismatch')
   data=(C.c_uint8*view.nbytes).from_buffer(view) if view.nbytes else None
   with self._operation() as h:
    if probe:
     count=n*len(self.labels)*(len(self.labels)-1)//2
     if count>8000000:raise ValueError('probe too large')
     out=(C.c_double*count)();self._check(self._lib.lm_probe(h,data,n,self.features,out,count));return bytes(out)
    out=(C.c_int*n)();stats=(C.c_uint64*4)();self._check(self._lib.lm_run(h,data,n,self.features,MODES[mode],out,stats,4))
    labels=[self.labels[i] for i in out]
    return (labels,dict(zip(('kernel_lookups','evaluated_pairs','coefficient_terms','certificate_checks'),stats))) if inspect else labels
  finally:del data;view.release()
 def predict_buffer(self,values,*,mode='compiled'):return self._invoke(values,mode,False,False)
 def inspect_buffer(self,values,*,mode='compiled'):return self._invoke(values,mode,False,True)
 def probe(self,values):return self._invoke(values,'compiled',True,False)
