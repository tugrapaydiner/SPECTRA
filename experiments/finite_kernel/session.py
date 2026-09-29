"""Framework-free finite-grid experiment. Same explicit native ownership contract."""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import sys
from spectra.svm_shared import _decode, MAX_BYTES, MAX_ELEMENTS
from spectra.svm_lifetime import _NativeOwner

MODES={'compiled':1,'scalar_integer':2,'prior_adaptive':0,'compiled_exhaustive':3}
class FiniteSession(_NativeOwner):
 def __init__(self,model,library):
  self._init_lifetime('adaptive session')
  with Path(model).open('rb') as stream:raw=stream.read(MAX_BYTES+1)
  self.features,self.labels=_decode(raw)
  self._lib=lib=C.CDLL(str(Path(library).resolve(strict=True)))
  lib.fk_abi.restype=C.c_int
  if lib.fk_abi()!=1:raise ValueError('unsupported adaptive ABI')
  lib.et_error.restype=C.c_char_p
  lib.fk_create.argtypes=[C.c_void_p,C.c_uint64];lib.fk_create.restype=C.c_void_p
  lib.fk_destroy.argtypes=[C.c_void_p];lib.fk_destroy.restype=None
  lib.fk_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];lib.fk_info.restype=C.c_int
  lib.fk_run.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_uint64),C.c_int];lib.fk_run.restype=C.c_int
  lib.fk_probe.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_int,C.c_int,C.POINTER(C.c_double),C.c_uint64];lib.fk_probe.restype=C.c_int
  blob=C.create_string_buffer(raw,len(raw));pointer=lib.fk_create(blob,len(raw))
  if not pointer:raise ValueError(lib.et_error().decode(errors='replace'))
  self._adopt(pointer,lib,'fk_destroy')
  with self._operation() as handle:
   info=(C.c_uint64*9)();self._check(lib.fk_info(handle,info,9));self.info=dict(zip(('features','classes','supports','denominator','binary32_encoding','exact_grid','table_entries','added_bytes','prior_bytes'),info))
 def _check(self,status):
  if status:raise ValueError(self._lib.et_error().decode(errors='replace'))
 def _invoke(self,values,mode,probe,inspect=False):
  if type(mode) is not str or mode not in MODES:raise ValueError('invalid execution mode')
  try:view=memoryview(values)
  except TypeError as err:raise ValueError('writable contiguous binary64 input required') from err
  data=None
  try:
   native='<d' if sys.byteorder=='little' else '>d'
   if view.format not in ('d','@d','=d',native) or view.itemsize!=8 or view.readonly or not view.c_contiguous or view.ndim not in (1,2):raise ValueError('writable contiguous native binary64 input required')
   if view.ndim==2 and view.shape[1]!=self.features:raise ValueError('feature count mismatch')
   elems=view.nbytes//8
   if elems%self.features or elems>MAX_ELEMENTS:raise ValueError('invalid element count')
   n=elems//self.features
   if n>65536:raise ValueError('row cap exceeded')
   data=(C.c_double*elems).from_buffer(view) if elems else None
   if data is not None and C.addressof(data)%8:raise ValueError('unaligned input')
   with self._operation() as h:
    if probe:
     cells=n*len(self.labels)*(len(self.labels)-1)//2*4
     if cells>16_000_000:raise ValueError('probe output exceeds element cap')
     out=(C.c_double*cells)();self._check(self._lib.fk_probe(h,data,n,self.features,out,cells));return bytes(out)
    out=(C.c_int*n)();stats=(C.c_uint64*5)()
    self._check(self._lib.fk_run(h,data,n,self.features,MODES[mode],out,stats,5))
    result=[self.labels[i] for i in out]
    return (result,tuple(stats)) if inspect else result
  finally:
   del data;view.release()
 def predict_buffer(self,values,*,mode='compiled'):
  return self._invoke(values,mode,False)
 def inspect_buffer(self,values,*,mode='compiled'):
  result,values=self._invoke(values,mode,False,True)
  return result,dict(zip(('signature_lookups','accepted_pairs','exact_fallback_pairs','domain_fallback_rows','exact_kernel_evaluations'),values))
 def probe(self,values):return self._invoke(values,'compiled',True)
