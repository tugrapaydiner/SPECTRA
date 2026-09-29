"""Explicit builds and owned, bounded inference for learned finite-table models."""
from __future__ import annotations
import ctypes as C
import hashlib,json,struct,subprocess,sys,time,zlib
from pathlib import Path
from spectra.svm_lifetime import _NativeOwner
from .model import HEADER,MAGIC,MAX_BYTES
ROOT=Path(__file__).resolve().parents[2]
MODES={'exhaustive':0,'selective':1,'scalar':2}

def build(folder,target='avx2',ubsan=False):
 folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False)
 if target not in ('avx2','portable'):raise ValueError('target')
 source=Path(__file__).with_name('runtime.cpp');base=ROOT/'spectra/_native/ovo/runtime.cpp';lib=folder/'learned.so'
 cmd=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fPIC','-shared']
 if target=='avx2':cmd+=['-mavx2']
 if ubsan:cmd+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all']
 cmd+=[f'-DLS_BASE_RUNTIME="{base}"',str(source),'-o',str(lib)]
 start=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
 files=[source]+list(source.parent.glob('*.hpp'))+[f for f in base.parent.rglob('*') if f.suffix in ('.hpp','.cpp')]
 receipt={'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'seconds':time.perf_counter()-start,'source_sha256':{f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
 if p.returncode==0:receipt['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
 (folder/'build.json').write_text(json.dumps(receipt,indent=2))
 if p.returncode:raise RuntimeError(p.stderr)
 return lib

class Session(_NativeOwner):
 """Inputs are bounded native uint8 domain codes, not silently rounded floats.

    The source library must be trusted. Each call owns a lease and uses private
    mutable state; no cross-request kernel cache or hidden model fitting.
 """
 def __init__(self,path,library):
  self._init_lifetime('learned signature session')
  with Path(path).open('rb') as f:raw=f.read(MAX_BYTES+1)
  if not HEADER.size<=len(raw)<=MAX_BYTES:raise ValueError('model size')
  magic,d,c,n,cap,nt,ne,terms,mlen,crc=HEADER.unpack_from(raw)
  if magic!=MAGIC or not 2<=c<=128 or mlen>65536 or len(raw)<HEADER.size+mlen:raise ValueError('model header')
  def unique(pairs):
   out={}
   for k,v in pairs:
    if k in out:raise ValueError('duplicate metadata key')
    out[k]=v
   return out
  meta=json.loads(raw[HEADER.size:HEADER.size+mlen],object_pairs_hook=unique)
  labels=meta.get('labels')
  if not isinstance(labels,list) or len(labels)!=c or not (all(type(x)is int for x in labels) or all(type(x)is str and len(x.encode())<=256 for x in labels)) or len(set(labels))!=c:raise ValueError('invalid labels')
  self.labels=tuple(labels);self.features=d;self.domain_max=cap;self.metadata=meta
  self._lib=lib=C.CDLL(str(Path(library).resolve(strict=True)))
  lib.ls_abi.restype=C.c_int
  if lib.ls_abi()!=1:raise ValueError('ABI mismatch')
  lib.et_error.restype=C.c_char_p
  lib.ls_create.argtypes=[C.c_void_p,C.c_uint64];lib.ls_create.restype=C.c_void_p
  lib.ls_destroy.argtypes=[C.c_void_p];lib.ls_destroy.restype=None
  lib.ls_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];lib.ls_info.restype=C.c_int
  lib.ls_run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_uint64),C.c_int];lib.ls_run.restype=C.c_int
  lib.ls_probe.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.POINTER(C.c_double),C.c_uint64];lib.ls_probe.restype=C.c_int
  blob=C.create_string_buffer(raw,len(raw));h=lib.ls_create(blob,len(raw))
  if not h:raise ValueError(lib.et_error().decode(errors='replace'))
  self._adopt(h,lib,'ls_destroy')
  with self._operation() as handle:
   info=(C.c_uint64*7)();self._check(lib.ls_info(handle,info,7));self.info=dict(zip(('features','classes','supports','profiles','table_length','counted_bytes','domain_max'),info))
 def _check(self,status):
  if status:raise ValueError(self._lib.et_error().decode(errors='replace'))
 def _invoke(self,values,mode,probe):
  if type(mode) is not str or mode not in MODES:raise ValueError('unknown mode')
  try:view=memoryview(values)
  except TypeError as e:raise ValueError('uint8 buffer required') from e
  data=None
  try:
   if view.format!='B' or view.itemsize!=1 or view.readonly or not view.c_contiguous or view.ndim not in (1,2):raise ValueError('writable contiguous uint8 buffer required')
   if view.ndim==2 and view.shape[1]!=self.features:raise ValueError('feature dimension mismatch')
   if view.nbytes%self.features or view.nbytes>8000000:raise ValueError('element cap or incomplete row')
   n=view.nbytes//self.features
   if n>65536:raise ValueError('row cap')
   data=(C.c_uint8*view.nbytes).from_buffer(view) if n else None
   with self._operation() as handle:
    if probe:
     cells=n*len(self.labels)*(len(self.labels)-1)//2
     if cells>16000000:raise ValueError('probe cap')
     out=(C.c_double*cells)();self._check(self._lib.ls_probe(handle,data,n,self.features,out,cells));return bytes(out)
    out=(C.c_int*n)();stats=(C.c_uint64*4)();self._check(self._lib.ls_run(handle,data,n,self.features,MODES[mode],out,stats,4))
    return [self.labels[i] for i in out],tuple(stats)
  finally:
   del data;view.release()
 def predict_buffer(self,values,*,mode='selective'):return self._invoke(values,mode,False)[0]
 def inspect_buffer(self,values,*,mode='selective'):return self._invoke(values,mode,False)
 def probe(self,values):return self._invoke(values,'exhaustive',True)
