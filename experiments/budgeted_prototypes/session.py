"""Framework-free prototype inference. Own-model scores, not old-SVM certificates."""
from __future__ import annotations
import ctypes as C, hashlib, json, struct, zlib, math
from pathlib import Path
from spectra.svm_lifetime import _NativeOwner
HEADER=struct.Struct('<8sIIIIIIIIII')
MODES={'compiled':0,'scalar':1,'direct_exp':2}

def _unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate metadata key')
        out[k]=v
    return out

def decode(raw):
    if type(raw) is not bytes or not 48<=len(raw)<=64*1024**2:raise ValueError('model byte cap')
    magic,d,p,c,D,Q,U,bits,nmeta,payload,crc=HEADER.unpack_from(raw)
    if magic!=b'SPPRO001' or not 1<=d<=256 or not 1<=p<=4096 or not 2<=c<=128 or not 1<=D<=255 or not 1<=Q<=16 or not 1<=U<=16:raise ValueError('prototype geometry')
    if not 1<=nmeta<=65536 or payload!=8+4*p*d+8*(p*c+c)+nmeta or len(raw)!=48+payload or zlib.crc32(raw[48:])!=crc:raise ValueError('prototype inventory/CRC')
    try:meta=json.loads(raw[-nmeta:].decode('utf-8'),object_pairs_hook=_unique)
    except (UnicodeError,RecursionError) as e:raise ValueError('bad labels') from e
    if type(meta) is not dict or set(meta)!={'labels'}:raise ValueError('label metadata')
    labels=meta['labels']
    if type(labels) is not list or len(labels)!=c:raise ValueError('label count')
    if not (all(type(v) is int and -(2**63)<=v<2**63 for v in labels) or all(type(v) is str and len(v.encode('utf-8'))<=256 for v in labels)):raise ValueError('label type/length')
    if len(set(labels))!=c:raise ValueError('duplicate labels')
    return d,p,D,Q,U,tuple(labels)

class PrototypeSession(_NativeOwner):
    def __init__(self,model,library):
        self._init_lifetime('prototype model')
        with Path(model).open('rb') as f:raw=f.read(64*1024**2+1)
        self.features,self.prototypes,self.maximum,self.quarter,self.units,self.labels=decode(raw)
        self.sha256=hashlib.sha256(raw).hexdigest();self._lib=lib=C.CDLL(str(Path(library).resolve(strict=True)))
        lib.bp_abi.restype=C.c_int
        if lib.bp_abi()!=1:raise ValueError('prototype ABI')
        lib.bp_error.restype=C.c_char_p
        lib.bp_create.argtypes=[C.c_void_p,C.c_uint64];lib.bp_create.restype=C.c_void_p
        lib.bp_destroy.argtypes=[C.c_void_p];lib.bp_destroy.restype=None
        lib.bp_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];lib.bp_info.restype=C.c_int
        lib.bp_run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int)];lib.bp_run.restype=C.c_int
        lib.bp_scores.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_double),C.c_uint64];lib.bp_scores.restype=C.c_int
        b=C.create_string_buffer(raw,len(raw));h=lib.bp_create(b,len(raw))
        if not h:raise ValueError(lib.bp_error().decode(errors='replace'))
        self._adopt(h,lib,'bp_destroy')
        with self._operation() as handle:
            out=(C.c_uint64*8)();self._check(lib.bp_info(handle,out,8))
            self.info=dict(zip(('features','prototypes','classes','table_entries','full_table_entries','prepared_bytes','uniform_metric','narrow_distance'),out))
    def _check(self,status):
        if status:raise ValueError(self._lib.bp_error().decode(errors='replace'))
    def _invoke(self,values,mode,scores):
        if type(mode) is not str or mode not in MODES:raise ValueError('unknown execution mode')
        try:view=memoryview(values)
        except TypeError as e:raise ValueError('uint8 buffer required') from e
        data=None
        try:
            if view.format!='B' or view.readonly or not view.c_contiguous or view.ndim not in (1,2):raise ValueError('writable contiguous uint8 required')
            if view.ndim==2 and view.shape[1]!=self.features:raise ValueError('feature width')
            n=view.nbytes//self.features
            if view.nbytes%self.features or view.nbytes>8000000 or n>65536:raise ValueError('input cap/shape')
            data=(C.c_uint8*view.nbytes).from_buffer(view) if view.nbytes else None
            with self._operation() as h:
                if scores:
                    cells=n*len(self.labels)
                    if cells>8000000:raise ValueError('score cap')
                    out=(C.c_double*cells)();self._check(self._lib.bp_scores(h,data,n,self.features,MODES[mode],out,cells));return bytes(out)
                out=(C.c_int*n)();self._check(self._lib.bp_run(h,data,n,self.features,MODES[mode],out));return [self.labels[i] for i in out]
        finally:del data;view.release()
    def predict_buffer(self,values,*,mode='compiled'):return self._invoke(values,mode,False)
    def scores(self,values,*,mode='compiled'):return self._invoke(values,mode,True)
