"""Framework-free input-to-label interface for the experimental integer embedding.

Supplied libraries and model files must be trusted. No implicit input snapping,
compilation, training or old-model numerical-equivalence claim.
"""
from __future__ import annotations
import ctypes as C,struct,json,zlib,hashlib
from pathlib import Path
from spectra.svm_lifetime import _NativeOwner
from spectra.svm_shared import _labels_valid
MODES={'compiled':1,'scalar_integer':2,'exhaustive':0,'direct_exp':3,'projected':4}

def decode(raw):
    if not 44<=len(raw)<=64*1024**2:raise ValueError('invalid model length')
    magic,d,r,D,c,n,bits,mlen,plen,crc=struct.unpack_from('<8sIIIIIIIII',raw)
    if magic!=b'SPINT001' or not 1<=d<=64 or not 1<=r<=128 or not 1<=D<=255 or not 2<=c<=128 or not 1<=n<=100000 or r*n>8000000:raise ValueError('invalid model geometry')
    expected=8+2*r*d+n*d+4*c+8*((c-1)*n+c*(c-1)//2)+mlen
    if plen!=expected or len(raw)!=44+expected or mlen>65536 or not mlen or zlib.crc32(raw[44:])!=crc:raise ValueError('model inventory/CRC mismatch')
    def obj(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate label field')
            result[k]=v
        return result
    try:meta=json.loads(raw[-mlen:].decode('utf-8'),object_pairs_hook=obj)
    except (UnicodeError,RecursionError) as e:raise ValueError('invalid labels') from e
    if type(meta) is not dict or set(meta)!={'labels'} or not _labels_valid(meta['labels'],c):raise ValueError('invalid class labels')
    return d,r,D,tuple(meta['labels'])

class InteractionSession(_NativeOwner):
    def __init__(self,model,library):
        self._init_lifetime('integer interaction session')
        with Path(model).open('rb') as f:raw=f.read(64*1024**2+1)
        self.features,self.rank,self.maximum,self.labels=decode(raw);self.sha256=hashlib.sha256(raw).hexdigest()
        self._lib=lib=C.CDLL(str(Path(library).resolve(strict=True)));lib.ii_abi.restype=C.c_int
        if lib.ii_abi()!=1:raise ValueError('unsupported interaction ABI')
        lib.et_error.restype=C.c_char_p
        lib.ii_create.argtypes=[C.c_void_p,C.c_uint64];lib.ii_create.restype=C.c_void_p
        lib.ii_destroy.argtypes=[C.c_void_p];lib.ii_destroy.restype=None
        lib.ii_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];lib.ii_info.restype=C.c_int
        lib.ii_run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_uint64),C.c_int];lib.ii_run.restype=C.c_int
        lib.ii_probe.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.POINTER(C.c_double),C.c_uint64];lib.ii_probe.restype=C.c_int
        buf=C.create_string_buffer(raw,len(raw));ptr=lib.ii_create(buf,len(raw))
        if not ptr:raise ValueError(lib.et_error().decode(errors='replace'))
        self._adopt(ptr,lib,'ii_destroy')
        with self._operation() as h:
            out=(C.c_uint64*9)();self._check(lib.ii_info(h,out,9))
            self.info=dict(zip(('raw_features','embedding_features','classes','support_vectors','maximum','table_entries','full_table_entries','counted_prepared_worker_bytes','projection_terms'),out))
    def _check(self,v):
        if v:raise ValueError(self._lib.et_error().decode(errors='replace'))
    def _call(self,values,mode,probe,inspect):
        if type(mode) is not str or mode not in MODES:raise ValueError('unsupported mode')
        try:view=memoryview(values)
        except TypeError as e:raise ValueError('uint8 buffer required') from e
        data=None
        try:
            if view.format!='B' or view.readonly or not view.c_contiguous or view.ndim not in (1,2):raise ValueError('writable contiguous uint8 buffer required')
            if view.ndim==2 and view.shape[1]!=self.features:raise ValueError('wrong feature width')
            n=view.nbytes//self.features
            if view.nbytes%self.features or view.nbytes>8000000 or n>65536:raise ValueError('input count cap')
            data=(C.c_uint8*view.nbytes).from_buffer(view) if view.nbytes else None
            with self._operation() as h:
                if probe:
                    cells=n*len(self.labels)*(len(self.labels)-1)//2
                    if cells>8000000:raise ValueError('probe cap')
                    out=(C.c_double*cells)();self._check(self._lib.ii_probe(h,data,n,self.features,out,cells));return bytes(out)
                out=(C.c_int*n)();stats=(C.c_uint64*4)();self._check(self._lib.ii_run(h,data,n,self.features,MODES[mode],out,stats,4))
                labels=[self.labels[i] for i in out]
                return (labels,dict(zip(('kernels','pairs','terms','projection_terms'),stats))) if inspect else labels
        finally:del data;view.release()
    def predict_buffer(self,values,*,mode='compiled'):return self._call(values,mode,False,False)
    def inspect_buffer(self,values,*,mode='compiled'):return self._call(values,mode,False,True)
    def probe(self,values):return self._call(values,'compiled',True,False)
