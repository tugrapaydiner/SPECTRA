"""Explicit ahead-of-time export of small dense LinearSVC decision functions.

No training or compiler invocation on import/install, no numerical frameworks at
inference. Numerical contract: ordered binary64 dot products and original class
mapping, not bitwise equivalence to every BLAS implementation or exact-real dots.
This does not supply an SVM tournament certificate.
"""
from __future__ import annotations
from array import array
import ctypes as C
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys

from ._json import load_file

FORMAT = 'spectra.compiled-linear.v1'
MAX_ELEMENTS = 8_000_000
MAX_ROWS = 65536
EXPORTS = ('sp_linear_abi','sp_linear_features','sp_linear_classes','sp_linear_outputs',
           'sp_linear_identity','sp_linear_run')


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')


def _identity(features, labels, parameter_sha256):
    return hashlib.sha256(_canonical({'format':FORMAT,'features':features,'labels':labels,
                                     'parameters_sha256':parameter_sha256})).hexdigest()


def _load(folder):
    meta=load_file(Path(folder)/'model.json',max_bytes=65536)
    fields={'format','features','classes','outputs','labels','parameters_sha256','model_sha256','source_sha256'}
    if type(meta) is not dict or set(meta)!=fields or meta['format']!=FORMAT:
        raise ValueError('unsupported compiled-linear metadata')
    d,c,o=meta['features'],meta['classes'],meta['outputs']
    if any(type(v) is not int for v in (d,c,o)) or not 1<=d<=4096 or not 2<=c<=128 or o!=(1 if c==2 else c):
        raise ValueError('unsupported linear dimensions')
    labels=meta['labels']
    if type(labels) is not list or len(labels)!=c:
        raise ValueError('invalid linear labels')
    if not (all(type(v) is int and -(2**63)<=v<2**63 for v in labels) or
            all(type(v) is str and len(v.encode('utf-8'))<=256 for v in labels)):
        raise ValueError('labels must be bounded integers or UTF-8 strings of one type')
    if len(set(labels))!=c:
        raise ValueError('duplicate linear labels')
    for name in ('parameters_sha256','model_sha256','source_sha256'):
        value=meta[name]
        if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid linear digest')
    if meta['model_sha256']!=_identity(d,labels,meta['parameters_sha256']):
        raise ValueError('linear label/parameter binding mismatch')
    return meta


_TEMPLATE = r'''// Generated immutable model. Build with strict noncontracted arithmetic.
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <limits>
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,"binary64 required");
constexpr int D=@D@, CLASSES=@C@, OUTPUTS=@O@;
static const double weights[D][OUTPUTS]={@W@};
static const double biases[OUTPUTS]={@B@};
extern "C" {
int sp_linear_abi(){return 1;}
int sp_linear_features(){return D;}
int sp_linear_classes(){return CLASSES;}
int sp_linear_outputs(){return OUTPUTS;}
const char* sp_linear_identity(){return "@IDENTITY@";}
int sp_linear_run(const double* inputs,int rows,int features,int* labels,int capacity,
                  double* scores,int score_capacity){
    if(rows<0 || rows>65536 || features!=D || capacity!=rows ||
       uint64_t(rows)*D>8000000 || (rows && (!inputs || !labels)) ||
       (scores ? score_capacity!=rows*OUTPUTS : score_capacity!=0))return 1;
    if(std::fegetround()!=FE_TONEAREST)return 2;
    for(uint64_t i=0;i<uint64_t(rows)*D;++i)if(!std::isfinite(inputs[i]))return 3;
    for(int row=0;row<rows;++row){
        double sums[OUTPUTS]={};
        for(int f=0;f<D;++f){
            const double x=inputs[uint64_t(row)*D+f];
            for(int c=0;c<OUTPUTS;++c)sums[c]+=x*weights[f][c];
        }
        for(int c=0;c<OUTPUTS;++c){
            sums[c]+=biases[c];
            if(!std::isfinite(sums[c]))return 4;
        }
        int winner=0;
        if(CLASSES==2)winner=sums[0]>0?1:0;
        else for(int c=1;c<OUTPUTS;++c)if(sums[c]>sums[winner])winner=c;
        if(scores)for(int c=0;c<OUTPUTS;++c)scores[uint64_t(row)*OUTPUTS+c]=sums[c];
        labels[row]=winner;
    }
    return 0;
}
}
'''


def export_linear_svc(model, destination: str | Path) -> dict:
    """Export a trusted fitted stock LinearSVC without rounding its parameters.

    One generated native library is specific to one fixed model. The output folder
    must be new. No feature extraction/scaling is embedded. Crammer-Singer and
    arbitrary estimators are rejected. Export needs sklearn; prediction does not.
    """
    import numpy as np
    from sklearn.svm import LinearSVC
    from sklearn.utils.validation import check_is_fitted
    if type(model) is not LinearSVC or model.multi_class!='ovr':
        raise ValueError('expected a fitted stock one-vs-rest LinearSVC')
    check_is_fitted(model)
    labels=model.classes_.tolist();d=int(model.n_features_in_);c=len(labels);o=1 if c==2 else c
    parameters=np.concatenate((np.asarray(model.coef_,dtype='<f8').ravel(),
                               np.asarray(model.intercept_,dtype='<f8').ravel()))
    if not 1<=d<=4096 or not 2<=c<=128 or model.coef_.shape!=(o,d) or model.intercept_.shape!=(o,):
        raise ValueError('unsupported fitted linear geometry')
    if not np.isfinite(parameters).all():raise ValueError('nonfinite linear parameter')
    raw=parameters.tobytes();param_sha=hashlib.sha256(raw).hexdigest();ident=_identity(d,labels,param_sha)
    rows=[','.join(float(model.coef_[k,f]).hex() for k in range(o)) for f in range(d)]
    source=_TEMPLATE.replace('@D@',str(d)).replace('@C@',str(c)).replace('@O@',str(o))
    source=source.replace('@W@',',\n'.join('{'+r+'}' for r in rows))
    source=source.replace('@B@',','.join(float(v).hex() for v in model.intercept_)).replace('@IDENTITY@',ident)
    encoded=source.encode('ascii')
    meta={'format':FORMAT,'features':d,'classes':c,'outputs':o,'labels':labels,
          'parameters_sha256':param_sha,'model_sha256':ident,'source_sha256':hashlib.sha256(encoded).hexdigest()}
    # Validate label constraints before creating output using the same explicit contract.
    from .svm_shared import _labels_valid
    if not _labels_valid(labels,c):raise ValueError('unsupported linear class labels')
    folder=Path(destination);folder.mkdir(parents=True,exist_ok=False)
    (folder/'parameters.f64').write_bytes(raw);(folder/'model.cpp').write_bytes(encoded)
    (folder/'model.json').write_bytes(_canonical(meta)+b'\n')
    return {**meta,'parameter_bytes':len(raw),'source_bytes':len(encoded)}


def build_linear(bundle: str | Path, out: str | Path, *, compiler: str | None=None) -> Path:
    """Explicit per-model compilation using the platform's strict build helper.

    A trusted compiler and source bundle are required. File hashes establish only
    identity. This is not a sandbox or a signature authenticating the model.
    """
    from .svm_build import command_for,execute
    folder=Path(bundle).resolve();meta=_load(folder)
    expected=(meta['features']*meta['outputs']+meta['outputs'])*8
    with (folder/'parameters.f64').open('rb') as f:raw=f.read(expected+1)
    if len(raw)!=expected or hashlib.sha256(raw).hexdigest()!=meta['parameters_sha256']:
        raise ValueError('linear parameter file identity mismatch')
    if any(not math.isfinite(v[0]) for v in struct.iter_unpack('<d',raw)):
        raise ValueError('nonfinite stored linear parameter')
    with (folder/'model.cpp').open('rb') as f:source=f.read(16*1024*1024+1)
    if len(source)>16*1024*1024 or hashlib.sha256(source).hexdigest()!=meta['source_sha256']:
        raise ValueError('linear generated-source identity mismatch')
    destination=Path(out).resolve()
    library=destination/('spectra_linear.dll' if sys.platform=='win32' else 'libspectra_linear.so')
    command=command_for(folder/'model.cpp',library,target='portable',compiler=compiler,exports=EXPORTS)
    destination.mkdir(parents=True,exist_ok=False)
    return execute(command,library,{'model_sha256':meta['model_sha256'],'source_sha256':meta['source_sha256'],
                                    'target':'portable','scope':'fixed LinearSVC ordered binary64 evaluator'})


class CompiledLinear:
    """Immutable fixed-model evaluator. No mutable native cache or native heap.

    Input must already have the original training-time feature extraction/scaling.
    Every call owns fresh output buffers; the native code is reentrant. Libraries
    are trusted executable code. No explicit unload, vote certificate, probability
    output or universal BLAS/exact-real decision equivalence is offered.
    """
    def __init__(self, bundle: str | Path, library: str | Path):
        if sys.maxsize<=2**32 or sys.byteorder!='little':raise ValueError('64-bit little-endian host required')
        meta=_load(bundle);lib=C.CDLL(str(Path(library).resolve(strict=True)))
        for name in EXPORTS[:4]:
            getattr(lib,name).argtypes=[];getattr(lib,name).restype=C.c_int
        lib.sp_linear_identity.argtypes=[];lib.sp_linear_identity.restype=C.c_char_p
        if (lib.sp_linear_abi()!=1 or lib.sp_linear_features()!=meta['features'] or
            lib.sp_linear_classes()!=meta['classes'] or lib.sp_linear_outputs()!=meta['outputs'] or
            lib.sp_linear_identity()!=meta['model_sha256'].encode('ascii')):
            raise ValueError('linear library/model binding mismatch')
        lib.sp_linear_run.argtypes=[C.POINTER(C.c_double),C.c_int,C.c_int,C.POINTER(C.c_int),C.c_int,C.POINTER(C.c_double),C.c_int]
        lib.sp_linear_run.restype=C.c_int
        self._features=meta['features'];self._labels=tuple(meta['labels']);self._outputs=meta['outputs']
        self._identity=meta['model_sha256'];self._lib=lib
    @property
    def features(self):return self._features
    @property
    def labels(self):return self._labels
    @property
    def sha256(self):return self._identity
    def _run_buffer(self, values, scores=False):
        try:view=memoryview(values)
        except TypeError as e:raise ValueError('native binary64 buffer required') from e
        data=None
        try:
            if (view.format not in ('d','@d','=d','<d') or view.itemsize!=8 or view.readonly or
                not view.c_contiguous or view.ndim not in (1,2)):
                raise ValueError('writable C-contiguous little-endian binary64 buffer required')
            if (view.ndim==2 and view.shape[1]!=self.features) or (view.nbytes//8)%self.features:
                raise ValueError('linear input shape mismatch')
            elements=view.nbytes//8;count=elements//self.features
            if count>MAX_ROWS or elements>MAX_ELEMENTS:raise ValueError('linear input limit exceeded')
            data=(C.c_double*elements).from_buffer(view) if elements else None
            if data is not None and C.addressof(data)%C.alignment(C.c_double):raise ValueError('unaligned binary64 buffer')
            result=(C.c_int*count)();scored=(C.c_double*(count*self._outputs))() if scores else None
            status=self._lib.sp_linear_run(data,count,self.features,result,count,scored,len(scored) if scored is not None else 0)
            if status:raise ValueError('linear inference failed: '+str(status))
            if any(not 0<=i<len(self.labels) for i in result):raise RuntimeError('invalid native class index')
            return list(scored) if scores else [self.labels[i] for i in result]
        finally:
            del data;view.release()
    def predict_buffer(self, values):
        """Borrow input through call completion; caller must not mutate concurrently."""
        return self._run_buffer(values)
    def scores_buffer(self, values):
        """Return flat, fresh ordered binary64 scores for inspection, not probabilities."""
        return self._run_buffer(values,True)
    def predict_many(self, rows):
        packed=array('d');cap=min(MAX_ROWS,MAX_ELEMENTS//self.features)
        for index,row in enumerate(rows):
            if index>=cap:raise ValueError('linear row limit exceeded')
            values=list(itertools.islice(iter(row),self.features+1))
            if len(values)!=self.features:raise ValueError('linear row width mismatch')
            try:
                if any(isinstance(v,(str,bytes,bool)) or not math.isfinite(v) for v in values):raise ValueError('finite numeric input required')
                packed.extend(values)
            except (TypeError,OverflowError) as e:raise ValueError('invalid numeric input') from e
        return self.predict_buffer(packed)
    def predict(self,row):return self.predict_many([row])[0]
