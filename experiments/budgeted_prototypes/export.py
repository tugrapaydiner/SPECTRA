"""Export validated integer geometry and exact float64 fitted output head."""
from __future__ import annotations
import json,struct,zlib,math
from pathlib import Path
import numpy as np
from .session import HEADER,decode
from .learning import kernel_config

def encode(arrays,maximum,s):
    if s.affine or s.adaptive_gamma or s.normalized:
        raise ValueError("this native format supports only the locked constant-head fixed-width unnormalized model")
    C=np.asarray(arrays['centers']);W=np.asarray(arrays['weights']);H=np.asarray(arrays['head']);b=np.asarray(arrays['bias']);labels=np.asarray(arrays['classes']).tolist()
    if C.ndim!=2 or C.shape!=W.shape or H.ndim!=2 or H.shape[0]!=len(C) or len(b)!=H.shape[1]:raise ValueError('model array shape')
    p,d=C.shape;c=H.shape[1]
    if C.dtype.kind not in 'iu' or W.dtype.kind not in 'iu' or np.any(C<0) or np.any(C>maximum*s.quarter) or np.any(W<1) or np.any(W.sum(1)!=s.units*d) or np.any(W>65535):raise ValueError('integer geometry violation')
    if not np.isfinite(H).all() or not np.isfinite(b).all():raise ValueError('nonfinite head')
    k=kernel_config(d,maximum,s.quarter,s.units,s.gamma)
    meta=json.dumps({'labels':labels},ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()
    payload=struct.pack('<d',k['alpha'])+C.astype('<u2').tobytes()+W.astype('<u2').tobytes()+H.astype('<f8').tobytes()+b.astype('<f8').tobytes()+meta
    result=HEADER.pack(b'SPPRO001',d,p,c,maximum,s.quarter,s.units,k['bits'],len(meta),len(payload),zlib.crc32(payload))+payload
    decode(result);return result
