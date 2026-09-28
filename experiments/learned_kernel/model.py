"""Training/export of nonnegative radial mixtures; original inputs never pickled at inference.

Mixtures and kernel alignment are established methods. This module defines the
restricted same-function format used by this experiment, not an arbitrary SVC loader.
"""
from __future__ import annotations
import hashlib,json,math,struct,zlib
from pathlib import Path

MAGIC=b'SPKERN01'
MAX_BYTES=64*1024*1024
SCHEMA='spectra.learned_kernel.v1'


def _unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate metadata key')
        out[k]=v
    return out


def decode(raw):
    from spectra.svm_shared import _decode
    if type(raw) is not bytes or not 16<=len(raw)<=MAX_BYTES:raise ValueError('invalid learned-kernel size')
    magic,meta_n,body_n=struct.unpack('<8sII',raw[:16])
    if magic!=MAGIC or not 1<=meta_n<=16384 or len(raw)!=16+meta_n+body_n:raise ValueError('invalid learned-kernel inventory')
    try:meta=json.loads(raw[16:16+meta_n].decode('utf-8'),object_pairs_hook=_unique)
    except (UnicodeError,RecursionError) as e:raise ValueError('invalid metadata encoding') from e
    if type(meta) is not dict or set(meta)!={'format','power','denominator','gammas','weights','inner_sha256','feature_counts'} or meta['format']!=SCHEMA:raise ValueError('invalid learned-kernel fields')
    if type(meta['power']) is not int or meta['power'] not in (1,2):raise ValueError('distance power must be one or two')
    D=meta['denominator']
    if type(D) is not int or not 1<=D<=128 or D&(D-1):raise ValueError('denominator must be a power of two <=128')
    arrays=[]
    for key in ('gammas','weights'):
        seq=meta[key]
        if type(seq) is not list or not 1<=len(seq)<=16:raise ValueError('invalid component inventory')
        vals=[]
        for v in seq:
            if type(v) is not str or len(v)>32:raise ValueError('canonical binary64 constants required')
            try:x=float.fromhex(v)
            except (ValueError,OverflowError) as e:raise ValueError('invalid constant') from e
            if not math.isfinite(x) or x.hex()!=v:raise ValueError('noncanonical or nonfinite constant')
            vals.append(x)
        arrays.append(tuple(vals))
    gamma,weight=arrays
    if len(gamma)!=len(weight) or any(g<=0 for g in gamma) or any(not 0<=w<=1 for w in weight) or not 0<sum(weight)<=1+2**-40:raise ValueError('invalid nonnegative normalized kernel mixture')
    body=raw[16+meta_n:]
    if meta['inner_sha256']!=hashlib.sha256(body).hexdigest():raise ValueError('model digest mismatch')
    features,labels=_decode(body)
    counts=meta['feature_counts']
    if type(counts) is not list or len(counts)!=features or any(type(v) is not int or not 0<=v<=4096 for v in counts) or not 1<=sum(counts)<=4096:
        raise ValueError('invalid finite feature-count budget')
    return meta,body,features,labels,gamma,weight


def export_pairs(X,y,pairs,*,gammas,weights,power,denominator,destination,feature_counts=None):
    """All pair coefficients share one union support bank; no duplicate models.

    Positive pair margins mean class j, including computed zero, for ordered i<j.
    Counts/coefficients use an inner standard SRT layout only as a bounded storage
    container. The outer magic prevents accidental use as an ordinary RBF model.
    """
    import numpy as np
    classes=np.unique(y);c=len(classes);d=X.shape[1]
    if len(pairs)!=c*(c-1)//2:raise ValueError('missing pair models')
    required=np.unique(np.concatenate([p[2] for p in pairs]));ids=np.concatenate([required[y[required]==cl] for cl in classes])
    lookup=np.full(len(X),-1,dtype=int);lookup[ids]=np.arange(len(ids))
    counts=np.array([np.sum(y[ids]==cl) for cl in classes],dtype='<u4');dual=np.zeros((c-1,len(ids)),dtype='<f8');bias=[]
    expected=[(i,j) for i in range(c) for j in range(i+1,c)]
    for pair,(i,j) in zip(pairs,expected):
        pi,pj,support,coef,b=pair
        if (pi,pj)!=(i,j):raise ValueError('pair order mismatch')
        if len(support)!=len(coef) or len(set(map(int,support)))!=len(support):raise ValueError('invalid pair support inventory')
        globalids=lookup[support]
        for group,ci,row in ((classes[i],i,j-1),(classes[j],j,i)):
            mask=y[support]==group
            dual[row,globalids[mask]]=np.asarray(coef)[mask]
        if not np.all(np.isin(y[support],[classes[i],classes[j]])):raise ValueError('pair contains wrong class support')
        # Native order is class i then class j and increasing support-bank index.
        if list(globalids)!=sorted(map(int,globalids)):raise ValueError('pair coefficient order is not canonical')
        bias.append(float(b))
    labels=classes.tolist();label_bytes=json.dumps({'labels':labels},ensure_ascii=False,separators=(',',':')).encode()
    payload=struct.pack('<d',1.)+counts.tobytes()+np.asarray(X[ids],dtype='<f8').tobytes()+dual.tobytes()+np.asarray(bias,dtype='<f8').tobytes()
    body=label_bytes+payload
    inner=struct.pack('<8sIIIIII',b'SPCSVM02',c,len(ids),d,len(label_bytes),len(payload),zlib.crc32(body))+body
    meta={'format':SCHEMA,'power':int(power),'denominator':int(denominator),'gammas':[float(g).hex() for g in gammas],
          'weights':[float(w).hex() for w in weights],'feature_counts':list(map(int,feature_counts)) if feature_counts is not None else [1]*d,'inner_sha256':hashlib.sha256(inner).hexdigest()}
    encoded=json.dumps(meta,sort_keys=True,separators=(',',':')).encode()
    raw=struct.pack('<8sII',MAGIC,len(encoded),len(inner))+encoded+inner
    decode(raw)
    with Path(destination).open('xb') as f:f.write(raw)
    return {'format':SCHEMA,'features':d,'classes':c,'supports':len(ids),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),**meta}


def stock_pairs(model):
    """Convert trained stock SVC's pair layout without retraining or reordering."""
    import numpy as np
    c=len(model.classes_);starts=np.r_[0,np.cumsum(model.n_support_)];pairs=[];k=0
    for i in range(c):
        for j in range(i+1,c):
            ii=np.arange(starts[i],starts[i+1]);jj=np.arange(starts[j],starts[j+1]);inds=np.r_[ii,jj]
            co=np.r_[model.dual_coef_[j-1,ii],model.dual_coef_[i,jj]]
            bias=model.intercept_[k];k+=1
            if c>2:co=-co;bias=-bias
            keep=co!=0
            pairs.append((i,j,model.support_[inds][keep],co[keep].copy(),float(bias)))
    return pairs
