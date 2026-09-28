"""Inert integer-metric table model; exact integer signatures, ordered FP64 scores.

Training and deployment use the same serialized table. The intended output is
this NEW trained function, not an asserted correction to arbitrary old SVMs.
"""
from __future__ import annotations
from array import array
import hashlib,json,math,struct,sys,zlib
from pathlib import Path
MAX_BYTES=64*1024*1024
MAGIC=b'SPLFK001'
HEADER=struct.Struct('<8sIIIIIII')
HEADER2=struct.Struct('<8sIIIIIIII')

def encode(x,labels,weights,lut,counts,dual,bias,signature='hamming'):
    # Only training calls this NumPy-based encoder; decode/reference is stdlib.
    import numpy as np
    x=np.asarray(x);w=np.asarray(weights);lut=np.asarray(lut);c=len(labels);n,d=x.shape
    if not np.isin(x,[0,1]).all() or not np.equal(w,np.rint(w)).all() or (w<0).any() or (w>7).any():raise ValueError('binary/integer model required')
    words=(d+63)//64;packed=np.zeros((n,words),dtype='<u8')
    for f in range(d):packed[:,f//64]|=x[:,f].astype(np.uint64)<<(f%64)
    meta=json.dumps({'labels':list(labels)},ensure_ascii=True,separators=(',',':')).encode('ascii')
    raw=meta+np.asarray(w,dtype=np.uint8).tobytes()+np.asarray(counts,dtype='<u4').tobytes()+packed.tobytes()
    for values in (lut,dual,bias):raw+=np.asarray(values,dtype='<f8',order='C').tobytes()
    if signature not in ('hamming','intersection'):raise ValueError('unknown signature')
    encoded=(HEADER.pack(MAGIC,d,c,n,len(lut),len(meta),len(raw),zlib.crc32(raw)) if signature=='hamming'
             else HEADER2.pack(b'SPLFK002',d,c,n,len(lut),len(meta),len(raw),zlib.crc32(raw),1))+raw
    Model(encoded) # Apply the independent loader bounds before publishing.
    return encoded

class Model:
    def __init__(self,raw):
        if type(raw) is not bytes or not HEADER.size<=len(raw)<=MAX_BYTES:raise ValueError('model byte bound')
        magic,d,c,n,t,meta_size,size,crc=HEADER.unpack_from(raw)
        header=HEADER.size if magic==MAGIC else HEADER2.size
        if magic not in (MAGIC,b'SPLFK002') or len(raw)<header:raise ValueError('model format')
        kind=0 if magic==MAGIC else struct.unpack_from('<I',raw,36)[0]
        if kind not in (0,1):raise ValueError('unknown signature')
        if not 1<=d<=4096 or not 2<=c<=128 or not c<=n<=100000 or not 1<=meta_size<=65536:raise ValueError('model geometry')
        words=(d+63)//64;pairs=c*(c-1)//2
        if n*d>8_000_000 or not 2<=t<=7*d+1:raise ValueError('model resource limit')
        if size!=meta_size+d+4*c+8*n*words+8*(t+(c-1)*n+pairs) or len(raw)!=header+size:raise ValueError('model inventory')
        body=raw[header:]
        if zlib.crc32(body)!=crc:raise ValueError('model CRC')
        def unique(pairs):
            out={}
            for k,v in pairs:
                if k in out:raise ValueError('duplicate metadata')
                out[k]=v
            return out
        meta=json.loads(body[:meta_size].decode('ascii'),object_pairs_hook=unique)
        labels=meta.get('labels')
        if set(meta)!={'labels'} or type(labels) is not list or len(labels)!=c:raise ValueError('labels geometry')
        if not (all(type(v) is int and -(2**63)<=v<2**63 for v in labels) or all(type(v) is str and len(v.encode())<=256 for v in labels)):raise ValueError('label types')
        if len(set(labels))!=c:raise ValueError('duplicate labels')
        pos=meta_size;weights=tuple(body[pos:pos+d]);pos+=d
        if any(v>7 for v in weights) or sum(weights)+1!=t:raise ValueError('weight/table geometry')
        counts=struct.unpack_from('<'+'I'*c,body,pos);pos+=4*c
        if not all(counts) or sum(counts)!=n:raise ValueError('support counts')
        supports=struct.unpack_from('<'+'Q'*(n*words),body,pos);pos+=8*n*words
        if d%64 and any(supports[i*words+words-1]>>(d%64) for i in range(n)):raise ValueError('nonzero bit padding')
        def take(count):
            nonlocal pos
            v=struct.unpack_from('<'+'d'*count,body,pos);pos+=8*count
            if not all(math.isfinite(x) for x in v):raise ValueError('nonfinite model value')
            return v
        table=take(t);dual=take((c-1)*n);bias=take(pairs)
        if any(abs(x)>2**20 for x in table):raise ValueError('kernel table range')
        self.kind=kind;self.raw=raw;self.sha256=hashlib.sha256(raw).hexdigest();self.d=d;self.c=c;self.n=n;self.words=words
        self.labels=tuple(labels);self.weights=weights;self.table=table;self.supports=supports
        starts=[0]
        for v in counts:starts.append(starts[-1]+v)
        self.pairs=[]
        for i in range(c):
            for j in range(i+1,c):
                terms=[]
                for cls,coefrow in ((i,j-1),(j,i)):
                    for k in range(starts[cls],starts[cls+1]):
                        coefficient=dual[coefrow*n+k]
                        if coefficient:terms.append((k,coefficient))
                b=bias[len(self.pairs)]
                if sum(abs(a) for _,a in terms)*max(map(abs,table))+abs(b)>=sys.float_info.max/4:raise ValueError('unsafe score bound')
                self.pairs.append((i,j,terms,b))

    @classmethod
    def load(cls,path):
        with Path(path).open('rb') as f:return cls(f.read(MAX_BYTES+1))

    def margins(self,row):
        if len(row)!=self.d or any(type(v) not in (int,bool) or v not in (0,1) for v in row):raise ValueError('binary row required')
        # Intentionally direct per-feature reference, not the optimized bitplanes.
        kernels=[]
        for k in range(self.n):
            distance=0
            for f,(x,w) in enumerate(zip(row,self.weights)):
                y=(self.supports[k*self.words+f//64]>>(f%64))&1
                distance+=w*(int(x)*y if self.kind else (int(x)-y)**2)
            kernels.append(self.table[distance])
        result=[]
        for i,j,terms,b in self.pairs:
            score=0.
            for k,a in terms:score+=a*kernels[k]
            result.append(score+b)
        return result

    def predict(self,row):
        margins=self.margins(row)
        if self.c==2:return self.labels[1 if margins[0]>=0 else 0]
        wins=[0]*self.c
        for (i,j,_,_),score in zip(self.pairs,margins):wins[i if score>0 else j]+=1
        return self.labels[max(range(self.c),key=lambda i:wins[i])]
