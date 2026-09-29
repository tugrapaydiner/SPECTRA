"""Inert finite-domain classifier format and independent ordered interpretation."""
from __future__ import annotations
import hashlib,json,math,struct,zlib
from pathlib import Path
MAGIC=b'SPLKT001'
HEADER=struct.Struct('<8s9I')
MAX_BYTES=128*1024*1024

def pack_model(path, *, codes, classes, counts, weights, table, pairs, metadata=None):
 import numpy as np
 codes=np.asarray(codes)
 if codes.dtype!=np.uint8 or codes.ndim!=2:raise ValueError('support codes must be a uint8 matrix')
 c=len(classes);n,d=codes.shape;weights=np.asarray(weights,dtype='<u4')
 tables=np.asarray(table,dtype='<f8');tables=tables[None,:] if tables.ndim==1 else tables
 cap=int((metadata or {}).get('domain_max',int(codes.max())))
 if c<2 or len(pairs)!=c*(c-1)//2 or len(counts)!=c or sum(counts)!=n:raise ValueError('invalid classes/pair counts')
 if tables.shape[1]!=int(weights.sum())*cap*cap+1:raise ValueError('table/metric geometry mismatch')
 meta={'labels':[v.item() if hasattr(v,'item') else v for v in classes],**(metadata or {})}
 # Labels are deliberately not allowed to be replaced through user metadata.
 meta['labels']=[v.item() if hasattr(v,'item') else v for v in classes]
 rawmeta=json.dumps(meta,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
 sizes=[];profiles=[];biases=[];ids=[];coef=[]
 for pair in pairs:
  pi=np.asarray(pair['ids'],dtype='<u4');pv=np.asarray(pair['coef'],dtype='<f8')
  if len(pi)!=len(pv) or np.any(pi>=n) or not np.isfinite(pv).all():raise ValueError('bad pair')
  sizes.append(len(pi));profiles.append(int(pair['profile']));biases.append(float(pair['bias']));ids.extend(pi.tolist());coef.extend(pv.tolist())
 body=(rawmeta+np.asarray(counts,dtype='<u4').tobytes()+weights.tobytes()+codes.tobytes()+
       np.asarray(sizes,dtype='<u4').tobytes()+np.asarray(profiles,dtype='<u4').tobytes()+
       np.asarray(biases,dtype='<f8').tobytes()+np.asarray(ids,dtype='<u4').tobytes()+
       np.asarray(coef,dtype='<f8').tobytes()+tables.tobytes())
 raw=HEADER.pack(MAGIC,d,c,n,cap,len(tables),tables.shape[1],len(ids),len(rawmeta),zlib.crc32(body))+body
 if len(raw)>MAX_BYTES:raise ValueError('model byte cap')
 Path(path).write_bytes(raw)
 return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'supports':n,'features':d,'classes':c,'terms':len(ids),'profiles':len(tables),'table_entries':int(tables.size)}

def from_precomputed(svc,q,weights,table,path,metadata):
 """Serialize a fitted precomputed SVC without its n*n kernel matrix."""
 import numpy as np
 counts=svc.n_support_.astype(int).tolist();starts=np.cumsum([0]+counts);c=len(svc.classes_);pairs=[]
 for i in range(c):
  for j in range(i+1,c):
   idx=np.r_[np.arange(starts[i],starts[i+1]),np.arange(starts[j],starts[j+1])]
   coefficients=np.r_[svc.dual_coef_[j-1,starts[i]:starts[i+1]],svc.dual_coef_[i,starts[j]:starts[j+1]]]
   keep=coefficients!=0;pair_index=len(pairs)
   pairs.append({'ids':idx[keep],'coef':coefficients[keep],'bias':svc.intercept_[pair_index],'profile':0})
 return pack_model(path,codes=q[svc.support_],classes=svc.classes_,counts=counts,weights=weights,table=table,pairs=pairs,metadata=metadata)

class ReferenceModel:
 """Standard-library-only integer distance, ordered binary64 margin and vote oracle."""
 def __init__(self,path):
  raw=Path(path).read_bytes()
  if len(raw)<HEADER.size or len(raw)>MAX_BYTES:raise ValueError('model size')
  magic,self.d,self.c,self.n,self.cap,self.nt,self.ne,terms,mlen,crc=HEADER.unpack_from(raw)
  if magic!=MAGIC or not 1<=self.d<=4096 or not 2<=self.c<=128 or not 1<=self.n<=100000 or not 1<=self.cap<=255 or not 1<=self.nt<=64 or not 1<=self.ne<=4000001 or mlen>65536:raise ValueError('header')
  body=raw[HEADER.size:]
  if zlib.crc32(body)!=crc:raise ValueError('CRC mismatch')
  try:self.metadata=json.loads(body[:mlen]);self.labels=self.metadata['labels']
  except Exception as e:raise ValueError('bad metadata') from e
  if len(self.labels)!=self.c:raise ValueError('labels')
  off=mlen
  def take(fmt,n):
   nonlocal off
   values=struct.unpack_from('<'+str(n)+fmt,body,off);off+=struct.calcsize(fmt)*n;return values
  self.counts=take('I',self.c);self.weights=take('I',self.d)
  if sum(self.counts)!=self.n or any(not 1<=w<=8 for w in self.weights) or self.ne!=sum(self.weights)*self.cap*self.cap+1:raise ValueError('geometry')
  self.codes=take('B',self.d*self.n)
  if any(v>self.cap for v in self.codes):raise ValueError('code outside domain')
  npairs=self.c*(self.c-1)//2;sizes=take('I',npairs);self.profiles=take('I',npairs);self.bias=take('d',npairs)
  if sum(sizes)!=terms or any(v>=self.nt for v in self.profiles):raise ValueError('pair inventory')
  ids=take('I',terms);coef=take('d',terms);self.table=take('d',self.nt*self.ne)
  if off!=len(body) or any(v>=self.n for v in ids) or any(not math.isfinite(v) for v in coef+self.bias+self.table):raise ValueError('invalid parameter')
  self.pairs=[];start=0
  for n in sizes:self.pairs.append((ids[start:start+n],coef[start:start+n]));start+=n
 def margins(self,row):
  if len(row)!=self.d or any(type(x) is not int or not 0<=x<=self.cap for x in row):raise ValueError('integer input required')
  cache={};result=[]
  for pi,(ids,coefs) in enumerate(self.pairs):
   total=0.;base=self.profiles[pi]*self.ne
   for i,a in zip(ids,coefs):
    if i not in cache:cache[i]=sum(w*(x-self.codes[i*self.d+j])**2 for j,(x,w) in enumerate(zip(row,self.weights)))
    total+=a*self.table[base+cache[i]]
   result.append(total+self.bias[pi])
  return result
 def predict(self,row):
  margins=self.margins(row)
  if self.c==2:return self.labels[int(margins[0]>=0)]
  votes=[0]*self.c;k=0
  for i in range(self.c):
   for j in range(i+1,self.c):votes[i if margins[k]>0 else j]+=1;k+=1
  return self.labels[max(range(self.c),key=lambda i:votes[i])]
