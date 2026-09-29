"""Separately decoded, row-vectorized ordered interpreter for stored table models.

Uses NumPy for batch verification, not the deployment or training implementation.
Distance accumulation is integer; coefficient summation preserves serialized order.
"""
from __future__ import annotations
import json,struct,zlib
from pathlib import Path
import numpy as np

class OrderedOracle:
 def __init__(self,path):
  raw=Path(path).read_bytes()
  if len(raw)<44 or len(raw)>128*1024*1024:raise ValueError('size')
  magic,d,c,n,cap,nt,ne,terms,meta,crc=struct.unpack_from('<8s9I',raw)
  if magic!=b'SPLKT001' or zlib.crc32(raw[44:])!=crc:raise ValueError('identity')
  pairs=c*(c-1)//2;expected=44+meta+4*c+4*d+n*d+16*pairs+12*terms+8*nt*ne
  if len(raw)!=expected:raise ValueError('inventory')
  self.labels=json.loads(raw[44:44+meta])['labels'];self.d,self.cap,self.n,self.ne=d,cap,n,ne
  offset=44+meta
  def take(dtype,count):
   nonlocal offset
   v=np.frombuffer(raw,dtype=dtype,count=count,offset=offset);offset+=v.nbytes;return v
  counts=take('<u4',c);self.weights=take('<u4',d).astype(np.int64);self.codes=take('u1',n*d).reshape(n,d)
  sizes=take('<u4',pairs);self.profiles=take('<u4',pairs);self.bias=take('<f8',pairs);ids=take('<u4',terms);coefs=take('<f8',terms);self.tables=take('<f8',nt*ne).reshape(nt,ne)
  if self.ne!=self.weights.sum()*cap*cap+1 or counts.sum()!=n:raise ValueError('geometry')
  self.pairs=[];first=0
  for size in sizes:size=int(size);self.pairs.append((ids[first:first+size],coefs[first:first+size]));first+=size
 def margins(self,q,chunk=128):
  q=np.asarray(q)
  if q.dtype!=np.uint8 or q.ndim!=2 or q.shape[1]!=self.d or np.any(q>self.cap):raise ValueError('domain')
  out=np.empty((len(q),len(self.pairs)),dtype=np.float64)
  for first in range(0,len(q),chunk):
   x=q[first:first+chunk].astype(np.int64);dist=np.zeros((len(x),self.n),dtype=np.int64)
   for j,w in enumerate(self.weights):
    delta=x[:,j,None]-self.codes[None,:,j];dist+=w*delta*delta
   if dist.max(initial=0)>=self.ne:raise ValueError('distance')
   for k,(ids,coeff) in enumerate(self.pairs):
    bank=self.tables[self.profiles[k]];score=np.zeros(len(x),dtype=np.float64)
    for sid,a in zip(ids,coeff):score+=a*bank[dist[:,sid]]
    out[first:first+len(x),k]=score+self.bias[k]
  return out
 def predictions(self,margins):
  c=len(self.labels)
  if c==2:return np.asarray(self.labels)[(margins[:,0]>=0).astype(int)]
  wins=np.zeros((len(margins),c),dtype=np.int32);k=0
  for i in range(c):
   for j in range(i+1,c):wins[:,i]+=margins[:,k]>0;wins[:,j]+=margins[:,k]<=0;k+=1
  return np.asarray(self.labels)[wins.argmax(1)]
