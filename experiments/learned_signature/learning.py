"""Training-only metric/mixture learning for integer-distance lookup classifiers."""
from __future__ import annotations
import hashlib, io, json, time, zipfile
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize, nnls
from scipy.special import expit
from sklearn.model_selection import train_test_split
MULTIPLIERS=(.125,.25,.5,1.,2.,4.,8.)
C_VALUES=(1.,10.,100.)
RADIAL_SCALES=(.25,.5,1.,2.,4.)
SEEDS=(1401,2402,3403)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def load_part(data_dir,task,*,test=False):
 """Evaluation part is explicit; development callers never request it."""
 data_dir=Path(data_dir)
 if task=='letter':
  with zipfile.ZipFile(data_dir/'letter.zip') as z:raw=z.read('letter-recognition.data')
  lines=raw.splitlines();chosen=lines[16000:] if test else lines[:16000]
  a=np.loadtxt(io.BytesIO(b'\n'.join(chosen)),delimiter=',',dtype=str)
  q=a[:,1:].astype(np.uint8);y=np.array([ord(v)-ord('A') for v in a[:,0]],dtype=np.int64);cap=15
 elif task=='pendigits':
  with zipfile.ZipFile(data_dir/'pendigits.zip') as z:raw=z.read('pendigits.tes' if test else 'pendigits.tra')
  a=np.loadtxt(io.BytesIO(raw),delimiter=',',dtype=np.int64);q=a[:,:-1].astype(np.uint8);y=a[:,-1];cap=100
 else:raise ValueError('unknown task')
 if q.ndim!=2 or len(q)!=len(y) or int(q.max())>cap:raise ValueError('invalid data')
 return np.ascontiguousarray(q),y,cap

def split_indices(y,seed,cap=5000):
 f,v=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
 if len(f)>cap:f,_=train_test_split(f,train_size=cap,stratify=y[f],random_state=seed+1)
 return np.sort(f),np.sort(v)

def stratified_subset(y,size,seed):
 if len(y)<=size:return np.arange(len(y))
 ids,_=train_test_split(np.arange(len(y)),train_size=size,stratify=y,random_state=seed)
 return np.sort(ids)

def signatures(a,b,weights):
 """Integer dot products exactly representable in binary64 under these caps."""
 if a.dtype!=np.uint8 or b.dtype!=np.uint8:raise ValueError('integer codes required')
 w=np.asarray(weights,dtype=np.int64)
 if w.shape!=(a.shape[1],) or np.any(w<1) or np.any(w>8):raise ValueError('weights 1..8 required')
 af,bf=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64);aw,bw=af*w,bf*w
 r=aw@bf.T;r*=-2.;r+=(aw*af).sum(axis=1)[:,None];r+=(bw*bf).sum(axis=1)[None,:]
 if np.any(r<0) or np.any(r>np.iinfo(np.uint32).max) or not np.array_equal(r,np.rint(r)):raise ArithmeticError('inexact signature')
 return r.astype(np.uint32)

def base_gamma(q,cap):
 v=float((q.astype(np.float64)/cap).var());return 1./(q.shape[1]*v) if v else 1.

def learn_metric(q,y,cap,seed):
 """Three training-only hit/miss mining rounds with trace-normalized diagonal metric."""
 start,cpu=time.perf_counter(),time.process_time()
 bi=stratified_subset(y,4096,seed+10);ai=stratified_subset(y,2048,seed+11)
 b,a=q[bi].astype(np.float64)/cap,q[ai].astype(np.float64)/cap;by,ay=y[bi],y[ai]
 d=q.shape[1];z=np.zeros(d);history=[]
 for iteration in range(3):
  e=np.exp(z-z.max());w=d*e/e.sum()
  dist=-2*(a*w)@b.T;dist+=((a*a)*w).sum(axis=1)[:,None];dist+=((b*b)*w).sum(axis=1)[None,:]
  same=ay[:,None]==by[None,:];selfpair=ai[:,None]==bi[None,:]
  pos=np.where(same&~selfpair,dist,np.inf).argmin(axis=1);neg=np.where(~same,dist,np.inf).argmin(axis=1)
  pp=(a-b[pos])**2;nn=(a-b[neg])**2
  scale=max(float(np.median(((pp+nn)*w).sum(axis=1))),1e-8);diff=(pp-nn)/scale;pull=pp/scale
  def objective(u):
   e=np.exp(u-u.max());v=d*e/e.sum();margin=.5+diff@v
   loss=np.logaddexp(0.,margin).mean()+.05*(pull@v).mean()+.01*np.mean(u*u)
   gw=(expit(margin)[:,None]*diff).mean(axis=0)+.05*pull.mean(axis=0)
   return loss,v*(gw-np.dot(gw,v)/d)+.02*u/d
  done=minimize(objective,z,jac=True,method='L-BFGS-B',bounds=[(-2.,2.)]*d,options={'maxiter':150,'ftol':1e-12,'gtol':1e-8})
  z=done.x;history.append({'round':iteration,'success':bool(done.success),'iterations':int(done.nit),'objective':float(done.fun),'scale':scale,'message':str(done.message)})
 e=np.exp(z-z.max());continuous=d*e/e.sum();weights=np.clip(np.rint(4.*continuous),1,8).astype(np.uint32)
 return weights,{'weights':weights.tolist(),'continuous':continuous.tolist(),'rounds':history,'anchors':len(ai),'bank':len(bi),'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start}

def table_profile(cap,weights,gamma,coefficients=(1.,),scales=(1.,)):
 """One stored table regardless of the number of nonnegative radial components."""
 w=np.asarray(weights);alpha=np.asarray(coefficients,dtype=np.float64);rates=np.asarray(scales,dtype=np.float64)
 if w.ndim!=1 or not np.all(w==np.floor(w)) or np.any(w<1) or np.any(w>8):raise ValueError('invalid weights')
 if alpha.ndim!=1 or len(alpha)!=len(rates) or np.any(alpha<0) or not np.isfinite(alpha).all() or alpha.sum()<=0:raise ValueError('invalid mixture')
 if gamma<=0 or not np.isfinite(gamma) or np.any(rates<=0) or not np.isfinite(rates).all():raise ValueError('invalid rates')
 alpha=alpha/alpha.sum();size=int(w.sum())*cap*cap+1
 if size>4_000_001:raise ValueError('table exceeds 32 MB budget')
 dist=np.arange(size,dtype=np.float64)/(cap*cap*float(w.mean()));table=np.zeros(size,dtype=np.float64)
 for a,r in zip(alpha,rates):table+=a*np.exp(-(gamma*r)*dist)
 return table

def learn_mixture(q,y,cap,weights,gamma,seed):
 """Centered-alignment nonnegative QP; an established MKL comparator."""
 ids=stratified_subset(y,1024,seed+21)
 dist=signatures(q[ids],q[ids],weights).astype(np.float64)/(cap*cap*float(np.mean(weights)))
 label=(y[ids,None]==y[ids][None,:]).astype(np.float64);label-=label.mean(axis=0,keepdims=True);label-=label.mean(axis=1,keepdims=True)
 kernels=[]
 for scale in RADIAL_SCALES:
  k=np.exp(-(gamma*scale)*dist);k-=k.mean(axis=0,keepdims=True);k-=k.mean(axis=1,keepdims=True);kernels.append(k)
 f=np.stack([k.ravel() for k in kernels]);M=f@f.T;target=f@label.ravel();ridge=max(float(np.trace(M)/len(kernels)*1e-8),1e-12)
 L=np.linalg.cholesky(M+ridge*np.eye(len(kernels)))
 coeff,res=nnls(L.T,solve_triangular(L,target,lower=True),maxiter=10000)
 if coeff.sum()<=0:coeff=np.ones(len(kernels))
 coeff/=coeff.sum()
 return coeff,{'samples':len(ids),'ridge':ridge,'residual':float(res),'coefficients':coeff.tolist(),'scales':list(RADIAL_SCALES)}
