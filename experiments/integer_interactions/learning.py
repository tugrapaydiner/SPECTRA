"""Full/diagonal supervised geometry compiled into a fixed integer embedding.

Training only. Kernel fitting uses the exact projected map and two-table product,
not a continuous model secretly approximated at deployment.
"""
from __future__ import annotations
import math, time
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


def subset(y, cap, seed):
    index = np.arange(len(y))
    if len(index) <= cap: return index
    selected, _ = train_test_split(index, train_size=cap, stratify=y, random_state=seed)
    return np.sort(selected)


def split(q, y, seed, fit_cap=6000, val_cap=2000):
    _, group = np.unique(q, axis=0, return_inverse=True)
    fit, val = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(q, y, group))
    fit = fit[subset(y[fit], fit_cap, seed+1)]
    val = val[subset(y[val], val_cap, seed+2)]
    assert not set(group[fit]) & set(group[val])
    return fit, val


class NeighborhoodObjective:
    """Exact analytic gradient of a normalized positive metric factor.

    Effective M = d*(B.T B + floor*I)/trace(B.T B + floor*I).
    Both diagonal and full candidates optimize the same objective and samples.
    """
    def __init__(self, x, y, seed, diagonal=False, regularizer=.02, floor=.05):
        self.d=x.shape[1]; self.diagonal=diagonal; self.reg=regularizer; self.floor=floor
        gi=subset(y,2048,seed); ai=subset(y,512,seed+1)
        self.a=np.ascontiguousarray(x[ai],dtype=np.float64)
        self.b=np.ascontiguousarray(x[gi],dtype=np.float64)
        self.same=y[ai,None]==y[gi][None,:]
        self.self_match=ai[:,None]==gi[None,:]; self.same[self.self_match]=False
        dd=cdist(self.a,self.b,'sqeuclidean'); dd[~self.same]=np.inf
        nn=np.partition(dd,2,axis=1)[:,2]
        if not np.isfinite(nn).all(): raise ValueError('three same-label neighbors required')
        self.beta=1/max(float(np.median(nn)),1e-4)
        self.calls=0
    def __call__(self, p):
        B=np.diag(p) if self.diagonal else p.reshape(self.d,self.d)
        H=B.T@B+self.floor*np.eye(self.d); tr=float(np.trace(H)); scale=self.d/tr; M=H*scale
        ax=self.a@M; bx=self.b@M
        distance=np.sum(ax*self.a,axis=1)[:,None]+np.sum(bx*self.b,axis=1)[None,:]-2*ax@self.b.T
        logits=-self.beta*distance; logits[self.self_match]=-np.inf
        z=logsumexp(logits,axis=1,keepdims=True)
        positive=np.where(self.same,logits,-np.inf); zp=logsumexp(positive,axis=1,keepdims=True)
        residual=np.exp(positive-zp)-np.exp(logits-z)
        grad=(self.a.T@(residual.sum(axis=1)[:,None]*self.a)+
              self.b.T@(residual.sum(axis=0)[:,None]*self.b)-
              self.a.T@residual@self.b-self.b.T@residual.T@self.a)*(self.beta/len(self.a))
        change=M-np.eye(self.d)
        value=float(np.mean(z-zp)+self.reg*np.sum(change*change)/self.d)
        grad+=(2*self.reg/self.d)*change
        gh=scale*(grad-np.sum(grad*H)/tr*np.eye(self.d))
        gb=2*B@gh
        self.calls+=1
        return value, np.diag(gb).copy() if self.diagonal else gb.ravel()


def fit_map(q,y,maximum,method,seed=20260928,integer_scale=8,maxiter=100):
    q=np.asarray(q); y=np.asarray(y)
    if q.ndim!=2 or len(q)!=len(y) or q.dtype!=np.uint8 or not 1<=maximum<=255 or np.any(q>maximum):
        raise ValueError('expected valid uint8 fitting codes and labels')
    d=q.shape[1]; x=q.astype(float)/maximum; start=time.process_time(); history={}
    if method=='uniform': B=np.eye(d)
    elif method in ('diagonal','full'):
        objective=NeighborhoodObjective(x,y,seed,diagonal=method=='diagonal')
        initial=np.ones(d) if method=='diagonal' else np.eye(d).ravel()
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',
                        options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-6,'maxls':30})
        B=np.diag(result.x) if method=='diagonal' else result.x.reshape(d,d)
        history={'objective':float(result.fun),'iterations':int(result.nit),'success':bool(result.success),
                 'message':str(result.message),'objective_calls':objective.calls,'beta':objective.beta,
                 'gallery':len(objective.b),'anchors':len(objective.a)}
    elif method in ('local_supervised','local_unsupervised'):
        gallery=subset(y,6000,seed); anchors=subset(y,2048,seed+1)
        scatter=np.zeros((d,d)); n=0
        for start_anchor in range(0,len(anchors),128):
            ids=anchors[start_anchor:start_anchor+128]
            dd=cdist(x[ids],x[gallery],'sqeuclidean')
            dd[ids[:,None]==gallery[None,:]]=np.inf
            if method=='local_supervised': dd[y[ids,None]!=y[gallery][None,:]]=np.inf
            near=np.argsort(dd,axis=1,kind='stable')[:,:3]
            if not np.isfinite(np.take_along_axis(dd,near,axis=1)).all():
                raise ValueError('three local neighbors required')
            dif=(x[ids,None,:]-x[gallery[near]]).reshape(-1,d)
            scatter+=dif.T@dif; n+=len(dif)
        scatter/=n
        ev,V=np.linalg.eigh(scatter);ev=np.maximum(ev,.05*max(float(ev.mean()),1e-12))
        B=(V/np.sqrt(ev)[None,:]).T
        history={'local_eigenvalues':ev.tolist(),'shrinkage_floor':.05,
                 'anchors':len(anchors),'gallery':len(gallery),'neighbor_pairs':n,
                 'same_label_neighbors':method=='local_supervised'}
    elif method=='whitening':
        within=np.zeros((d,d)); n=0
        for label in np.unique(y):
            z=x[y==label]; z=z-z.mean(axis=0); within+=z.T@z; n+=len(z)-1
        within/=max(n,1)
        ev,V=np.linalg.eigh(within); ev=np.maximum(ev,.05*max(float(ev.mean()),1e-12))
        B=(V/np.sqrt(ev)[None,:]).T
        # Rows are ordered deterministically by eigh; sign does not affect distances.
        history={'within_eigenvalues':ev.tolist(),'shrinkage_floor':.05}
    else: raise ValueError('unknown method')
    if not np.isfinite(B).all() or np.linalg.norm(B)==0: raise ValueError('invalid learned factor')
    # One scale fixes trace across candidates before integer projection.
    B=B*math.sqrt(d/float(np.sum(B*B)))
    quantized=np.clip(np.rint(integer_scale*B),-31,31).astype(np.int16)
    A=np.concatenate((quantized,np.eye(d,dtype=np.int16)),axis=0)
    if np.any(A==0): pass # Zero individual coefficients are normal; identity preserves rank.
    # Remove all-zero rows without changing the mathematical function.
    A=A[np.any(A!=0,axis=1)]
    lengths=np.abs(A.astype(np.int64)).sum(axis=1)*maximum
    if np.any(lengths>32767): raise ValueError('projection exceeds int16 code range')
    if int(lengths@lengths)>=2**48: raise ValueError('squared-signature bound exceeds exact binary64 integers')
    rec={'method':method,'seed':seed,'raw_features':d,'embedding_features':len(A),'integer_scale':integer_scale,
         'integer_matrix':A.tolist(),'continuous_factor':B.tolist(),'trace_integer':int(np.sum(A.astype(np.int64)**2)),
         'signature_bound':int(lengths@lengths),'fit_rows':len(q),'cpu_seconds':time.process_time()-start,**history}
    return A,rec


def project(q,A,maximum):
    if np.asarray(q).dtype!=np.uint8 or np.asarray(q).ndim!=2 or np.asarray(A).ndim!=2 or q.shape[1]!=A.shape[1]:
        raise ValueError('uint8 codes / matrix geometry required')
    if np.any(q>maximum): raise ValueError('query outside integer domain')
    A=np.asarray(A,dtype=np.int64); offset=-maximum*np.minimum(A,0).sum(axis=1)
    z=q.astype(np.int64)@A.T+offset
    if np.any(z<0) or np.any(z>32767): raise ValueError('projection outside checked int16 bounds')
    return np.ascontiguousarray(z,dtype=np.int16)


@dataclass
class ProductKernel:
    coefficient: float
    bits: int
    high: np.ndarray
    low: np.ndarray
    bound: int
    def evaluate(self,s):
        s=np.asarray(s)
        if s.dtype.kind not in 'iu' or np.any(s<0) or np.any(s>self.bound): raise ValueError('invalid integer signature')
        s=s.astype(np.uint64,copy=False)
        return self.high[s>>self.bits]*self.low[s&((1<<self.bits)-1)]


def make_kernel(A,maximum,gamma):
    A=np.asarray(A,dtype=np.int64)
    trace=int(np.sum(A*A)); d=A.shape[1]
    if trace<=0 or not math.isfinite(gamma) or gamma<=0: raise ValueError('invalid kernel settings')
    lengths=np.abs(A).sum(axis=1)*maximum; bound=int(lengths@lengths)
    if bound<=0 or bound>=2**48: raise ValueError('invalid signature/table bound')
    bits=(bound.bit_length()+1)//2; B=1<<bits
    if B+(bound>>bits)+1>1048576: raise ValueError('table entry cap exceeded')
    alpha=float(gamma)/(maximum*maximum*(trace/d))
    high=np.fromiter((math.exp(-alpha*float(k*B)) for k in range((bound>>bits)+1)),dtype=np.float64)
    low=np.fromiter((math.exp(-alpha*float(k)) for k in range(B)),dtype=np.float64)
    return ProductKernel(alpha,bits,high,low,bound)


def gram(a,b,kernel,path=None):
    a=np.ascontiguousarray(a,dtype=float); b=np.ascontiguousarray(b,dtype=float)
    result=np.empty((len(a),len(b))) if path is None else np.memmap(path,mode='w+',dtype=np.float64,shape=(len(a),len(b)))
    for start in range(0,len(a),128):
        distance=cdist(a[start:start+128],b,'sqeuclidean'); s=distance.astype(np.uint64)
        if not np.equal(distance,s).all(): raise ValueError('inexact integer training distance')
        result[start:start+128]=kernel.evaluate(s)
    if path is not None: result.flush()
    return result
