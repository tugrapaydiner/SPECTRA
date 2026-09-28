"""Training-only local margin metric under a fixed integer replication budget.

This is a diagonal large-margin metric experiment, not a claim to invent metric
learning. A nonnegative integer weight is an exact feature replication count.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.special import expit


def integer_budget(weights,budget):
    w=np.asarray(weights,dtype=np.float64)
    if w.ndim!=1 or not len(w) or not np.isfinite(w).all() or (w<0).any() or not w.sum()>0:raise ValueError('invalid weights')
    if type(budget) is not int or not 1<=budget<=4096:raise ValueError('invalid budget')
    scaled=w/w.sum()*budget;out=np.floor(scaled).astype(np.int64)
    remain=budget-int(out.sum())
    order=np.lexsort((np.arange(len(w)),-(scaled-out)))
    out[order[:remain]]+=1
    if int(out.sum())!=budget or (out<0).any():raise RuntimeError('allocation invariant')
    return out


def train_metric(X,y,*,seed=20260928,regularization=.1,anchors=1536,rounds=2):
    if len(X)!=len(y) or X.ndim!=2 or not np.isfinite(X).all():raise ValueError('invalid fitting data')
    n,d=X.shape;rng=np.random.default_rng(seed);ii=rng.choice(n,min(anchors,n),False)
    w=np.ones(d,dtype=np.float64);hist=[]
    for step in range(rounds):
        triplets=[]
        for start in range(0,len(ii),128):
            ids=ii[start:start+128];dist=cdist(X[ids]*np.sqrt(w),X*np.sqrt(w),'sqeuclidean')
            dist[np.arange(len(ids)),ids]=np.inf
            same=y[ids,None]==y[None,:]
            pos=np.where(same,dist,np.inf);neg=np.where(~same,dist,np.inf)
            pp=np.argpartition(pos,2,axis=1)[:,:3];nn=np.argpartition(neg,2,axis=1)[:,:3]
            for k,i in enumerate(ids):
                for j in pp[k]:
                    for l in nn[k]:triplets.append((int(i),int(j),int(l)))
        tri=np.asarray(triplets,dtype=np.int64)
        pos=(X[tri[:,0]]-X[tri[:,1]])**2;neg=(X[tri[:,0]]-X[tri[:,2]])**2
        scale=max(float(np.median(pos.sum(1))),1e-4);delta=(pos-neg)/scale
        def fun(a):
            z=delta@a+1
            loss=float(np.logaddexp(0,z).mean()+regularization*np.mean((a-1)**2))
            grad=delta.T@expit(z)/len(z)+2*regularization*(a-1)/d
            return loss,grad
        result=minimize(fun,w,jac=True,method='SLSQP',bounds=[(0.,4.)]*d,
                        constraints=[{'type':'eq','fun':lambda a:np.sum(a)-d,'jac':lambda a:np.ones(d)}],
                        options={'maxiter':150,'ftol':1e-8})
        w=np.maximum(result.x,0);w=w/w.sum()*d
        hist.append({'round':step,'objective':float(result.fun),'success':bool(result.success),'message':str(result.message),
                     'iterations':int(result.nit),'scale':scale,'weights':w.tolist(),'triplets':len(tri)})
    counts=integer_budget(w,2*d)
    return counts,{'continuous_weights':w.tolist(),'counts':counts.tolist(),'budget':2*d,'regularization':regularization,'seed':seed,'anchors':ii.tolist(),'history':hist}
