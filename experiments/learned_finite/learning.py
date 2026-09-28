"""Deployment-compatible group metric and radial table learning (training only).

NCA-style neighbors and centered-alignment mixtures are established techniques.
The finite integer projection is part of the model, not a post-test approximation.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import nnls

GAMMA_MULTIPLIERS=np.array([.125,.25,.5,1.,2.,4.,8.,16.])
CS=(.1,1.,10.,100.)

def distances(x,z,weights):
    # Binary inputs make this expression exact in these small integer ranges.
    a=np.asarray(x,dtype=np.float64);b=np.asarray(z,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    d=(a*w).sum(1)[:,None]+(b*w).sum(1)[None,:]-2*(a*w)@b.T
    if not np.isfinite(d).all() or (d<0).any() or not np.equal(d,np.rint(d)).all():raise ValueError('noninteger binary distance')
    return d.astype(np.int32)

def learn_metric(x,y,feature_groups):
    import torch
    torch.set_num_threads(1)
    started=time.process_time();G=int(feature_groups.max())+1
    component=np.stack([distances(x[:,feature_groups==g],x[:,feature_groups==g],np.ones((feature_groups==g).sum())) for g in range(G)])
    d0=component.sum(0); median=float(np.median(d0[d0>0])); beta=8./median
    D=torch.tensor(component,dtype=torch.float64)
    yy=torch.tensor(y);same=yy[:,None].eq(yy[None,:]);same.fill_diagonal_(False)
    if not same.any(1).all():raise ValueError('metric requires at least two examples per class')
    eye=torch.eye(len(y),dtype=torch.bool)
    theta=torch.zeros(G,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.Adam([theta],lr=.05);curve=[]
    for step in range(120):
        optimizer.zero_grad();weight=torch.softmax(theta,dim=0)*G
        logit=-beta*torch.einsum('g,gij->ij',weight,D)
        logit=logit.masked_fill(eye,-torch.inf)
        logp=torch.logsumexp(logit.masked_fill(~same,-torch.inf),dim=1)-torch.logsumexp(logit,dim=1)
        loss=-logp.mean()+.01*torch.log(weight).square().mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite metric loss')
        loss.backward();optimizer.step();curve.append(float(loss.detach()))
        if time.process_time()-started>60:raise TimeoutError('metric optimization exceeds prospective 60 CPU seconds')
    continuous=(torch.softmax(theta,dim=0)*G).detach().numpy()
    projected=np.clip(np.rint(2*continuous),0,7).astype(np.uint8)
    if not projected.any():projected[np.argmax(continuous)]=1
    return projected[feature_groups], {'steps':120,'loss':curve,'continuous':continuous.tolist(),
        'group_integer_weights':projected.tolist(),'temperature':beta,'cpu_seconds':time.process_time()-started}

def table(scales,weights,max_distance):
    # These serialized float64 table values define the fitted kernel.
    result=np.zeros(max_distance+1,np.float64)
    grid=np.arange(max_distance+1,dtype=np.float64)
    for gamma,beta in zip(scales,weights):result+=float(beta)*np.exp(-float(gamma)*grid)
    return result

def learn_mixture(d,y,scales):
    started=time.process_time()
    kernels=np.stack([np.exp(-float(g)*d) for g in scales])
    centered=kernels-kernels.mean(2,keepdims=True)-kernels.mean(1,keepdims=True)+kernels.mean((1,2),keepdims=True)
    norms=np.linalg.norm(centered.reshape(len(scales),-1),axis=1)
    if (norms<=0).any():raise ValueError('degenerate centered kernel')
    normalized=centered/norms[:,None,None]
    counts=np.bincount(y);Y=np.eye(len(counts))[y]/np.sqrt(counts)[None,:]
    target=Y@Y.T;target=target-target.mean(0)[None,:]-target.mean(1)[:,None]+target.mean()
    target/=np.linalg.norm(target)
    flat=normalized.reshape(len(scales),-1)
    M=flat@flat.T;a=flat@target.ravel()
    ridge=1e-8;R=np.linalg.cholesky(M+ridge*np.eye(len(scales))).T
    coefficient,residual=nnls(R,np.linalg.solve(R.T,a),maxiter=1000)
    beta=coefficient/norms
    if beta.sum()<=0:raise ValueError('alignment weights vanished')
    beta/=beta.sum()
    return beta,{'coefficients':beta.tolist(),'system':M.tolist(),'target':a.tolist(),
                 'ridge':ridge,'residual':float(residual),'cpu_seconds':time.process_time()-started}

def align_tables(tables,d,y):
    """The same centered-alignment objective for any fixed PSD table bases."""
    kernels=tables[:,d]
    centered=kernels-kernels.mean(2,keepdims=True)-kernels.mean(1,keepdims=True)+kernels.mean((1,2),keepdims=True)
    norms=np.linalg.norm(centered.reshape(len(tables),-1),axis=1)
    if (norms<=0).any():raise ValueError('degenerate base kernel')
    flat=(centered/norms[:,None,None]).reshape(len(tables),-1)
    counts=np.bincount(y);Y=np.eye(len(counts))[y]/np.sqrt(counts)[None,:]
    target=Y@Y.T;target=target-target.mean(0)[None,:]-target.mean(1)[:,None]+target.mean();target/=np.linalg.norm(target)
    M=flat@flat.T;a=flat@target.ravel();R=np.linalg.cholesky(M+1e-8*np.eye(len(tables))).T
    coef,residual=nnls(R,np.linalg.solve(R.T,a),maxiter=1000);beta=coef/norms
    if beta.sum()<=0:raise ValueError('zero mixture')
    beta/=beta.sum()
    # Sum in an explicit fixed order; resulting table bytes are the kernel.
    lut=np.zeros(tables.shape[1],dtype=np.float64)
    for b,t in zip(beta,tables):lut+=b*t
    return lut,{'weights':beta.tolist(),'ridge':1e-8,'residual':float(residual)}

def spectral_tables(d,orders):
    """Normalized Krawtchouk recurrence, average parity-feature inner products."""
    if d<1 or any(r<1 or r>d for r in orders):raise ValueError('invalid spectral order')
    s=np.arange(d+1,dtype=np.float64);previous=np.ones(d+1);current=1.-2.*s/d
    output={1:current.copy()}
    for r in range(1,max(orders)):
        following=((d-2*s)*current-r*previous)/(d-r)
        previous,current=current,following;output[r+1]=current.copy()
    return np.stack([output[r] for r in orders])
