"""Supervised quantized RBF prototypes; no teacher or test fitting.
Training-only PyTorch. Exact integer geometry with a float64 head is exported.
"""
from __future__ import annotations
import dataclasses, hashlib, json, math, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
import torch
from torch import nn

@dataclasses.dataclass(frozen=True)
class Settings:
    prototypes: int = 256
    gamma: float = 8.0
    epochs: int = 120
    batch: int = 256
    lr: float = .01
    reg: float = 1e-4
    metric_reg: float = .001
    quarter: int = 4
    units: int = 4
    seed: int = 611
    arm: str = 'local'
    affine: bool = False
    adaptive_gamma: bool = False
    normalized: bool = False
    average_tail: bool = False

    def __post_init__(self):
        for name in ('prototypes','epochs','batch','quarter','units'):
            value=getattr(self,name)
            if type(value) is not int or value<1:raise ValueError(name+' must be a positive integer')
        if self.prototypes>4096 or self.batch>65536 or self.quarter>16 or self.units>16:
            raise ValueError('configuration exceeds compiled limits')
        if type(self.seed) is not int or not 0<=self.seed<2**32:raise ValueError('invalid seed')
        for name in ('gamma','lr','reg','metric_reg'):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('nonfinite/negative optimization setting')
        if self.gamma==0 or self.lr==0:raise ValueError('gamma and learning rate must be positive')
        if self.arm not in ('fixed','centers','local'):raise ValueError('unknown learner')
        for name in ('affine','adaptive_gamma','normalized','average_tail'):
            if type(getattr(self,name)) is not bool:raise ValueError('experimental flags must be boolean')

def checked(q, maximum):
    q=np.asarray(q)
    if q.ndim!=2 or q.dtype!=np.uint8 or len(q)<1 or not 1<=q.shape[1]<=256 or type(maximum) is not int or not 1<=maximum<=255 or np.any(q>maximum):
        raise ValueError('bounded original uint8 codes required')
    return q

def grouped_split(q,y,seed,cap=6000,val_cap=2000):
    _,groups=np.unique(q,axis=0,return_inverse=True)
    a,b=next(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(q,y,groups))
    from sklearn.model_selection import train_test_split
    if len(a)>cap: a,_=train_test_split(a,train_size=cap,stratify=y[a],random_state=seed+1)
    if len(b)>val_cap:b,_=train_test_split(b,train_size=val_cap,stratify=y[b],random_state=seed+2)
    a,b=np.sort(a),np.sort(b)
    assert not set(groups[a])&set(groups[b])
    return a,b

def integer_weights(logits,units=4):
    p=np.asarray(logits,dtype=np.float64)
    if p.ndim!=2 or not np.isfinite(p).all(): raise ValueError('finite metric logits required')
    d=p.shape[1];soft=np.exp(p-p.max(axis=1,keepdims=True));soft/=soft.sum(axis=1,keepdims=True)
    extras=soft*((units-1)*d);floor=np.floor(extras).astype(np.int64);left=(units-1)*d-floor.sum(axis=1)
    for i in range(len(p)):
        order=np.lexsort((np.arange(d),-(extras-floor)[i]));floor[i,order[:left[i]]]+=1
    result=1+floor
    if result.max()>65535:raise ValueError('metric overflow')
    assert np.all(result.sum(axis=1)==units*d)
    return result.astype(np.uint16)

def center_init(x,y,p,seed):
    labels=np.unique(y)
    if p<len(labels) or p>len(x):raise ValueError('prototype budget outside fitting size')
    out=[];assignments=[]
    for i,label in enumerate(labels):
        n=p//len(labels)+(i<p%len(labels));xx=x[y==label]
        if n>len(xx):raise ValueError('not enough class members')
        km=KMeans(n_clusters=n,n_init=1,max_iter=40,random_state=seed+i,algorithm='lloyd').fit(xx)
        out.extend(km.cluster_centers_);assignments.extend([i]*n)
    return np.asarray(out,np.float32),np.asarray(assignments)

class Model(nn.Module):
    def __init__(self,centers,classes,assignments,maximum,s):
        super().__init__();self.s=s;self.maximum=maximum
        self.centers=nn.Parameter(torch.as_tensor(centers.copy()),requires_grad=s.arm!='fixed')
        self.metric=nn.Parameter(torch.zeros_like(self.centers),requires_grad=s.arm=='local')
        self.log_gamma=nn.Parameter(torch.full((len(centers),),math.log2(s.gamma)),requires_grad=s.adaptive_gamma)
        self.head=nn.Parameter(torch.zeros((len(centers)*(centers.shape[1]+1 if s.affine else 1),classes)));self.bias=nn.Parameter(torch.zeros(classes))
        with torch.no_grad():self.head[torch.arange(len(centers))*(centers.shape[1]+1 if s.affine else 1),torch.as_tensor(assignments)]=3.
    def geometry(self):
        Q=self.maximum*self.s.quarter;center=self.centers.clamp(0,1);rounded=torch.round(center*Q)/Q
        center=center+(rounded-center).detach();d=center.shape[1]
        relaxed=(1+torch.softmax(self.metric,dim=1)*((self.s.units-1)*d))/self.s.units
        extra=relaxed*self.s.units-1;floor=torch.floor(extra);left=((self.s.units-1)*d-floor.sum(1)).long()
        order=torch.argsort(extra-floor,dim=1,descending=True,stable=True);ranks=torch.argsort(order,dim=1,stable=True)
        quant=(1+floor+(ranks<left[:,None]))/self.s.units;weights=relaxed+(quant-relaxed).detach()
        return center,weights,relaxed
    def forward(self,x):
        c,w,relaxed=self.geometry();d=x.square()@w.T-2*x@(w*c).T+(w*c.square()).sum(1)
        lg=self.log_gamma.clamp(-2,7)
        if self.s.adaptive_gamma: lg=lg+(torch.round(lg*2)/2-lg).detach()
        g=torch.pow(2.,lg) if self.s.adaptive_gamma else self.s.gamma
        k=torch.softmax(-g*d.clamp_min(0),dim=1) if self.s.normalized else torch.exp(-g*d.clamp_min(0))
        if self.s.affine:
            k=torch.cat((k[:,:,None],k[:,:,None]*(x[:,None,:]-c[None,:,:])),dim=2).reshape(len(x),-1)
        return k@self.head+self.bias,relaxed
    def arrays(self):
        with torch.no_grad():
            centers=torch.round(self.centers.clamp(0,1)*(self.maximum*self.s.quarter)).to(torch.int32).cpu().numpy().astype(np.uint16)
            _,w,_=self.geometry();weights=torch.round(w*self.s.units).to(torch.int32).cpu().numpy().astype(np.uint16)
        result={'centers':centers,'weights':weights,'head':self.head.detach().double().numpy().copy(),'bias':self.bias.detach().double().numpy().copy()}
        if self.s.adaptive_gamma:
            result['gamma_code']=torch.round(self.log_gamma.detach().clamp(-2,7)*2+4).to(torch.int16).numpy().copy()
        return result

def kernel_config(d,maximum,quarter,units,gamma):
    if not 1<=quarter<=16 or not 1<=units<=16 or not math.isfinite(gamma) or gamma<=0:raise ValueError('invalid kernel setting')
    bound=int(units*d*(maximum*quarter)**2)
    if bound>=2**48:raise ValueError('signature bound exceeded')
    bits=(bound.bit_length()+1)//2;B=1<<bits
    if B+(bound>>bits)+1>1048576:raise ValueError('table cap exceeded')
    alpha=float(gamma)/(units*(maximum*quarter)**2)
    high=np.fromiter((math.exp(-alpha*float(i*B)) for i in range((bound>>bits)+1)),np.float64)
    low=np.fromiter((math.exp(-alpha*float(i)) for i in range(B)),np.float64)
    return {'alpha':alpha,'bits':bits,'bound':bound,'high':high,'low':low}

def features(q,arrays,maximum,s):
    q=checked(q,maximum);c=np.asarray(arrays['centers'],dtype=np.int64);w=np.asarray(arrays['weights'],dtype=np.int64)
    if c.ndim!=2 or c.shape!=w.shape or c.shape[1]!=q.shape[1] or np.any(c>maximum*s.quarter) or np.any(w<1) or np.any(w.sum(1)!=s.units*q.shape[1]):raise ValueError('bad integer model geometry')
    k=kernel_config(q.shape[1],maximum,s.quarter,s.units,s.gamma);out=np.empty((len(q),len(c)*(q.shape[1]+1 if s.affine else 1)),dtype=np.float64)
    for i in range(0,len(q),256):
        x=q[i:i+256].astype(np.int64)*s.quarter;S=(x*x)@w.T-2*x@(w*c).T+(w*c*c).sum(1)
        if np.any(S<0) or np.any(S>k['bound']):raise ValueError('signature outside bound')
        if s.adaptive_gamma:
            phi=np.empty(S.shape,dtype=np.float64)
            codes=np.asarray(arrays['gamma_code'])
            for code in np.unique(codes):
                kk=kernel_config(q.shape[1],maximum,s.quarter,s.units,2.**((int(code)-4)/2))
                cols=np.flatnonzero(codes==code);SS=S[:,cols]
                phi[:,cols]=kk['high'][SS>>kk['bits']]*kk['low'][SS&((1<<kk['bits'])-1)]
        else: phi=k['high'][S>>k['bits']]*k['low'][S&((1<<k['bits'])-1)]
        if s.normalized:
            totals=phi.sum(axis=1,keepdims=True)
            if np.any(totals==0):
                # Stable scalar reconstruction only for complete underflow.
                gs=np.asarray([2.**((int(t)-4)/2) for t in arrays['gamma_code']]) if s.adaptive_gamma else np.full(len(c),s.gamma)
                energy=-S*(gs/(s.units*(maximum*s.quarter)**2))
                ix=np.flatnonzero(totals[:,0]==0); pp=np.exp(energy[ix]-energy[ix].max(axis=1,keepdims=True));phi[ix]=pp;totals[ix]=pp.sum(axis=1,keepdims=True)
            phi/=totals
        if s.affine:
            delta=(x[:,None,:]-c[None,:,:]).astype(np.float64)/(maximum*s.quarter)
            phi=np.concatenate((phi[:,:,None],phi[:,:,None]*delta),axis=2).reshape(len(x),-1)
        out[i:i+256]=phi
    return out

def refit_head(q,y,arrays,maximum,s,maxiter=150):
    start=time.process_time();F=features(q,arrays,maximum,s);classes=arrays['head'].shape[1];p=F.shape[1]
    def obj(theta):
        H=theta[:p*classes].reshape(p,classes);b=theta[p*classes:];z=F@H+b
        zsum=logsumexp(z,axis=1);prob=np.exp(z-zsum[:,None])
        value=float(np.mean(zsum-z[np.arange(len(y)),y])+.5*s.reg*np.sum(H*H))
        prob[np.arange(len(y)),y]-=1;prob/=len(y);grad=F.T@prob+s.reg*H
        return value,np.r_[grad.ravel(),prob.sum(0)]
    initial=np.r_[arrays['head'].ravel(),arrays['bias']]
    res=minimize(obj,initial,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-6,'maxls':30})
    result={**arrays,'head':res.x[:p*classes].reshape(p,classes).copy(),'bias':res.x[p*classes:].copy()}
    return result,{'cpu_seconds':time.process_time()-start,'success':bool(res.success),'iterations':int(res.nit),'message':str(res.message),'objective':float(res.fun)}

def train(q,y,maximum,s:Settings,validation=None,final_head=True):
    q=checked(q,maximum);labels,yi=np.unique(y,return_inverse=True);start=time.process_time();wall=time.perf_counter()
    torch.set_num_threads(1);torch.manual_seed(s.seed)
    if s.arm not in ('fixed','centers','local'):raise ValueError('unknown learner')
    with threadpool_limits(1):
        c,assign=center_init(q.astype(np.float32)/maximum,np.asarray(y),s.prototypes,s.seed)
        model=Model(c,len(labels),assign,maximum,s);X=torch.from_numpy(q.astype(np.float32)/maximum);Y=torch.from_numpy(yi)
        optimizer=torch.optim.Adam([{'params':[model.head,model.bias],'lr':s.lr*3},{'params':[model.centers,model.metric,model.log_gamma],'lr':s.lr}],eps=1e-8)
        rng=np.random.default_rng(s.seed);trace=[];averages=None;average_count=0
        for epoch in range(s.epochs):
            model.train();order=rng.permutation(len(q));total=0.
            factor=.2 if epoch>=s.epochs*.8 else .5 if epoch>=s.epochs*.6 else 1.
            optimizer.param_groups[0]['lr']=s.lr*3*factor;optimizer.param_groups[1]['lr']=s.lr*factor
            for i in range(0,len(order),s.batch):
                ids=order[i:i+s.batch];optimizer.zero_grad(set_to_none=True);logits,w=model(X[ids])
                loss=nn.functional.cross_entropy(logits,Y[ids])+.5*s.reg*model.head.square().sum()+s.metric_reg*(w-1).square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
                with torch.no_grad():model.centers.clamp_(0,1);model.metric.clamp_(-6,6);model.log_gamma.clamp_(-2,7)
                total+=float(loss.detach())*len(ids)
            if s.average_tail and epoch>=int(.8*s.epochs):
                state={k:v.detach().clone() for k,v in model.state_dict().items()}
                if averages is None:averages=state
                else:
                    for k,v in state.items():averages[k]+=v
                average_count+=1
            if epoch==s.epochs-1 or (epoch+1)%20==0:
                record={'epoch':epoch+1,'objective':total/len(q)}
                if validation is not None:
                    xv,yv=validation
                    with torch.no_grad():lp,_=model(torch.from_numpy(xv.astype(np.float32)/maximum));pred=labels[lp.argmax(1).numpy()]
                    record.update(correct=int(np.sum(pred==yv)),rows=len(yv))
                trace.append(record)
        if averages is not None:model.load_state_dict({k:v/average_count for k,v in averages.items()})
        arrays=model.arrays();head_record=None
        if final_head:arrays,head_record=refit_head(q,yi,arrays,maximum,s)
        arrays['classes']=labels
        report={'settings':dataclasses.asdict(s),'trace':trace,'head_refit':head_record,'cpu_seconds':time.process_time()-start,'wall_seconds':time.perf_counter()-wall,'rows':len(q),'features':q.shape[1],'parameters':int(sum(a.size for a in arrays.values())),'fitting_labels_only':True}
        if validation is not None:
            xv,yv=validation;score=features(xv,arrays,maximum,s)@arrays['head']+arrays['bias'];pred=labels[score.argmax(1)]
            report['validation']={'correct':int(np.sum(pred==yv)),'rows':len(yv)}
        return arrays,report
