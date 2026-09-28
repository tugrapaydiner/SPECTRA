"""Logged development-only probe; no final partitions are read."""
from pathlib import Path
import argparse,json,time,pickle,sys,warnings
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import nnls
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


def radial(X,Y,gammas,weights,p=2):
    D=cdist(X,Y,'sqeuclidean' if p==2 else 'cityblock')
    K=np.zeros_like(D)
    for g,w in zip(gammas,weights):
        if w: K += w*np.exp(-g*D)
    return K


def align_weights(X,y,gammas,p):
    D=cdist(X,X,'sqeuclidean' if p==2 else 'cityblock')
    Ks=[]
    for g in gammas:
        K=np.exp(-g*D);K-=K.mean(0)[None,:];K-=K.mean(1)[:,None];Ks.append(K)
    T=(y[:,None]==y[None,:]).astype(float);T-=T.mean(0)[None,:];T-=T.mean(1)[:,None]
    gram=np.array([[np.sum(a*b) for b in Ks] for a in Ks]);target=np.array([np.sum(K*T) for K in Ks])
    ridge=1e-6*np.trace(gram)/len(gram)
    L=np.linalg.cholesky(gram+np.eye(len(gram))*ridge)
    z=np.linalg.solve(L,target)
    beta=nnls(L.T,z)[0]
    if beta.sum()==0: beta[:]=1
    return beta/beta.sum()


def fit_ovo(X,y,V,gammas,weights,C,p):
    classes=np.unique(y);votes=np.zeros((len(V),len(classes)),np.int32);pairs=[]
    for i,ci in enumerate(classes):
      for j in range(i+1,len(classes)):
        cj=classes[j];idx=np.flatnonzero((y==ci)|(y==cj));xx=X[idx];yy=(y[idx]==cj).astype(int)
        model=SVC(C=C,kernel='precomputed',tol=1e-3,cache_size=64).fit(radial(xx,xx,gammas,weights,p),yy)
        sv=idx[model.support_];coef=model.dual_coef_[0].copy();bias=float(model.intercept_[0]);
        val=radial(V,X[sv],gammas,weights,p)@coef+bias
        # Positive binary score votes for class j; multiclass ties go to j in libsvm.
        winner=np.where(val>=0,j,i);votes[np.arange(len(V)),winner]+=1
        pairs.append((i,j,sv,coef,bias))
    return classes[votes.argmax(1)],pairs


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    for task in a.tasks.split(','):
      with np.load(a.data/task/'development.npz') as d:
        raw=d['x'].reshape(len(d['x']),-1);y=d['y'];fi=d['fit'];vi=d['val']
      D0=15 if task=='letter' else 100;D=16 if task=='letter' else 128
      X=np.rint(raw*D0).astype(np.float64)/D;Tx=X[fi];Ty=y[fi];V=X[vi];Vy=y[vi]
      rng=np.random.default_rng(20260928);subset=rng.choice(len(Tx),min(1536,len(Tx)),False)
      records=[]
      for power in (2,1):
        dd=cdist(Tx[subset],Tx[subset], 'sqeuclidean' if power==2 else 'cityblock')
        med=np.median(dd[np.triu_indices(len(subset),1)]);gammas=np.array([.25,1,4,16])/med
        learned=align_weights(Tx[subset],Ty[subset],gammas,power)
        print(task,'p',power,'gammas',gammas,'alignment',learned,flush=True)
        for family in ('single','uniform','alignment'):
          ws=list(np.eye(4)) if family=='single' else [np.ones(4)/4] if family=='uniform' else [learned]
          for wi,w in enumerate(ws):
            for C in (1.,10.,100.):
              start=time.perf_counter();cpu=time.process_time()
              # One-v-one training keeps full-data Gram matrices bounded.
              if family=='single' and power==2:
                model=SVC(C=C,gamma=gammas[wi],tol=1e-3,cache_size=128).fit(Tx,Ty);pred=model.predict(V);pairs=None;nsv=len(model.support_)
              else:
                pred,pairs=fit_ovo(Tx,Ty,V,gammas,w,C,power);model=None;nsv=len(np.unique(np.concatenate([p[2] for p in pairs])))
              rec={'task':task,'power':power,'family':family,'wi':wi,'C':C,'gammas':gammas.tolist(),'weights':w.tolist(),'correct':int(np.sum(pred==Vy)),'n':len(Vy),'nsv':nsv,'cpu':time.process_time()-cpu,'wall':time.perf_counter()-start}
              records.append(rec);tag=f'{task}-{power}-{family}-{wi}-{int(C)}'
              (a.out/(tag+'.pkl')).write_bytes(pickle.dumps({'model':model,'pairs':pairs,'recipe':rec,'train_x':Tx,'train_y':Ty,'fit_indices':fi,'val_indices':vi,'predictions':pred},protocol=4))
              with (a.out/'rows.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
              print(tag,rec['correct'],len(Vy),'sv',nsv,'sec',round(rec['wall'],2),flush=True)
      (a.out/(task+'-summary.json')).write_text(json.dumps(records,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tasks',default='letter,pendigits');a=p.parse_args()
 with threadpool_limits(1):run(a)
