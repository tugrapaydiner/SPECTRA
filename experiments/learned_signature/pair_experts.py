"""Nested training-only pair-specific kernel choice with a shared finite table bank."""
from __future__ import annotations
import argparse,json,time,sys,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from experiments.learned_signature.learning import *
from experiments.learned_signature.model import pack_model


def fit_pair_experts(q,y,cap,seed,out,valid_q=None,valid_y=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 cpu_start,wall_start=time.process_time(),time.perf_counter()
 labels=np.unique(y);c=len(labels);d=q.shape[1];weights=np.ones(d,dtype=np.uint32)
 gamma=base_gamma(q,cap)
 tables=np.stack([table_profile(cap,weights,gamma*m) for m in MULTIPLIERS])
 inner_f,inner_v=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed+31)
 # The common default uses a bounded global inner fit, never any outer validation.
 gf=inner_f
 if len(gf)>5000:
  gf,_=train_test_split(gf,train_size=5000,stratify=y[gf],random_state=seed+32)
 global_sig=signatures(q[gf],q[gf],weights);global_vsig=signatures(q[inner_v],q[gf],weights)
 global_rows=[];fits=0;fit_cpu=0.
 with (out/'inner_trials.jsonl').open('x') as log:
  for gi,m in enumerate(MULTIPLIERS):
   K=tables[gi,global_sig];V=tables[gi,global_vsig]
   for ci,C in enumerate(C_VALUES):
    start=time.process_time();svc=SVC(C=C,kernel='precomputed',cache_size=128).fit(K,y[gf]);cost=time.process_time()-start
    pred=svc.predict(V);correct=int(np.sum(pred==y[inner_v]));fits+=1;fit_cpu+=cost
    row={'pair':None,'C':C,'profile':gi,'correct':correct,'rows':len(inner_v),'supports':len(svc.support_),'fit_cpu':cost,'index':gi*3+ci}
    global_rows.append(row);log.write(json.dumps(row)+'\n')
   del K,V
  default=max(global_rows,key=lambda r:(r['correct'],-r['supports'],-r['index']))
  del global_sig,global_vsig
  configured=[];pair_models=[];decisions=[]
  for i in range(c):
   for j in range(i+1,c):
    ix=np.flatnonzero((y==labels[i])|(y==labels[j]))
    fi=inner_f[(y[inner_f]==labels[i])|(y[inner_f]==labels[j])]
    vi=inner_v[(y[inner_v]==labels[i])|(y[inner_v]==labels[j])]
    sig=signatures(q[fi],q[fi],weights);vsig=signatures(q[vi],q[fi],weights);trials=[]
    for gi in range(len(MULTIPLIERS)):
     K=tables[gi,sig];V=tables[gi,vsig]
     for ci,C in enumerate(C_VALUES):
      start=time.process_time();svc=SVC(C=C,kernel='precomputed',cache_size=32).fit(K,y[fi]);cost=time.process_time()-start
      correct=int(np.sum(svc.predict(V)==y[vi]));fits+=1;fit_cpu+=cost
      row={'pair':[int(labels[i]),int(labels[j])],'C':C,'profile':gi,'correct':correct,'rows':len(vi),'supports':len(svc.support_),'fit_cpu':cost,'index':gi*3+ci}
      trials.append(row);log.write(json.dumps(row)+'\n')
    old=next(r for r in trials if r['C']==default['C'] and r['profile']==default['profile'])
    best=max(trials,key=lambda r:(r['correct'],-r['supports'],-r['index']))
    chosen=best if best['correct']>=old['correct']+2 else old
    gi=chosen['profile'];K=tables[gi,signatures(q[ix],q[ix],weights)]
    start=time.process_time();svc=SVC(C=chosen['C'],kernel='precomputed',cache_size=64).fit(K,y[ix]);cost=time.process_time()-start;fits+=1;fit_cpu+=cost
    global_support=ix[svc.support_]
    # Binary SVC scores are positive for the second label; multiclass vote oracle
    # stores positive-for-first signs. Negate both coefficients and bias together.
    sign=1. if c==2 else -1.
    pair_models.append({'support_indices':global_support,'coef':sign*svc.dual_coef_[0],
                        'bias':sign*svc.intercept_[0],'profile':gi})
    configured.append({'pair':[int(labels[i]),int(labels[j])],'C':chosen['C'],'profile':gi,
                       'changed':chosen['C']!=default['C'] or gi!=default['profile'],
                       'inner_base_correct':old['correct'],'inner_selected_correct':chosen['correct'],
                       'inner_rows':len(vi),'supports':len(svc.support_),'refit_cpu':cost})
    if valid_q is not None:
     V=tables[gi,signatures(valid_q,q[ix],weights)]
     decisions.append(sign*svc.decision_function(V))
    if len(pair_models)%20==0:print('pairs',len(pair_models),'/',c*(c-1)//2,flush=True)
    del svc,K,sig,vsig
  log.flush()
 unique=np.unique(np.concatenate([p['support_indices'] for p in pair_models]))
 unique=unique[np.lexsort((unique,y[unique]))]
 mapping=np.full(len(q),-1,dtype=np.int32);mapping[unique]=np.arange(len(unique))
 counts=[int(np.sum(y[unique]==label)) for label in labels]
 pairs=[{'ids':mapping[p['support_indices']],'coef':p['coef'],'bias':p['bias'],'profile':p['profile']} for p in pair_models]
 metadata={'task':out.name,'family':'pair_experts','domain_max':cap,'gamma_base':gamma,
           'scales':list(MULTIPLIERS),'default_C':default['C'],'default_profile':default['profile'],
           'seed':seed,'definition':'stored finite radial table, integer squared-distance signature'}
 info=pack_model(out/'model.lkt',codes=q[unique],classes=labels,counts=counts,weights=weights,table=tables,pairs=pairs,metadata=metadata)
 report={'default':default,'pair_configuration':configured,'fits':fits,'fit_cpu':fit_cpu,
         'cpu_seconds':time.process_time()-cpu_start,'wall_seconds':time.perf_counter()-wall_start,
         'changed_pairs':sum(p['changed'] for p in configured),'model':info}
 if valid_q is not None:
  margins=np.column_stack(decisions)
  votes=np.zeros((len(valid_q),c),dtype=np.int32);col=0
  if c==2:pred=labels[(margins[:,0]>=0).astype(int)]
  else:
   for i in range(c):
    for j in range(i+1,c):votes[:,i]+=(margins[:,col]>0);votes[:,j]+=(margins[:,col]<=0);col+=1
   pred=labels[votes.argmax(axis=1)]
  np.savez(out/'validation.npz',pred=pred,labels=valid_y,margins=margins)
  report.update(correct=int(np.sum(pred==valid_y)),rows=len(valid_y),accuracy=float(np.mean(pred==valid_y)))
 write_json(out/'summary.json',report);return report

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--task',choices=['letter','pendigits'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--final-fit',action='store_true');a=p.parse_args()
 with threadpool_limits(1):
  q,y,cap=load_part(a.data,a.task)
  if a.final_fit:r=fit_pair_experts(q,y,cap,a.seed,a.out)
  else:
   f,v=split_indices(y,a.seed);r=fit_pair_experts(q[f],y[f],cap,a.seed,a.out,q[v],y[v])
  print(json.dumps({k:v for k,v in r.items() if k!='pair_configuration'},indent=2),flush=True)
