"""Matched training-only selection and fixed full-training refits.

Disk-backed integer-signature Gram matrices avoid an extra in-memory quadratic
copy. This remains ordinary quadratic-cost LIBSVM training, not a scalable claim.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json,os,pickle,struct,sys,time,zlib
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.learned_metric.learning import fit_weights,kernel_table,kernel_matrix,split_development
SEEDS=(611,977,1543);ARMS=('uniform','variance','nca')
GRID=tuple(itertools.product((1.,10.,100.),(.5,2.,8.)))

def write(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def source_hashes():
 return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}

def encode_model(model,training_codes,weights,maximum,coefficient):
 """Distinct format prevents a weighted model from being used as ordinary RBF."""
 support=np.asarray(training_codes[model.support_],dtype=np.float64)
 classes=len(model.classes_);d=support.shape[1];n=len(support)
 meta=json.dumps({'labels':model.classes_.tolist()},separators=(',',':')).encode()
 payload=struct.pack('<d',coefficient)+np.asarray(model.n_support_,dtype='<u4').tobytes()
 payload+=b''.join(np.asarray(a,dtype='<f8',order='C').tobytes() for a in (support,model.dual_coef_,model.intercept_))
 body=meta+payload
 inner=struct.pack('<8sIIIIII',b'SPCSVM02',classes,n,d,len(meta),len(payload),zlib.crc32(body))+body
 outer=np.asarray(weights,dtype='<u2').tobytes()+inner
 return struct.pack('<8sIIIII',b'SPLMET01',maximum,d,int(np.sum(weights)),len(inner),zlib.crc32(outer))+outer

def selection(args):
 args.out.mkdir(parents=True,exist_ok=False)
 write(args.out/'protocol.json',{'seeds':SEEDS,'arms':ARMS,'grid':GRID,'source_sha256':source_hashes(),
  'data_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
  'scope':'training/validation only; no evaluation reads','numerical_threads':1})
 t=time.process_time();wall=time.perf_counter();records=[]
 with threadpool_limits(1),(args.out/'selection.jsonl').open('x') as log:
  for task in ('letter','pendigits'):
   data=np.load(args.data/(task+'-train.npz'));q=data['q'];y=data['y'];D=int(data['maximum'])
   for seed in (SEEDS[:1] if args.pilot else SEEDS):
    fit,val=split_development(q,y,seed)
    np.savez_compressed(args.out/f'{task}-{seed}-split.npz',fit=fit,validation=val)
    qt,yt,qv,yv=q[fit],y[fit],q[val],y[val]
    for arm in ARMS:
     w,metric=fit_weights(qt,yt,D,method=arm,seed=seed)
     write(args.out/f'{task}-{seed}-{arm}-metric.json',metric)
     print(task,seed,arm,'metric',w.tolist(),flush=True)
     for gamma in ((2.,) if args.pilot else (.5,2.,8.)):
      table,coefficient=kernel_table(w,D,gamma);path=args.out/'scratch-gram.f64'
      start=time.process_time();gram=kernel_matrix(qt,qt,w,table,path=path)
      valgram=kernel_matrix(qv,qt,w,table);matrix_sec=time.process_time()-start
      for C in ((10.,) if args.pilot else (1.,10.,100.)):
       start=time.process_time();begin=time.perf_counter()
       model=SVC(C=C,kernel='precomputed',tol=1e-3,shrinking=True,cache_size=128).fit(gram,yt)
       seconds=time.process_time()-start;wallfit=time.perf_counter()-begin
       start=time.process_time();pred=model.predict(valgram);predsec=time.process_time()-start
       record={'task':task,'seed':seed,'arm':arm,'C':C,'gamma':gamma,'fit_rows':len(fit),'validation_rows':len(val),
        'correct':int(np.sum(pred==yv)),'accuracy':float(np.mean(pred==yv)), 'support_vectors':len(model.support_),
        'kernel_cpu_seconds':matrix_sec,'fit_cpu_seconds':seconds,'fit_wall_seconds':wallfit,'predict_cpu_seconds':predsec}
       records.append(record);log.write(json.dumps(record)+'\n');log.flush()
       tag=f'{task}-{seed}-{arm}-C{C:g}-g{gamma:g}'
       np.savez_compressed(args.out/(tag+'-predictions.npz'),prediction=pred,expected=yv)
       print(tag,record['correct'],'/',len(yv),'sv',len(model.support_),'fit',round(seconds,3),flush=True)
      del model,valgram,gram;path.unlink()
 write(args.out/'cost.json',{'cpu_seconds':time.process_time()-t,'wall_seconds':time.perf_counter()-wall,'fits':len(records)})
 if not args.pilot:
  chosen={}
  for task in ('letter','pendigits'):
   chosen[task]={}
   for arm in ARMS:
    options=[]
    for order,(C,gamma) in enumerate(GRID):
     rr=[r for r in records if r['task']==task and r['arm']==arm and r['C']==C and r['gamma']==gamma]
     if len(rr)!=3:raise ValueError('missing candidate')
     score=sum(r['correct'] for r in rr)/sum(r['validation_rows'] for r in rr)
     options.append((score,-sum(r['support_vectors'] for r in rr),-order,C,gamma))
    best=max(options);chosen[task][arm]={'C':best[3],'gamma':best[4],'validation_accuracy':best[0]}
  write(args.out/'selected.json',chosen)

def refit(args):
 from sklearn.neural_network import MLPClassifier
 from sklearn.svm import LinearSVC
 args.out.mkdir(parents=True,exist_ok=False)
 selected=json.loads((args.selection/'selected.json').read_text())
 write(args.out/'refit-lock.json',{'selection_sha256':hashlib.sha256((args.selection/'selected.json').read_bytes()).hexdigest(),
  'source':source_hashes(),'no_evaluation_read':True})
 t=time.process_time();wall=time.perf_counter();fits=[]
 with threadpool_limits(1):
  for task in ('letter','pendigits'):
   data=np.load(args.data/(task+'-train.npz'));q=data['q'];y=data['y'];D=int(data['maximum'])
   for arm in ARMS:
    folder=args.out/(task+'-'+arm);folder.mkdir()
    w,metric=fit_weights(q,y,D,method=arm,seed=20260928)
    choice=selected[task][arm];table,coefficient=kernel_table(w,D,choice['gamma'])
    write(folder/'metric.json',metric)
    path=args.out/'refit-gram.f64';start=time.process_time();gram=kernel_matrix(q,q,w,table,path=path);ksec=time.process_time()-start
    print(task,arm,'kernel ready',gram.shape,flush=True)
    begin=time.perf_counter();start=time.process_time()
    model=SVC(C=choice['C'],kernel='precomputed',tol=1e-3,shrinking=True,cache_size=128).fit(gram,y)
    fsec=time.process_time()-start;raw=encode_model(model,q,w,D,coefficient)
    (folder/'model.sgm').write_bytes(raw);(folder/'sklearn.pkl').write_bytes(pickle.dumps(model,protocol=4))
    np.savez_compressed(folder/'reference.npz',support_codes=q[model.support_],support_indices=model.support_,
      coefficients=model.dual_coef_,intercepts=model.intercept_,counts=model.n_support_,classes=model.classes_,
      weights=w,table=table,maximum=D,gamma=coefficient)
    record={'task':task,'arm':arm,**choice,'training_rows':len(y),'support_vectors':len(model.support_),
      'learned_weight_entries':len(w),'weight_mass':int(w.sum()),'table_entries':len(table),'kernel_cpu_seconds':ksec,
      'fit_cpu_seconds':fsec,'fit_wall_seconds':time.perf_counter()-begin,'metric_cpu_seconds':metric['cpu_seconds'],
      'model_bytes':len(raw),'model_sha256':hashlib.sha256(raw).hexdigest()}
    write(folder/'fit.json',record);fits.append(record)
    print('final',task,arm,'sv',len(model.support_),'fit',fsec,flush=True)
    del model,gram;path.unlink()
   x=np.ascontiguousarray(q/D,dtype=np.float64)
   controls=[('linear',LinearSVC(C=10.,max_iter=5000,random_state=20260928,dual='auto')),
    ('mlp',MLPClassifier(hidden_layer_sizes=(128,128),activation='relu',solver='adam',alpha=1e-4,
       batch_size=128,learning_rate_init=.002,max_iter=240,early_stopping=True,n_iter_no_change=20,
       validation_fraction=.1,random_state=20260928))]
   for arm,model in controls:
    folder=args.out/(task+'-'+arm);folder.mkdir();start=time.process_time();model.fit(x,y)
    fsec=time.process_time()-start;(folder/'model.pkl').write_bytes(pickle.dumps(model,protocol=4))
    if arm=='mlp':np.savez_compressed(folder/'weights.npz',**{f'w{i}':v for i,v in enumerate(model.coefs_)},**{f'b{i}':v for i,v in enumerate(model.intercepts_)},classes=model.classes_,D=D)
    else:np.savez_compressed(folder/'weights.npz',w=model.coef_,b=model.intercept_,classes=model.classes_,D=D)
    record={'task':task,'arm':arm,'fit_cpu_seconds':fsec,'training_rows':len(y),
      'parameters':int(sum(v.size for v in model.coefs_)+sum(v.size for v in model.intercepts_)) if arm=='mlp' else int(model.coef_.size+model.intercept_.size)}
    write(folder/'fit.json',record);fits.append(record);print('control',record,flush=True)
 files={p.relative_to(args.out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.rglob('*')) if p.is_file()}
 write(args.out/'FINAL_LOCK.json',{'files':files,'source':source_hashes(),'cpu_seconds':time.process_time()-t,
   'wall_seconds':time.perf_counter()-wall,'fits':fits,'scope':'all final models frozen before retained evaluation'})

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['selection','refit'])
 p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--selection',type=Path);p.add_argument('--pilot',action='store_true');a=p.parse_args()
 (selection if a.operation=='selection' else refit)(a)
