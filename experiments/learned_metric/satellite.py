"""Predeclared transfer of the fixed integer-metric learner to UCI Satellite.

Official test is never used here: three training-only validation splits, then one
full-training refit per selected arm. This does not replace the official split
with cross-validation or establish spatial independence.
"""
from __future__ import annotations
import argparse,hashlib,io,json,pickle,struct,sys,time,warnings,zipfile
from pathlib import Path
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.learned_metric.learning import fit_weights,kernel_table,kernel_matrix,split_development
from experiments.learned_metric.train import write,encode_model
ARMS=('uniform','variance','nca');SEEDS=(611,977,1543);GRID=tuple((c,g) for c in (1.,10.,100.) for g in (2.,8.,32.))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(archive,out):
 out.mkdir(parents=True,exist_ok=False)
 entries={}
 with zipfile.ZipFile(archive) as z:
  for part,name in (('train','sat.trn'),('test','sat.tst')):
   raw=z.read(name);matrix=np.loadtxt(io.BytesIO(raw),dtype=np.int64)
   q=matrix[:,:-1];y=matrix[:,-1]
   if q.shape[1]!=36 or q.min()<0 or q.max()>255 or len(q)!=(4435 if part=='train' else 2000):raise ValueError('wrong Satellite inventory')
   np.savez_compressed(out/(part+'.npz'),q=q.astype(np.uint8),y=y,maximum=255)
   entries[part]={'rows':len(q),'classes':sorted(set(y.tolist())),'raw_sha256':hashlib.sha256(raw).hexdigest(),'npz_sha256':sha(out/(part+'.npz'))}
  (out/'sat.doc').write_bytes(z.read('sat.doc'))
 write(out/'manifest.json',{'archive_sha256':sha(archive),'official_protocol':'original sat.trn/sat.tst; no redistribution into official cross-validation folds','files':entries,
  'source':'https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite','license':'CC BY4.0','author':'Ashwin Srinivasan (1993)'})

def weights(q,y,arm,seed):
 if arm=='uniform':return np.ones(q.shape[1],dtype=np.uint16),{'method':'uniform','integer_weights':[1]*q.shape[1],'cpu_seconds':0.}
 _,record=fit_weights(q,y,255,method=arm,seed=seed)
 relaxed=np.asarray(record['continuous_weights']);budget=64-len(relaxed)
 scaled=budget*relaxed/relaxed.sum();extra=np.floor(scaled).astype(int)
 order=np.lexsort((np.arange(len(extra)),-(scaled-extra)));extra[order[:budget-extra.sum()]]+=1
 integer=(1+extra).astype(np.uint16)
 assert integer.sum()==64 and integer.min()>=1
 record['initial_units2_integer_weights']=record['integer_weights'];record['integer_weights']=integer.tolist();record['satellite_projection']='one per dimension plus28 largest-remainder units proportional to the relaxed weights'
 return integer,record

def select(data,out):
 out.mkdir(parents=True,exist_ok=False);d=np.load(data/'train.npz');q=d['q'];y=d['y'];t=time.process_time();start=time.perf_counter();records=[]
 write(out/'protocol.json',{'arms':ARMS,'seeds':SEEDS,'grid':GRID,'data_manifest_sha256':sha(data/'manifest.json'),'script_sha256':sha(__file__),'learning_sha256':sha(Path(__file__).with_name('learning.py')),'scope':'only original training rows; do not infer spatial independence'})
 with threadpool_limits(1),(out/'selection.jsonl').open('x') as log:
  for seed in SEEDS:
   fit,val=split_development(q,y,seed);np.savez_compressed(out/f'split-{seed}.npz',fit=fit,validation=val)
   for arm in ARMS:
    w,metric=weights(q[fit],y[fit],arm,seed);write(out/f'{seed}-{arm}-metric.json',metric)
    print(seed,arm,'weights',w.tolist(),flush=True)
    for gamma in (2.,8.,32.):
     table,coefficient=kernel_table(w,255,gamma);path=out/'scratch.f64';begin=time.process_time()
     gram=kernel_matrix(q[fit],q[fit],w,table,path=path);vgram=kernel_matrix(q[val],q[fit],w,table);ksec=time.process_time()-begin
     for C in (1.,10.,100.):
      begin=time.process_time();model=SVC(C=C,kernel='precomputed',tol=1e-3,cache_size=128).fit(gram,y[fit]);fsec=time.process_time()-begin
      pred=model.predict(vgram);r={'seed':seed,'arm':arm,'C':C,'gamma':gamma,'correct':int(np.sum(pred==y[val])),'validation_rows':len(val),'fit_rows':len(fit),'support_vectors':len(model.support_),'fit_cpu_seconds':fsec,'kernel_cpu_seconds':ksec}
      records.append(r);log.write(json.dumps(r)+'\n');log.flush();np.savez_compressed(out/f'{seed}-{arm}-C{C:g}-g{gamma:g}.npz',prediction=pred,expected=y[val]);print(r,flush=True)
     del model,gram,vgram,table;path.unlink()
 chosen={}
 for arm in ARMS:
  options=[]
  for index,(C,g) in enumerate(GRID):
   rr=[r for r in records if r['arm']==arm and r['C']==C and r['gamma']==g]
   assert len(rr)==3
   options.append((sum(r['correct'] for r in rr)/sum(r['validation_rows'] for r in rr),-sum(r['support_vectors'] for r in rr),-index,C,g))
  best=max(options);chosen[arm]={'C':best[3],'gamma':best[4],'validation_accuracy':best[0]}
 write(out/'selected.json',chosen);write(out/'cost.json',{'cpu_seconds':time.process_time()-t,'wall_seconds':time.perf_counter()-start,'fits':len(records)})

def fit(data,selection,out):
 out.mkdir(parents=True,exist_ok=False);chosen=json.loads((selection/'selected.json').read_text());t=time.process_time();start=time.perf_counter();records=[]
 train=np.load(data/'train.npz');q=train['q'];y=train['y']
 write(out/'pre-fit-lock.json',{'selection_sha256':sha(selection/'selected.json'),'training_sha256':sha(data/'train.npz'),'script_sha256':sha(__file__),'no_test_predictions':True})
 with threadpool_limits(1):
  for arm in ARMS:
   folder=out/arm;folder.mkdir();w,metric=weights(q,y,arm,20260928);write(folder/'metric.json',metric)
   table,coefficient=kernel_table(w,255,chosen[arm]['gamma']);path=out/'scratch.f64';begin=time.process_time();gram=kernel_matrix(q,q,w,table,path=path);ksec=time.process_time()-begin
   begin=time.process_time();model=SVC(C=chosen[arm]['C'],kernel='precomputed',tol=1e-3,cache_size=128).fit(gram,y);seconds=time.process_time()-begin
   raw=encode_model(model,q,w,255,coefficient);(folder/'model.sgm').write_bytes(raw);(folder/'sklearn.pkl').write_bytes(pickle.dumps(model,protocol=4))
   np.savez_compressed(folder/'reference.npz',support_codes=q[model.support_],support_indices=model.support_,coefficients=model.dual_coef_,intercepts=model.intercept_,counts=model.n_support_,classes=model.classes_,weights=w,table=table,maximum=255,gamma=coefficient)
   record={'arm':arm,**chosen[arm],'training_rows':len(y),'support_vectors':len(model.support_),'weight_mass':int(w.sum()),'table_entries':len(table),'model_bytes':len(raw),'model_sha256':sha(folder/'model.sgm'),'fit_cpu_seconds':seconds,'metric_cpu_seconds':metric['cpu_seconds'],'kernel_cpu_seconds':ksec}
   write(folder/'fit.json',record);records.append(record);print('final',record,flush=True)
   del model,table,gram;path.unlink()
  x=q.astype(float)/255.
  controls=[('linear',LinearSVC(C=10,dual='auto',max_iter=5000,random_state=20260928)),
   ('mlp',MLPClassifier((128,128),activation='relu',solver='adam',learning_rate_init=.002,batch_size=128,alpha=1e-4,max_iter=240,early_stopping=True,validation_fraction=.1,n_iter_no_change=20,random_state=20260928)),
   ('qda',QuadraticDiscriminantAnalysis(reg_param=.01))]
  for arm,model in controls:
   folder=out/arm;folder.mkdir();begin=time.process_time()
   with warnings.catch_warnings(record=True) as messages:
    warnings.simplefilter('always');model.fit(x,y)
   seconds=time.process_time()-begin;(folder/'model.pkl').write_bytes(pickle.dumps(model,protocol=4))
   if arm=='mlp':np.savez_compressed(folder/'weights.npz',**{f'w{i}':v for i,v in enumerate(model.coefs_)},**{f'b{i}':v for i,v in enumerate(model.intercepts_)},classes=model.classes_,D=255)
   elif arm=='linear':np.savez_compressed(folder/'weights.npz',w=model.coef_,b=model.intercept_,classes=model.classes_,D=255)
   else:np.savez_compressed(folder/'weights.npz',means=model.means_,rotations=np.asarray(model.rotations_),scalings=np.asarray(model.scalings_),priors=model.priors_,classes=model.classes_,D=255)
   record={'arm':arm,'fit_cpu_seconds':seconds,'training_rows':len(y),'warnings':[str(m.message) for m in messages]};write(folder/'fit.json',record);records.append(record);print('control',record,flush=True)
 files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
 write(out/'FINAL_LOCK.json',{'files':files,'fits':records,'script_sha256':sha(__file__),'learning_sha256':sha(Path(__file__).with_name('learning.py')),'cpu_seconds':time.process_time()-t,'wall_seconds':time.perf_counter()-start,'scope':'all selected satellite models frozen before any satellite test prediction'})

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['prepare','selection','refit']);p.add_argument('--archive',type=Path);p.add_argument('--data',type=Path);p.add_argument('--selection',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.operation=='prepare':prepare(a.archive,a.out)
 elif a.operation=='selection':select(a.data,a.out)
 else:fit(a.data,a.selection,a.out)
