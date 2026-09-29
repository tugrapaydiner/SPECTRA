"""Descriptive final evaluation of already-frozen models; no fitting or selection."""
from __future__ import annotations
import argparse,datetime,hashlib,json,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits
from experiments.learned_signature.learning import load_part,signatures,sha,write_json
from experiments.learned_signature.native import Session
from experiments.learned_signature.dense import DenseSession
from experiments.learned_signature.oracle import OrderedOracle
FAMILIES=('rbf','metric','alignment','uniform','metric_alignment','pair_experts','margin','linear','mlp')

def paired(candidate,baseline,y):
 d=(candidate==y).astype(int)-(baseline==y).astype(int);better=int(np.sum(d==1));worse=int(np.sum(d==-1));ties=len(d)-better-worse
 draws=np.random.default_rng(20260928).multinomial(len(d),[better/len(d),worse/len(d),ties/len(d)],size=20000)
 diff=(draws[:,0]-draws[:,1])/len(d)*100
 p=binomtest(better,better+worse,.5).pvalue if better+worse else 1.
 return {'gain_pp':100*float(d.mean()),'corrected':better,'new_errors':worse,'paired_bootstrap_95_pp':np.quantile(diff,[.025,.975]).tolist(),'paired_exact_p':p,'bonferroni_12_p':min(1.,12*p),'scope':'fixed-model row bootstrap and paired test, historically consumed data; not seed/writer/population uncertainty'}

def run(a):
 a.out.mkdir(parents=True,exist_ok=False);frozen=json.loads(a.freeze.read_text())
 for name,info in frozen['files'].items():
  path=a.models/name
  if path.stat().st_size!=info['bytes'] or sha(path)!=info['sha256']:raise ValueError('changed frozen model '+name)
 write_json(a.out/'started.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha(a.freeze),'selection_sha256':sha(a.selection),'script_sha256':sha(__file__),'library_sha256':sha(a.library),'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'scope':'first final evaluation in this continuation; test partitions were consumed historically'})
 summary={};full_pairs=full_margins=0
 for task in ('letter','pendigits'):
  train,train_y,cap=load_part(a.data,task);q,y,cap2=load_part(a.data,task,test=True);assert cap==cap2
  train_vectors={bytes(r) for r in train};nonoverlap=np.array([bytes(r) not in train_vectors for r in q],dtype=bool)
  np.savez(a.out/f'{task}-cases.npz',q=q,y=y,nonoverlap=nonoverlap)
  results={};predictions={};dataout={}
  old=np.fromfile(a.legacy/task/'X.f64',dtype='<f8').reshape(-1,q.shape[1]);oldpred=np.asarray(json.loads((a.legacy/task/'expected.json').read_text()))
  if old.shape!=q.shape or not np.array_equal(np.rint(old*cap).astype(np.uint8),q):raise ValueError('legacy case alignment differs')
  if oldpred.shape!=y.shape:raise ValueError('legacy labels mismatch')
  for family in FAMILIES:
   folder=a.models/task/family;cpu=time.process_time();start=time.perf_counter()
   if family in ('linear','mlp'):
    model=pickle.loads((folder/'model.pkl').read_bytes());expected=model.predict(q.astype(np.float64)/cap)
    with DenseSession(folder/'control.json',a.library) as session:pred=np.asarray(session.predict_buffer(q));info={'parameters':session.parameter_count,'payload_bytes':session.parameter_count*8,'serialized_bytes':(folder/'control.json').stat().st_size}
    margins_checked=0;maxgap=None
   else:
    with Session(folder/'model.lkt',a.library) as session:
     pred=np.asarray(session.predict_buffer(q));scalar=np.asarray(session.predict_buffer(q,mode='scalar'));exhaustive=np.asarray(session.predict_buffer(q,mode='exhaustive'))
     if not np.array_equal(pred,scalar) or not np.array_equal(pred,exhaustive):raise ValueError('native path mismatch')
     native_margins=np.frombuffer(session.probe(q),dtype='<f8').reshape(len(q),-1).copy();info=session.info
    oracle=OrderedOracle(folder/'model.lkt');om=oracle.margins(q)
    if om.tobytes()!=native_margins.tobytes():raise ValueError('ordered margin mismatch '+task+'/'+family)
    expected=oracle.predictions(om);margins_checked=om.size;maxgap=0.
    if family!='pair_experts':
     model=pickle.loads((folder/'model.pkl').read_bytes());table=np.load(folder/'table.npy');weights=np.load(folder/'weights.npy');classes=model.classes_;model.decision_function_shape='ovo';sk=[];gap=0.
     for first in range(0,len(q),128):
      K=table[signatures(q[first:first+128],train,weights)];sk.extend(model.predict(K).tolist());m=model.decision_function(K).reshape(len(K),-1);gap=max(gap,float(np.max(np.abs(m-native_margins[first:first+len(K)]))))
     if not np.array_equal(sk,pred):raise ValueError('sklearn label mismatch '+task+'/'+family)
     maxgap=gap
    np.save(a.out/f'{task}-{family}-margins.npy',native_margins);full_margins+=margins_checked
   if not np.array_equal(pred,expected):raise ValueError('reference label mismatch '+task+'/'+family)
   full_pairs+=len(q);predictions[family]=pred;np.save(a.out/f'{task}-{family}-pred.npy',pred)
   r={'correct':int(np.sum(pred==y)),'rows':len(y),'accuracy':float(np.mean(pred==y)),'nonoverlap_correct':int(np.sum((pred==y)&nonoverlap)),'nonoverlap_rows':int(nonoverlap.sum()),'per_class':{str(label):{'rows':int(np.sum(y==label)),'correct':int(np.sum((pred==y)&(y==label)))} for label in np.unique(y)},'model':info,'margin_values_checked':int(margins_checked),'max_sklearn_margin_difference':maxgap,'verification_cpu':time.process_time()-cpu,'verification_wall':time.perf_counter()-start}
   results[family]=r;print(task,family,r['correct'],r['rows'],r['accuracy'],r['model'],flush=True)
  base=predictions['rbf']
  for family,pred in predictions.items():results[family]['versus_tuned_rbf']=paired(pred,base,y);results[family]['versus_legacy']=paired(pred,oldpred,y)
  results['legacy']={'correct':int(np.sum(oldpred==y)),'rows':len(y),'accuracy':float(np.mean(oldpred==y)),'source_model_sha256':sha(a.legacy/task/'model.srt'),'expected_sha256':sha(a.legacy/task/'expected.json'),'scope':'previous frozen SVM; different fitting budget/configuration from all new models'}
  summary[task]={'families':results,'train_rows':len(train),'train_test_feature_overlap_rows':int((~nonoverlap).sum())}
 selection=json.loads(a.selection.read_text())
 report={'format':'spectra.learned-quality.evaluation.v1','tasks':summary,'matched_new_model_input_pairs':full_pairs,'bitwise_margin_values':int(full_margins),'original_primary_validation_gate':selection['primary_validation_gate'],'primary_promotion':False,'scope':'descriptive task accuracy, not general intelligence or independent confirmation; validation gate cannot be overridden by test results'}
 write_json(a.out/'report.json',report);return report

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('data','models','freeze','selection','library','legacy','out'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args()
 with threadpool_limits(1):run(a)
