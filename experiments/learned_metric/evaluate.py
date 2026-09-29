"""Final frozen-model evaluation; no selection, retraining or threshold fitting.

These official partitions were used in earlier SPECTRA research. 'Held out from
this fit/selection' is not fresh independent confirmation. All comparisons survive.
"""
from __future__ import annotations
import argparse,ctypes as C,hashlib,json,pickle,struct,subprocess,sys,time,zlib
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix,f1_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.learned_metric.session import MetricSession
from experiments.learned_metric.learning import kernel_matrix
from experiments.learned_metric.controls import ControlSession

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

def expanded_model(reference):
 d=reference['support_codes'].shape[1];w=reference['weights'].astype(int);sv=np.repeat(reference['support_codes'],w,axis=1).astype('<f8')
 labels=reference['classes'].tolist();nc=len(labels);meta=json.dumps({'labels':labels},separators=(',',':')).encode()
 payload=struct.pack('<d',float(reference['gamma']))+reference['counts'].astype('<u4').tobytes()
 payload+=sv.tobytes()+reference['coefficients'].astype('<f8').tobytes()+reference['intercepts'].astype('<f8').tobytes()
 body=meta+payload
 return struct.pack('<8sIIIIII',b'SPCSVM02',nc,len(sv),sv.shape[1],len(meta),len(payload),zlib.crc32(body))+body

class Reference:
 def __init__(self,raw,path,nc,d):
  self.lib=lib=C.CDLL(str(path));self.nc=nc;self.d=d
  lib.ref_create.argtypes=[C.c_void_p,C.c_uint64];lib.ref_create.restype=C.c_void_p
  lib.ref_probe.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_int,C.c_int,C.POINTER(C.c_double),C.c_uint64];lib.ref_probe.restype=C.c_int
  lib.ref_destroy.argtypes=[C.c_void_p];lib.ref_destroy.restype=None
  buf=C.create_string_buffer(raw,len(raw));self.h=lib.ref_create(buf,len(raw))
  if not self.h:raise ValueError('reference model load')
 def probe(self,x):
  x=np.ascontiguousarray(x,dtype=np.float64);out=np.empty((len(x),self.nc*(self.nc-1)//2),dtype=np.float64)
  rc=self.lib.ref_probe(self.h,x.ctypes.data_as(C.POINTER(C.c_double)),len(x),self.d,out.ctypes.data_as(C.POINTER(C.c_double)),out.size)
  if rc:raise ValueError('reference margin failure')
  return out
 def close(self):
  if self.h:self.lib.ref_destroy(self.h);self.h=None

def summarize(pred,y):
 return {'correct':int(np.sum(pred==y)),'rows':len(y),'accuracy':float(np.mean(pred==y)),
  'macro_f1':float(f1_score(y,pred,average='macro')),'confusion':confusion_matrix(y,pred).tolist()}

def differences(a,b,y,q,seed):
 delta=(a==y).astype(int)-(b==y).astype(int);wins=int(np.sum(delta==1));losses=int(np.sum(delta==-1))
 rng=np.random.default_rng(seed);samples=np.empty(4000)
 for i in range(4000):samples[i]=delta[rng.integers(len(y),size=len(y))].mean()
 _,group=np.unique(q,axis=0,return_inverse=True);den=np.bincount(group);num=np.bincount(group,weights=delta)
 clusters=np.empty(4000)
 for i in range(4000):ind=rng.integers(len(den),size=len(den));clusters[i]=num[ind].sum()/den[ind].sum()
 return {'difference_points':float(delta.mean()*100),'candidate_only_correct':wins,'baseline_only_correct':losses,
  'exact_discordance_pvalue':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
  'row_bootstrap_95_points':(np.quantile(samples,[.025,.975])*100).tolist(),
  'feature_group_bootstrap_95_points':(np.quantile(clusters,[.025,.975])*100).tolist(),
  'replications':4000,'scope':'descriptive paired intervals on previously consumed test partitions; feature grouping does not identify writers or fonts'}

def main(a):
 a.out.mkdir(parents=True,exist_ok=False)
 lock=json.loads((a.models/'FINAL_LOCK.json').read_text())
 for file,digest in lock['files'].items():
  if sha(a.models/file)!=digest:raise ValueError('final model changed')
 source=Path(__file__).parent/'reference_probe.cpp';lib=a.out/'reference.so'
 cmd=['g++','-std=c++17','-O3','-mavx2','-fno-fast-math','-ffp-contract=off','-shared','-fPIC',f'-DLM_BASE_RUNTIME="{ROOT}/spectra/_native/ovo/runtime.cpp"',str(source),'-o',str(lib)]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
 write(a.out/'reference-build.json',{'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'source_sha256':sha(source)})
 if p.returncode:raise RuntimeError(p.stderr)
 # This record is written BEFORE any held-out model call in this script.
 write(a.out/'EVALUATION_OPENING.json',{'final_lock_sha256':sha(a.models/'FINAL_LOCK.json'),
  'data':{p.name:sha(p) for p in a.data.glob('*.npz')},'code':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
  'native_library_sha256':sha(a.library),'reference_library_sha256':sha(lib),
  'scope':'first evaluation of these frozen models; official partitions previously consumed by older research'})
 results={};allchecks=[];start=time.process_time()
 with threadpool_limits(1):
  for task in ('letter','pendigits'):
   tr=np.load(a.data/(task+'-train.npz'));te=np.load(a.data/(task+'-test.npz'));q=te['q'];y=te['y'];D=int(te['maximum'])
   folder=a.out/task;folder.mkdir();predictions={}
   trainkeys=set(map(bytes,tr['q']));nonoverlap=np.array([bytes(row) not in trainkeys for row in q])
   for arm in ('uniform','variance','nca'):
    modeldir=a.models/(task+'-'+arm);ref=np.load(modeldir/'reference.npz');sk=pickle.loads((modeldir/'sklearn.pkl').read_bytes())
    expected=[]
    for s in range(0,len(q),128):
     matrix=kernel_matrix(q[s:s+128],tr['q'],ref['weights'],ref['table']);expected.extend(sk.predict(matrix).tolist())
    expected=np.array(expected);native_modes={}
    with MetricSession(modeldir/'model.sgm',a.library) as engine:
     for mode in ('compiled','scalar_integer','scalar_exp','exhaustive'):
      pred=np.asarray(engine.predict_buffer(q,mode=mode));assert np.array_equal(pred,expected),(task,arm,mode)
      native_modes[mode]={'matched':len(q),'prediction_sha256':hashlib.sha256(pred.astype('<i8').tobytes()).hexdigest()}
     _,work=engine.inspect_buffer(q)
     expanded=expanded_model(ref);(folder/(arm+'-expanded-reference.srt')).write_bytes(expanded)
     original=Reference(expanded,lib,len(ref['classes']),int(ref['weights'].sum()))
     count=0;observed=hashlib.sha256();reference=hashlib.sha256()
     try:
      for s in range(0,len(q),64):
       xb=np.repeat(q[s:s+64],ref['weights'].astype(int),axis=1).astype(np.float64)
       old=original.probe(xb);new=engine.probe(q[s:s+64]);assert new==old.tobytes(),(task,arm,'margin',s)
       observed.update(new);reference.update(old.tobytes());count+=old.size
     finally:original.close()
     check={'task':task,'arm':arm,'native_modes':native_modes,'pair_margins':count,
       'margin_sha256':observed.hexdigest(),'independent_margin_sha256':reference.hexdigest(),
       'work':work,'runtime_info':engine.info,'features_repeated':int(ref['weights'].sum())}
     allchecks.append(check)
    predictions[arm]=expected
    np.savez_compressed(folder/(arm+'-predictions.npz'),prediction=expected,expected=y,nonoverlap=nonoverlap)
    print(task,arm,'correct',int(np.sum(expected==y)),'/',len(y),'margin checks',count,flush=True)
   for arm in ('linear','mlp'):
    sk=pickle.loads((a.models/(task+'-'+arm)/'model.pkl').read_bytes());pred=sk.predict(q.astype(float)/D)
    engine=ControlSession(a.controls/(task+'-'+arm));native=np.asarray(engine.predict_buffer(q))
    assert np.array_equal(pred,native),(task,arm,'control fidelity')
    predictions[arm]=pred;np.savez_compressed(folder/(arm+'-predictions.npz'),prediction=pred,expected=y,nonoverlap=nonoverlap)
    print(task,arm,'correct',int(np.sum(pred==y)),'/',len(y),flush=True)
   records={arm:{**summarize(pred,y),'nonoverlap':summarize(pred[nonoverlap],y[nonoverlap])} for arm,pred in predictions.items()}
   records['difference']=differences(predictions['nca'],predictions['uniform'],y,q,20260928)
   records['nonoverlap_rows']=int(nonoverlap.sum());records['overlap_rows']=int((~nonoverlap).sum())
   records['nca_vs_variance']=differences(predictions['nca'],predictions['variance'],y,q,20260929)
   results[task]=records
 write(a.out/'quality.json',results);write(a.out/'fidelity.json',allchecks)
 write(a.out/'cost.json',{'cpu_seconds':time.process_time()-start})
 print(json.dumps({t:{k:v for k,v in r.items() if k in ('difference','overlap_rows','nonoverlap_rows')} for t,r in results.items()},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('data','models','library','controls','out'):p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
