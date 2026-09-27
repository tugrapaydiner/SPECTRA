"""Fixed HAR models; never read test features/labels or select by their outcomes."""
from __future__ import annotations
import argparse,hashlib,json,platform,sys,time,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)

def main(a):
 import numpy as np,sklearn,scipy,joblib
 from sklearn.svm import SVC,LinearSVC
 from threadpoolctl import threadpool_limits
 from spectra.svm_export import export_prepared_svc
 from spectra.svm_pipeline import SCHEMA
 a.out.mkdir(parents=True,exist_ok=False)
 acquisition=json.loads((a.data/'ACQUISITION.json').read_text())
 for name,entry in acquisition['files'].items():
  if sha(a.data/name)!=entry['sha256']:raise ValueError('source data identity mismatch')
 x=np.loadtxt(a.data/'train/X_train.txt',dtype=np.float64)
 y=np.loadtxt(a.data/'train/y_train.txt',dtype=np.int64)
 subject=np.loadtxt(a.data/'train/subject_train.txt',dtype=np.int64)
 assert x.shape==(7352,561) and y.shape==subject.shape==(7352,) and np.isfinite(x).all()
 models=[('rbf_c1',SVC(C=1.,gamma='scale',probability=False,break_ties=False,cache_size=128)),
         ('rbf_c10',SVC(C=10.,gamma='scale',probability=False,break_ties=False,cache_size=128)),
         ('linear_c1',LinearSVC(C=1.,dual='auto',max_iter=20000,random_state=0))]
 results=[]
 with threadpool_limits(1):
  for name,model in models:
   begin=time.perf_counter();cpu=time.process_time()
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always');model.fit(x,y)
   elapsed=time.perf_counter()-begin;cpu=time.process_time()-cpu
   folder=a.out/name;folder.mkdir()
   joblib.dump(model,folder/'model.joblib',compress=3)
   record={'name':name,'train_seconds':elapsed,'train_cpu_seconds':cpu,'warnings':[str(w.message) for w in caught],
           'training_rows':len(y),'training_subjects':sorted(map(int,set(subject))),'classes':list(map(int,model.classes_)),
           'source_sha256':sha(__file__),'model_joblib_sha256':sha(folder/'model.joblib')}
   if name.startswith('rbf'):
    record.update(export_prepared_svc(model,folder/'model.srt'))
    plan={'schema':SCHEMA,'columns':[f'feature_{i+1:03}' for i in range(561)],'features':561,
      'model_sha256':sha(folder/'model.srt'),'operations':[{'kind':'numeric','column':i,'fill':(0.).hex(),
      'mean':(0.).hex(),'scale':(1.).hex()} for i in range(561)]}
    # Identity transform, not refitted scaling of UCI's published feature vectors.
    write(folder/'preprocessing.json',plan)
    record['nonzero_coefficients']=int(np.count_nonzero(model.dual_coef_))
   else:
    np.savez(folder/'linear_weights.npz',coef=model.coef_,intercept=model.intercept_,classes=model.classes_)
    record['learned_values']=int(model.coef_.size+model.intercept_.size)
   results.append(record);write(folder/'TRAINING.json',record);print(json.dumps(record),flush=True)
 write(a.out/'FROZEN_MODELS.json',{'models':results,'python':sys.version,'numpy':np.__version__,
  'sklearn':sklearn.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
  'acquisition_sha256':sha(a.data/'ACQUISITION.json'),'test_opened':False,
  'scope':'all models fixed before test predictions; joblib is a trusted training artifact, not a deployment input'})
 np.save(a.out/'train_probe.npy',x[np.linspace(0,len(x)-1,128,dtype=int)])

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('data','out'):p.add_argument('--'+key,type=Path,required=True)
 main(p.parse_args())
