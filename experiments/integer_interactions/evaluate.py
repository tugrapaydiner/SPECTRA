"""Locked-model exposed-partition evaluation plus separately ordered native reference.

No fit or model selection occurs here. These official tests were consumed in
prior SPECTRA work; new-model evaluation is not independent confirmation.
"""
from __future__ import annotations
import argparse,ctypes as C,hashlib,json,pickle,struct,sys,zlib,time,subprocess
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.integer_interactions.learning import ProductKernel,project,gram
from experiments.integer_interactions.session import InteractionSession

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def data_path(root,t,part):return root/('satellite/data/'+part+'.npz' if t=='satellite' else 'datasets/'+t+'-'+part+'.npz')
def original_model(ref):
    A=ref['A'];D=int(ref['maximum']);raw=ref['raw_supports']
    # Independent scalar projection on MODEL supports, not imported project().
    shifts=[-D*sum(min(0,int(c)) for c in row) for row in A]
    z=np.array([[shifts[r]+sum(int(c)*int(x) for c,x in zip(row,q)) for r,row in enumerate(A)] for q in raw],dtype='<f8')
    n,d=z.shape;c=len(ref['classes']);meta=json.dumps({'labels':ref['classes'].tolist()},separators=(',',':')).encode()
    payload=struct.pack('<d',float(ref['coefficient']))+ref['counts'].astype('<u4').tobytes()+z.tobytes()+ref['coefficients'].astype('<f8').tobytes()+ref['intercepts'].astype('<f8').tobytes()
    body=meta+payload
    return struct.pack('<8sIIIIII',b'SPCSVM02',c,n,d,len(meta),len(payload),zlib.crc32(body))+body

class Reference:
    def __init__(self,model,bits,path,classes,rank):
        self._lib=lib=C.CDLL(str(path));self.classes=classes;self.rank=rank
        lib.ref_create.argtypes=[C.c_void_p,C.c_uint64,C.c_int];lib.ref_create.restype=C.c_void_p
        lib.ref_destroy.argtypes=[C.c_void_p];lib.ref_destroy.restype=None
        lib.ref_probe.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_int,C.c_int,C.POINTER(C.c_double),C.c_uint64];lib.ref_probe.restype=C.c_int
        raw=C.create_string_buffer(model,len(model));self.h=lib.ref_create(raw,len(model),bits)
        if not self.h:raise ValueError('reference loading failed')
    def probe(self,x):
        x=np.ascontiguousarray(x,dtype=np.float64);out=np.empty((len(x),self.classes*(self.classes-1)//2),dtype=np.float64)
        rc=self._lib.ref_probe(self.h,x.ctypes.data_as(C.POINTER(C.c_double)),len(x),self.rank,out.ctypes.data_as(C.POINTER(C.c_double)),out.size)
        if rc:raise ValueError('independent reference failed')
        return out
    def close(self):
        if self.h:self._lib.ref_destroy(self.h);self.h=None

def scores(pred,y):return {'correct':int(np.sum(pred==y)),'rows':len(y),'accuracy':float(np.mean(pred==y)),'macro_f1':float(f1_score(y,pred,average='macro')),'confusion':confusion_matrix(y,pred).tolist()}
def difference(a,b,y,q,seed):
    d=(a==y).astype(np.int8)-(b==y).astype(np.int8);wins=int(np.sum(d==1));losses=int(np.sum(d==-1))
    _,g=np.unique(q,axis=0,return_inverse=True);counts=np.bincount(g);values=np.bincount(g,weights=d);rng=np.random.default_rng(seed)
    trials=[]
    for _ in range(4000):
        ids=rng.integers(len(counts),size=len(counts));trials.append(values[ids].sum()/counts[ids].sum())
    return {'points':float(np.mean(d)*100),'candidate_only_correct':wins,'baseline_only_correct':losses,
        'paired_pvalue':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
        'feature_group_bootstrap_95_points':(np.quantile(trials,[.025,.975])*100).tolist(),
        'scope':'exploratory on previously consumed partitions; feature rows are not proven independent writers/scenes'}

def main(a):
    a.out.mkdir(parents=True,exist_ok=False);lock=json.loads((a.models/'FINAL_LOCK.json').read_text())
    for name,h in lock['files'].items():
        if sha(a.models/name)!=h:raise ValueError('frozen model changed')
    sr=Path(__file__).with_name('reference.cpp');lr=a.out/'reference.so'
    command=['g++','-std=c++17','-O3','-mavx2','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',f'-DII_BASE_RUNTIME="{ROOT}/spectra/_native/ovo/runtime.cpp"',str(sr),'-o',str(lr)]
    p=subprocess.run(command,capture_output=True,text=True,timeout=120);write(a.out/'reference-build.json',{'command':command,'exitcode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'source_sha256':sha(sr)})
    if p.returncode:raise RuntimeError(p.stderr)
    write(a.out/'EVALUATION_OPENING.json',{'final_lock_sha256':sha(a.models/'FINAL_LOCK.json'),'native_library_sha256':sha(a.library),
        'reference_library_sha256':sha(lr),'source':{p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()},
        'data':{t+'-'+part:sha(data_path(a.data,t,part)) for t in ('letter','pendigits','satellite') for part in ('train','test')},
        'scope':'new frozen models, previously consumed official test partitions; not fresh confirmation'})
    quality={};fidelity=[];cpu=time.process_time()
    with threadpool_limits(1):
        for task in ('letter','pendigits','satellite'):
            tr=np.load(data_path(a.data,task,'train'));te=np.load(data_path(a.data,task,'test'));q=te['q'];y=te['y'];D=int(te['maximum'])
            training_keys=set(map(bytes,tr['q']));mask=np.array([bytes(v) not in training_keys for v in q]);preds={};quality[task]={}
            for arm in ('uniform','diagonal','full','whitening','local_supervised','local_unsupervised'):
                folder=a.models/(task+'-'+arm)
                with np.load(folder/'reference.npz') as f:ref={k:f[k] for k in f.files}
                model=pickle.loads((folder/'reference.pkl').read_bytes()) # Trusted locally generated frozen model only.
                A=ref['A'];k=ProductKernel(float(ref['coefficient']),int(ref['bits']),ref['high'],ref['low'],int(ref['bound']))
                zx=project(q,A,D);zt=project(tr['q'],A,D);pred=[]
                for start in range(0,len(q),128):pred.extend(model.predict(gram(zx[start:start+128],zt,k)).tolist())
                pred=np.array(pred);preds[arm]=pred
                with InteractionSession(folder/'model.sik',a.library) as w:
                    for mode in ('compiled','scalar_integer','exhaustive'):
                        assert w.predict_buffer(q,mode=mode)==pred.tolist(),(task,arm,mode)
                    direct=w.predict_buffer(q,mode='direct_exp');direct_diff=int(np.sum(np.array(direct)!=pred))
                    raw=original_model(ref);(a.out/f'{task}-{arm}-original.srt').write_bytes(raw)
                    original=Reference(raw,int(ref['bits']),lr,len(ref['classes']),len(A));h1=hashlib.sha256();h2=hashlib.sha256();count=0
                    try:
                        # An independent scalar projection for all observed queries.
                        shifts=[-D*sum(min(0,int(v)) for v in row) for row in A]
                        for s in range(0,len(q),64):
                            independent=np.array([[shifts[r]+sum(int(c)*int(x) for c,x in zip(row,query)) for r,row in enumerate(A)] for query in q[s:s+64]],dtype=np.float64)
                            assert independent.tobytes()==zx[s:s+64].astype(np.float64).tobytes()
                            old=original.probe(independent);new=w.probe(q[s:s+64]);assert new==old.tobytes(),(task,arm,s,'margin')
                            h1.update(old.tobytes());h2.update(new);count+=old.size
                    finally:original.close()
                    _,work=w.inspect_buffer(q)
                    fidelity.append({'task':task,'arm':arm,'rows':len(q),'pair_scores':count,'original_sha256':h1.hexdigest(),'candidate_sha256':h2.hexdigest(),
                                     'direct_exp_label_disagreements':direct_diff,'info':w.info,'work':work})
                np.savez_compressed(a.out/f'{task}-{arm}-predictions.npz',prediction=pred,expected=y,nonoverlap=mask,direct_prediction=np.array(direct))
                quality[task][arm]={**scores(pred,y),'nonoverlap':scores(pred[mask],y[mask])}
                print(task,arm,quality[task][arm]['correct'],'/',len(y),'margins',count,'direct disagreements',direct_diff,flush=True)
            quality[task]['full_vs_diagonal']=difference(preds['full'],preds['diagonal'],y,q,20260928)
            quality[task]['full_vs_uniform']=difference(preds['full'],preds['uniform'],y,q,20260929)
            quality[task]['whitening_vs_diagonal']=difference(preds['whitening'],preds['diagonal'],y,q,20260930)
            quality[task]['local_vs_diagonal']=difference(preds['local_supervised'],preds['diagonal'],y,q,20260931)
            quality[task]['local_vs_unsupervised']=difference(preds['local_supervised'],preds['local_unsupervised'],y,q,20260932)
            quality[task]['overlap_rows']=int((~mask).sum())
    write(a.out/'quality.json',quality);write(a.out/'fidelity.json',fidelity);write(a.out/'cost.json',{'cpu_seconds':time.process_time()-cpu})
if __name__=='__main__':
    p=argparse.ArgumentParser();
    for n in ('data','models','library','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
