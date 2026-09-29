"""Open frozen official tests only after all selected model artifacts are locked.

Prototype scores are checked by a separately compiled scalar observer that does
not include the candidate runtime or reuse its tables. Reference Python/BLAS
scores need not be bit-identical; their predicted labels must agree here.
"""
from __future__ import annotations
import argparse,ctypes as C,hashlib,io,json,pickle,struct,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.budgeted_prototypes.learning import Settings,features
from experiments.budgeted_prototypes.session import PrototypeSession
from experiments.budgeted_prototypes.controls import ControlSession
from experiments.budgeted_prototypes.study import TASKS,ARMS,sha,write,train_path,source_hashes

class Observer:
    def __init__(self,library):
        self.lib=C.CDLL(str(library));self.fn=self.lib.pr_scores
        self.fn.argtypes=[C.c_void_p,C.c_uint64,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.POINTER(C.c_double),C.c_uint64];self.fn.restype=C.c_int
    def scores(self,raw,q,classes):
        blob=C.create_string_buffer(raw,len(raw));out=np.empty((len(q),classes),dtype=np.float64)
        status=self.fn(blob,len(raw),q.ctypes.data_as(C.POINTER(C.c_uint8)),len(q),q.shape[1],out.ctypes.data_as(C.POINTER(C.c_double)),out.size)
        if status:raise ValueError('independent scalar observer failed')
        return out

def quality(pred,y):
    labels=np.unique(y)
    return {'correct':int(np.sum(pred==y)),'rows':len(y),'accuracy':float(np.mean(pred==y)),
            'macro_f1':float(f1_score(y,pred,average='macro')),'labels':labels.tolist(),
            'confusion':confusion_matrix(y,pred,labels=labels).tolist()}
def paired(pred,base,y):
    wins=int(np.sum((pred==y)&(base!=y)));losses=int(np.sum((pred!=y)&(base==y)))
    return {'gain_points':100.*(wins-losses)/len(y),'candidate_only_correct':wins,'baseline_only_correct':losses,
            'paired_binomial_p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
            'scope':'descriptive row-paired statistic; not writer-cluster or multiple-comparison-adjusted confirmation'}
def main(a):
    a.out.mkdir(parents=True,exist_ok=False);lock=json.loads((a.models/'FINAL_LOCK.json').read_text())
    for name,h in lock['files'].items():
        if sha(a.models/name)!=h:raise ValueError('final artifact changed: '+name)
    lib=a.out/'reference.so';src=Path(__file__).with_name('reference.cpp')
    command=['g++','-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',str(src),'-o',str(lib)]
    done=subprocess.run(command,capture_output=True,text=True,timeout=120)
    write(a.out/'reference-build.json',{'command':command,'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr,'source_sha256':sha(src)})
    if done.returncode:raise RuntimeError(done.stderr)
    # This is written before parsing the new optical-digit test file.
    write(a.out/'TEST_OPENING.json',{'final_lock_sha256':sha(a.models/'FINAL_LOCK.json'),
        'source':source_hashes(),'native_sha256':sha(a.library),'control_sha256':sha(a.controls),
        'observer_sha256':sha(lib),'optical_archive_sha256':sha(a.new/'optdigits.zip'),
        'selection_lock_commit':'622241a499ca01b95bf6446402df9277bbd5212c',
        'scope':'one opening of newly fitted models; three historical tasks exposed, optical test not previously opened in this continuation'})
    observer=Observer(lib);reports={};fidelity=[]
    with threadpool_limits(1):
        for task in TASKS:
            if task=='optdigits':
                with zipfile.ZipFile(a.new/'optdigits.zip') as z:raw=z.read('optdigits.tes')
                matrix=np.loadtxt(io.BytesIO(raw),delimiter=',',dtype=np.int64)
                if matrix.shape!=(1797,65) or matrix[:,:64].min()<0 or matrix[:,:64].max()>16:raise ValueError('optical test schema differs')
                q=matrix[:,:64].astype(np.uint8);y=matrix[:,64];D=16
                np.savez_compressed(a.out/'optdigits-test.npz',q=q,y=y,maximum=D)
                write(a.out/'optical-test-data.json',{'raw_sha256':hashlib.sha256(raw).hexdigest(),'rows':len(q),'features':64,'parsed_after_final_lock':True})
            else:
                path=a.prior/('satellite/data/test.npz' if task=='satellite' else f'datasets/{task}-test.npz')
                with np.load(path) as f:q=f['q'];y=f['y'];D=int(f['maximum'])
            q=np.ascontiguousarray(q)
            with np.load(train_path(a.prior,a.new,task)) as f:train_q=f['q']
            known=set(map(bytes,train_q));nonoverlap=np.asarray([bytes(r) not in known for r in q])
            folder=a.out/task;folder.mkdir();q.tofile(folder/'input.u8');np.save(folder/'truth.npy',y)
            outputs={};rowreports={}
            for arm in (*ARMS,'linear'):
                model=a.models/f'{task}-{arm}';fit=json.loads((model/'fit.json').read_text())
                if arm in ('fixed','centers','local'):
                    with np.load(model/'model.npz') as f:arrays={k:f[k] for k in f.files}
                    s=Settings(**json.loads((model/'settings.json').read_text()))
                    scores=features(q,arrays,D,s)@arrays['head']+arrays['bias'];pred=arrays['classes'][scores.argmax(1)]
                    raw=(model/'model.spp').read_bytes();obs_hash=hashlib.sha256();score_count=0
                    with PrototypeSession(model/'model.spp',a.library) as native:
                        for mode in ('compiled','scalar'):
                            if native.predict_buffer(q,mode=mode)!=pred.tolist():raise ValueError(f'{task}/{arm}/{mode} label disagreement')
                        for start in range(0,len(q),128):
                            block=q[start:start+128];reference=observer.scores(raw,block,len(native.labels));got=native.scores(block)
                            if got!=reference.tobytes():raise ValueError(f'{task}/{arm} independent ordered score disagreement at {start}')
                            obs_hash.update(got);score_count+=reference.size
                        info=native.info
                    details={'exact_observer_scores':score_count,'score_sha256':obs_hash.hexdigest(),'model_bytes':len(raw),
                             'deployed_scalar_values':int(2*arrays['centers'].size+arrays['head'].size+arrays['bias'].size),**info}
                else:
                    reference=pickle.loads((model/'model.pkl').read_bytes());pred=reference.predict(q.astype(float)/D)
                    path=model/('model.srt' if arm=='svc' else 'model.snn')
                    with ControlSession(path,a.controls,maximum=D) as native:
                        if native.predict_buffer(q)!=pred.tolist():raise ValueError(f'{task}/{arm} native control mismatch')
                        if arm!='svc' and native.scores(q)!=native.scores(q,mode='scalar'):raise ValueError('vector/scalar control score mismatch')
                        details={**native.info,'model_bytes':path.stat().st_size}
                outputs[arm]=np.asarray(pred)
                np.savez_compressed(folder/f'{arm}-predictions.npz',prediction=pred,expected=y,nonoverlap=nonoverlap)
                rowreports[arm]={**quality(pred,y),'nonoverlap':quality(pred[nonoverlap],y[nonoverlap]),'deployment':details}
                fidelity.append({'task':task,'arm':arm,'rows':len(q),'matched':True,**details})
                print(task,arm,rowreports[arm]['correct'],'/',len(y),flush=True)
            reports[task]={'arms':rowreports,'overlap_rows':int((~nonoverlap).sum()),'nonoverlap_rows':int(nonoverlap.sum()),
                          'local_vs_fixed':paired(outputs['local'],outputs['fixed'],y),
                          'local_vs_centers':paired(outputs['local'],outputs['centers'],y),
                          'local_vs_svc':paired(outputs['local'],outputs['svc'],y),
                          'local_vs_mlp':paired(outputs['local'],outputs['mlp'],y)}
    write(a.out/'quality.json',reports);write(a.out/'fidelity.json',fidelity)
    print(json.dumps({t:{arm:v['correct'] for arm,v in r['arms'].items()} for t,r in reports.items()},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('models','prior','new','library','controls','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
