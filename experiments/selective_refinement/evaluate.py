"""One final evaluation after all models and calibration policies are frozen."""
from __future__ import annotations
import argparse,ctypes as C,hashlib,io,json,pickle,subprocess,time,zipfile
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
from experiments.budgeted_prototypes.session import PrototypeSession
from experiments.budgeted_prototypes.float_control import FloatSession
from experiments.budgeted_prototypes.evaluate import Observer
from experiments.conditioned_heads.pilot import training
from .session import RefinementSession
from .study import TASKS,sha,write,source_snapshot

def test_data(data,task):
    if task=='optdigits':
        with zipfile.ZipFile(data/'optdigits.zip') as z:m=np.loadtxt(io.BytesIO(z.read('optdigits.tes')),delimiter=',',dtype=int)
        assert m.shape==(1797,65) and m[:,:64].min()>=0 and m[:,:64].max()<=16
        return m[:,:64].astype(np.uint8),m[:,64],16
    with np.load(data/(task+'-test.npz')) as d:return d['q'],d['y'],int(d['maximum'])

def quality(pred,truth):
    return {'rows':len(truth),'correct':int(np.sum(pred==truth)),'accuracy':float(np.mean(pred==truth)),
            'macro_f1':float(f1_score(truth,pred,average='macro')),
            'confusion':confusion_matrix(truth,pred).tolist()}

def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    lock=json.loads((a.calibration/'CALIBRATION_LOCK.json').read_text())
    for name,h in lock['files'].items():
        if sha(a.calibration/name)!=h:raise ValueError('calibration artifact changed')
    if sha(a.models/'FINAL_MODELS.json')!=lock['models_sha256']:raise ValueError('different model inventory')
    model_lock=json.loads((a.models/'FINAL_MODELS.json').read_text())
    for name,h in model_lock['files'].items():
        if sha(a.models/name)!=h:raise ValueError('model changed after fitting')
    policies=json.loads((a.calibration/'POLICIES.json').read_text())
    write(a.out/'OPENING.json',{'model_lock_sha256':sha(a.models/'FINAL_MODELS.json'),
        'calibration_lock_sha256':sha(a.calibration/'CALIBRATION_LOCK.json'),
        'source':source_snapshot(a.out/'source'),'native_sha256':sha(a.library),'mlp_sha256':sha(a.mlplib),
        'scope':'all four tests previously exposed; no new model/threshold selection after opening'})
    root=Path(__file__).resolve().parents[2];src=root/'experiments/budgeted_prototypes/reference.cpp'
    reference=a.out/'observer.so';cmd=['g++','-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',str(src),'-o',str(reference)]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    write(a.out/'reference-build.json',{'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'source_sha256':sha(src)})
    if p.returncode:raise RuntimeError(p.stderr)
    observer=Observer(reference);results={};score_cells=0;start=time.process_time()
    with threadpool_limits(1):
        for task in TASKS:
            q,y,D=test_data(a.data,task);q=np.ascontiguousarray(q);folder=a.out/task;folder.mkdir()
            np.savez_compressed(folder/'input.npz',q=q,truth=y,maximum=D)
            q.tofile(folder/'input.u8');preds={};work={};scores=None
            with ExitStack() as stack:
                policy_engines={key:stack.enter_context(RefinementSession(a.models/task/'fast/model.spp',a.models/task/'strong/model.srt',a.library,
                                    threshold=policies[task][key]['threshold'],accept_fraction=policies[task][key]['accept_fraction'])) for key in ('0.005','0.01','0.02')}
                engine=policy_engines['0.01']
                for name,mode in (('fast','fast'),('strong','strong'),('primary','calibrated'),('blind','blind')):
                    values,stats=engine.inspect_buffer(q,mode=mode);preds[name]=np.asarray(values);work[name]=stats
                for name,key in (('strict','0.005'),('loose','0.02')):
                    values,stats=policy_engines[key].inspect_buffer(q);preds[name]=np.asarray(values);work[name]=stats
                fast=stack.enter_context(PrototypeSession(a.models/task/'fast/model.spp',a.library))
                scores=np.frombuffer(fast.scores(q),dtype=np.float64).reshape(len(q),-1).copy()
                assert fast.predict_buffer(q)==preds['fast'].tolist()
                ordered=np.sort(scores,axis=1);gap=ordered[:,-1]-ordered[:,-2]
                for name,key in (('primary','0.01'),('strict','0.005'),('loose','0.02')):
                    t=policies[task][key]['threshold'];accept=gap>=(np.inf if t is None else t)
                    expected=np.where(accept,preds['fast'],preds['strong'])
                    if not np.array_equal(preds[name],expected):raise ValueError('native routing differs from independent threshold expression')
                    assert work[name]['strong_evaluations']==int((~accept).sum())
                h=hashlib.sha256();raw=(a.models/task/'fast/model.spp').read_bytes()
                for i in range(0,len(q),128):
                    reference_scores=observer.scores(raw,q[i:i+128],len(fast.labels))
                    candidate=fast.scores(q[i:i+128]);assert candidate==reference_scores.tobytes()
                    h.update(candidate);score_cells+=reference_scores.size
                write(folder/'score-fidelity.json',{'score_values':scores.size,'sha256':h.hexdigest(),'all_equal':True})
                svm=pickle.loads((a.models/task/'strong/model.pkl').read_bytes())
                if not np.array_equal(preds['strong'],svm.predict(q.astype(float)/D)):raise ValueError('strong native differs from trained SVM')
                mlp=stack.enter_context(FloatSession(a.models/task/'mlp/model.sfn',a.mlplib));preds['mlp32']=np.asarray(mlp.predict_buffer(q))
                with np.load(a.models/task/'mlp/weights.npz') as f:arr={k:f[k] for k in f.files}
                z=(q.astype(np.float32)/np.float32(D)-arr['mean'].astype(np.float32))/arr['scale'].astype(np.float32)
                for i in range(3):
                    z=z@arr[f'w{i}'].astype(np.float32)+arr[f'b{i}'].astype(np.float32)
                    if i<2:z=np.maximum(z,0)
                ref=arr['classes'][z.argmax(1)]
                if not np.array_equal(preds['mlp32'],ref):raise ValueError('FP32 native disagrees with separate forward pass')
                original=pickle.loads((a.models/task/'mlp/model.pkl').read_bytes()).predict(q.astype(float)/D)
                original_mlp_disagreements=int(np.sum(original!=preds['mlp32']))
                info=engine.info
            np.savez_compressed(folder/'predictions.npz',**preds,truth=y,gap=gap,fast_scores=scores)
            rec={'quality':{name:quality(pred,y) for name,pred in preds.items()},'work':work,
                 'policies':policies[task],'runtime_info':info,'mlp_fp32_disagreements':original_mlp_disagreements}
            for name,key in (('primary','0.01'),('strict','0.005'),('loose','0.02')):
                t=policies[task][key]['threshold'];accept=gap>=(np.inf if t is None else t)
                harm=int(np.sum(accept&(preds['fast']!=y)&(preds['strong']==y)))
                benefit=int(np.sum(accept&(preds['fast']==y)&(preds['strong']!=y)))
                fixes=int(np.sum((preds[name]==y)&(preds['fast']!=y)));regressions=int(np.sum((preds[name]!=y)&(preds['fast']==y)))
                rec[name]={'accepted_fast':int(accept.sum()),'escalated':int((~accept).sum()),'observed_added_harm':harm/len(y),
                           'harmful_accepted':harm,'beneficial_accepted':benefit,'fixed_fast_errors':fixes,'introduced_fast_errors':regressions,
                           'gain_over_fast_points':100*float(np.mean(preds[name]==y)-np.mean(preds['fast']==y)),
                           'loss_against_strong_points':100*float(np.mean(preds['strong']==y)-np.mean(preds[name]==y))}
            # Keep exact-row overlap strata without inferring writer/scene independence.
            trainq,_,_=training(a.data,task)
            with np.load(a.models/task/'roles.npz') as f:dev=f['development'];cal=f['calibration']
            devrows=set(map(bytes,trainq[dev]));calrows=set(map(bytes,trainq[cal]))
            rec['overlap']={'test_rows_seen_in_development':sum(bytes(row) in devrows for row in q),
                            'test_rows_seen_in_calibration':sum(bytes(row) in calrows for row in q)}
            results[task]=rec
            print(task,{name:r['correct'] for name,r in rec['quality'].items()},'primary',rec['primary'],flush=True)
    write(a.out/'RESULTS.json',{'tasks':results,'independent_prototype_scores':score_cells,'cpu_seconds':time.process_time()-start,
                              'scope':'new model pair and fixed calibration policy; reused public tests, no iid/shift validation'})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('models','calibration','data','library','mlplib','out'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
