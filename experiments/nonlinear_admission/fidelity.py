"""All held-out predictions plus independent numerical receipt probes.

No timing acceptance here. Frozen pickles are trusted local training outputs.
Native dense arithmetic may differ in final bits from BLAS, but no changed class
is excused. Eight explicit receipt probes are not a full independent SVM replay.
"""
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.nonlinear_admission.study import load,verify_freeze,save,sha,TASKS
from experiments.nonlinear_admission.native import DenseSession
from experiments.native_baselines.runtime import NativeSession
from spectra.svm_shared import PreparedModel
from spectra.svm_receipt import create_receipt,verify_receipt

def run(a):
    from threadpoolctl import threadpool_limits
    verify_freeze(a.root);build=json.loads((a.builds/'BUILD.json').read_text());result={}
    a.out.mkdir(exist_ok=False)
    with threadpool_limits(1):
        for task in TASKS:
            x,y,_=load(a.root/'data'/task/'test.npz')
            with (a.root/'final'/task/'svm/model.pkl').open('rb') as f:obj=pickle.load(f)
            z=np.ascontiguousarray(obj['scaler'].transform(x));expected=np.load(a.root/'evaluation'/(task+'-svm.npy')).tolist()
            models={};rec=[];pairs=len(obj['model'].classes_)*(len(obj['model'].classes_)-1)//2
            with PreparedModel(a.builds/task/'model.srt',build['libraries']['spectra'],input_dtype='float64') as owner,owner.session() as w:
                for mode in ('beretta_cert','exhaustive'):
                    observed=w.predict_buffer(z,schedule=mode)
                    if observed!=expected:raise AssertionError((task,mode,'changed label'))
                for i in (0,len(y)//3,2*len(y)//3,len(y)-1):
                    receipt=create_receipt(w,z[i]);check=verify_receipt(a.builds/task/'model.srt',receipt,expected_input=z[i],input_dtype='float64')
                    if not check['verified'] or check['label']!=expected[i]:raise AssertionError('receipt mismatch')
                    save(a.out/(task+'-receipt-'+str(i)+'.json'),receipt);rec.append({'row':i,**check})
            with NativeSession(build['libraries']['libsvm'],x.shape[1],obj['model'].classes_,a.builds/task/'libsvm.model') as lib:
                labels,scores=lib.predict_buffer(z,return_scores=True)
                if labels!=expected:raise AssertionError('LIBSVM changed class')
                scores=np.frombuffer(scores,dtype='d').reshape(len(y),pairs)
                reference=obj['model']._decision_function(z)
                if not np.allclose(scores,reference,rtol=1e-10,atol=1e-10):raise AssertionError('unexpected margin error')
                models['svm']={'rows':len(y),'max_libsvm_margin_error':float(np.max(np.abs(scores-reference))), 'score_values':scores.size}
            for name in ('linear','mlp-101','mlp-202','mlp-303'):
                with (a.root/'final'/task/name/'model.pkl').open('rb') as f:obj=pickle.load(f)
                if not np.array_equal(obj['scaler'].transform(x),z):raise AssertionError('different scaler')
                model=obj['model'];expected=np.load(a.root/'evaluation'/(task+'-'+name+'.npy')).tolist()
                with DenseSession(build['libraries']['dense'],a.builds/task/(name+'.weights'),x.shape[1],model.classes_) as dense:
                    observed,raw=dense.predict_buffer(z,return_scores=True)
                    if observed!=expected:raise AssertionError((task,name,'changed class'))
                    actual=np.frombuffer(raw,dtype='d').reshape(len(y),len(model.classes_))
                    ref=z@model.coef_.T+model.intercept_ if name=='linear' else np.maximum(0.,z@model.coefs_[0]+model.intercepts_[0])@model.coefs_[1]+model.intercepts_[1]
                    if not np.allclose(actual,ref,rtol=1e-10,atol=1e-10):raise AssertionError('unexpected dense score error')
                    models[name]={'rows':len(y),'score_values':actual.size,'max_score_error':float(np.max(np.abs(actual-ref)))}
            result[task]={'models':models,'receipts':rec}
            print(task,'fidelity PASS',flush=True)
    save(a.out/'SUMMARY.json',{'status':'PASS','model_freeze_sha256':sha(a.root/'MODEL_FREEZE.json'),'build_sha256':sha(a.builds/'BUILD.json'),
        'script_sha256':sha(__file__),'tasks':result,'scope':'all native model class decisions; limited independent SVM receipt probes; same-host floating-point contracts'})
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','builds','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
