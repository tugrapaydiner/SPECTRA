"""Freeze old-model identity and one predetermined HAR SVC/linear control.

Trusted prior pickle deserialization is confined to this export experiment.
No prediction on the new HAR test set occurs in this script.
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, shutil, sys, time, warnings
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.svm import SVC, LinearSVC
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from spectra.svm_export import export_prepared_svc
from spectra.svm_pipeline import Preprocessor

TASKS=('wine','wdbc','chess','penguins','titanic','zoo')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def export_libsvm(model,path):
    """Exact decimal roundtrip, with sklearn's binary sign convention converted."""
    c=len(model.classes_);coef=model.dual_coef_;bias=model.intercept_
    if c==2: coef=-coef;bias=-bias
    f=lambda v:format(float(v),'.17g')
    with Path(path).open('x',encoding='ascii') as out:
        out.write('svm_type c_svc\nkernel_type rbf\ngamma '+f(model._gamma)+'\n')
        out.write(f'nr_class {c}\ntotal_sv {len(model.support_vectors_)}\n')
        out.write('rho '+' '.join(f(-v) for v in bias)+'\n')
        out.write('label '+' '.join(str(i) for i in range(c))+'\n')
        out.write('nr_sv '+' '.join(str(int(i)) for i in model.n_support_)+'\nSV\n')
        for i,sv in enumerate(model.support_vectors_):
            terms=[f(coef[j,i]) for j in range(c-1)]
            terms.extend(f'{j+1}:{f(v)}' for j,v in enumerate(sv) if v!=0.)
            out.write(' '.join(terms)+'\n')

def main(a):
    import sklearn, scipy
    if not a.trust_frozen_pickles: raise ValueError('explicit prior-pickle trust required')
    a.out.mkdir(parents=True,exist_ok=False)
    old=json.loads((a.old_inputs/'manifest.json').read_text())
    manifest={'protocol_commit':'0ca7b08fc1a8aaa7e2cb2044938024fb08ff76ab','numpy':np.__version__,
              'sklearn':sklearn.__version__,'scipy':scipy.__version__,'models':{},'new_test_predictions_performed':False}
    for task in TASKS:
        name=task+'-101';src=a.old_inputs/name;entry=old['models'][name]
        for filename in ('reference_svc.pkl','model.srt','preprocessing.json','original.jsonl','expected.json'):
            wanted=entry['files'][filename];p=src/filename
            assert p.stat().st_size==wanted['bytes'] and sha(p)==wanted['sha256'],p
        with (src/'reference_svc.pkl').open('rb') as f:model=pickle.load(f)
        dest=a.out/name;dest.mkdir()
        for filename in ('preprocessing.json','original.jsonl','expected.json'):
            shutil.copyfile(src/filename,dest/filename)
        export_prepared_svc(model,dest/'model.srt')
        assert sha(dest/'model.srt')==entry['original_model_sha256']
        shutil.copyfile(src/'reference_svc.pkl',dest/'model.pkl')
        raw=[json.loads(l) for l in (src/'original.jsonl').read_text().splitlines()]
        x=np.asarray(Preprocessor.load(src/'preprocessing.json').transform(raw),dtype=np.float64).reshape(len(raw),-1)
        x.tofile(dest/'X.f64')
        export_libsvm(model,dest/'libsvm.model')
        manifest['models'][name]={'rows':len(x),'features':x.shape[1],'classes':model.classes_.tolist(),
            'supports':len(model.support_vectors_),'origin':'retained seed101','files':{p.name:sha(p) for p in sorted(dest.iterdir())}}
    dest=a.out/'har';dest.mkdir()
    train=a.har/'train';test=a.har/'test'
    x=np.loadtxt(train/'X_train.txt',dtype=np.float64);y=np.loadtxt(train/'y_train.txt',dtype=np.int64)
    train_subject=np.loadtxt(train/'subject_train.txt',dtype=np.int64)
    test_subject=np.loadtxt(test/'subject_test.txt',dtype=np.int64)
    assert x.shape==(7352,561) and np.isfinite(x).all() and len(y)==len(x)
    assert set(train_subject).isdisjoint(set(test_subject))
    times={}
    with threadpool_limits(1):
        for name,model in [('svc',SVC(C=10,gamma='scale',kernel='rbf',probability=False,break_ties=False,cache_size=256)),
                           ('linear',LinearSVC(C=1,dual='auto',max_iter=10000,random_state=20260927))]:
            start=time.perf_counter();cpu=time.process_time()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(x,y)
            times[name]={'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
                         'warnings':[str(w.message) for w in caught]}
            with (dest/(name+'.pkl')).open('xb') as f:pickle.dump(model,f,protocol=5)
            if name=='svc':
                export_prepared_svc(model,dest/'model.srt');export_libsvm(model,dest/'libsvm.model')
                shutil.copyfile(dest/'svc.pkl',dest/'model.pkl');svc=model
            else:
                np.savez(dest/'linear_parameters.npz',coef=model.coef_,intercept=model.intercept_,classes=model.classes_)
    manifest['models']['har']={'rows':2947,'features':561,'classes':svc.classes_.tolist(),'supports':len(svc.support_vectors_),
        'origin':'original subject-disjoint UCI HAR split; supplied feature matrix','training':times,
        'train_subjects':sorted(map(int,set(train_subject))),'test_subjects':sorted(map(int,set(test_subject))),
        'source_sha256':{str(p.relative_to(a.har)):sha(p) for p in sorted(a.har.rglob('*.txt'))},
        'files':{p.name:sha(p) for p in sorted(dest.iterdir())}}
    save(a.out/'MODEL_FREEZE.json',manifest)
    print(json.dumps(manifest['models']['har'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('old-inputs','har','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--trust-frozen-pickles',action='store_true');main(p.parse_args())
