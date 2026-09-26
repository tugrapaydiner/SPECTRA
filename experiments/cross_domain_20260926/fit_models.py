"""Fit only the fixed development portions, then seal every model before tests."""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from spectra.svm_export import export_prepared_svc
from panel_data import SEED,TASKS,sha


def libsvm_text(model, path):
    """Exact float64->17-digit decimal export, canonical class-index labels.

    sklearn flips binary public dual/intercept signs; use its original private
    LIBSVM parameters. A full fidelity check is mandatory before any timings.
    """
    classes=len(model.classes_)
    n=len(model.support_vectors_)
    coef=np.asarray(model._dual_coef_, dtype=np.float64)
    intercept=np.asarray(model._intercept_, dtype=np.float64)
    lines=['svm_type c_svc','kernel_type rbf',f'gamma {model._gamma:.17g}',
           f'nr_class {classes}',f'total_sv {n}',
           'rho '+' '.join(format(-v,'.17g') for v in intercept),
           'label '+' '.join(map(str,range(classes))),
           'nr_sv '+' '.join(map(str,model.n_support_)), 'SV']
    for k,row in enumerate(model.support_vectors_):
        lines.append(' '.join(format(coef[j,k],'.17g') for j in range(classes-1))+' '+
                     ' '.join(f'{j+1}:{v:.17g}' for j,v in enumerate(row)))
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')


def fit_task(folder):
    split=json.loads((folder/'split.json').read_text())
    original=np.load(folder/'original.npz',allow_pickle=False)
    x,y=original['x'],original['y']
    train,val=split['fit'],split['validation']
    scaler=StandardScaler().fit(x[train])
    xf,xv=scaler.transform(x[train]),scaler.transform(x[val])
    specs=[(f'svc_C{c}_g{factor}',SVC(C=c,gamma=factor/x.shape[1],cache_size=256,probability=False,break_ties=False))
           for c in (1,10) for factor in (1.,.25)]
    specs += [('logistic',LogisticRegression(C=1,max_iter=2000)),
              ('mlp',MLPClassifier(hidden_layer_sizes=(64,64),max_iter=200,batch_size=128,
                 solver='adam',learning_rate_init=.001,alpha=.0001,early_stopping=False,
                 tol=0,n_iter_no_change=201,random_state=SEED))]
    records=[]; models={}
    destination=folder/'models';destination.mkdir(exist_ok=False)
    for name,model in specs:
        wall,cpu=time.perf_counter(),time.process_time()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            with threadpool_limits(limits=1):
                model.fit(xf,y[train]);pred=model.predict(xv)
        records.append(dict(name=name,validation_correct=int((pred==y[val]).sum()),validation_rows=len(val),
                    fit_cpu_seconds=time.process_time()-cpu,fit_wall_seconds=time.perf_counter()-wall,
                    warnings=[str(w.message) for w in seen],parameters=model.get_params(deep=False)))
        models[name]=model
        with (destination/(name+'.pkl')).open('xb') as stream:pickle.dump(model,stream,protocol=5)
        np.save(destination/(name+'_validation.npy'),pred,allow_pickle=False)
        print(folder.name,name,records[-1]['validation_correct'],'/',len(val),f"{records[-1]['fit_cpu_seconds']:.3f}s",flush=True)
    best=max(range(4),key=lambda i:records[i]['validation_correct'])
    selected=records[best]['name'];model=models[selected]
    export=export_prepared_svc(model,destination/'selected.spc')
    libsvm_text(model,destination/'selected.libsvm')
    np.savez(destination/'scaler.npz',mean=scaler.mean_,scale=scaler.scale_)
    with (destination/'selected.pkl').open('xb') as stream:pickle.dump(model,stream,protocol=5)
    (destination/'fitting.json').write_text(json.dumps(dict(selected=selected,export=export,models=records),indent=2,default=str)+'\n')
    return records


def seal(panel, source):
    if any(not (panel/task/'models'/'fitting.json').exists() for task in TASKS):
        raise ValueError('not every task has frozen models')
    paths=[]
    for task in TASKS:
        paths += sorted((panel/task/'models').glob('*')) + [panel/task/'split.json',panel/task/'original.npz']
    source_files = [p for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    source_files += [ROOT/'spectra/svm.py', ROOT/'spectra/svm_shared.py', ROOT/'spectra/svm_export.py']
    source_files += [p for p in sorted((ROOT/'spectra/_native/ovo').rglob('*')) if p.suffix in ('.cpp','.hpp')]
    report=dict(schema='spectra.cross_domain.freeze.v1',protocol_commit='2ed5eabfeb8e0f88da9c50fa84ede8de0b05a9da',
                files={str(p.relative_to(panel)):sha(p) for p in paths},
                source={str(p.relative_to(ROOT)):sha(p) for p in source_files})
    with (panel/'TEST_OPENING.json').open('x') as f:json.dump(report,f,indent=2)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--task',choices=TASKS)
    p.add_argument('--seal',action='store_true');a=p.parse_args()
    if a.seal:seal(a.panel,Path(__file__).parent)
    else:fit_task(a.panel/a.task)
