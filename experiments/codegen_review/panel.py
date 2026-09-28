"""Fixed synthetic compatibility models; output bytes, not predictive quality."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import warnings


def prepare(out):
    import numpy as np
    import sklearn
    from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
    from sklearn.svm import SVC, NuSVC, SVR, NuSVR, OneClassSVM
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.neural_network import MLPClassifier
    out.mkdir(parents=True, exist_ok=False)
    rng=np.random.default_rng(20260928);x=rng.normal(size=(72,7));y=np.arange(72)%3
    regression=np.sin(x[:,0])+x[:,2]*.3
    estimators={
      'linear':(LinearRegression(),regression), 'ridge':(Ridge(),regression),
      'logistic':(LogisticRegression(max_iter=200),y),
      'svc_binary':(SVC(C=2),y%2), 'svc_rbf':(SVC(C=2),y),
      'svc_linear':(SVC(kernel='linear'),y),'svc_poly':(SVC(kernel='poly',degree=2),y),
      'svc_sigmoid':(SVC(kernel='sigmoid'),y),'nu_svc':(NuSVC(nu=.3),y),
      'svr':(SVR(),regression),'nu_svr':(NuSVR(),regression),'one_class':(OneClassSVM(),None),
      'tree_class':(DecisionTreeClassifier(max_depth=4,random_state=7),y),
      'tree_reg':(DecisionTreeRegressor(max_depth=4,random_state=7),regression),
      'forest_class':(RandomForestClassifier(n_estimators=5,max_depth=4,random_state=7,n_jobs=1),y),
      'forest_reg':(RandomForestRegressor(n_estimators=5,max_depth=4,random_state=7,n_jobs=1),regression),
      'extra_class':(ExtraTreesClassifier(n_estimators=5,max_depth=4,random_state=7,n_jobs=1),y),
      'extra_reg':(ExtraTreesRegressor(n_estimators=5,max_depth=4,random_state=7,n_jobs=1),regression),
      'unsupported_mlp':(MLPClassifier(hidden_layer_sizes=(4,),max_iter=5,random_state=7),y),
    }
    records={};start=time.process_time()
    for name,(model,target) in estimators.items():
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always');model.fit(x,target)
        raw=pickle.dumps(model,protocol=4);(out/(name+'.pkl')).write_bytes(raw)
        records[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'warnings':[str(w.message) for w in captured]}
    report={'models':records,'seed':20260928,'numpy':np.__version__,'sklearn':sklearn.__version__,'cpu_seconds':time.process_time()-start,
            'scope':'synthetic export compatibility only; no accuracy measurement or benchmark-model fitting'}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');return report


def evaluate(models,out):
    import m2cgen
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((models/'manifest.json').read_text())
    languages=sorted(n[len('export_to_'):] for n in dir(m2cgen) if n.startswith('export_to_'))
    records={};sys.setrecursionlimit(12000)
    for name,entry in manifest['models'].items():
        raw=(models/(name+'.pkl')).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==entry['sha256']
        model=pickle.loads(raw) # Only locally constructed, hash-bound test objects.
        for language in languages:
            key=name+'--'+language
            try:
                code=getattr(m2cgen,'export_to_'+language)(model).encode('utf-8')
            except Exception as error:
                records[key]={'status':'ERROR','type':type(error).__name__,'message':str(error)}
            else:
                (out/(key+'.txt')).write_bytes(code)
                records[key]={'status':'OK','bytes':len(code),'sha256':hashlib.sha256(code).hexdigest()}
    result={'languages':languages,'models':list(manifest['models']),'results':records,'package':str(Path(m2cgen.__file__).resolve()),
            'package_sha256':{str(f.relative_to(Path(m2cgen.__file__).parent)):hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in sorted(Path(m2cgen.__file__).parent.rglob('*.py'))}}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'outputs':sum(r['status']=='OK' for r in records.values()),'errors':sum(r['status']=='ERROR' for r in records.values()),'languages':len(languages)}))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['prepare','evaluate'])
    p.add_argument('--models',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.operation=='prepare': print(json.dumps(prepare(a.out),indent=2))
    else:evaluate(a.models,a.out)
