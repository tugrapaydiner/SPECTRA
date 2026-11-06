"""Reconstruct frozen public SVM fixtures; abort unless prior bytes match exactly.

This performs the already specified small fitting jobs solely to reconstruct an
existing corpus for remote tests. No new model selection, accuracy claim, downloads
or replacement of old evidence. Inference targets receive only inert output files.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
TASKS=('wine','wdbc','chess','penguins','titanic','zoo')
SEEDS=(101,202,303)
OPENML={'chess':3,'penguins':42585,'titanic':40945,'zoo':62}


def sha(raw): return hashlib.sha256(raw).hexdigest()
def json_bytes(value): return (json.dumps(value,ensure_ascii=True,separators=(',',':'),allow_nan=False)+'\n').encode('ascii')


def dataset(task, package, expected):
    import numpy as np
    import pandas as pd
    from sklearn.externals import _arff
    if task in ('wine','wdbc'):
        name={'wine':'wine_data.csv','wdbc':'breast_cancer.csv'}[task]
        raw=(package/'datasets/data'/name).read_bytes()
        if sha(raw)!=expected['files'][task+'.csv']: raise ValueError('original CSV differs')
        entries=list(csv.reader(raw.decode('utf-8').splitlines()))
        first=entries[0]; matrix=np.asarray(entries[1:],dtype=np.float64)
        if len(matrix)!=int(first[0]): raise ValueError('CSV row count differs')
        labels=np.array([first[2+int(v)] for v in matrix[:,-1]])
        frame=pd.DataFrame(matrix[:,:-1],columns=[f'feature_{i}' for i in range(matrix.shape[1]-1)])
        return frame,labels,list(frame.columns),[]
    did=OPENML[task]; path=package/f'datasets/tests/data/openml/id_{did}'
    raw=next(path.glob('*.arff.gz')).read_bytes()
    if sha(raw)!=expected['files'][task+'.arff.gz']: raise ValueError('original ARFF differs')
    metadata=gzip.decompress((path/f'api-v1-jd-{did}.json.gz').read_bytes())
    if sha(metadata)!=expected['files'][task+'.metadata.json']: raise ValueError('original metadata differs')
    decoded=gzip.decompress(raw)
    if hashlib.md5(decoded).hexdigest()!=json.loads(metadata)['data_set_description']['md5_checksum']:
        raise ValueError('ARFF does not match its public metadata')
    parsed=_arff.loads(decoded.decode('utf-8'))
    frame=pd.DataFrame(parsed['data'],columns=[a[0] for a in parsed['attributes']])
    if len(frame)!=expected['sources'][task]['expected_rows']: raise ValueError('original row count differs')
    target={'chess':'class','penguins':'species','titanic':'survived','zoo':'type'}[task]
    labels=frame.pop(target).to_numpy(dtype=str)
    if task=='titanic':
        frame=frame[['pclass','sex','age','sibsp','parch','fare','embarked']]
        numeric=['age','sibsp','parch','fare']; categorical=['pclass','sex','embarked'];labels=labels.astype(int)
    elif task=='chess': numeric=[];categorical=list(frame.columns)
    elif task=='penguins':
        frame.loc[frame.sex=='_','sex']=None
        numeric=['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g'];categorical=['island','sex']
    else:
        frame=frame.drop(columns=['animal']);numeric=['legs'];categorical=[c for c in frame if c!='legs']
    for name in numeric: frame[name]=pd.to_numeric(frame[name],errors='raise').astype(float)
    for name in categorical: frame[name]=frame[name].map(lambda v:'<missing>' if v is None else str(v))
    return frame,labels,numeric,categorical


def reconstruct(destination):
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler,OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from threadpoolctl import threadpool_limits
    from spectra.svm_pipeline_export import export_pipeline
    if (np.__version__,pd.__version__,sklearn.__version__)!=('2.3.5','2.2.3','1.8.0'):
        raise ValueError('requires pinned NumPy2.3.5/pandas2.2.3/sklearn1.8.0')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    expected=json.loads(Path(__file__).with_name('data_manifest.json').read_text())
    corpus=json.loads(Path(__file__).with_name('corpus_manifest.json').read_text())
    package=Path(sklearn.__file__).parent
    started=time.perf_counter();cpu=time.process_time();results=[]
    with threadpool_limits(limits=1):
        for task in TASKS:
            frame,y,numeric,categorical=dataset(task,package,expected)
            for seed in SEEDS:
                train,test=train_test_split(np.arange(len(frame)),test_size=.25,stratify=y,random_state=seed)
                parts=[]
                if numeric: parts.append(('numeric',Pipeline([('missing',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric))
                if categorical: parts.append(('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical))
                prep=ColumnTransformer(parts,sparse_threshold=0.)
                trainx=np.ascontiguousarray(prep.fit_transform(frame.iloc[train]),dtype=np.float64)
                model=SVC(C=10.,gamma='scale',kernel='rbf',probability=False,break_ties=False,cache_size=128).fit(trainx,y[train])
                name=f'{task}-{seed}';folder=destination/name
                export_pipeline(prep,model,folder)
                transformed=np.ascontiguousarray(prep.transform(frame.iloc[test]),dtype=np.float64)
                rows=frame.iloc[test].astype(object).where(pd.notna(frame.iloc[test]),None).values.tolist()
                (folder/'input.jsonl').write_bytes(b''.join(json_bytes(row) for row in rows))
                (folder/'transformed.f64').write_bytes(transformed.tobytes())
                (folder/'expected.json').write_bytes(json_bytes(model.predict(transformed).tolist()))
                entry=corpus['models'][name]
                if len(rows)!=entry['rows']: raise ValueError('retained row count differs')
                for filename,identity in entry['files'].items():
                    value=(folder/filename).read_bytes()
                    if len(value)!=identity['bytes'] or sha(value)!=identity['sha256']:
                        raise ValueError(f'reconstruction differs from frozen corpus: {name}/{filename}')
                results.append({'model':name,'rows':len(rows),'files_matched':len(entry['files'])})
                print(name,'exact original bytes matched',flush=True)
    report={'status':'PASS','models':results,'cpu_seconds':time.process_time()-cpu,
            'wall_seconds':time.perf_counter()-started,'platform':platform.platform(),'python':sys.version,
            'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
            'manifest_sha256':sha(Path(__file__).with_name('corpus_manifest.json').read_bytes()),
            'generator_sha256':sha(Path(__file__).read_bytes()),
            'scope':'exact reconstruction of previous fixed models and data; not a new training/accuracy study'}
    (destination/'RECONSTRUCTION.json').write_bytes(json_bytes(report))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();print(json.dumps(reconstruct(args.out),indent=2))
