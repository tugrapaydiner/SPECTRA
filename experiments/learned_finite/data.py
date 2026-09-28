"""Prepare the two prospective datasets; never fit/evaluate a classifier here."""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

CAR_CATEGORIES = [ ['vhigh','high','med','low'], ['vhigh','high','med','low'],
                   ['2','3','4','5more'], ['2','4','more'], ['small','med','big'], ['low','med','high'] ]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)

def groups(x,y):
    _,first,inverse=np.unique(x,axis=0,return_index=True,return_inverse=True)
    gy=y[first]
    if not np.array_equal(gy[inverse],y): raise ValueError('identical inputs with conflicting labels')
    return inverse,gy

def split_groups(x,y,fraction,seed):
    inverse,gy=groups(x,y)
    left,right=train_test_split(np.arange(len(gy)),test_size=fraction,stratify=gy,random_state=seed)
    return np.flatnonzero(np.isin(inverse,left)),np.flatnonzero(np.isin(inverse,right))

def prepare(inputs,out):
    out.mkdir(parents=True,exist_ok=False)
    acquisition=json.loads((inputs/'ACQUISITION.json').read_text())
    summary={}
    for name in ('semeion','car'):
        archive=inputs/(name+'.zip')
        if sha(archive)!=acquisition['data'][name]['sha256']: raise ValueError('acquisition hash mismatch')
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            raw=z.read(name+'.data')
            if name=='semeion':
                values=np.loadtxt(raw.decode().splitlines())
                if values.shape!=(1593,266) or not np.isin(values,[0,1]).all(): raise ValueError('unexpected Semeion geometry')
                if not (values[:,256:].sum(1)==1).all(): raise ValueError('invalid one-hot labels')
                x=values[:,:256].astype(np.uint8);y=values[:,256:].argmax(1)
                feature_groups=np.array([(i//16//4)*4+(i%16//4) for i in range(256)],np.int32)
                labels=list(range(10));encoder={'kind':'binary_pixels','shape':[16,16]}
            else:
                rows=[line.split(',') for line in raw.decode('ascii').splitlines() if line]
                if len(rows)!=1728 or any(len(r)!=7 for r in rows): raise ValueError('unexpected Car geometry')
                labels=sorted({r[-1] for r in rows});y=np.array([labels.index(r[-1]) for r in rows])
                x=np.zeros((len(rows),sum(map(len,CAR_CATEGORIES))),np.uint8)
                feature_groups=[];offset=0
                for g,cat in enumerate(CAR_CATEGORIES):
                    for i,row in enumerate(rows):x[i,offset+cat.index(row[g])]=1
                    feature_groups += [g]*len(cat);offset+=len(cat)
                feature_groups=np.array(feature_groups,np.int32)
                encoder={'kind':'one_hot','categories':CAR_CATEGORIES}
            names=z.read(name+'.names') if name+'.names' in z.namelist() else b''
        dev,test=split_groups(x,y,.5,20260928)
        folder=out/name;folder.mkdir()
        np.savez_compressed(folder/'development.npz',x=x[dev],y=y[dev],original_indices=dev,feature_groups=feature_groups)
        np.savez_compressed(folder/'holdout.npz',x=x[test],y=y[test],original_indices=test)
        (folder/'source.names').write_bytes(names)
        splits={}
        for seed in (101,202,303):
            tr,va=split_groups(x[dev],y[dev],.25,seed)
            splits[str(seed)]={'fit':tr.tolist(),'validation':va.tolist()}
        record={'dataset':name,'source_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':sha(archive),
                'rows':len(x),'unique_inputs':len(np.unique(x,axis=0)),'features':x.shape[1],
                'development_rows':len(dev),'holdout_rows':len(test),'labels':labels,'encoder':encoder,
                'development_sha256':sha(folder/'development.npz'),'holdout_sha256':sha(folder/'holdout.npz'),
                'inner_splits':splits,'outer_seed':20260928,
                'scope':'new local test split; duplicate input groups separated; no writer IDs or production generalization claim'}
        write_json(folder/'manifest.json',record);summary[name]=record
    write_json(out/'PREPARATION.json',summary)
    print(json.dumps({n:{k:v for k,v in r.items() if k not in ('inner_splits','encoder')} for n,r in summary.items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.inputs,a.out)
