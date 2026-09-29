"""Policy binding and receipt checks; synthetic files are not scientific evidence."""
import hashlib, json
from pathlib import Path
import pytest
from .deployment import read_policy
from .audit import cp_upper

def policy(tmp_path):
    models={}
    for name in ('fast.spp','strong.srt'):
        data=b'synthetic file; not a trained model';(tmp_path/name).write_bytes(data)
        models[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    doc={'format':'spectra.refinement.policy.v1','experimental':True,'models':models,
         'policies':{k:{'threshold':1.,'accept_fraction':.5} for k in ('primary','strict','loose')}}
    p=tmp_path/'policy.json';p.write_text(json.dumps(doc));return doc,p

def test_valid_identity(tmp_path):
    doc,p=policy(tmp_path);h=hashlib.sha256(p.read_bytes()).hexdigest()
    assert read_policy(tmp_path,h)==(doc,h)

@pytest.mark.parametrize('change',['model','policy','digest','role','path','duplicate','nan'])
def test_policy_changes_rejected(tmp_path,change):
    doc,p=policy(tmp_path);h=hashlib.sha256(p.read_bytes()).hexdigest()
    if change=='model':(tmp_path/'strong.srt').write_bytes(b'replacement')
    elif change=='policy':doc['policies']['primary']['threshold']=0.;p.write_text(json.dumps(doc))
    elif change=='digest':h='0'*64
    elif change=='role':doc['models']['other']=doc['models'].pop('fast.spp');p.write_text(json.dumps(doc));h=None
    elif change=='path':(tmp_path/'strong.srt').unlink();(tmp_path/'strong.srt').symlink_to(tmp_path/'fast.spp')
    elif change=='duplicate':p.write_text(p.read_text().replace('"experimental": true','"experimental": true,"experimental": true'));h=None
    else:p.write_text(p.read_text().replace('"threshold": 1.0','"threshold": NaN'));h=None
    with pytest.raises(ValueError):read_policy(tmp_path,h)

@pytest.mark.parametrize('k,n',[(0,10),(0,3200),(1,1499),(13,3200),(6,887),(99,100),(100,100)])
def test_independent_binomial_inversion(k,n):
    from scipy.stats import beta
    wanted=1. if k==n else float(beta.ppf(1-.05/13,k+1,n-k))
    assert cp_upper(k,n,.05/13)==pytest.approx(wanted,rel=1e-9,abs=1e-12)
