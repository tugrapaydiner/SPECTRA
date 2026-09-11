import gzip
import json
from pathlib import Path
import copy
import numpy as np
import pytest
from data.cnf import CNF
from .rank import pairs_from_snapshots
from .audit import records,validate_row
from .followup import run_variant
from .runtime import W


def test_pairwise_labels_are_within_state_and_state_balanced():
    target=np.array([0,0.5,1,0,1,1,1])
    snaps=[{'candidate_start':0,'candidate_count':3},{'candidate_start':3,'candidate_count':2},{'candidate_start':5,'candidate_count':2}]
    l,r,w,count=pairs_from_snapshots(target,snaps)
    assert count==2 and np.all(target[l]>target[r])
    assert np.isclose(w[l<3].sum(),1) and np.isclose(w[l>=3].sum(),1)
    assert not np.any(l>=5) and not np.any(r>=5)


def test_pairwise_training_rejects_no_signal():
    with pytest.raises(ValueError):pairs_from_snapshots(np.zeros(3),[{'candidate_start':0,'candidate_count':3}])

@pytest.mark.parametrize('raw',['{"x": 1, "x": 2}\n','{"x": NaN}\n','{"x": Infinity}\n','\n','{"x":1}\n'])
def test_strict_jsonl_rejects_ambiguity(tmp_path,raw):
    p=tmp_path/'x.gz'
    with gzip.open(p,'wt') as f:f.write(raw)
    with pytest.raises((ValueError,json.JSONDecodeError)):list(records(p))


def fixture_row():
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    p=CNF(1,((1,),));c={'id':'unit','formula':p.record(),'ordered_sha256':p.sha256()}
    banks={'bce':{},'rank':{}}
    row=run_variant(c,'probsat',1901,17001,0,cfg,banks,True);row['round']=0
    return c,row,cfg


def test_real_native_record_has_independent_valid_witness():
    c,r,cfg=fixture_row();validate_row(c,r,cfg,work=True)

@pytest.mark.parametrize('corruption',['witness','boolean','hash','counter','scope','variant','budget'])
def test_semantic_tampering_is_rejected_even_without_hashes(corruption):
    c,r,cfg=fixture_row();r=copy.deepcopy(r)
    if corruption=='witness':r['witness']=(False,)
    elif corruption=='boolean':r['within_budget']=1
    elif corruption=='hash':r['ordered_sha256']='0'*64
    elif corruption=='counter':r['work']['flips']=-1
    elif corruption=='scope':r['setup_ns']=r['complete_ns']+1
    elif corruption=='variant':r['patch_interval']=32
    else:r['budget_ms']=2
    with pytest.raises(ValueError):validate_row(c,r,cfg,work=True)
