from itertools import product
import json
from pathlib import Path
import pytest
from data.cnf import CNF
from experiments.residual_patch.data import assignment,cases,formula

@pytest.fixture
def config():
    c=json.loads(Path(__file__).with_name('config.json').read_text())
    c.update(train_per_stratum=2,validation_per_stratum=2,development_per_stratum=2)
    return c

@pytest.mark.parametrize('family',['uniform','community'])
def test_deterministic_unfiltered_construction(config,family):
    p=formula(64,2309,family,config)
    assert p==formula(64,2309,family,config)
    assert len(p.clauses)==64*21//5
    assert len(set(p.clauses))==len(p.clauses)
    assert all(len({abs(v) for v in c})==3 for c in p.clauses)
    assert all(len(c)==3 for c in p.clauses)


def test_complete_split_inventory_and_input_identity(config):
    combined=[]
    for split in ('train','validation','development'):
        items=cases(config,split)
        assert len(items)==8
        assert items==cases(config,split)
        combined.extend(items)
    assert len({c['id'] for c in combined})==len(combined)
    assert len({c['seed'] for c in combined})==len(combined)
    assert len({c['normalized_sha256'] for c in combined})==len(combined)
    for c in combined:
        p=CNF.from_record(c['formula'])
        assert c['ordered_sha256']==p.sha256()


def test_no_confirmation_route(config):
    with pytest.raises(ValueError):cases(config,'confirmation')
    with pytest.raises(ValueError):formula(64,1,'invented',config)


def test_initialization_rng_boundaries():
    assert assignment(64,234)==assignment(64,234)
    assert assignment(64,234)!=assignment(64,235)
    assert all(type(x) is bool for x in assignment(64,234))
    with pytest.raises(ValueError):assignment(1,-1)
