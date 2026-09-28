"""Protocol and freeze contracts; synthetic records are never experiment evidence."""
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('admission_study',Path(__file__).with_name('study.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

def record(c,score,status='COMPLETE'):
    return {'job':{'config':c},'status':status,'fit':{'validation_accuracy':score}}

def test_candidate_inventory():
    c=s.configs()
    assert len(c)==18
    assert sum(x['family']=='linear' for x in c)==4
    assert sum(x['family']=='svm' for x in c)==6
    assert sum(x['family']=='mlp' for x in c)==6
    assert sum(x['family']=='tree' for x in c)==2
    assert {x.get('seed') for x in c if x['family']=='mlp'}=={101,202,303}

def test_selection_never_picks_a_lucky_seed():
    rows=[]
    for c in s.configs():
        score=.9
        if c['family']=='mlp':
            score=.95 if c['width']==64 else {101:1.,202:.8,303:.8}[c['seed']]
        if c['family']=='svm':score=.97
        rows.append(record(c,score))
    selected=s.choose(rows)
    assert selected['configs']['mlp']['width']==64
    assert 'seed' not in selected['configs']['mlp']
    assert selected['configs']['linear']['C']==.01 # declared first tie
    assert selected['configs']['svm']=={'family':'svm','C':1.,'gamma_multiplier':.25}
    assert selected['svm_admitted_validation']

def test_failed_seed_not_removed_from_mean():
    rows=[record(c,.9) for c in s.configs()]
    rows[10]['status']='TIMEOUT'
    assert s.choose(rows)['configs']['mlp']['width']==256

def test_no_successful_family_fails_closed():
    rows=[record(c,.9,'ERROR' if c['family']=='svm' else 'COMPLETE') for c in s.configs()]
    with pytest.raises(ValueError,match='no completed svm'):s.choose(rows)

@pytest.mark.parametrize('text',['1 1:2 1:3','1 0:1','1 129:1','7 1:0','1 1:nan','1 1:inf'])
def test_invalid_sparse_data(text):
    with pytest.raises(ValueError):s.sparse_rows(text.encode())

def test_sparse_feature_indices_not_file_order():
    x,y=s.sparse_rows(b'2 128:4 1:3\n')
    assert x.shape==(1,128) and y.tolist()==[2]
    assert x[0,0]==3 and x[0,-1]==4

def test_saved_data_safe_roundtrip(tmp_path):
    s.write_data(tmp_path,'x',[[1,2],[3,4]],[1,2],[9,10])
    x,y,g=s.load(tmp_path/'x.npz')
    assert x.dtype==np.float64 and g.tolist()==[9,10] and y.tolist()==[1,2]

def freeze_fixture(root):
    s.save(root/'SELECTED.json',{'synthetic':True})
    splits={}
    for task in s.TASKS:
        d=root/'data'/task;d.mkdir(parents=True)
        s.write_data(d,'test',[[1,2]],[1],[0])
        splits[task]={'test':{'sha256':s.sha(d/'test.npz')}}
    s.save(root/'DATA.json',{'splits':splits})
    (root/'model').write_bytes(b'fixture, not an executable pickle')
    doc={'selected_sha256':s.sha(root/'SELECTED.json'),'data_sha256':s.sha(root/'DATA.json'),'files':{'model':s.sha(root/'model')}}
    s.save(root/'MODEL_FREEZE.json',doc)
    return doc

@pytest.mark.parametrize('path',['SELECTED.json','DATA.json','model','data/isolet/test.npz'])
def test_freeze_rejects_mutation(tmp_path,path):
    freeze_fixture(tmp_path);assert s.verify_freeze(tmp_path)
    with (tmp_path/path).open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError):s.verify_freeze(tmp_path)

def test_freeze_refuses_escape(tmp_path):
    doc=freeze_fixture(tmp_path);doc['files']={'../secret':'0'*64}
    (tmp_path/'MODEL_FREEZE.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError):s.verify_freeze(tmp_path)
