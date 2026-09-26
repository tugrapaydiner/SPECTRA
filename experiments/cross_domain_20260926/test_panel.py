"""Partition, export and harness checks; no held-out panel rows are evaluated."""
import ctypes as C
import io
import json
from pathlib import Path
import sys
import os
import numpy as np
import pytest
from sklearn.svm import SVC
sys.path.insert(0,str(Path(__file__).parent))
from panel_data import split_indices,cap_indices,SEED
from panel_runtime import Timer,Libsvm,Arm,DP
from fit_models import libsvm_text
from spectra.svm_export import export_prepared_svc

BUILD=Path(os.environ.get('SPECTRA_PANEL_BUILD', '.work/cross-domain/build'))

@pytest.mark.parametrize('task',['wine','vehicle','satellite','har','sensorless'])
def test_partition_contract(task):
    n=600;x=np.arange(n*3,dtype=float).reshape(n,3);y=np.tile(np.arange(3),n//3)
    groups=np.repeat(np.arange(20),30)
    kwargs={'boundary':480} if task in ('satellite','har') else {}
    if task=='har':kwargs['groups']=groups
    report=split_indices(x,y,task,**kwargs)
    a,b,c=[set(report[key]) for key in ('fit','validation','test')]
    assert not a&b and not a&c and not b&c
    assert report==split_indices(x,y,task,**kwargs)
    assert sorted(report['timing_test_positions'])==report['timing_test_positions']
    if task in ('satellite','har'):assert c==set(range(480,600))
    if task=='sensorless':
        for label in range(3):
            assert max(i for i in a if y[i]==label)<min(i for i in b if y[i]==label)
            assert max(i for i in b if y[i]==label)<min(i for i in c if y[i]==label)


def test_subject_overlap_rejected():
    with pytest.raises(ValueError,match='subject overlap'):
        split_indices(np.ones((200,3)),np.tile([0,1],100),'har',np.zeros(200),100)


def test_cap_and_duplicate_audit():
    y=np.repeat([0,1,2],4000);ids=cap_indices(np.arange(len(y)),y,4096)
    assert len(ids)==4096 and len(set(ids))==4096 and set(y[ids])=={0,1,2}
    report=split_indices(np.zeros((1200,13)),np.tile([0,1,2],400),'wine')
    assert len(report['duplicate_feature_test_ids'])==len(report['test'])

@pytest.mark.skipif(not (BUILD/'libpanel.so').exists(), reason='build the comparison harness or set SPECTRA_PANEL_BUILD')
@pytest.mark.parametrize('d,c',[(1,2),(13,3),(18,4),(36,6),(48,11),(561,6)])
def test_real_libsvm_roundtrip(d,c,tmp_path):
    rng=np.random.default_rng(d+c);n=max(64,c*12)
    x=rng.normal(size=(n,d));labels=np.asarray(['class_'+str(i) for i in range(c)])
    y=labels[np.arange(n)%c]
    m=SVC(C=2,gamma=1/d).fit(x,y)
    text=tmp_path/'model.txt';srt=tmp_path/'model.spc';libsvm_text(m,text);export_prepared_svc(m,srt)
    timer=Timer(BUILD/'libpanel.so');external=Libsvm(timer,text,d,m.classes_.tolist())
    arm=Arm(srt,BUILD/'spectra/libspectra_svm.so','cert_tables')
    try:
        expected=np.asarray([list(m.classes_).index(v) for v in m.predict(x)])
        assert np.array_equal(external.batch(x),expected)
        assert np.array_equal(arm.batch(x),expected)
        poison=x[:3].copy();poison[-1,-1]=np.nan
        with pytest.raises(ValueError):external.batch(poison)
        with pytest.raises(ValueError):arm.batch(poison)
    finally:external.close();arm.close()
