from array import array
import ctypes as C
import math,os
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import binom
from sklearn.svm import SVC
from spectra.svm_export import export_prepared_svc
from experiments.budgeted_prototypes.learning import Settings
from experiments.budgeted_prototypes.export import encode
from experiments.budgeted_prototypes.session import PrototypeSession
from .calibration import upper_bound,calibrate,THRESHOLDS
from .session import RefinementSession
from .build import build

@pytest.fixture(scope='session')
def library(tmp_path_factory):
    return Path(os.environ['REFINEMENT_LIBRARY']) if 'REFINEMENT_LIBRARY' in os.environ else build(tmp_path_factory.mktemp('refine')/'build')
@pytest.fixture
def models(tmp_path):
    rng=np.random.default_rng(61);q=rng.integers(0,16,(60,3),dtype=np.uint8);y=np.arange(60)%3
    p=7;arr={'centers':rng.integers(0,61,(p,3),dtype=np.uint16),
             'weights':np.full((p,3),4,dtype=np.uint16),
             'head':rng.normal(size=(p,3)),'bias':rng.normal(size=3),'classes':np.arange(3)}
    fast=tmp_path/'fast.spp';fast.write_bytes(encode(arr,15,Settings(prototypes=p,gamma=2.)))
    svm=SVC(C=10,gamma=2).fit(q.astype(float)/15,y)
    slow=tmp_path/'slow.srt';export_prepared_svc(svm,slow)
    return fast,slow,q,svm

@pytest.mark.parametrize('threshold',[0.,.25,1.,4.,32.,None])
def test_policy_matches_separate_implementations(models,library,threshold):
    fast,slow,q,svm=models
    with PrototypeSession(fast,library) as base:
        scores=np.frombuffer(base.scores(q),dtype=np.float64).reshape(len(q),3)
        fp=np.asarray(base.predict_buffer(q));parts=np.sort(scores,axis=1);gap=parts[:,-1]-parts[:,-2]
    sp=svm.predict(q.astype(float)/15)
    expected=np.where(gap>=(math.inf if threshold is None else threshold),fp,sp)
    with RefinementSession(fast,slow,library,threshold=threshold,accept_fraction=.4) as w:
        got,stats=w.inspect_buffer(q)
        assert got==expected.tolist()
        assert stats['strong_evaluations']==int(np.sum(gap<(math.inf if threshold is None else threshold)))
        assert w.predict_buffer(q,mode='fast')==fp.tolist()
        assert w.predict_buffer(q,mode='strong')==sp.tolist()
        for mode in ('calibrated','fast','strong','blind'):
            assert w.predict_buffer(np.empty((0,3),dtype=np.uint8),mode=mode)==[]
        assert w.predict_buffer(q,mode='blind')==w.predict_buffer(q.copy(),mode='blind')
    with pytest.raises(ValueError):w.predict_buffer(q)

@pytest.mark.parametrize('bad',[float('nan'),-1,True,'foo'])
def test_invalid_threshold(models,library,bad):
    f,s,q,m=models
    with pytest.raises(ValueError):RefinementSession(f,s,library,threshold=bad)
@pytest.mark.parametrize('bad',[float('nan'),-1,True,1.1])
def test_invalid_random_fraction(models,library,bad):
    f,s,q,m=models
    with pytest.raises(ValueError):RefinementSession(f,s,library,threshold=1.,accept_fraction=bad)

def test_confidence_bound_coverage_small_binomial():
    # Exact enumerated frequentist coverage, not Monte Carlo model evidence.
    for n in (10,30,100):
        for p in (.01,.1,.5,.9):
            failed=[k for k in range(n+1) if upper_bound(k,n,.05/13)<p]
            probability=float(binom.pmf(failed,n,p).sum())
            assert probability<=.05/13+1e-12

def test_calibration_targets_added_harm_not_all_fast_errors():
    n=2000;gap=np.full(n,4.);y=np.zeros(n,dtype=int);fast=y.copy();slow=y.copy()
    fast[:20]=1;slow[:15]=1
    r=calibrate(gap,fast,slow,y)
    assert r['rows'][0]['harmful']==5
    policy=r['policies']['0.01']
    assert policy['threshold']==0.
    assert policy['upper_added_harm']<=.01
    fast[:]=1;slow[:]=0
    r=calibrate(gap,fast,slow,y)
    assert r['policies']['0.005']['accepted']==0

def test_too_little_calibration_falls_back():
    r=calibrate(np.ones(3),np.zeros(3),np.zeros(3),np.zeros(3))
    assert all(p['always_strong'] for p in r['policies'].values())

@pytest.mark.parametrize('k,n,d',[(-1,4,.05),(5,4,.05),(0,0,.05),(True,4,.05),(0,4,1),(0,4,0)])
def test_bad_bound_arguments(k,n,d):
    with pytest.raises(ValueError):upper_bound(k,n,d)

def test_native_rejects_late_bad_row_before_write(models,library):
    f,s,q,m=models
    with RefinementSession(f,s,library,threshold=1.) as w:
        inp=(C.c_uint8*6)(0,0,0,0,0,16);out=(C.c_int*2)(43,44);stats=(C.c_uint64*3)()
        assert w._lib.sr_run(w._handle,inp,2,3,0,out,stats,3)!=0
        assert list(out)==[43,44]
        for x in (bytes([0,0,0]),np.zeros((1,3)),np.zeros((2,4),dtype=np.uint8)):
            with pytest.raises(ValueError):w.predict_buffer(x)

def test_reentrant_close_and_buffer_release(models,library):
    f,s,q,m=models;w=RefinementSession(f,s,library,threshold=1.)
    fn=w._lib.sr_run
    def close_inside(*args):w.close();assert w._handle;return fn(*args)
    x=array('B',[0,0,0]);w._lib.sr_run=close_inside
    try:assert len(w.predict_buffer(x))==1
    finally:w._lib.sr_run=fn;w.close()
    x.extend([1,1,1]);assert not w._handle
