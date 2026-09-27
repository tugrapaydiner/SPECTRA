"""Opt-in fused execution: complete labels, bounded tiles and live-pointer safety."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import struct
import weakref
import zlib

import pytest
from spectra.svm_pipeline import PreparedPipeline, SCHEMA
from spectra.svm_preprocess_native import build_preprocessor
from spectra.svm_shared import SHARED_SCHEDULES


@pytest.fixture(scope='module')
def prelib(tmp_path_factory):
    p=os.environ.get('SPECTRA_PREPROCESS_LIBRARY')
    return Path(p) if p else build_preprocessor(tmp_path_factory.mktemp('fused-pre')/'build')


def fixture_folder(path, features=5, categories=True):
    path.mkdir()
    labels=['left','middle','right'];classes=3
    # Fully specified valid model with unequal signed coefficients and biases.
    vectors=[(i+1)/(features+7) for i in range(features*3)]
    coeff=[.5,-.125,.25, -.5,.75,-.25]
    payload=struct.pack('<dIII',.5,1,1,1)
    values=vectors+coeff+[.1,-.2,.3]
    payload+=struct.pack('<'+'d'*len(values),*values)
    meta=json.dumps({'labels':labels}).encode();body=meta+payload
    model=struct.pack('<8sIIIIII',b'SPCSVM02',3,3,features,len(meta),len(payload),zlib.crc32(body))+body
    (path/'model.srt').write_bytes(model)
    if categories:
        columns=['x','kind','z']
        operations=[{'kind':'onehot','column':1,'categories':['α','b',''],'unknown':'ignore'},
                    {'kind':'numeric','column':2,'fill':(2.).hex(),'mean':(-0.).hex(),'scale':(3.).hex()},
                    {'kind':'numeric','column':0,'fill':(-1.).hex(),'mean':(1.).hex(),'scale':(.3).hex()}]
    else:
        columns=[f'x{i}' for i in range(features)]
        operations=[{'kind':'numeric','column':i,'fill':(0.).hex(),'mean':(0.).hex(),'scale':(1.).hex()}
                    for i in range(features)]
    plan={'schema':SCHEMA,'columns':columns,'features':features,
          'model_sha256':hashlib.sha256(model).hexdigest(),'operations':operations}
    (path/'preprocessing.json').write_text(json.dumps(plan))
    return path


@pytest.fixture
def pipeline(tmp_path,library,prelib):
    with PreparedPipeline(fixture_folder(tmp_path/'p'),library,preprocessor_library=prelib) as p:
        yield p


@pytest.mark.parametrize('tile',[1,3,8,128])
@pytest.mark.parametrize('schedule',list(SHARED_SCHEDULES))
def test_same_complete_labels_and_fresh_lists(pipeline,tile,schedule):
    rows=[[1.,'α',None],[None,'b',math.nan],[7,'unknown',-0.],[-1.,'',1e150]]*37
    with pipeline.session() as w:
        expected=w.predict_many(rows,schedule='exhaustive')
        a=w.predict_fused(rows,schedule=schedule,hint=0,tile_rows=tile)
        b=w.predict_fused(tuple(map(tuple,rows)),schedule=schedule,tile_rows=tile)
        assert a==b==expected and a is not b
        assert w.predict_fused([])==[]
        assert w.predict_with_certificate(rows[0]).label==expected[0]


@pytest.mark.parametrize('value',[float('inf'),-float('inf'),True,'1',1j,10**999])
def test_late_invalid_input_raises_and_worker_reusable(pipeline,value):
    rows=[[1.,'b',2.] for _ in range(257)];rows[-1][0]=value
    with pipeline.session() as w:
        with pytest.raises(ValueError):w.predict_fused(rows,tile_rows=8)
        assert w.predict_fused([[1.,'b',2.]])==w.predict_many([[1.,'b',2.]])


@pytest.mark.parametrize('tile',[0,129,-1,True,1.5,'4'])
def test_bad_tile_rejected(pipeline,tile):
    with pipeline.session() as w:
        with pytest.raises(ValueError):w.predict_fused([],tile_rows=tile)


def test_schema_and_iterator_fallback_consumption(pipeline):
    visited=[]
    def source():
        for i in range(13):
            visited.append(i);yield [i,'b',2.]
    with pipeline.session() as w:
        assert w.predict_fused(source())==w.predict_many([[i,'b',2.] for i in range(13)])
        assert visited==list(range(13))
        with pytest.raises(ValueError):w.predict_fused([],columns=['kind','x','z'])
        with pytest.raises(ValueError):w.predict_fused([[1.,'b']])
        with pytest.raises(ValueError):w.predict_fused([],schedule='unsafe')
        with pytest.raises(ValueError):w.predict_fused([],hint=True)


def test_custom_conversion_happens_once(pipeline):
    calls=[]
    class Custom(float):
        def __float__(self):calls.append(1);return 7.
    with pipeline.session() as w:
        result=w.predict_fused([[Custom(2.),'b',Fraction(1,3)]])
        assert calls==[1]
        assert result==w.predict_many([[7.,'b',1/3]])


def test_no_compiled_component_rejected(tmp_path,library):
    with PreparedPipeline(fixture_folder(tmp_path/'p'),library) as p,p.session() as w:
        with pytest.raises(ValueError,match='compiled'):w.predict_fused([[1.,'b',2.]])


@pytest.mark.parametrize('features',[1,17,512,4096])
def test_feature_tile_geometry(tmp_path,library,prelib,features):
    with PreparedPipeline(fixture_folder(tmp_path/'p',features,False),library,preprocessor_library=prelib) as p,p.session() as w:
        rows=[[i/5.]*features for i in range(139)]
        assert w.predict_fused(rows)==w.predict_many(rows)


def test_parallel_workers_survive_owner_close(pipeline):
    workers=[pipeline.session() for _ in range(4)]
    rows=[[i/3.,'b',2.] for i in range(513)]
    expected=workers[0].predict_many(rows)
    pipeline.close()
    try:
        with ThreadPoolExecutor(4) as pool:
            answers=list(pool.map(lambda w:w.predict_fused(rows,tile_rows=8),workers))
        assert answers==[expected]*4
    finally:
        for w in workers:w.close()
    with pytest.raises(ValueError):workers[0].predict_fused(rows)


def test_same_worker_mixed_calls_serialized(pipeline):
    with pipeline.session() as w:
        rows=[[1.,'α',2.]]*17;expected=w.predict_many(rows)
        def call(i):
            return w.predict_fused(rows) if i%2 else w.predict_many(rows)
        with ThreadPoolExecutor(4) as pool:
            assert list(pool.map(call,range(40)))==[expected]*40


def test_binding_does_not_keep_unreachable_worker_alive(pipeline):
    w=pipeline.session();ref=weakref.ref(w._worker)
    w.predict_fused([[1.,'b',2.]]);w.close();del w;gc.collect()
    assert ref() is None


def test_reentrant_close_is_deferred_and_reentry_rejected(pipeline,monkeypatch):
    # Deterministic simulation of a same-thread signal handler. RLock alone is
    # insufficient: it would allow freeing the pointer during a tiled C call.
    w=pipeline.session();original=pipeline.preprocessor._module.predict_fused
    def signal_like(*args):
        w.close();assert w._worker._handle
        with pytest.raises(ValueError,match='reentrant'):w.predict_many([[1.,'b',2.]])
        with pytest.raises(ValueError,match='reentrant'):w.predict_fused([])
        return original(*args)
    monkeypatch.setattr(pipeline.preprocessor._module,'predict_fused',signal_like)
    assert len(w.predict_fused([[1.,'b',2.]]*13,tile_rows=1))==13
    assert w._worker._handle is None


@pytest.mark.skipif(not hasattr(signal,'setitimer'),reason='requires POSIX signals')
@pytest.mark.parametrize('action',['close','interrupt','mutate'])
def test_real_signal_during_tiles(pipeline,action):
    w=pipeline.session();rows=[[1.,'b',2.] for _ in range(12000)];seen=[]
    expected=w.predict_many(rows[:1])[0]
    def handler(*_):
        seen.append(w._worker._native_active)
        if action=='close':w.close()
        elif action=='interrupt':raise KeyboardInterrupt('test interruption')
        else:rows.clear()
    previous=signal.signal(signal.SIGALRM,handler)
    try:
        signal.setitimer(signal.ITIMER_REAL,.002)
        if action=='interrupt':
            with pytest.raises(KeyboardInterrupt):w.predict_fused(rows,tile_rows=1)
            assert w.predict_fused([[1.,'b',2.]])==[expected]
        elif action=='mutate':
            with pytest.raises(ValueError,match='mutated'):w.predict_fused(rows,tile_rows=1)
            assert w.predict_fused([[1.,'b',2.]])==[expected]
        else:
            assert w.predict_fused(rows,tile_rows=1)==[expected]*12000
            assert not w._worker._handle
        assert seen==[True]
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous);w.close()


def test_close_before_busy_flag_cannot_use_stale_pointer(pipeline):
    import inspect
    import sys
    from spectra.svm_lifetime import _NativeOwner
    w=pipeline.session();w.predict_fused([[1.,'b',2.]])
    source,start=inspect.getsourcelines(_NativeOwner._operation.__wrapped__)
    target=start+next(i for i,s in enumerate(source) if 'self._native_active = True' in s)
    seen=[]
    def tracer(frame,event,arg):
        if event=='line' and frame.f_code is _NativeOwner._operation.__wrapped__.__code__ and frame.f_lineno==target:
            seen.append(True);w.close()
        return tracer
    previous=sys.gettrace();sys.settrace(tracer)
    try:
        with pytest.raises(ValueError,match='closed'):w.predict_fused([[1.,'b',2.]])
    finally:sys.settrace(previous)
    assert seen==[True] and not w._worker._native_active and not w._worker._handle
