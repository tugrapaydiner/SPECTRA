"""Shared ownership, generic model fidelity and ordering-only prototype checks."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import gc
import itertools
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest
from spectra.svm import Session, verify_certificate
from spectra.svm_shared import PreparedModel, SHARED_SCHEDULES, _decode
from spectra.svm_export import export_prepared_svc


def encoded(path, features=16, classes=3, labels=None, count_each=1):
    supports=classes*count_each
    values=[0.]*(features*supports + (classes-1)*supports + classes*(classes-1)//2)
    raw=struct.pack('<d',.5)+struct.pack('<'+'I'*classes,*([count_each]*classes))
    raw+=struct.pack('<'+'d'*len(values),*values)
    meta=json.dumps({'labels':list(range(classes)) if labels is None else labels}).encode()
    body=meta+raw
    path.write_bytes(struct.pack('<8sIIIIII',b'SPCSVM02',classes,supports,features,len(meta),len(raw),zlib.crc32(body))+body)
    return path


@pytest.mark.parametrize('features',[1,3,16,17,64,257])
@pytest.mark.parametrize('labels',[[10,30],[-99,0,21],['ant','tür','z']])
def test_sklearn_generic_roundtrip(tmp_path,library,features,labels):
    np=pytest.importorskip('numpy');SVC=pytest.importorskip('sklearn.svm').SVC
    rng=np.random.default_rng(features+200)
    x=rng.integers(-3,4,size=(60,features)).astype('float64')/8
    y=np.asarray([labels[i%len(labels)] for i in range(60)])
    svc=SVC(C=3,gamma=.5/features).fit(x,y)
    path=tmp_path/'model.srt';info=export_prepared_svc(svc,path)
    probe=np.concatenate((x,rng.normal(size=(30,features))))
    assert info['features']==features
    for tables in (False,True):
        with PreparedModel(path,library,tables=tables,input_dtype='float64') as m,m.session() as w:
            assert m.labels==tuple(svc.classes_.tolist())
            expected=svc.predict(probe).tolist()
            for schedule in SHARED_SCHEDULES:
                assert w.predict_many(probe,schedule=schedule)==expected
            for row,label in zip(probe[:5],expected):
                p=w.predict_with_certificate(row,schedule='cost_aware',hint=len(labels)-1)
                assert p.label==label and verify_certificate(len(labels),p.class_index,p.pair_outcomes)
                assert p.cost_scan_terms>=0
    with pytest.raises(FileExistsError):export_prepared_svc(svc,path)


@pytest.mark.parametrize('features',[1,16,4096])
def test_shared_ownership_survives_closing_owner(tmp_path,library,features):
    path=encoded(tmp_path/'model.srt',features=features)
    owner=PreparedModel(path,library)
    workers=[owner.session() for _ in range(8)]
    assert len({w.info['model_id'] for w in workers})==1
    other=PreparedModel(path,library)
    assert other.info['model_id']!=owner.info['model_id']
    owner.close();owner.close()
    with pytest.raises(ValueError):owner.session()
    path.unlink();del owner;gc.collect()
    x=[0.]*features
    expected=workers[0].predict(x,schedule='exhaustive')
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers=list(pool.map(lambda w:w.predict_many([x]*8),workers))
    assert all(answer==[expected]*8 for answer in answers)
    workers[0].close()
    assert workers[-1].predict(x)==expected
    for w in workers:w.close()
    other.close()


def test_legacy_format_and_new_session_match(tmp_path,library):
    np=pytest.importorskip('numpy');SVC=pytest.importorskip('sklearn.svm').SVC
    from spectra.svm_export import export_svc
    rng=np.random.default_rng(21);x=rng.normal(size=(80,16)).astype('float32')
    svc=SVC().fit(x,np.arange(80)%4);path=tmp_path/'old.srt';export_svc(svc,path)
    with Session(path,library) as old,PreparedModel(path,library) as m,m.session() as w:
        assert w.predict_many(x)==old.predict_many(x)
        assert m.labels==tuple(range(4))


def test_explicit_precision_boundary(tmp_path,library):
    # Two support vectors around a binary decision boundary; x differs only below FP32 precision.
    d=1;c=2;sv=[0.,1.];dual=[1.,-1.];bias=[0.]
    meta=b'{"labels":[10,20]}'
    payload=struct.pack('<dII5d',.5,1,1,*(sv+dual+bias))
    body=meta+payload;path=tmp_path/'boundary.srt'
    path.write_bytes(struct.pack('<8sIIIIII',b'SPCSVM02',c,2,d,len(meta),len(payload),zlib.crc32(body))+body)
    x=[.5+2**-25]
    with PreparedModel(path,library,input_dtype='float32') as a,PreparedModel(path,library,input_dtype='float64') as b:
        with a.session() as wa,b.session() as wb:
            assert wa.predict(x)==20
            assert wb.predict(x)==10


@pytest.mark.parametrize('bad',[[],[0.]*15,[0.]*17,[math.nan]*16,[math.inf]*16,[True]*16,['0']*16,[1e100]*16])
def test_bad_inputs_rejected(tmp_path,library,bad):
    path=encoded(tmp_path/'m.srt')
    with PreparedModel(path,library) as m,m.session() as w:
        with pytest.raises(ValueError):w.predict(bad)


def test_bounded_infinite_row_iterator(tmp_path,library):
    path=encoded(tmp_path/'m.srt')
    with PreparedModel(path,library) as m,m.session() as w:
        with pytest.raises(ValueError,match='features'):w.predict(itertools.repeat(0.))


@pytest.mark.parametrize('kind',['magic','crc','tail','truncated','width','count','labels','duplicate','mixed','nonfinite'])
def test_invalid_model_rejected(tmp_path,library,kind):
    path=encoded(tmp_path/'m.srt');raw=bytearray(path.read_bytes())
    if kind=='magic':raw[0]=0
    elif kind=='crc':raw[-1]^=1
    elif kind=='tail':raw+=b'x'
    elif kind=='truncated':raw=raw[:26]
    elif kind=='width':struct.pack_into('<I',raw,16,0)
    elif kind=='count':struct.pack_into('<I',raw,12,100001)
    elif kind in ('labels','duplicate','mixed'):
        labels={'labels':['a','a','a'],'duplicate':[True,False,True],'mixed':[1,'x',3]}[kind]
        encoded(path,labels=labels);raw=bytearray(path.read_bytes())
    elif kind=='nonfinite':
        label_size=struct.unpack_from('<I',raw,20)[0]
        struct.pack_into('<d',raw,32+label_size,float('nan'))
        struct.pack_into('<I',raw,28,zlib.crc32(raw[32:]))
    path.write_bytes(raw)
    with pytest.raises(ValueError):PreparedModel(path,library)


def test_native_late_nan_preserves_output_and_invalidates_certificate(tmp_path,library):
    path=encoded(tmp_path/'m.srt')
    with PreparedModel(path,library) as m,m.session() as w:
        w.predict_with_certificate([0.]*16)
        data=(C.c_double*32)(*([0.]*31+[math.nan]));out=(C.c_int*2)(123,456);stats=(C.c_uint64*5)()
        assert w._lib.sp_worker_run(w._handle,data,2,16,5,-1,out,2,stats,5,0)!=0
        assert list(out)==[123,456]
        trace=(C.c_int8*3)()
        assert w._lib.sp_worker_certificate(w._handle,trace,3)!=0


def test_batch_one_does_not_leave_certificate(tmp_path,library):
    path=encoded(tmp_path/'m.srt')
    with PreparedModel(path,library) as m,m.session() as w:
        w.predict_with_certificate([0.]*16)
        w.predict_many([[0.]*16])
        trace=(C.c_int8*3)()
        assert w._lib.sp_worker_certificate(w._handle,trace,3)!=0


def test_same_worker_mixed_threads_and_close(tmp_path,library):
    path=encoded(tmp_path/'m.srt');m=PreparedModel(path,library);w=m.session()
    def call(i):
        return w.predict_with_certificate([i/100]*16).class_index if i%2 else w.predict_many([[0.]*16])[0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert len(set(pool.map(call,range(100))))==1
    w.close();m.close()
    with pytest.raises(ValueError):w.predict([0.]*16)
    with pytest.raises(ValueError):w.__enter__()


def test_module_import_is_framework_free():
    root=Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable,'-S','-c',
        'import spectra.svm_shared,sys; assert not ({"torch","sklearn","numpy"} & sys.modules.keys())'],cwd=root,check=True)


@pytest.mark.parametrize('kwargs',[{'input_dtype':'bad'},{'tables':1},{'input_dtype':False}])
def test_invalid_owner_settings(tmp_path,library,kwargs):
    path=encoded(tmp_path/'m.srt')
    with pytest.raises(ValueError):PreparedModel(path,library,**kwargs)


@pytest.mark.parametrize('kwargs',[{'hint':True},{'hint':3},{'schedule':'unsafe'},{'schedule':[]}])
def test_invalid_worker_settings(tmp_path,library,kwargs):
    path=encoded(tmp_path/'m.srt')
    with PreparedModel(path,library) as m,m.session() as w:
        with pytest.raises(ValueError):w.predict([0.]*16,**kwargs)


def test_high_cardinality_lossless_fallback(tmp_path,library):
    np=pytest.importorskip('numpy');SVC=pytest.importorskip('sklearn.svm').SVC
    rng=np.random.default_rng(500);x=rng.normal(size=(360,1));y=np.arange(360)%3
    svc=SVC(C=.001,gamma=.5).fit(x,y)
    path=tmp_path/'m.srt';export_prepared_svc(svc,path)
    with PreparedModel(path,library,tables=True,input_dtype='float64') as m,m.session() as w:
        assert not m.info['tables_enabled']
        assert w.predict_many(x)==svc.predict(x).tolist()


@pytest.mark.parametrize('field',['features','labels','input_dtype','sha256'])
def test_public_model_metadata_is_readonly(tmp_path,library,field):
    path=encoded(tmp_path/'m.srt')
    with PreparedModel(path,library) as m,m.session() as w:
        for obj in (m,w):
            with pytest.raises(AttributeError):setattr(obj,field,None)


def test_native_rounding_rejected_without_output(tmp_path,library):
    libc=C.CDLL(None)
    if not hasattr(libc,'fegetround') or not hasattr(libc,'fesetround'):
        pytest.skip('C floating-point environment functions unavailable')
    path=encoded(tmp_path/'m.srt')
    with PreparedModel(path,library) as m,m.session() as w:
        old=libc.fegetround()
        # FE_DOWNWARD on the tested Linux x86-64 platform.
        try:
            if libc.fesetround(0x400):pytest.skip('FE_DOWNWARD unsupported')
            with pytest.raises(ValueError,match='round-to-nearest'):w.predict([0.]*16)
        finally:libc.fesetround(old)
