"""Boolean-domain specialization never rounds an input into its fast domain."""
from array import array
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import hashlib
import itertools
import json
import math
import random
import struct
import sys
import zlib

import pytest
from spectra.svm_shared import PreparedModel, SHARED_SCHEDULES
from spectra.svm_receipt import create_receipt, verify_receipt


def model_file(path, d=73, classes=3, count=4, gamma=.25, perturb=False, zero=False):
    rng=random.Random(d+classes*79)
    n=classes*count
    vectors=[float(rng.randrange(2)) for _ in range(d*n)]
    vectors[0]=-0.0
    if perturb: vectors[-1]=math.nextafter(0.,1.)
    coef=[0. if zero else rng.randint(-8,8)/8 for _ in range((classes-1)*n)]
    bias=[0. if zero else rng.randint(-4,4)/8 for _ in range(classes*(classes-1)//2)]
    labels=[f'class-{k}é' for k in range(classes)]
    metadata=json.dumps({'labels':labels}).encode()
    payload=struct.pack('<d'+'I'*classes,gamma,*([count]*classes))
    values=vectors+coef+bias
    payload+=struct.pack('<'+'d'*len(values),*values)
    body=metadata+payload
    path.write_bytes(struct.pack('<8sIIIIII',b'SPCSVM02',classes,n,d,len(metadata),len(payload),zlib.crc32(body))+body)
    return path


@pytest.mark.parametrize('d',[1,3,63,64,65,73,128,257,4096])
@pytest.mark.parametrize('tables',[False,True])
@pytest.mark.parametrize('boolean',['packed','lookup'])
def test_modes_boundaries_fallback_and_receipt(tmp_path,library,d,tables,boolean):
    path=model_file(tmp_path/'model.srt',d=d)
    rng=random.Random(d)
    rows=[[float(rng.randrange(2)) for _ in range(d)] for _ in range(9)]
    rows[0][0]=-0.
    for x in (.5,math.nextafter(1.,2.),math.nextafter(0.,1.),sys.float_info.max):
        row=rows[-1].copy();row[-1]=x;rows.append(row)
    with PreparedModel(path,library,input_dtype='float64') as old,old.session() as ref, \
         PreparedModel(path,library,input_dtype='float64',tables=tables,boolean=boolean) as model,model.session() as w:
        assert model.info['boolean_enabled']==1
        assert model.info['boolean_words']==(d+63)//64
        expected=ref.predict_many(rows,schedule='exhaustive')
        for mode in ('exhaustive','beretta_cert','binary_stream'):
            assert w.predict_many(rows,schedule=mode)==expected
            for n in (0,1,3,4,5,9,13):
                packed=array('d',(v for r in rows[:n] for v in r))
                assert w.predict_buffer(packed,schedule=mode)==expected[:n]
        r=create_receipt(w,rows[0]);assert verify_receipt(path,r,expected_input=rows[0],input_dtype='float64')['verified']
        assert w.boolean_stats['active']==1
        assert w.boolean_stats['exp_calls']<=d+1 if boolean=='lookup' else w.boolean_stats['exp_calls']>=0
        w.predict(rows[-1]);assert w.boolean_stats['active']==0
        w.predict(rows[0]);assert w.boolean_stats['active']==1


@pytest.mark.parametrize('classes',[2,3,10,128])
@pytest.mark.parametrize('zero',[False,True])
def test_every_schedule_and_zero_convention(tmp_path,library,classes,zero):
    path=model_file(tmp_path/'model',d=3,classes=classes,count=1,zero=zero)
    rows=[list(map(float,x)) for x in itertools.product((0,1),repeat=3)]
    with PreparedModel(path,library,input_dtype='float64') as m,m.session() as ref, \
         PreparedModel(path,library,boolean='lookup',input_dtype='float64') as b,b.session() as w:
        expected=ref.predict_many(rows,schedule='exhaustive')
        for mode in SHARED_SCHEDULES:
            assert w.predict_many(rows,schedule=mode)==expected
        for row,wanted in zip(rows,expected):
            p=w.predict_with_certificate(row)
            assert p.label==wanted


@pytest.mark.parametrize('gamma',[2**-1074,2**-1000,.25,100.,740.,sys.float_info.max])
def test_subnormal_and_extreme_gamma(tmp_path,library,gamma):
    path=model_file(tmp_path/'m',d=3,classes=2,gamma=gamma)
    rows=[list(map(float,x)) for x in itertools.product((0,1),repeat=3)]
    with PreparedModel(path,library,input_dtype='float64') as m,m.session() as ref, \
         PreparedModel(path,library,input_dtype='float64',boolean='lookup') as b,b.session() as w:
        assert w.predict_many(rows)==ref.predict_many(rows)
        for row in rows:
            r=create_receipt(w,row)
            assert verify_receipt(path,r,expected_input=row,input_dtype='float64')['verified']


@pytest.mark.parametrize('tables',[False,True])
def test_non_boolean_support_no_allocation_or_approximation(tmp_path,library,tables):
    path=model_file(tmp_path/'m',perturb=True)
    with PreparedModel(path,library,tables=tables,boolean='lookup',input_dtype='float64') as m,m.session() as w:
        assert m.info['boolean_enabled']==0 and m.info['boolean_support_bytes']==0
        row=[0.]*73
        r=create_receipt(w,row)
        assert verify_receipt(path,r,expected_input=row,input_dtype='float64')['verified']
        assert w.boolean_stats['active']==0


def test_cache_is_per_input_and_workers_outlive_owner(tmp_path,library):
    path=model_file(tmp_path/'m',d=73)
    owner=PreparedModel(path,library,boolean='lookup',input_dtype='float64')
    workers=[owner.session() for _ in range(4)]
    rows=[[float((f+i)%3==0) for f in range(73)] for i in range(9)]
    first=workers[0].predict_many(rows,schedule='exhaustive')
    workers[0].predict(rows[0],schedule='exhaustive');stats=workers[0].boolean_stats
    workers[0].predict(rows[0],schedule='exhaustive');assert stats==workers[0].boolean_stats
    assert stats['exp_calls']>0 and stats['lookup_hits']>0
    owner.close()
    try:
        with ThreadPoolExecutor(4) as pool:
            assert list(pool.map(lambda w:w.predict_many(rows),workers))==[first]*4
    finally:
        for w in workers:w.close()


@pytest.mark.parametrize('boolean',[None,True,1,[],{},'automatic'])
def test_invalid_mode_rejected(tmp_path,library,boolean):
    with pytest.raises(ValueError):PreparedModel(tmp_path/'missing',library,boolean=boolean)


def test_float32_rounding_precedes_domain_check(tmp_path,library):
    path=model_file(tmp_path/'m',d=1,classes=2)
    with PreparedModel(path,library,boolean='lookup',input_dtype='float32') as m,m.session() as w:
        w.predict([1.+2**-25]);assert w.boolean_stats['active']==1
    with PreparedModel(path,library,boolean='lookup',input_dtype='float64') as m,m.session() as w:
        w.predict([1.+2**-25]);assert w.boolean_stats['active']==0


def test_native_invalid_mode_and_late_nan(tmp_path,library):
    path=model_file(tmp_path/'m',d=3,classes=2)
    with PreparedModel(path,library,boolean='lookup',input_dtype='float64') as m,m.session() as w:
        raw=path.read_bytes();blob=C.create_string_buffer(raw)
        for bad in (0,3,-1):assert not w._lib.sp_model_create_boolean(blob,len(raw),0,bad)
        data=(C.c_double*15)(*([0.]*14+[math.nan]));out=(C.c_int*5)(*([99]*5));stats=(C.c_uint64*5)()
        assert w._lib.sp_worker_run(w._handle,data,5,3,7,-1,out,5,stats,5,0)!=0
        assert list(out)==[99]*5


def test_raw_fused_pipeline(tmp_path,library):
    import os
    from spectra.svm_pipeline import PreparedPipeline,SCHEMA
    from spectra.svm_preprocess_native import build_preprocessor
    folder=tmp_path/'bundle';folder.mkdir()
    path=model_file(folder/'model.srt',d=3)
    plan={'schema':SCHEMA,'columns':['x','y','z'],'features':3,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'operations':[{'kind':'numeric','column':f,'fill':(0.).hex(),'mean':(0.).hex(),'scale':(1.).hex()} for f in range(3)]}
    (folder/'preprocessing.json').write_text(json.dumps(plan))
    pre=os.environ.get('SPECTRA_PREPROCESS_LIBRARY') or build_preprocessor(tmp_path/'pre')
    rows=[[1.,0.,1.],[0.,1.,0.],[None,0.,1.],[.5,0.,1.]]*35
    with PreparedPipeline(folder,library,preprocessor_library=pre) as a,a.session() as ref, \
         PreparedPipeline(folder,library,preprocessor_library=pre,boolean='lookup') as b,b.session() as w:
        assert w.predict_fused(rows)==w.predict_many(rows)==ref.predict_many(rows)
