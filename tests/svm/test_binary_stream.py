"""Exact per-row order, tails, tie rules and interleaving for binary streaming."""
from array import array
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import json
import math
from pathlib import Path
import random
import struct
import sys
import zlib

import pytest
from spectra.svm_shared import PreparedModel


def encoded(path, d, *, coefficients=None, sv=None, bias=0., gamma=.25):
    coefficients=[1.,-.5,.125,-.25,0.] if coefficients is None else coefficients
    n=len(coefficients)
    assert n>=2
    sv=[(k%13-6)/16 for k in range(n*d)] if sv is None else sv
    metadata=json.dumps({'labels':[-9,27]},separators=(',',':')).encode()
    payload=struct.pack('<dII',gamma,n//2,n-n//2)
    values=list(sv)+list(coefficients)+[bias]
    payload+=struct.pack('<'+'d'*len(values),*values)
    body=metadata+payload
    path.write_bytes(struct.pack('<8sIIIIII',b'SPCSVM02',2,n,d,len(metadata),len(payload),zlib.crc32(body))+body)
    return path


def ordered_score(x,coeff,sv,bias,gamma):
    total=0.
    for i,a in enumerate(coeff):
        if a==0.:continue
        distance=0.
        for f,value in enumerate(x):
            difference=value-sv[i*len(x)+f]
            distance+=difference*difference
        total+=a*math.exp(-gamma*distance)
    return total+bias


@pytest.mark.parametrize('d',[1,3,4,13,16,17,30,73,257,4096])
@pytest.mark.parametrize('tables',[False,True])
def test_odd_dimensions_packets_and_original_order(tmp_path,library,d,tables):
    rng=random.Random(997+d)
    coeff=[rng.uniform(-3,3) for _ in range(9)];coeff[2]=0.
    sv=[rng.randint(-2,2)/8 for _ in range(d*len(coeff))]
    bias=.125;gamma=.25/d
    path=encoded(tmp_path/'model.srt',d,coefficients=coeff,sv=sv,bias=bias,gamma=gamma)
    rows=[[rng.uniform(-1,1) for _ in range(d)] for __ in range(13)]
    expected=[27 if ordered_score(x,coeff,sv,bias,gamma)>=0 else -9 for x in rows]
    with PreparedModel(path,library,tables=tables,input_dtype='float64') as model,model.session() as w:
        for n in (0,1,2,3,4,5,7,8,9,13):
            values=array('d',(v for row in rows[:n] for v in row))
            assert w.predict_buffer(values,schedule='binary_stream')==expected[:n]
            assert w.predict_many(rows[:n],schedule='binary_stream',hint=0)==expected[:n]
        assert w.predict_many(rows,schedule='exhaustive')==expected
        # Alternate streaming/cache modes: stale epochs must not affect other calls.
        for j,row in enumerate(rows):
            for mode in ('binary_stream','beretta_cert','exhaustive','cost_aware'):
                p=w.predict_with_certificate(row,schedule=mode,hint=j%2)
                assert p.label==expected[j] and p.pair_outcomes==(p.class_index,)
                assert p.evaluated_pairs==1
                assert p.evaluated_kernels==8 and p.evaluated_terms==8


@pytest.mark.parametrize('bias',[0.,-0.,2**-1074,-2**-1074])
def test_zero_coefficients_and_subnormal_bias(tmp_path,library,bias):
    path=encoded(tmp_path/'model.srt',3,coefficients=[0.,-0.],bias=bias)
    with PreparedModel(path,library,input_dtype='float64') as model,model.session() as w:
        rows=[[0.,0.,0.]]*9
        assert w.predict_many(rows,schedule='binary_stream')==[27 if bias>=0 else -9]*9
        p=w.predict_with_certificate(rows[0],schedule='binary_stream')
        assert p.evaluated_kernels==p.evaluated_terms==0
        assert p.evaluated_pairs==1


@pytest.mark.parametrize('tables',[False,True])
def test_extreme_finite_inputs_and_cancellation(tmp_path,library,tables):
    maximum=sys.float_info.max;tiny=2**-1074
    coeff=[1e290,-1e290,1.,-1.,tiny,-tiny]
    path=encoded(tmp_path/'model.srt',3,coefficients=coeff,sv=[0.]*18,bias=-tiny)
    rows=[[v,v,-v] for v in (0.,-0.,tiny,-tiny,1.,-1.,maximum,-maximum)]
    with PreparedModel(path,library,input_dtype='float64',tables=tables) as m,m.session() as w:
        expected=w.predict_many(rows,schedule='exhaustive')
        assert w.predict_many(rows,schedule='binary_stream')==expected
        assert [w.predict(row,schedule='binary_stream') for row in rows]==expected


def test_late_invalid_batch_does_not_write_outputs(tmp_path,library):
    path=encoded(tmp_path/'model.srt',3)
    with PreparedModel(path,library,input_dtype='float64') as model,model.session() as w:
        p=w.predict_with_certificate([0.,0.,0.],schedule='binary_stream')
        values=(C.c_double*15)(*([0.]*14+[math.nan]));output=(C.c_int*5)(*([123]*5));stats=(C.c_uint64*5)()
        assert w._lib.sp_worker_run(w._handle,values,5,3,7,-1,output,5,stats,5,0)!=0
        assert list(output)==[123]*5
        trace=(C.c_int8*1)()
        assert w._lib.sp_worker_certificate(w._handle,trace,1)!=0
        assert w.predict_buffer(array('d'),schedule='binary_stream')==[]
        assert w._lib.sp_worker_certificate(w._handle,trace,1)!=0


def test_independent_workers_after_owner_close(tmp_path,library):
    path=encoded(tmp_path/'model.srt',17)
    owner=PreparedModel(path,library,input_dtype='float64');workers=[owner.session() for _ in range(4)]
    rng=random.Random(73);rows=[[rng.uniform(-1,1) for _ in range(17)] for __ in range(129)]
    expected=workers[0].predict_many(rows,schedule='exhaustive');owner.close()
    try:
        with ThreadPoolExecutor(4) as pool:
            assert list(pool.map(lambda w:w.predict_many(rows,schedule='binary_stream'),workers))==[expected]*4
    finally:
        for w in workers:w.close()
