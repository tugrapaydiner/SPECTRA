"""Numerical contracts and actual learner-gradient checks, not accuracy thresholds."""
from __future__ import annotations
from array import array
from concurrent.futures import ThreadPoolExecutor
import ctypes as C,math,struct,zlib,os,json
from pathlib import Path
import numpy as np
import pytest
from sklearn.svm import SVC
from .learning import NeighborhoodObjective,fit_map,project,make_kernel,gram,split
from .study import encode
from .session import InteractionSession,decode,MODES
from .build import build

@pytest.fixture(scope='session')
def library(tmp_path_factory):
    return Path(os.environ['INTERACTION_LIBRARY']) if 'INTERACTION_LIBRARY' in os.environ else build(tmp_path_factory.mktemp('lib')/'build')

def fitted(tmp_path,d=16,c=3,D=15,dense=True):
    rng=np.random.default_rng(821+d+c);q=rng.integers(0,D+1,(48,d),dtype=np.uint8);y=np.arange(48)%c*3-7
    A=rng.integers(-1,2,(d,d),dtype=np.int16) if dense else np.eye(d,dtype=np.int16)*8
    A=np.concatenate((A,np.eye(d,dtype=np.int16)));A=A[np.any(A!=0,axis=1)]
    k=make_kernel(A,D,2);z=project(q,A,D);m=SVC(C=10,kernel='precomputed').fit(gram(z,z,k),y)
    path=tmp_path/'model.sik';path.write_bytes(encode(m,q,A,D,k));return path,q,y,A,k,m

@pytest.mark.parametrize('diagonal',[False,True])
def test_objective_gradient(diagonal):
    rng=np.random.default_rng(17);x=rng.normal(size=(48,5));y=np.arange(48)%3
    fun=NeighborhoodObjective(x,y,11,diagonal=diagonal)
    p=np.ones(5) if diagonal else np.eye(5).ravel();p=p+rng.normal(0,.1,size=p.size)
    loss,g=fun(p);h=1e-5
    for i in range(len(p)):
        u=np.zeros(len(p));u[i]=h
        observed=(fun(p+u)[0]-fun(p-u)[0])/(2*h)
        assert math.isclose(observed,float(g[i]),rel_tol=1e-5,abs_tol=1e-7)

@pytest.mark.parametrize('method',['uniform','diagonal','full','whitening','local_supervised','local_unsupervised'])
def test_projected_learning_is_deterministic_label_renaming_invariant(method):
    rng=np.random.default_rng(101);q=rng.integers(0,16,(64,4),dtype=np.uint8);y=(q[:,0]>q[:,1]).astype(int)
    a,rec=fit_map(q,y,15,method,seed=11,maxiter=25)
    b,_=fit_map(q,1-y,15,method,seed=11,maxiter=25)
    assert np.array_equal(a,b) and rec['trace_integer']>0
    assert np.linalg.matrix_rank(a)==4

def test_full_learner_responds_to_labels():
    rng=np.random.default_rng(99);q=rng.integers(0,16,(96,4),dtype=np.uint8);y=(q[:,0]>q[:,1]).astype(int)
    a,_=fit_map(q,y,15,'full',seed=4);b,_=fit_map(q,rng.permutation(y),15,'full',seed=4)
    assert not np.array_equal(a.T@a,b.T@b)

def test_group_boundaries():
    rng=np.random.default_rng(92);q=np.repeat(rng.integers(0,16,(80,3),dtype=np.uint8),2,axis=0);y=np.arange(160)//2%4
    train,val=split(q,y,611)
    assert set(map(bytes,q[train])).isdisjoint(set(map(bytes,q[val])))

@pytest.mark.parametrize('d,D',[(1,1),(3,15),(16,15),(17,100),(36,255),(64,255)])
@pytest.mark.parametrize('classes',[2,3,6])
def test_all_backends_and_ordered_scores(tmp_path,library,d,D,classes):
    path,q,y,A,k,m=fitted(tmp_path,d,classes,D)
    rng=np.random.default_rng(233);test=rng.integers(0,D+1,(13,d),dtype=np.uint8)
    expected=m.predict(gram(project(test,A,D),project(q,A,D),k)).tolist()
    zz=project(test,A,D);sv=project(q[m.support_],A,D);starts=np.r_[0,np.cumsum(m.n_support_)]
    with InteractionSession(path,library) as w:
        for mode in ('compiled','scalar_integer','exhaustive','projected'):
            assert w.predict_buffer(test,mode=mode)==expected
            assert w.predict_buffer(np.empty((0,d),dtype=np.uint8),mode=mode)==[]
        observed=np.frombuffer(w.probe(test),dtype=np.float64).reshape(len(test),-1)
        for row,z in enumerate(zz):
            kernels=[]
            for support in sv:
                S=sum((int(a)-int(b))**2 for a,b in zip(z,support))
                kernels.append(math.exp(-k.coefficient*float((S>>k.bits)<<k.bits))*math.exp(-k.coefficient*float(S&((1<<k.bits)-1))))
            p=0
            for i in range(classes):
                for j in range(i+1,classes):
                    value=0.
                    for cl,coefrow in ((i,j-1),(j,i)):
                        for t in range(starts[cl],starts[cl+1]):
                            coef=float(m.dual_coef_[coefrow,t])
                            if coef!=0:value+=coef*kernels[t]
                    value+=float(m.intercept_[p]);assert struct.pack('<d',value)==struct.pack('<d',observed[row,p]);p+=1
        assert w.info['table_entries']==len(k.high)+len(k.low)
        with ThreadPoolExecutor(3) as pool:assert list(pool.map(lambda _:w.predict_buffer(test),range(6)))==[expected]*6
    with pytest.raises(ValueError):w.predict_buffer(test)


def test_wide_uint64_accumulation(tmp_path,library):
    d=36;D=255;q=np.stack([np.full(d,j,dtype=np.uint8) for j in (0,1,2,253,254,255)])
    A=np.ones((72,d),dtype=np.int16);A[::2,::2]=-1
    k=make_kernel(A,D,2);assert k.bound>2**31
    z=project(q,A,D);m=SVC(C=10,kernel='precomputed').fit(gram(z,z,k),[0,0,0,1,1,1])
    path=tmp_path/'wide.sik';path.write_bytes(encode(m,q,A,D,k))
    with InteractionSession(path,library) as s:
        assert s.predict_buffer(q)==[0,0,0,1,1,1]
        assert s.predict_buffer(q,mode='scalar_integer')==[0,0,0,1,1,1]
        assert s.probe(q)==s.probe(q.copy())

@pytest.fixture
def small(tmp_path):return fitted(tmp_path,3,3,15)[0]

@pytest.mark.parametrize('damage',['magic','crc','short','trailing','d','rank','maximum','classes','count','bits','alpha','projection','support','label','infinite_coeff'])
def test_corrupt_models_rejected(small,library,damage):
    data=bytearray(small.read_bytes())
    header=struct.unpack_from('<8sIIIIIIIII',data);_,d,r,D,c,n,bits,mlen,plen,crc=header
    if damage=='magic':data[0]^=1
    elif damage=='crc':data[-1]^=1
    elif damage=='short':data.pop()
    elif damage=='trailing':data+=b'x'
    elif damage in ('d','rank','maximum','classes','count','bits'):
        offset={'d':8,'rank':12,'maximum':16,'classes':20,'count':24,'bits':28}[damage]
        struct.pack_into('<I',data,offset,999999)
    elif damage=='alpha':struct.pack_into('<d',data,44,float('nan'))
    elif damage=='projection':struct.pack_into('<h',data,52,32)
    elif damage=='support':data[52+2*d*r]=D+1
    elif damage=='label':data[-mlen]=ord('[')
    else:struct.pack_into('<d',data,52+2*d*r+n*d+4*c,float('inf'))
    if damage not in ('crc','short','trailing'):struct.pack_into('<I',data,40,zlib.crc32(data[44:]))
    small.write_bytes(data)
    with pytest.raises(ValueError):InteractionSession(small,library)

@pytest.mark.parametrize('data',[np.zeros((1,3),dtype=float),b'abc',np.zeros((2,4),dtype=np.uint8),np.zeros((2,6),dtype=np.uint8)[:,::2],np.full((1,3),16,dtype=np.uint8)])
def test_input_rejection(small,library,data):
    with InteractionSession(small,library) as s:
        with pytest.raises(ValueError):s.predict_buffer(data)
        assert len(s.predict_buffer(np.zeros((1,3),dtype=np.uint8)))==1

def test_late_bad_code_preserves_native_output(small,library):
    with InteractionSession(small,library) as s:
        data=(C.c_uint8*6)(0,0,0,0,0,16);output=(C.c_int*2)(777,888);stats=(C.c_uint64*4)()
        assert s._lib.ii_run(s._handle,data,2,3,1,output,stats,4)!=0
        assert list(output)==[777,888]

def test_buffer_export_released(small,library):
    a=array('B',[1,2,3])
    with InteractionSession(small,library) as s:s.predict_buffer(a)
    a.extend([1,2,3]);assert len(a)==6

def test_same_thread_close_leases_pointer(small,library):
    s=InteractionSession(small,library);original=s._lib.ii_run
    def close_inside(*args):
        s.close();assert s._handle is not None
        return original(*args)
    s._lib.ii_run=close_inside
    try:assert len(s.predict_buffer(array('B',[1,2,3])))==1
    finally:s._lib.ii_run=original;s.close()
    assert s._handle is None

@pytest.mark.parametrize('value',[None,True,[],42,'unsupported'])
def test_modes_rejected(small,library,value):
    with InteractionSession(small,library) as s:
        with pytest.raises(ValueError):s.predict_buffer(array('B',[1,2,3]),mode=value)


def test_two_tables_preserve_signature_definition():
    A=np.array([[8,-5,3],[-7,8,1],[1,0,0],[0,1,0],[0,0,1]],dtype=np.int16);k=make_kernel(A,255,8)
    assert len(k.high)+len(k.low)<k.bound+1
    rng=np.random.default_rng(444)
    samples=np.r_[np.arange(100),np.arange(max(0,k.bound-100),k.bound+1),rng.integers(0,k.bound+1,2000)].astype(np.uint64)
    values=k.evaluate(samples)
    for S,v in zip(samples,values):
        S=int(S)
        assert v==math.exp(-k.coefficient*float((S>>k.bits)<<k.bits))*math.exp(-k.coefficient*float(S&((1<<k.bits)-1)))


@pytest.fixture(scope='session')
def target_fenv(tmp_path_factory):
    import subprocess
    folder=tmp_path_factory.mktemp('target-fenv');src=folder/'env.cpp';lib=folder/'env.so'
    src.write_text('#include <cfenv>\nextern "C" { int env_get(){return std::fegetround();} int env_down(){return FE_DOWNWARD;} int env_set(int v){return std::fesetround(v);} }\n')
    subprocess.run(['g++','-std=c++17','-fPIC','-shared',str(src),'-o',str(lib)],check=True,capture_output=True)
    r=C.CDLL(str(lib));r.env_get.argtypes=[];r.env_get.restype=C.c_int;r.env_down.argtypes=[];r.env_down.restype=C.c_int
    r.env_set.argtypes=[C.c_int];r.env_set.restype=C.c_int;return r

def test_non_nearest_rounding_rejected(small,library,target_fenv):
    f=target_fenv;original=f.env_get()
    with InteractionSession(small,library) as session:
        try:
            assert f.env_set(f.env_down())==0
            with pytest.raises(ValueError,match='round-to-nearest'):session.predict_buffer(array('B',[1,2,3]))
            with pytest.raises(ValueError,match='round-to-nearest'):session.probe(array('B',[1,2,3]))
        finally:assert f.env_set(original)==0


@pytest.mark.parametrize('d,D',[(3,15),(16,100),(36,255)])
def test_diagonal_specialization_matches_projected_route(tmp_path,library,d,D):
    path,q,y,A,k,m=fitted(tmp_path,d,3,D,dense=False)
    with InteractionSession(path,library) as s:
        fast,work=s.inspect_buffer(q)
        assert fast==s.predict_buffer(q,mode='projected')
        assert work['projection_terms']==0
        _,plain=s.inspect_buffer(q,mode='projected');assert plain['projection_terms']>0
