"""Synthetic numerical, signature, ownership, input and spectral contracts."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import itertools,json,math,os,struct
from pathlib import Path
import numpy as np
import pytest
from experiments.learned_finite.model import Model,encode
from experiments.learned_finite.session import Session
from experiments.learned_finite.learning import spectral_tables
from experiments.learned_finite.validate import reference

@pytest.fixture(scope='session')
def library():return Path(os.environ['LF_LIBRARY'])

def fixture(tmp_path,d=17,c=3,kind='hamming',signed=False):
    rng=np.random.default_rng(d+c);x=rng.integers(0,2,size=(2*c,d),dtype=np.uint8)
    w=rng.integers(0,8,size=d,dtype=np.uint8);w[0]=1
    if signed:lut=(-1.)**np.arange(int(w.sum())+1)
    else:lut=np.exp(-.1*np.arange(int(w.sum())+1))
    dual=rng.normal(size=(c-1,len(x)))*.1;bias=rng.normal(size=c*(c-1)//2)*.01
    raw=encode(x,list(range(c)),w,lut,np.full(c,2),dual,bias,kind)
    path=tmp_path/'model.lfk';path.write_bytes(raw)
    probe=rng.integers(0,2,size=(19,d),dtype=np.uint8)
    return path,probe

@pytest.mark.parametrize('d',[1,17,64,65,256,4096])
@pytest.mark.parametrize('kind',['hamming','intersection'])
@pytest.mark.parametrize('c',[2,3,10])
def test_complete_scores_and_modes(tmp_path,library,d,kind,c):
    path,x=fixture(tmp_path,d,c,kind);wanted,margins=reference(Model.load(path),x)
    with Session(path,library) as s:
        for mode in ('selective','exhaustive','scalar'):
            assert s.predict_buffer(x,mode)==wanted
            assert s.predict_buffer(x[:0],mode)==[]
        assert s.margins(x)==margins.astype('<f8').tobytes()
        assert s.predict_buffer(x.tobytes())==wanted
        first=s.predict_buffer(x[:1]);s.predict_buffer(x[2:]);assert s.predict_buffer(x[:1])==first

@pytest.mark.parametrize('c',[2,4,10])
def test_signed_parity_tables(tmp_path,library,c):
    path,x=fixture(tmp_path,16,c,signed=True)
    wanted,margin=reference(Model.load(path),x)
    with Session(path,library) as s:
        assert s.predict_buffer(x)==wanted and s.margins(x)==margin.tobytes()

@pytest.mark.parametrize('bad',[2,255])
def test_bad_input_no_native_write(tmp_path,library,bad):
    path,x=fixture(tmp_path);x[-1,-1]=bad
    with Session(path,library) as s:
        with pytest.raises(ValueError,match='nonbinary'):s.predict_buffer(x)
        out=(C.c_int*len(x))(*([123]*len(x)));stats=(C.c_uint64*3)()
        raw=(C.c_uint8*x.size).from_buffer(x)
        assert s._lib.lf_run(s._handle,raw,len(x),17,0,out,stats,3)!=0
        assert list(out)==[123]*len(x)

@pytest.mark.parametrize('kind',['shape','float','noncontiguous','mode','closed'])
def test_api_rejections(tmp_path,library,kind):
    path,x=fixture(tmp_path)
    with Session(path,library) as s:
        with pytest.raises(ValueError):
            if kind=='shape':s.predict_buffer(x[:,:16].copy())
            elif kind=='float':s.predict_buffer(x.astype(float))
            elif kind=='noncontiguous':s.predict_buffer(x[::2])
            elif kind=='mode':s.predict_buffer(x,'fake')
            else:s.close();s.predict_buffer(x)

@pytest.mark.parametrize('where',[0,8,12,16,20,24,28,32,40,-1])
def test_corrupt_model_rejected(tmp_path,where):
    path,x=fixture(tmp_path);raw=bytearray(path.read_bytes());raw[where]^=127
    with pytest.raises(ValueError):Model(bytes(raw))

def test_worker_lock_and_lifetime(tmp_path,library):
    path,x=fixture(tmp_path)
    s=Session(path,library);wanted=s.predict_buffer(x);path.unlink()
    with ThreadPoolExecutor(4) as pool:assert list(pool.map(lambda _:s.predict_buffer(x),range(20)))==[wanted]*20
    s.close()
    with pytest.raises(ValueError):s.predict_buffer(x)

@pytest.mark.parametrize('d',[2,3,4,5,6])
def test_spectral_recurrence_equals_explicit_feature_map(d):
    x=np.array(list(itertools.product([0,1],repeat=d)),dtype=np.int32)
    dist=(x[:,None,:]!=x[None,:,:]).sum(2);tables=spectral_tables(d,list(range(1,d+1)))
    z=1-2*x
    for order in range(1,d+1):
        phi=np.array([np.prod(z[:,subset],axis=1) for subset in itertools.combinations(range(d),order)]).T
        exact=phi@phi.T/phi.shape[1];K=tables[order-1][dist]
        assert np.allclose(K,exact,rtol=0,atol=1e-12)
        assert np.linalg.eigvalsh(K).min()>-1e-10
