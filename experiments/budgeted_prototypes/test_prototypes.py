"""Own-model fidelity, integer bounds and ownership; no accuracy thresholds."""
from array import array
import ctypes as C, dataclasses, json, math, os, struct, subprocess, sys, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pytest
from .learning import Settings,Model,integer_weights,kernel_config,features,grouped_split,train
from .export import encode
from .session import PrototypeSession,HEADER,decode
from .build import build
from .controls import ControlSession,export_network,build as build_controls

@pytest.fixture(scope='session')
def library(tmp_path_factory):
    p=os.environ.get('PROTOTYPE_LIBRARY')
    return Path(p) if p else build(tmp_path_factory.mktemp('proto')/'build','portable')
@pytest.fixture(scope='session')
def controls_library(tmp_path_factory):
    p=os.environ.get('PROTOTYPE_CONTROL_LIBRARY')
    return Path(p) if p else build_controls(tmp_path_factory.mktemp('control')/'build','portable')

def random_model(d=16,p=17,c=10,D=15,Q=4,U=4,uniform=False):
    rng=np.random.default_rng(72+d+p+c)
    centers=rng.integers(0,D*Q+1,(p,d),dtype=np.uint16)
    weights=np.full((p,d),U,dtype=np.uint16) if uniform else integer_weights(rng.normal(size=(p,d)),U)
    arrays={'centers':centers,'weights':weights,'head':rng.normal(size=(p,c)),'bias':rng.normal(size=c),'classes':np.arange(c)*7-19}
    s=Settings(prototypes=p,gamma=2.,quarter=Q,units=U)
    return arrays,s

def python_scores(q,arrays,D,s):
    k=kernel_config(q.shape[1],D,s.quarter,s.units,s.gamma);out=[]
    for row in q:
        scores=[0.]*len(arrays['bias'])
        for j in range(len(arrays['centers'])):
            S=sum(int(arrays['weights'][j,f])*(int(row[f])*s.quarter-int(arrays['centers'][j,f]))**2 for f in range(len(row)))
            value=math.exp(-k['alpha']*float((S>>k['bits'])<<k['bits']))*math.exp(-k['alpha']*float(S&((1<<k['bits'])-1)))
            for c in range(len(scores)):scores[c]+=value*float(arrays['head'][j,c])
        for c in range(len(scores)):scores[c]+=float(arrays['bias'][c])
        out.extend(scores)
    return struct.pack('<'+'d'*len(out),*out)

@pytest.mark.parametrize('d,p,c,D,Q,U',[(1,1,2,1,1,1),(3,3,3,15,4,4),(8,7,10,100,4,4),(16,17,26,15,4,4),(17,33,10,100,4,4),(64,17,10,16,4,4),(256,3,128,255,16,16)])
@pytest.mark.parametrize('uniform',[False,True])
def test_exact_ordered_scores(tmp_path,library,d,p,c,D,Q,U,uniform):
    a,s=random_model(d,p,c,D,Q,U,uniform);path=tmp_path/'model';path.write_bytes(encode(a,D,s))
    q=np.random.default_rng(88).integers(0,D+1,(13,d),dtype=np.uint8)
    raw=python_scores(q,a,D,s);expected=np.frombuffer(raw,dtype='<f8').reshape(len(q),c).argmax(1)
    with PrototypeSession(path,library) as w:
        for mode in ('compiled','scalar'):
            assert w.scores(q,mode=mode)==raw
            assert w.predict_buffer(q,mode=mode)==a['classes'][expected].tolist()
            assert w.predict_buffer(np.empty((0,d),dtype=np.uint8),mode=mode)==[]
        assert w.info['uniform_metric']==int(uniform or U==1)
        with ThreadPoolExecutor(3) as pool:assert list(pool.map(lambda _:w.predict_buffer(q),range(7)))==[a['classes'][expected].tolist()]*7
    with pytest.raises(ValueError):w.predict_buffer(q)

@pytest.mark.parametrize('seed',range(8))
def test_integer_metric_mass(seed):
    rng=np.random.default_rng(seed);logits=rng.normal(size=(31,36))*5
    a=integer_weights(logits,4);b=integer_weights(logits.copy(),4)
    assert np.array_equal(a,b) and a.min()>=1 and np.all(a.sum(1)==144)

@pytest.mark.parametrize('setting', [{'prototypes':True},{'prototypes':4097},{'gamma':0.},{'gamma':math.inf},{'reg':-1},{'lr':math.nan},{'epochs':0},{'quarter':17},{'units':0},{'seed':-1},{'affine':1},{'arm':'bad'}])
def test_bad_settings(setting):
    with pytest.raises(ValueError):Settings(**setting)

@pytest.mark.parametrize('flag',['affine','adaptive_gamma','normalized'])
def test_unsupported_pilot_models_not_silently_exported(flag):
    a,s=random_model()
    with pytest.raises(ValueError,match='only the locked'):encode(a,15,dataclasses.replace(s,**{flag:True}))

def test_train_geometry_is_label_sensitive_and_finite():
    rng=np.random.default_rng(21);q=rng.integers(0,16,(80,4),dtype=np.uint8);y=(q[:,0]>q[:,1]).astype(int)
    a,r=train(q,y,15,Settings(prototypes=8,epochs=5,gamma=2,reg=1e-5,arm='local'))
    b,_=train(q,rng.permutation(y),15,Settings(prototypes=8,epochs=5,gamma=2,reg=1e-5,arm='local'))
    assert np.all(a['weights'].sum(1)==16) and a['weights'].min()>=1
    assert np.isfinite(a['head']).all() and not np.array_equal(a['centers'],b['centers'])
    assert r['fitting_labels_only']

def test_duplicate_grouping():
    rng=np.random.default_rng(2);q=np.repeat(rng.integers(0,16,(100,5),dtype=np.uint8),2,axis=0);y=np.arange(200)//2%4
    tr,va=grouped_split(q,y,611)
    assert set(map(bytes,q[tr])).isdisjoint(set(map(bytes,q[va])))

@pytest.fixture
def small(tmp_path):
    a,s=random_model(d=3,p=3,c=3);path=tmp_path/'model';path.write_bytes(encode(a,15,s));return path

@pytest.mark.parametrize('kind',['magic','crc','tail','short','bits','d','p','c','maximum','quarter','units','alpha','center','weight','head','bias','labels'])
def test_bad_models(small,library,kind):
    data=bytearray(small.read_bytes());header=HEADER.unpack_from(data);_,d,p,c,D,Q,U,bits,nmeta,plen,crc=header
    if kind=='magic':data[0]^=1
    elif kind=='crc':data[-1]^=1
    elif kind=='tail':data+=b'x'
    elif kind=='short':data=data[:40]
    elif kind in ('bits','d','p','c','maximum','quarter','units'):
        off={'bits':32,'d':8,'p':12,'c':16,'maximum':20,'quarter':24,'units':28}[kind];struct.pack_into('<I',data,off,100000)
    elif kind=='alpha':struct.pack_into('<d',data,48,math.nan)
    elif kind=='center':struct.pack_into('<H',data,56,D*Q+1)
    elif kind=='weight':struct.pack_into('<H',data,56+2*p*d,0)
    elif kind=='head':struct.pack_into('<d',data,56+4*p*d,math.inf)
    elif kind=='bias':struct.pack_into('<d',data,56+4*p*d+8*p*c,math.nan)
    else:data[-nmeta]=ord('[')
    if kind not in ('crc','tail','short'):struct.pack_into('<I',data,44,zlib.crc32(data[48:]))
    small.write_bytes(data)
    with pytest.raises(ValueError):PrototypeSession(small,library)

@pytest.mark.parametrize('data',[b'abc',np.zeros((1,3),dtype=float),np.zeros((1,4),dtype=np.uint8),np.zeros((2,6),dtype=np.uint8)[:,::2],np.full((1,3),16,dtype=np.uint8)])
def test_bad_input(small,library,data):
    with PrototypeSession(small,library) as w:
        with pytest.raises(ValueError):w.predict_buffer(data)
        assert len(w.predict_buffer(array('B',[0,0,0])))==1

def test_late_invalid_c_input_does_not_write(small,library):
    with PrototypeSession(small,library) as w:
        data=(C.c_uint8*6)(0,0,0,0,0,16);out=(C.c_int*2)(117,118)
        assert w._lib.bp_run(w._handle,data,2,3,0,out)!=0
        assert list(out)==[117,118]

def test_buffer_and_lifetime(small,library):
    w=PrototypeSession(small,library);a=array('B',[1,2,3]);w.predict_buffer(a);a.extend([0,0,0])
    run=w._lib.bp_run
    def closing(*args):w.close();assert w._handle;return run(*args)
    w._lib.bp_run=closing
    try:assert len(w.predict_buffer(a))==2
    finally:w._lib.bp_run=run;w.close()
    assert w._handle is None

@pytest.fixture(scope='session')
def environment_probe(tmp_path_factory):
    p=tmp_path_factory.mktemp('fenv');(p/'e.cpp').write_text('#include <cfenv>\nextern "C"{int get(){return std::fegetround();} int set(int x){return std::fesetround(x);} int down(){return FE_DOWNWARD;}}\n')
    subprocess.run(['g++','-shared','-fPIC',str(p/'e.cpp'),'-o',str(p/'e.so')],check=True)
    lib=C.CDLL(str(p/'e.so'));lib.get.restype=C.c_int;lib.down.restype=C.c_int;lib.set.argtypes=[C.c_int];return lib

def test_rounding(small,library,environment_probe):
    f=environment_probe;old=f.get()
    with PrototypeSession(small,library) as w:
        try:
            assert f.set(f.down())==0
            with pytest.raises(ValueError):w.predict_buffer(array('B',[1,2,3]))
        finally:assert f.set(old)==0

@pytest.mark.parametrize('layers',[(3,3),(3,7,3),(3,8,5,3)])
def test_native_network_control(tmp_path,controls_library,layers):
    rng=np.random.default_rng(33);a={'mean':rng.normal(size=3),'scale':rng.uniform(.5,2,size=3),'maximum':15,'classes':np.array([-4,8,11])}
    for k,(i,o) in enumerate(zip(layers,layers[1:])):a[f'w{k}']=rng.normal(size=(i,o));a[f'b{k}']=rng.normal(size=o)
    p=tmp_path/'control';p.write_bytes(export_network(a));q=rng.integers(0,16,(17,3),dtype=np.uint8)
    expected=[]
    for row in q:
        x=[(int(row[j])/15-float(a['mean'][j]))/float(a['scale'][j]) for j in range(3)]
        for k in range(len(layers)-1):
            z=[]
            for j in range(layers[k+1]):
                val=0.
                for i in range(layers[k]):val+=x[i]*float(a[f'w{k}'][i,j])
                val+=float(a[f'b{k}'][j]);z.append(max(0.,val) if k+1<len(layers)-1 else val)
            x=z
        expected.extend(x)
    packed=struct.pack('<'+'d'*len(expected),*expected)
    with ControlSession(p,controls_library) as w:
        assert w.scores(q)==packed==w.scores(q,mode='scalar')
        assert w.predict_buffer(np.empty((0,3),dtype=np.uint8))==[]

def test_isolated_runtime_import(small,library):
    root=Path(__file__).resolve().parents[2]
    code='import sys;sys.path.insert(0,sys.argv[1]);from experiments.budgeted_prototypes.session import PrototypeSession;from array import array;w=PrototypeSession(sys.argv[2],sys.argv[3]);assert len(w.predict_buffer(array("B",[0,0,0])))==1;w.close();assert not ({"numpy","scipy","sklearn","torch"}&sys.modules.keys())'
    subprocess.run([sys.executable,'-I','-S','-c',code,str(root),str(small),str(library)],check=True)
