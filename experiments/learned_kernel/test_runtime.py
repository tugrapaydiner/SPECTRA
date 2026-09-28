"""Learned-function fidelity and finite-code compiler contracts; tiny fixed fixtures."""
from array import array
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,math,os,struct,zlib
from pathlib import Path
import pytest
import numpy as np
from experiments.learned_kernel.model import decode,export_pairs
from experiments.learned_kernel.session import KernelSession,MODES
from experiments.learned_kernel.metric import integer_budget,train_metric

@pytest.fixture(scope='session')
def library():
    return Path(os.environ['LEARNED_KERNEL_LIBRARY']).resolve(strict=True)

def fixture(path,d=17,D=16,power=2,counts=None,offgrid=False,components=3):
    rng=np.random.default_rng(219+d+D);classes=np.array([10,30,90]);y=np.repeat(classes,3)
    X=rng.integers(0,D+1,size=(9,d)).astype(float)/D
    if offgrid:X[0,0]=math.pi
    pairs=[]
    for i in range(3):
        for j in range(i+1,3):
            ids=np.flatnonzero((y==classes[i])|(y==classes[j]));co=rng.normal(size=len(ids));co[1]=0
            pairs.append((i,j,ids,co,.023*(i-j)))
    gamma=[.01,.3,2.][:components];weights=[.2,.3,.5] if components==3 else [1.]
    export_pairs(X,y,pairs,gammas=gamma,weights=weights,power=power,denominator=D,destination=path,feature_counts=counts)
    return X,y,pairs,gamma,weights

def ordered(X,pairs,gamma,weights,counts,power,row):
    margins=[];cache={}
    for i,j,ids,co,bias in pairs:
        total=0.
        for idx,a in zip(ids,co):
            if a==0:continue
            if idx not in cache:
                distance=0.
                for f,n in enumerate(counts):
                    z=float(row[f])-float(X[idx,f]);term=z*z if power==2 else abs(z)
                    for _ in range(n):distance+=term
                v=0.
                for g,w in zip(gamma,weights):
                    if w!=0:v+=w*math.exp(-g*distance)
                cache[idx]=v
            total+=float(a)*cache[idx]
        margins.append(total+float(bias))
    votes=[0]*3
    for score,(i,j,*_) in zip(margins,pairs):votes[j if score>=0 else i]+=1
    return [10,30,90][max(range(3),key=lambda i:votes[i])],margins

@pytest.mark.parametrize('d,D',[(1,1),(3,4),(16,16),(17,16),(73,1),(65,1),(257,1),(8,128)])
@pytest.mark.parametrize('power',[1,2])
@pytest.mark.parametrize('weighted',[False,True])
def test_exact_compilation_and_original_order(tmp_path,library,d,D,power,weighted):
    counts=[1]*d if not weighted else ([0]+[2]*(d-1) if d>1 else [3])
    path=tmp_path/'m.mkl';X,y,pairs,g,w=fixture(path,d,D,power,counts)
    rows=np.concatenate([X,np.ones((2,d)),np.zeros((1,d))])
    with KernelSession(path,library) as session:
        wanted=[];scores=[]
        for row in rows:
            label,margin=ordered(X,pairs,g,w,counts,power,row);wanted.append(label);scores.append(margin)
        for mode in MODES:assert session.predict_buffer(rows,mode=mode)==wanted
        observed=np.frombuffer(session.probe(rows),dtype=np.float64).reshape(len(rows),3,3)
        assert observed[:,:,0].tobytes()==np.asarray(scores).tobytes()
        assert observed[:,:,0].tobytes()==observed[:,:,1].tobytes()
        result,stats=session.inspect_buffer(rows)
        assert result==wanted and stats['query_exp_calls']==0 and stats['domain_fallback_rows']==0
        assert session.predict_buffer(array('d'))==[]

@pytest.mark.parametrize('model_offgrid',[False,True])
@pytest.mark.parametrize('power',[1,2])
def test_nonmember_uses_same_learned_function(tmp_path,library,model_offgrid,power):
    path=tmp_path/'m';X,y,pairs,g,w=fixture(path,3,4,power,offgrid=model_offgrid)
    rows=np.array([[math.pi,0.,.25],[.1,.2,.3],[-2.,1.,0.]])
    with KernelSession(path,library) as s:
        wanted=[ordered(X,pairs,g,w,[1]*3,power,row)[0] for row in rows]
        answer,stats=s.inspect_buffer(rows)
        assert answer==wanted==s.predict_buffer(rows,mode='direct')
        assert stats['domain_fallback_rows']==len(rows) and stats['query_exp_calls']>0
        a=np.frombuffer(s.probe(rows),np.float64).reshape(len(rows),3,3)
        assert a[:,:,0].tobytes()==a[:,:,1].tobytes() and not a[:,:,2].any()

@pytest.mark.parametrize('bad', [float('nan'),float('inf'),-float('inf')])
def test_late_nonfinite_input_rejected(tmp_path,library,bad):
    path=tmp_path/'m';fixture(path,d=3)
    with KernelSession(path,library) as s:
        with pytest.raises(ValueError):s.predict_buffer(array('d',[0.]*8+[bad]))
        assert len(s.predict_buffer(array('d',[0.]*3)))==1

@pytest.mark.parametrize('bad',[array('f',[0.]*3),array('d',[0.]*4),b'12345678',None])
def test_wrong_buffer(tmp_path,library,bad):
    path=tmp_path/'m';fixture(path,d=3)
    with KernelSession(path,library) as s:
        with pytest.raises(ValueError):s.predict_buffer(bad)

def test_owner_close_and_thread_state(tmp_path,library):
    path=tmp_path/'m';fixture(path,d=3)
    s=KernelSession(path,library);rows=array('d',[0.,1.,0.]*17);expected=s.predict_buffer(rows)
    with ThreadPoolExecutor(4) as pool:assert list(pool.map(lambda _:s.predict_buffer(rows),range(32)))==[expected]*32
    s.close();s.close()
    with pytest.raises(ValueError):s.predict_buffer(rows)

def rewrite(raw,mutate):
    magic,n,m=struct.unpack('<8sII',raw[:16]);doc=json.loads(raw[16:16+n]);mutate(doc)
    meta=json.dumps(doc).encode();return struct.pack('<8sII',magic,len(meta),m)+meta+raw[16+n:]

@pytest.mark.parametrize('field,value',[('power',True),('power',3),('denominator',3),('denominator',256),
 ('feature_counts',[0]*3),('feature_counts',[1,True,1]),('feature_counts',[5000,1,1]),('feature_counts',[1]),
 ('weights',['-0x1.0000000000000p+0']),('weights',[]),('gammas',['nan']),('gammas',['0x0.0p+0']),
 ('inner_sha256','0'*64),('format','wrong')])
def test_metadata_rejected(tmp_path,library,field,value):
    p=tmp_path/'m';fixture(p,d=3);raw=p.read_bytes();p.write_bytes(rewrite(raw,lambda d:d.__setitem__(field,value)))
    with pytest.raises(ValueError):KernelSession(p,library)

@pytest.mark.parametrize('kind',['short','trailing','magic','crc'])
def test_wire_inventory(tmp_path,library,kind):
    p=tmp_path/'m';fixture(p,d=3);raw=p.read_bytes()
    if kind=='short':raw=raw[:-1]
    elif kind=='trailing':raw+=b'x'
    elif kind=='magic':raw=b'BADMAGIC'+raw[8:]
    else:raw=raw[:-1]+bytes([raw[-1]^1])
    p.write_bytes(raw)
    with pytest.raises(ValueError):KernelSession(p,library)

@pytest.mark.parametrize('weights,budget',[([1,1,1],7),([0,2,1],9),([.1,.1,.1,.1],3),([1],4096)])
def test_integer_budget_invariants(weights,budget):
    a=integer_budget(weights,budget);b=integer_budget(weights,budget)
    assert np.array_equal(a,b) and sum(a)==budget and (a>=0).all()
    assert not np.any(a[np.asarray(weights)==0])

@pytest.mark.parametrize('weights,budget',[([-1,1],2),([0,0],2),([math.nan,1],2),([1],True),([1],0)])
def test_invalid_budget(weights,budget):
    with pytest.raises(ValueError):integer_budget(weights,budget)

def test_metric_deterministic_and_bounded():
    rng=np.random.default_rng(8);X=rng.normal(size=(32,4));y=np.arange(32)%2
    a,ra=train_metric(X,y,anchors=12,rounds=1);b,rb=train_metric(X,y,anchors=12,rounds=1)
    assert np.array_equal(a,b) and sum(a)==8 and (a>=0).all()
    assert ra==rb and ra['budget']==8
    assert all(math.isfinite(v) for v in ra['continuous_weights'])
