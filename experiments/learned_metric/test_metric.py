"""Learned geometry, exact native execution, malformed inputs and preserved labels."""
from array import array
import ctypes as C
import hashlib,math,os,struct,zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from sklearn.svm import SVC
from .learning import checked_codes,quantize_weights,kernel_matrix,kernel_table,split_development,fit_weights
from .train import encode_model
from .session import MetricSession,decode,MODES
from .build import build

@pytest.fixture(scope='session')
def library(tmp_path_factory):
 path=os.environ.get('METRIC_LIBRARY')
 return Path(path) if path else build(tmp_path_factory.mktemp('metric')/'build','portable')

@pytest.mark.parametrize('bad',[[0,0],[-1,2],[math.nan,1],[math.inf,1],[],[[1,2]]])
def test_bad_weight_vector(bad):
 with pytest.raises(ValueError):quantize_weights(bad)

@pytest.mark.parametrize('seed',range(10))
def test_integer_mass_and_determinism(seed):
 rng=np.random.default_rng(seed);x=rng.uniform(0,8,16)
 a=quantize_weights(x);b=quantize_weights(x)
 assert a.tolist()==b.tolist() and a.sum()==32 and np.all(a>=0)

@pytest.mark.parametrize('bad',[[[1.5,0]],[[math.inf,0]],[[-1,0]],[[101,0]],[]])
def test_non_grid_inputs_refused(bad):
 with pytest.raises(ValueError):checked_codes(bad,100)

def test_group_split_disjoint():
 q=np.repeat(np.arange(120)[:,None]%30,2,axis=1).astype(np.uint8);y=q[:,0]%3
 tr,va=split_development(q,y,611)
 assert set(map(tuple,q[tr])).isdisjoint(set(map(tuple,q[va])))

def test_signature_is_exact_and_kernel_psd():
 rng=np.random.default_rng(117);q=rng.integers(0,16,(32,16),dtype=np.uint8);w=quantize_weights(rng.uniform(0,3,16))
 table,gamma=kernel_table(w,15,2.)
 k=kernel_matrix(q,q,w,table)
 for i in range(len(q)):
  for j in range(len(q)):
   signature=sum(int(w[f])*(int(q[i,f])-int(q[j,f]))**2 for f in range(16))
   assert k[i,j]==math.exp(-gamma*signature)
 assert np.min(np.linalg.eigvalsh(k))>-1e-12

@pytest.mark.parametrize('d',[1,3,16,17,73])
@pytest.mark.parametrize('classes',[2,3,6])
def test_precomputed_native_parity(tmp_path,library,d,classes):
 rng=np.random.default_rng(772+d+classes)
 train=rng.integers(0,16,(72,d),dtype=np.uint8);y=np.asarray([100+7*(i%classes) for i in range(72)])
 w=quantize_weights(rng.uniform(.2,3,d));table,gamma=kernel_table(w,15,2.)
 model=SVC(C=10,kernel='precomputed').fit(kernel_matrix(train,train,w,table),y)
 raw=encode_model(model,train,w,15,gamma);path=tmp_path/'model.sgm';path.write_bytes(raw)
 probes=rng.integers(0,16,(19,d),dtype=np.uint8)
 expected=model.predict(kernel_matrix(probes,train,w,table)).tolist()
 with MetricSession(path,library) as s:
  assert decode(raw)[0]==d and s.weights==tuple(w.tolist())
  for mode in MODES:
   assert s.predict_buffer(probes,mode=mode)==expected
   assert s.predict_buffer(np.empty((0,d),dtype=np.uint8),mode=mode)==[]
  observed=np.frombuffer(s.probe(probes),dtype=np.float64).reshape(len(probes),-1)
  sv=train[model.support_];starts=np.r_[0,np.cumsum(model.n_support_)]
  for r,q in enumerate(probes):
   kernels=[]
   for x in sv:
    signature=sum(int(w[f])*(int(q[f])-int(x[f]))**2 for f in range(d))
    kernels.append(math.exp(-gamma*signature))
   p=0
   for i in range(classes):
    for j in range(i+1,classes):
     score=0.
     for a,row in [(i,j-1),(j,i)]:
      for k in range(starts[a],starts[a+1]):
       coefficient=model.dual_coef_[row,k]
       if coefficient!=0.:score+=float(coefficient)*kernels[k]
     score+=float(model.intercept_[p])
     assert struct.pack('d',score)==struct.pack('d',observed[r,p]);p+=1
  owned=array('B',probes.ravel().tolist());s.predict_buffer(owned);owned.extend([0]*d)
  with ThreadPoolExecutor(3) as pool:
   assert list(pool.map(lambda _:s.predict_buffer(probes),range(6)))==[expected]*6
 s.close()
 with pytest.raises(ValueError):s.predict_buffer(probes)

@pytest.fixture
def fixture_model(tmp_path,library):
 q=np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]],dtype=np.uint8);y=[0,0,1,1];w=np.array([2,2,2],dtype=np.uint16)
 table,gamma=kernel_table(w,3,2.)
 model=SVC(C=10,kernel='precomputed').fit(kernel_matrix(q,q,w,table),y)
 path=tmp_path/'m.sgm';path.write_bytes(encode_model(model,q,w,3,gamma));return path

@pytest.mark.parametrize('kind',['magic','crc','extra','truncated','weight','zero','dimension','maximum','mass'])
def test_malformed_model_rejected(fixture_model,library,kind):
 raw=bytearray(fixture_model.read_bytes())
 if kind=='magic':raw[0]=0
 elif kind=='crc':raw[-1]^=1
 elif kind=='extra':raw+=b'x'
 elif kind=='truncated':raw=raw[:24]
 elif kind=='weight':struct.pack_into('<H',raw,28,300)
 elif kind=='zero':raw[28:34]=b'\0'*6
 elif kind=='dimension':struct.pack_into('<I',raw,12,4097)
 elif kind=='maximum':struct.pack_into('<I',raw,8,256)
 else:struct.pack_into('<I',raw,16,4194304)
 fixture_model.write_bytes(raw)
 with pytest.raises(ValueError):MetricSession(fixture_model,library)

@pytest.mark.parametrize('data',[np.zeros((2,3),dtype=np.float64),b'123',np.zeros((2,4),dtype=np.uint8),np.zeros((2,6),dtype=np.uint8)[:,::2],np.full((2,3),4,dtype=np.uint8)])
def test_invalid_queries(fixture_model,library,data):
 with MetricSession(fixture_model,library) as s:
  with pytest.raises(ValueError):s.predict_buffer(data)
  assert len(s.predict_buffer(np.zeros((1,3),dtype=np.uint8)))==1

def test_late_invalid_native_output_untouched(fixture_model,library):
 with MetricSession(fixture_model,library) as s:
  x=(C.c_uint8*6)(0,0,0,0,0,4);out=(C.c_int*2)(117,118);stats=(C.c_uint64*4)()
  assert s._lib.lm_run(s._handle,x,2,3,1,out,stats,4)!=0
  assert list(out)==[117,118]

def test_wrong_geometry_does_not_overwrite(fixture_model,library):
 with MetricSession(fixture_model,library) as s:
  out=(C.c_int*2)(117,118);stats=(C.c_uint64*4)()
  assert s._lib.lm_run(s._handle,None,2,3,1,out,stats,4)!=0
  assert list(out)==[117,118]
