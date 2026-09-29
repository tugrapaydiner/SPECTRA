"""Learned table execution preserves its own model; no old-label fidelity claim."""
from array import array
import ctypes as C
import hashlib,json,math,os,struct,subprocess,sys,zlib
from pathlib import Path
import numpy as np
import pytest
from sklearn.svm import SVC
from experiments.learned_signature.learning import signatures,table_profile,learn_mixture
from experiments.learned_signature.model import pack_model,from_precomputed,ReferenceModel,HEADER
from experiments.learned_signature.native import Session,build

@pytest.fixture(scope='session')
def library(tmp_path_factory):
 return Path(os.environ['LEARNED_LIBRARY']) if 'LEARNED_LIBRARY' in os.environ else build(tmp_path_factory.mktemp('learned')/'native',target='portable')

def fitted(tmp_path,d,c,weighted=True):
 rng=np.random.default_rng(2026+d+c);q=rng.integers(0,8,size=(45,d),dtype=np.uint8);y=np.arange(45)%c
 w=(1+np.arange(d)%8).astype(np.uint32) if weighted else np.ones(d,dtype=np.uint32)
 table=table_profile(7,w,1./d,[.2,.3,.5],[.25,1.,4.])
 svc=SVC(C=3.,kernel='precomputed').fit(table[signatures(q,q,w)],y)
 path=tmp_path/'model.lkt';from_precomputed(svc,q,w,table,path,{'domain_max':7,'purpose':'test fixture'})
 probe=rng.integers(0,8,size=(17,d),dtype=np.uint8);return path,probe,svc,table[signatures(probe,q,w)]

@pytest.mark.parametrize('d',[1,7,8,9,16,17,65])
@pytest.mark.parametrize('classes',[2,3,5])
def test_precomputed_labels_margins_and_modes(tmp_path,library,d,classes):
 path,q,svc,K=fitted(tmp_path,d,classes);ref=ReferenceModel(path);expected=svc.predict(K).tolist()
 assert [ref.predict(row) for row in q.tolist()]==expected
 with Session(path,library) as native:
  for mode in ('selective','scalar','exhaustive'):
   assert native.predict_buffer(q,mode=mode)==expected
   assert native.predict_buffer(q[:1],mode=mode)==expected[:1]
   assert native.predict_buffer(q[:0],mode=mode)==[]
  observed=np.frombuffer(native.probe(q),dtype=np.float64).reshape(len(q),-1)
  oracle=np.array([ref.margins(row) for row in q.tolist()],dtype=np.float64)
  assert observed.tobytes()==oracle.tobytes()
  svc.decision_function_shape='ovo';original=svc.decision_function(K).reshape(len(K),-1)
  np.testing.assert_allclose(observed,original,atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('d',[3,16,127])
def test_pair_specific_profiles_share_signatures(tmp_path,library,d):
 rng=np.random.default_rng(d);q=rng.integers(0,5,size=(9,d),dtype=np.uint8);w=np.ones(d,dtype=np.uint32)
 tables=np.stack([table_profile(4,w,g) for g in (.1,.5,1.)]);pairs=[]
 for p in range(3):pairs.append({'ids':list(range(9)),'coef':[(-1.)**(k+p)*(k+1)/32 for k in range(9)],'bias':(p-1)/17,'profile':p})
 path=tmp_path/'pair.lkt';pack_model(path,codes=q,classes=['a','b','c'],counts=[3,3,3],weights=w,table=tables,pairs=pairs,metadata={'domain_max':4})
 rows=rng.integers(0,5,size=(13,d),dtype=np.uint8);ref=ReferenceModel(path)
 with Session(path,library) as session:
  got,counts=session.inspect_buffer(rows,mode='exhaustive')
  assert got==[ref.predict(x) for x in rows.tolist()]
  assert counts[0]==len(rows)*9 # not multiplied by three table profiles
  assert session.probe(rows)==np.asarray([ref.margins(x) for x in rows.tolist()],dtype='<f8').tobytes()

@pytest.mark.parametrize('bias',[0.,-0.,2**-1074,-2**-1074])
def test_binary_zero_and_subnormal_conventions(tmp_path,library,bias):
 path=tmp_path/'zero.lkt';pack_model(path,codes=np.zeros((2,1),dtype=np.uint8),classes=[-9,7],counts=[1,1],weights=[1],table=[1.,.5],pairs=[{'ids':[],'coef':[],'bias':bias,'profile':0}],metadata={'domain_max':1})
 with Session(path,library) as native:assert native.predict_buffer(np.zeros((7,1),dtype=np.uint8))==[7 if bias>=0 else -9]*7

@pytest.mark.parametrize('bad',['nan','float','readonly','strided','wide','domain','mode'])
def test_input_contract(tmp_path,library,bad):
 path,q,_,_=fitted(tmp_path,9,3)
 with Session(path,library) as session:
  with pytest.raises((ValueError,TypeError)):
   if bad=='nan':session.predict_buffer(np.full((1,9),np.nan))
   elif bad=='float':session.predict_buffer(q.astype(float))
   elif bad=='readonly':session.predict_buffer(bytes(q))
   elif bad=='strided':session.predict_buffer(q[::2])
   elif bad=='wide':session.predict_buffer(q[:,:8].copy())
   elif bad=='domain':session.predict_buffer(np.full((1,9),255,dtype=np.uint8))
   else:session.predict_buffer(q,mode=[])
  assert session.predict_buffer(q)==[ReferenceModel(path).predict(x) for x in q.tolist()]

@pytest.mark.parametrize('change',['magic','crc','short','tail','weights','profile','table','label'])
def test_invalid_model_rejected(tmp_path,library,change):
 path,q,_,_=fitted(tmp_path,3,3);raw=bytearray(path.read_bytes());fields=HEADER.unpack_from(raw);_,d,c,n,cap,nt,ne,terms,meta,crc=fields
 if change=='magic':raw[0]=0
 elif change=='crc':raw[-1]^=1
 elif change=='short':raw.pop()
 elif change=='tail':raw+=b'x'
 else:
  if change=='weights':struct.pack_into('<I',raw,44+meta+4*c,0)
  elif change=='profile':struct.pack_into('<I',raw,44+meta+4*c+4*d+n*d+4*3,nt)
  elif change=='table':struct.pack_into('<d',raw,len(raw)-8,math.nan)
  else:
   # metadata content length is retained while labels are made duplicates
   old=raw[44:44+meta];new=old.replace(b'[0,1,2]',b'[0,0,2]');assert len(new)==len(old);raw[44:44+meta]=new
  struct.pack_into('<I',raw,40,zlib.crc32(raw[44:]))
 path.write_bytes(raw)
 with pytest.raises(ValueError):Session(path,library)


def test_lifetime_concurrency_and_isolated_import(tmp_path,library):
 from concurrent.futures import ThreadPoolExecutor
 path,q,svc,K=fitted(tmp_path,9,3);session=Session(path,library);expected=svc.predict(K).tolist()
 with ThreadPoolExecutor(3) as pool:assert list(pool.map(lambda _:session.predict_buffer(q),range(9)))==[expected]*9
 session.close();session.close()
 with pytest.raises(ValueError):session.predict_buffer(q)
 root=Path(__file__).resolve().parents[2]
 cmd=[sys.executable,'-I','-S','-c',"import sys;sys.path.insert(0,sys.argv[1]);from experiments.learned_signature.native import Session;from array import array;s=Session(sys.argv[2],sys.argv[3]);assert len(s.predict_buffer(array('B',[0]*s.features)))==1;s.close();assert not ({'numpy','sklearn','torch','scipy'}&sys.modules.keys())",str(root),str(path),str(library)]
 done=subprocess.run(cmd,capture_output=True,text=True);assert done.returncode==0,done.stderr


def test_weighted_integer_signature_and_positive_mixture():
 rng=np.random.default_rng(97);q=rng.integers(0,101,size=(19,16),dtype=np.uint8);r=q[:7];w=(1+np.arange(16)%8).astype(np.uint32)
 got=signatures(q,r,w);expected=np.array([[sum(int(a)*(int(x)-int(y))**2 for a,x,y in zip(w,u,v)) for v in r] for u in q],dtype=np.uint32)
 assert np.array_equal(got,expected)
 coef=[.1,.6,.3];scales=[.25,1.,4.];combined=table_profile(100,w,.2,coef,scales)
 individual=sum(a*table_profile(100,w,.2*s) for a,s in zip(coef,scales))
 np.testing.assert_allclose(combined,individual,atol=1e-15,rtol=1e-15)

@pytest.mark.parametrize('cap,d,weight',[(127,16,1),(127,17,1),(127,65,1),(128,17,1),(255,17,2),(1,4096,8)])
def test_simd_exact_range_edges(tmp_path,library,cap,d,weight):
 codes=np.stack([np.zeros(d,dtype=np.uint8),np.full(d,cap,dtype=np.uint8)])
 w=np.full(d,weight,dtype=np.uint32);entries=d*weight*cap*cap+1
 table=np.exp(-np.arange(entries,dtype=np.float64)/max(1,entries-1))
 path=tmp_path/'range.lkt';pack_model(path,codes=codes,classes=[-2,3],counts=[1,1],weights=w,table=table,pairs=[{'ids':[0,1],'coef':[1.,-1.],'bias':0.,'profile':0}],metadata={'domain_max':cap})
 q=np.stack([codes[0],codes[1],np.full(d,cap//2,dtype=np.uint8)])
 oracle=ReferenceModel(path)
 with Session(path,library) as s:
  for mode in ('selective','exhaustive','scalar'):assert s.predict_buffer(q,mode=mode)==[oracle.predict(row) for row in q.tolist()]
  assert s.probe(q)==np.asarray([oracle.margins(row) for row in q.tolist()],dtype='<f8').tobytes()
