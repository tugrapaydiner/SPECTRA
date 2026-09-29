"""Synthetic numerical/ownership contracts; no new empirical model examples."""
from array import array
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import itertools,json,math,os,random,struct,subprocess,sys,zlib
from pathlib import Path
import pytest
from . import packed
from .session import TreeSession,VerifiedCompact
from .build import build
from .reference import certificate_oracle as co


def source(leaves,*,d=1,c=1,bias=None,scale=1.,splits=None):
    trees=[]
    for k,v in enumerate(leaves):
        ss=[(k%d,.5)] if splits is None else splits[k]
        trees.append({'splits':[{'split_type':'FloatFeature','float_feature_index':f,'border':b} for f,b in ss],'leaf_values':v})
    return co.canonical({'features_info':{'float_features':[{'feature_index':f,'flat_feature_index':f,'borders':[]} for f in range(d)]},
         'scale_and_bias':[scale,[0.]*c if bias is None else bias],'oblivious_trees':trees})

def verified(raw,bits=16,pair=False,D=3,d=None):
    binary,obj=packed.compile_bytes(raw,D,features=d,bits=bits,pairwise=pair)
    return VerifiedCompact(raw,binary),obj

@pytest.fixture(scope='session')
def library(tmp_path_factory):
    return Path(os.environ['TREE_LIBRARY']) if 'TREE_LIBRARY' in os.environ else build(tmp_path_factory.mktemp('tree')/'native')

@pytest.mark.parametrize('classes',[1,2,3,10])
@pytest.mark.parametrize('bits',[8,16])
@pytest.mark.parametrize('pair',[False,True])
def test_native_full_domain_equals_rational_oracle(library,classes,bits,pair):
    rng=random.Random(71+classes)
    leaves=[[rng.uniform(-2,2) for _ in range(4*classes)] for _ in range(11)]
    raw=source(leaves,d=2,c=classes,bias=[rng.uniform(-.1,.1) for _ in range(classes)],scale=1.25,splits=[[(0,.5),(1,1.5)]]*11)
    proof,obj=verified(raw,bits,pair,D=3);oracle=co.CertifiedOracle(raw,obj)
    rows=list(itertools.product(range(4),repeat=2));buf=array('B',[x for r in rows for x in r])
    with TreeSession(library,first=proof) as engine:
        for mode,checkpoint in itertools.product(('scalar','tiled'),(0,1,3,16)):
            native=engine.inspect_buffer(buf,mode=mode,checkpoint=checkpoint)
            expected=[oracle.predict(list(r),checkpoint=checkpoint) for r in rows]
            assert native['indices']==[r['class_index'] if r['class_index'] is not None else -1 for r in expected]
            assert native['trees_evaluated']==[r['trees_evaluated'] for r in expected]
            assert sum(native['trees_evaluated'])==native['work']['first_leaf_vectors']
            for row,index in zip(rows,native['indices']):
                if index>=0:
                    scores=co.source_scores(oracle.source,list(row));assert index==max(range(len(scores)),key=lambda c:scores[c])
        assert engine.predict_buffer(array('B'))==[]

def test_wrong_int8_refines_without_repeating_routes(library):
    raw=source([[.125,100.]])
    a,_=verified(raw,8,True,D=1);b,_=verified(raw,16,True,D=1)
    with TreeSession(library,first=a,second=b) as engine:
        assert engine.predict_buffer(array('B',[0]))==[-1]
        for mode in ('scalar','tiled'):
            got=engine.inspect_buffer(array('B',[0,1]),mode=mode,refine=True)
            assert got['indices']==[1,1] and got['work']['certified_first']==1 and got['work']['certified_second']==1
            assert got['work']['routed_trees']==2

def test_exact_zero_stays_unresolved(library):
    raw=source([[0.,0.]])
    a,_=verified(raw,8,D=1);b,_=verified(raw,16,D=1)
    with TreeSession(library,first=a,second=b) as engine:
        assert engine.predict_buffer(array('B',[0,1]),refine=True)==[-1,-1]

def test_initial_early_certificate(library):
    raw=source([[.01,-.01] for _ in range(40)],bias=[10.])
    a,_=verified(raw,D=1)
    with TreeSession(library,first=a) as engine:
        got=engine.inspect_buffer(array('B',[0,1]),checkpoint=8,mode='scalar')
        assert got['indices']==[1,1] and got['trees_evaluated']==[0,0] and got['work']['routed_trees']==0

def test_constant_and_repeated_predicates(library):
    raw=source([[float(i-3) for i in range(16)]],splits=[[(0,-2.),(0,1.),(0,1.),(0,20.)]])
    a,obj=verified(raw);oracle=co.CertifiedOracle(raw,obj)
    with TreeSession(library,first=a) as s:
        assert s.predict_buffer(array('B',[0,1,2,3]))==[oracle.predict([i],checkpoint=0)['class_index'] for i in range(4)]

@pytest.fixture
def small():return verified(source([[.125,100.]]),D=1)[0]

@pytest.mark.parametrize('damage',['magic','trailing','short','crc','bound','source','bias','leaf','predicate','tree','refs','exponent'])
def test_reconstruction_rejects_corruption(small,damage):
    raw=bytearray(small.raw);h=list(packed.HEADER.unpack_from(raw));off=packed.HEADER.size
    if damage=='magic':raw[0]^=1
    elif damage=='trailing':raw+=b'x'
    elif damage=='short':raw.pop()
    elif damage=='crc':raw[-1]^=1
    elif damage=='source':raw[60]^=1
    elif damage=='exponent':struct.pack_into('<i',raw,28,901)
    elif damage=='predicate':struct.pack_into('<H',raw,off,255)
    elif damage=='tree':struct.pack_into('<I',raw,off+4,100)
    elif damage=='refs':struct.pack_into('<H',raw,off+16,5)
    elif damage=='bias':struct.pack_into('<i',raw,off+18,1)
    elif damage=='bound':struct.pack_into('<q',raw,off+18+8,2**62)
    else:raw[-1]^=1
    if damage not in ('crc','trailing','short'):struct.pack_into('<I',raw,56,zlib.crc32(raw[off:]))
    with pytest.raises(ValueError):VerifiedCompact(source([[.125,100.]]),bytes(raw))

@pytest.mark.parametrize('bad',[None,b'\0',array('d',[0]),array('B',[2]),[0]])
def test_bad_input(small,library,bad):
    with TreeSession(library,first=small) as s:
        with pytest.raises(ValueError):s.predict_buffer(bad)
        assert s.predict_buffer(array('B',[0]))==[1]

@pytest.mark.parametrize('options',[{'checkpoint':True},{'checkpoint':-1},{'mode':[]},{'refine':True},{'fallback':True},{'refine':1}])
def test_bad_settings(small,library,options):
    with TreeSession(library,first=small) as s:
        with pytest.raises(ValueError):s.predict_buffer(array('B',[0]),**options)

def test_late_input_error_preserves_all_outputs(small,library):
    with TreeSession(library,first=small) as s:
        inp=(C.c_uint8*2)(0,2);out=(C.c_int32*2)(117,118);used=(C.c_uint32*2)(119,120);stats=(C.c_uint64*8)(*([333]*8))
        assert s._lib.tc_run(s._handle,inp,2,1,1,0,0,0,out,used,stats,8)!=0
        assert list(out)==[117,118] and list(used)==[119,120] and list(stats)==[333]*8

def test_concurrent_calls_close_and_buffer_lease(small,library):
    s=TreeSession(library,first=small);buf=array('B',[0,1]);s.predict_buffer(buf);buf.append(0)
    with ThreadPoolExecutor(4) as pool:assert list(pool.map(lambda _:s.predict_buffer(buf),range(40)))==[[1,1,1]]*40
    fn=s._lib.tc_run
    def close_during(*args):
        s.close();assert s._handle;return fn(*args)
    s._lib.tc_run=close_during
    try:assert s.predict_buffer(buf)==[1,1,1]
    finally:s._lib.tc_run=fn;s.close()
    assert s._handle is None
    with pytest.raises(ValueError):s.predict_buffer(buf)

def test_refinement_requires_same_source(library):
    a,_=verified(source([[.1,2.]]),8,D=1);b,_=verified(source([[.2,2.]]),16,D=1)
    with pytest.raises(ValueError,match='binding'):TreeSession(library,first=a,second=b)

def test_nearest_rounding_only(small,library,tmp_path):
    cpp=tmp_path/'f.cpp';lib=tmp_path/'f.so';cpp.write_text('#include <cfenv>\nextern "C"{int get(){return fegetround();}int set(int x){return fesetround(x);}int down(){return FE_DOWNWARD;}}')
    subprocess.run(['g++','-shared','-fPIC',str(cpp),'-o',str(lib)],check=True)
    f=C.CDLL(str(lib));f.get.restype=C.c_int;f.down.restype=C.c_int;f.set.argtypes=[C.c_int];old=f.get()
    with TreeSession(library,first=small) as s:
        try:
            assert f.set(f.down())==0
            with pytest.raises(ValueError,match='nearest'):s.predict_buffer(array('B',[0]))
        finally:assert f.set(old)==0

def test_no_framework_import_in_deployment(small,library,tmp_path):
    path=tmp_path/'c.sct';src=tmp_path/'source.json';path.write_bytes(small.raw);src.write_bytes(source([[.125,100.]]))
    root=Path(__file__).resolve().parents[2]
    code='import sys;sys.path.insert(0,sys.argv[1]);from experiments.certified_trees.session import VerifiedCompact,TreeSession;from array import array;p=VerifiedCompact.from_files(sys.argv[2],sys.argv[3]);s=TreeSession(sys.argv[4],first=p);assert s.predict_buffer(array("B",[0]))==[1];s.close();assert not ({"numpy","scipy","torch","sklearn","catboost"}&sys.modules.keys())'
    subprocess.run([sys.executable,'-I','-S','-c',code,str(root),str(src),str(path),str(library)],check=True)

@pytest.mark.parametrize('depth',[8,9,12])
def test_deep_repeated_predicate_vector_fallback(library,depth):
    rng=random.Random(481+depth)
    raw=source([[rng.uniform(-1,1) for _ in range((1<<depth)*3)]],d=2,c=3,
               splits=[[(j%2,(-2.,127.,128.,255.,300.)[j%5]) for j in range(depth)]])
    p,obj=verified(raw,bits=16,D=255)
    oracle=co.CertifiedOracle(raw,obj)
    rows=[list(r) for r in itertools.product((0,127,128,255),repeat=2)]*3
    with TreeSession(library,first=p) as s:
        got=s.predict_buffer(array('B',[v for r in rows for v in r]))
        assert got==[oracle.predict(r,checkpoint=0)['class_index'] for r in rows]

@pytest.mark.parametrize('classes',[6,8,16,26,32,33,64])
def test_register_padding_at_last_leaf(library,classes):
    rng=random.Random(871+classes)
    raw=source([[rng.uniform(-2,2) for _ in range(classes*2)]],c=classes)
    p,obj=verified(raw,bits=8,D=1);oracle=co.CertifiedOracle(raw,obj)
    with TreeSession(library,first=p) as s:
        buf=array('B',[1]*65)
        assert s.predict_buffer(buf)==[oracle.predict([1],checkpoint=0)['class_index']]*65

@pytest.mark.parametrize('version',['1.2.8','1.2.10'])
def test_actual_official_fallback_on_uncertified_zero_tie(library,tmp_path,version):
    # Synthetic training only: equal labels at both inputs produce exact zero logits.
    import catboost
    import numpy as np
    upstream=os.environ.get('TREE_UPSTREAM')
    if not upstream:pytest.skip('requires explicitly supplied official native library')
    native=Path(upstream)/f'libcatboostmodel-linux-x86_64-{version}.so'
    m=catboost.CatBoostClassifier(iterations=2,depth=1,bootstrap_type='No',random_strength=0,
        thread_count=1,verbose=False,allow_writing_files=False).fit(np.asarray([[0],[0],[1],[1]],dtype=np.uint8),[0,1,0,1])
    model=tmp_path/'m.cbm';doc=tmp_path/'m.json';m.save_model(str(model));m.save_model(str(doc),format='json')
    raw=doc.read_bytes();a,_=verified(raw,8,D=1);b,_=verified(raw,16,D=1)
    with TreeSession(library,first=a,second=b,official_model=model,official_library=native) as s:
        assert s.predict_buffer(array('B',[0,1]),refine=True)==[-1,-1]
        got=s.inspect_buffer(array('B',[0,1]),refine=True,fallback=True)
        assert got['indices']==[0,0] and got['work']['official_rows']==2
        assert got['work']['certified_first']==got['work']['certified_second']==0
        assert got['work']['routed_trees']==2*s.info['trees']
    model.write_bytes(b'not a CatBoost model')
    with pytest.raises(ValueError):TreeSession(library,first=a,official_model=model,official_library=native)
