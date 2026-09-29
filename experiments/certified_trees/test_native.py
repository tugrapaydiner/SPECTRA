"""Synthetic exhaustive certificate and native ABI contracts, not accuracy evidence."""
from __future__ import annotations
from array import array
from concurrent.futures import ThreadPoolExecutor
import ctypes as C,hashlib,itertools,math,os,random,struct,subprocess,sys,zlib
from pathlib import Path
import pytest
from . import certificate_oracle as co
from .packing import compile_binary,pack_original,verify_binary,HEADER
from .session import TreeSession
from .build import build

def source(trees,d=2,c=3,bias=None,scale=1.0,depth=2):
    return co.canonical({'features_info':{'float_features':[{'feature_index':i,'flat_feature_index':i} for i in range(d)]},
        'scale_and_bias':[scale,[0.0]*c if bias is None else bias],
        'oblivious_trees':[{'splits':[{'split_type':'FloatFeature','float_feature_index':j%d,'border':.5+j} for j in range(depth)],'leaf_values':t} for t in trees]})

def save(path,blob):path.write_bytes(blob);return path,hashlib.sha256(blob).hexdigest()
@pytest.fixture(scope='session')
def library(tmp_path_factory):
    p=os.environ.get('TREE_LIBRARY');return Path(p) if p else build(tmp_path_factory.mktemp('native')/'build','portable')

@pytest.mark.parametrize('classes',[1,2,3,6,10,26,64])
@pytest.mark.parametrize('bits',[8,16])
def test_exhaustive_native_oracle(tmp_path,library,classes,bits):
    rng=random.Random(317+classes);raw=source([[rng.uniform(-5,5) for _ in range(4*classes)] for _ in range(9)],c=classes,bias=[rng.uniform(-.1,.1) for _ in range(classes)],scale=1.25)
    blob,model=compile_binary(raw,3,bits=bits,pairwise=True);oracle=co.CertifiedOracle(raw,model)
    path,digest=save(tmp_path/'q.sct',blob)
    full=pack_original(oracle.source);fpath,fh=save(tmp_path/'f.sct',full)
    rows=list(itertools.product(range(4),repeat=2));x=bytearray(v for row in rows for v in row)
    with TreeSession(path,library,expected_sha256=digest) as w,TreeSession(fpath,library,expected_sha256=fh) as f:
        direct=[]
        for row in rows:
            scores=co.source_scores(oracle.source,row);direct.extend(scores)
        assert f.scores(x)==struct.pack('<'+'d'*len(direct),*direct)
        for cp in (0,1,3,16):
            for pair in (True,False):
                result=w.inspect_buffer(x,checkpoint=cp,pairwise=pair)
                scalar=w.inspect_buffer(x,checkpoint=cp,pairwise=pair,scalar=True)
                assert result==scalar
                wanted=[oracle.predict(row,checkpoint=cp,use_pairwise=pair) for row in rows]
                assert result['indices']==[-1 if r['class_index'] is None else r['class_index'] for r in wanted]
                assert result['trees_evaluated']==[r['trees_evaluated'] for r in wanted]
                assert result['approximate_indices']==[-1 if r['approximate_class_index'] is None else r['approximate_class_index'] for r in wanted]
                assert w.hybrid(f,x,checkpoint=cp)==f.predict_buffer(x)
        with ThreadPoolExecutor(4) as pool:assert list(pool.map(lambda _:w.predict_buffer(x),range(9)))==[w.predict_buffer(x)]*9
        assert w.predict_buffer(bytearray())==[]

@pytest.fixture
def fixture(tmp_path,library):
    raw=source([[.125,100.]],d=1,c=1,depth=1)
    pair={}
    for bits in (8,16):
        blob,m=compile_binary(raw,1,bits=bits,pairwise=True);pair[bits]=save(tmp_path/f'q{bits}.sct',blob)
    pair['full']=save(tmp_path/'full.sct',pack_original(co.parse_source(raw,1)))
    return raw,pair

def test_wrong_quantized_natural_boundary_refused(fixture,library):
    raw,paths=fixture
    with TreeSession(paths[8][0],library,expected_sha256=paths[8][1]) as q8,TreeSession(paths[16][0],library,expected_sha256=paths[16][1]) as q16:
        r=q8.inspect_buffer(bytearray([0]),checkpoint=0)
        assert r['indices']==[-1] and r['approximate_indices']==[0]
        assert q16.predict_buffer(bytearray([0]))==[1]
        assert q8.hybrid(q16,bytearray([0]),inspect=True)['fallback']==[1]
        assert q8.hybrid(q16,bytearray([0]))==[1]

def test_input_release_and_deferred_close(fixture,library):
    _,paths=fixture;path,h=paths[16];w=TreeSession(path,library,expected_sha256=h);x=array('B',[0,1])
    w.predict_buffer(x);x.extend([0]);run=w._lib.st_run
    def closing(*args):w.close();assert w._handle is not None;return run(*args)
    w._lib.st_run=closing
    try:assert w.predict_buffer(x)==[1,1,1]
    finally:w._lib.st_run=run;w.close()
    assert w._handle is None
    with pytest.raises(ValueError):w.predict_buffer(x)

@pytest.mark.parametrize('kind',['magic','crc','truncated','extra','features','classes','maximum','tree_count','predicates','splits','leaf_count','bits','fine','flags','bias','bounds','predicate_feature','predicate_threshold','tree_offset','tree_depth','split_index','leaf_min'])
def test_malformed_bytes_rejected(fixture,library,tmp_path,kind):
    _,paths=fixture;raw=bytearray(paths[8][0].read_bytes());h=HEADER.unpack_from(raw);c=h[2]
    pred=120+20*c+(8*c*c if h[10] else 0);desc=pred+4*h[5];split=desc+12*h[4];leaf=split+2*h[6]
    if kind=='magic':raw[0]^=1
    elif kind=='crc':raw[-1]^=1
    elif kind=='truncated':raw.pop()
    elif kind=='extra':raw+=b'x'
    elif kind in ('features','classes','maximum','tree_count','predicates','splits','leaf_count','bits','fine','flags'):
        index={'features':1,'classes':2,'maximum':3,'tree_count':4,'predicates':5,'splits':6,'leaf_count':7,'bits':8,'fine':9,'flags':10}[kind]
        struct.pack_into('<I',raw,8+(index-1)*4,999999)
    elif kind=='bias':struct.pack_into('<i',raw,120,2147483647)
    elif kind=='bounds':struct.pack_into('<q',raw,120+4*c,-2**63)
    elif kind=='predicate_feature':struct.pack_into('<H',raw,pred,256)
    elif kind=='predicate_threshold':struct.pack_into('<h',raw,pred+2,-2)
    elif kind=='tree_offset':struct.pack_into('<I',raw,desc,12)
    elif kind=='tree_depth':struct.pack_into('<I',raw,desc+8,13)
    elif kind=='split_index':struct.pack_into('<H',raw,split,65535)
    else:raw[leaf]=128
    if kind not in ('crc','truncated','extra'):struct.pack_into('<I',raw,52,zlib.crc32(raw[120:]))
    path,digest=save(tmp_path/'bad.sct',bytes(raw))
    with pytest.raises(ValueError):TreeSession(path,library,expected_sha256=digest)

@pytest.mark.parametrize('kind',['readonly','wrong_format','bad_code','bad_mode','bool_mode','nonbool_pair','bad_hash'])
def test_invalid_request(fixture,library,kind):
    _,paths=fixture;path,h=paths[8]
    if kind=='bad_hash':
        with pytest.raises(ValueError):TreeSession(path,library,expected_sha256='0'*64)
        return
    with TreeSession(path,library,expected_sha256=h) as w:
        with pytest.raises(ValueError):
            if kind=='readonly':w.predict_buffer(b'\x00')
            elif kind=='wrong_format':w.predict_buffer(array('h',[0]))
            elif kind=='bad_code':w.predict_buffer(bytearray([2]))
            elif kind=='bad_mode':w.predict_buffer(bytearray([0]),checkpoint=-1)
            elif kind=='bool_mode':w.predict_buffer(bytearray([0]),checkpoint=True)
            else:w.predict_buffer(bytearray([0]),pairwise=1)

def test_late_invalid_leaves_output_untouched(fixture,library):
    _,paths=fixture;path,h=paths[8]
    with TreeSession(path,library,expected_sha256=h) as w:
        x=(C.c_uint8*2)(0,2);out=(C.c_int*2)(117,118);steps=(C.c_uint32*2)(119,120);approx=(C.c_int*2)(121,122)
        assert w._lib.st_run(w._handle,x,2,1,0,1,0,out,steps,approx)!=0
        assert list(out)==[117,118] and list(steps)==[119,120] and list(approx)==[121,122]

def test_tie_fallback_and_zero_tree_early_exit(tmp_path,library):
    for i,bias in enumerate(([0.,0.,0.],[0.,100.,0.])):
        raw=source([[0.]*12]*3,c=3,bias=bias)
        blob,_=compile_binary(raw,3,bits=16);path,h=save(tmp_path/f'{i}.sct',blob)
        with TreeSession(path,library,expected_sha256=h) as w:
            r=w.inspect_buffer(bytearray([0,0]),checkpoint=1)
            if i==0:assert r['indices']==[-1] # Conservative source roundoff, no invented tie proof.
            else:assert r['indices']==[1] and r['trees_evaluated']==[0]

def test_mismatched_fallback_source(fixture,library,tmp_path):
    _,paths=fixture;raw=source([[.125,99.]],d=1,c=1,depth=1);path,h=save(tmp_path/'other.sct',pack_original(co.parse_source(raw,1)))
    with TreeSession(paths[8][0],library,expected_sha256=paths[8][1]) as a,TreeSession(path,library,expected_sha256=h) as b:
        with pytest.raises(ValueError):a.hybrid(b,bytearray([0]))

def test_binary_reconstruction_rejects_rebound_corruption(fixture):
    raw,paths=fixture;b=bytearray(paths[8][0].read_bytes());b[-1]^=1;struct.pack_into('<I',b,52,zlib.crc32(b[120:]))
    with pytest.raises(ValueError):verify_binary(raw,bytes(b))

def test_framework_free_import(fixture,library):
    _,paths=fixture;root=Path(__file__).resolve().parents[2]
    code='import sys;sys.path.insert(0,sys.argv[1]);from experiments.certified_trees.session import TreeSession;w=TreeSession(sys.argv[2],sys.argv[3],expected_sha256=sys.argv[4]);assert w.predict_buffer(bytearray([0]))==[1];w.close();assert not ({"numpy","scipy","catboost","sklearn","torch"}&sys.modules.keys())'
    subprocess.run([sys.executable,'-I','-S','-c',code,str(root),str(paths[16][0]),str(library),paths[16][1]],check=True)

@pytest.fixture(scope='session')
def fenv(tmp_path_factory):
    p=tmp_path_factory.mktemp('env');(p/'env.cpp').write_text('#include <cfenv>\nextern "C" {int get(){return std::fegetround();} int down(){return FE_DOWNWARD;} int set(int x){return std::fesetround(x);}}\n')
    subprocess.run(['g++','-shared','-fPIC',str(p/'env.cpp'),'-o',str(p/'env.so')],check=True)
    lib=C.CDLL(str(p/'env.so'));lib.get.restype=C.c_int;lib.down.restype=C.c_int;lib.set.argtypes=[C.c_int];return lib

def test_rounding_mode_rejected(fixture,library,fenv):
    _,paths=fixture
    with TreeSession(paths[8][0],library,expected_sha256=paths[8][1]) as w:
        old=fenv.get()
        try:
            assert fenv.set(fenv.down())==0
            with pytest.raises(ValueError):w.predict_buffer(bytearray([0]))
        finally:assert fenv.set(old)==0

@pytest.mark.parametrize('rows',[1,31,32,33,65,129])
def test_tree_major_tiles_and_tail_rows(tmp_path,library,rows):
    rng=random.Random(831);raw=source([[rng.uniform(-2,2) for _ in range(4*10)] for _ in range(7)],c=10)
    blob,model=compile_binary(raw,3,bits=16);path,h=save(tmp_path/'q.sct',blob)
    x=bytearray(rng.randrange(4) for _ in range(rows*2))
    with TreeSession(path,library,expected_sha256=h) as w:
        assert w.inspect_buffer(x)==w.inspect_buffer(x,scalar=True)
        assert len(w.predict_buffer(x))==rows

def test_constant_trees_no_predicate_storage(tmp_path,library):
    raw=source([[0.,1.,-1.]]*3,d=1,c=3,depth=0)
    blob,model=compile_binary(raw,1,bits=16);path,h=save(tmp_path/'q.sct',blob)
    with TreeSession(path,library,expected_sha256=h) as w:
        assert w.predict_buffer(bytearray([0,1]*36))==[1]*72
