"""All-domain synthetic tests for total source-arithmetic execution, not accuracy."""
from array import array
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import ctypes as C
import itertools
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import zlib
import pytest
from ..certified_trees.test_native import source
from ..certified_trees.reference import certificate_oracle as ref
from . import compiler
from .session import TotalSession
from .build import build


@pytest.fixture(scope='session')
def library(tmp_path_factory):
    override = os.environ.get('TOTAL_LIBRARY')
    return Path(override) if override else build(tmp_path_factory.mktemp('total')/'native')


def verified(raw, D=3, layout='interned'):
    return compiler.VerifiedTotal(raw, compiler.compile_bytes(raw, D, layout=layout))


def reference_scores(raw, rows, D):
    # The original independent Fraction parser validates routing. Float conversion
    # exactly recovers each original nonzero source leaf, not a contrast leaf.
    model = ref.parse_source(raw, D)
    values = [v for row in rows for v in ref.source_scores(model, list(row))]
    return struct.pack('<'+'d'*len(values), *values)


@pytest.mark.parametrize('classes', [1,2,3,6,10,26])
@pytest.mark.parametrize('seed', [7,113,911])
@pytest.mark.parametrize('layout', ['flat','interned'])
def test_full_small_domain_scores_and_total_labels(library, classes, seed, layout):
    rng = random.Random(seed)
    raw = source([[rng.uniform(-4,4) for _ in range(4*classes)] for t in range(11)],
                 d=2, c=classes, bias=[rng.uniform(-1,1) for _ in range(classes)],
                 scale=.3, splits=[[(0,.5),(1,1.5)]]*11)
    proof = verified(raw, layout=layout)
    rows = list(itertools.product(range(4), repeat=2))
    buf = array('B', [v for r in rows for v in r])
    scores = reference_scores(raw, rows, 3)
    c = proof.info['classes']
    f = struct.unpack('<'+'d'*(len(rows)*c), scores)
    expected = [max(range(c), key=lambda j:f[i*c+j]) for i in range(len(rows))]
    with TotalSession(proof, library) as session:
        for traversal in ('scalar','tiled'):
            assert session.scores(buf,traversal=traversal) == scores
            for policy in ('total','exact','audit'):
                got = session.inspect_buffer(buf,policy=policy,traversal=traversal)
                assert got['indices'] == expected
                assert got['work']['unresolved'] == 0
                assert got['work']['coarse_certified']+got['work']['exact_completed'] == len(rows)
                assert got['work']['routed_trees'] == len(rows)*11
                assert session.predict_buffer(array('B'),policy=policy) == []
        coarse = session.predict_buffer(buf,policy='certificate_only')
        assert all(a == -1 or a == b for a,b in zip(coarse,expected))
        with ThreadPoolExecutor(3) as pool:
            assert list(pool.map(lambda _:session.predict_buffer(buf),range(9))) == [expected]*9


@pytest.mark.parametrize('small', [0.,1e-14,-1e-14,1e-20,-1e-20])
def test_previous_unresolved_cancellation_has_total_original_answer(library, small):
    raw = source([[small,100.],[-100.,100.],[100.,-100.]])
    proof = verified(raw,D=1)
    model = ref.parse_source(raw,1)
    expected = max(range(2),key=ref.source_scores(model,[0]).__getitem__)
    with TotalSession(proof,library) as s:
        got = s.inspect_buffer(array('B',[0]))
        assert got['indices'] == [expected]
        assert got['work']['exact_completed'] == 1
        assert got['work']['routed_trees'] == 3
        assert got['work']['exact_leaf_vectors'] == 3
        assert s.predict_buffer(array('B',[0]),policy='certificate_only') == [-1]


def test_exact_real_argmax_is_not_a_substitute_for_source_rounding(library):
    # A real sum is positive; the source adds a tiny value to -100 then +100,
    # losing it. The source ties at zero; a mathematical exact-real sum differs.
    small=1e-20
    raw=source([[-100.,100.],[small,100.],[100.,-100.]])
    model=ref.parse_source(raw,1)
    assert ref.source_scores(model,[0],exact=True)[1] > 0
    assert ref.source_scores(model,[0])[1] == 0
    with TotalSession(verified(raw,D=1),library) as s:
        assert s.predict_buffer(array('B',[0])) == [0]


@pytest.mark.parametrize('classes',[1,2,10])
def test_ties_always_take_first_class(library,classes):
    raw=source([[0.]*(2*classes)]*5,c=classes)
    with TotalSession(verified(raw,D=1),library) as s:
        got=s.inspect_buffer(array('B',[0,1]))
        assert got['indices']==[0,0] and got['work']['exact_completed']==2


@pytest.mark.parametrize('exponent',[-880,-100,0,400,800])
@pytest.mark.parametrize('scale',[.1,1.,1.25])
def test_extreme_values_keep_original_rounding(library,exponent,scale):
    raw=source([[math.ldexp(v,exponent) for v in (1.,-1.,.5,-.25,2.,-2.)]],
               c=3,scale=scale,bias=[0.,0.,0.])
    with TotalSession(verified(raw),library) as s:
        assert s.scores(array('B',[0,1,2,3])) == reference_scores(raw,[(i,) for i in range(4)],3)
        assert -1 not in s.predict_buffer(array('B',[0,1,2,3]),policy='audit')


def test_subnormal_values_and_signed_zero_are_lossless(library):
    tiny=math.ulp(0.)
    raw=source([[tiny,-tiny],[-0.,0.],[1.,-1.]],bias=[tiny])
    p=verified(raw)
    with TotalSession(p,library) as s:
        assert s.scores(array('B',[0,1])) == reference_scores(raw,[(0,),(1,)],3)
    # Unlike a numeric-equality dict, bitwise interning does not conflate signs.
    raw=source([[0.,-0.]],c=1)
    p=verified(raw)
    assert p.info['unique_leaf_vectors']==2


@pytest.mark.parametrize('depth',[0,1,6,8,9,12])
def test_depths_constant_splits_and_tile_tails(library,depth):
    raw=source([[float(i%7-3) for i in range(1<<depth)]],
               splits=[[(0,(-1.,.5,1.5,20.)[i%4]) for i in range(depth)]])
    p=verified(raw)
    with TotalSession(p,library) as s:
        for n in (1,7,31,32,33,63,65):
            q=array('B',[i%4 for i in range(n)])
            assert s.scores(q,traversal='scalar')==s.scores(q,traversal='tiled')
            assert s.predict_buffer(q,policy='total')==s.predict_buffer(q,policy='exact')


def test_global_leaf_interning_is_exact_not_leaf_pruning(library):
    raw=source([[.1,.2,.1,.2],[.1,.2,.1,.2]],c=2)
    a,b=verified(raw,layout='flat'),verified(raw,layout='interned')
    assert b.info['unique_leaf_vectors']==1 and a.info['unique_leaf_vectors']==4
    with TotalSession(a,library) as x,TotalSession(b,library) as y:
        q=array('B',[0,1,2,3]);assert x.scores(q)==y.scores(q)
        assert y.inspect_buffer(q,policy='exact')['work']['exact_leaf_vectors']==8


@pytest.fixture
def small():
    text=source([[.0001,100.]])
    return text,verified(text,D=1)


@pytest.mark.parametrize('damage',['magic','crc','tail','short','source','basehash','class','rows','unique','width','scale','bias','index','leaf','base'])
def test_every_component_bound_to_source(small,damage):
    text,p=small;raw=bytearray(p.raw);m=compiler.metadata(p.raw)
    off=compiler.HEADER.size+m['base_bytes'];c=m['classes']
    if damage=='magic':raw[0]^=1
    elif damage=='crc':raw[-1]^=1
    elif damage=='tail':raw+=b'x'
    elif damage=='short':raw.pop()
    elif damage=='source':raw[40]^=1
    elif damage=='basehash':raw[72]^=1
    elif damage in ('class','rows','unique','width'):
        offset={'class':12,'rows':16,'unique':20,'width':24}[damage]
        struct.pack_into('<I',raw,offset,999999)
    elif damage=='scale':struct.pack_into('<d',raw,off,1.1)
    elif damage=='bias':struct.pack_into('<d',raw,off+16,.1)
    elif damage=='index':struct.pack_into('<H',raw,off+8*(c+1),m['unique_leaf_vectors'])
    elif damage=='leaf':raw[-8]^=1
    else:raw[compiler.HEADER.size+70]^=1
    if damage not in ('crc','tail','short'):struct.pack_into('<I',raw,32,zlib.crc32(raw[104:]))
    with pytest.raises(ValueError):compiler.VerifiedTotal(text,bytes(raw))


@pytest.mark.parametrize('damage',['index','nan','infinity','scale','width'])
def test_native_structural_checks(small,library,damage):
    _,proof=small;raw=bytearray(proof.raw);m=compiler.metadata(proof.raw);off=104+m['base_bytes']
    if damage=='index':struct.pack_into('<H',raw,off+8*(m['classes']+1),65535)
    elif damage=='nan':struct.pack_into('<d',raw,len(raw)-8,float('nan'))
    elif damage=='infinity':struct.pack_into('<d',raw,len(raw)-8,float('inf'))
    elif damage=='scale':struct.pack_into('<d',raw,off,0.)
    else:struct.pack_into('<I',raw,24,1)
    struct.pack_into('<I',raw,32,zlib.crc32(raw[104:]))
    with TotalSession(proof,library) as s:
        blob=C.create_string_buffer(bytes(raw),len(raw))
        assert not s._lib.tt_create(blob,len(raw))
        assert s.predict_buffer(array('B',[0]))==[1]


@pytest.mark.parametrize('value',[None,True,{},'wrong',2])
def test_bad_modes(small,library,value):
    with TotalSession(small[1],library) as s:
        with pytest.raises(ValueError):s.predict_buffer(array('B',[0]),policy=value)
        with pytest.raises(ValueError):s.scores(array('B',[0]),traversal=value)


@pytest.mark.parametrize('values',[None,[0],b'0',array('d',[0.]),array('B',[2])])
def test_input_rejection(small,library,values):
    with TotalSession(small[1],library) as s:
        with pytest.raises(ValueError):s.predict_buffer(values)
        with pytest.raises(ValueError):s.scores(values)


def test_atomic_native_outputs(small,library):
    with TotalSession(small[1],library) as s:
        data=(C.c_uint8*2)(0,2);out=(C.c_int32*2)(99,88);stats=(C.c_uint64*6)(*([77]*6))
        assert s._lib.tt_run(s._handle,data,2,1,0,1,out,stats,6)!=0
        assert list(out)==[99,88] and list(stats)==[77]*6
        scores=(C.c_double*4)(*([17.]*4))
        assert s._lib.tt_scores(s._handle,data,2,1,1,scores,4)!=0
        assert list(scores)==[17.]*4


def test_ownership_reentry_and_buffer_release(small,library):
    s=TotalSession(small[1],library);q=array('B',[0]);s.predict_buffer(q);q.extend([1])
    original=s._lib.tt_run
    def close(*args):
        s.close();assert s._handle is not None;return original(*args)
    s._lib.tt_run=close
    try:assert s.predict_buffer(q)==[1,1]
    finally:s._lib.tt_run=original;s.close()
    assert s._handle is None
    with pytest.raises(ValueError):s.predict_buffer(q)
    with pytest.raises(ValueError):s.scores(q)


def test_only_verified_immutable_object(small,library):
    with pytest.raises(ValueError):TotalSession(small[1].raw,library)
    with pytest.raises(FrozenInstanceError):small[1].raw=b''
    with pytest.raises(TypeError):small[1].info['classes']=99


def test_large_lossless_id_width(library):
    # 65,537+ distinct vectors makes the uint32 route-to-vector table necessary.
    depth=12;nt=17;c=2
    leaves=[[x for i in range(1<<depth) for x in (0.,float(t*(1<<depth)+i))] for t in range(nt)]
    raw=source(leaves,c=c,splits=[[(0,.5)]*depth]*nt)
    p=verified(raw,D=1)
    assert p.info['unique_leaf_vectors']==69632 and p.info['leaf_index_bytes']==4*69632
    with TotalSession(p,library) as s:
        assert s.predict_buffer(array('B',[0,1]),policy='audit')==[1,1]


@pytest.fixture(scope='session')
def fenv(tmp_path_factory):
    folder=tmp_path_factory.mktemp('fenv');src=folder/'probe.cpp';lib=folder/'probe.so'
    src.write_text('#include <cfenv>\n#include <xmmintrin.h>\nextern "C" { int get(){return std::fegetround();} int down(){return FE_DOWNWARD;} int set(int v){return std::fesetround(v);} unsigned mxget(){return _mm_getcsr();} void mxset(unsigned v){_mm_setcsr(v);} }\n')
    subprocess.run(['g++','-shared','-fPIC',str(src),'-o',str(lib)],check=True,capture_output=True)
    x=C.CDLL(str(lib));x.get.restype=C.c_int;x.down.restype=C.c_int;x.set.argtypes=[C.c_int]
    x.mxget.restype=C.c_uint;x.mxset.argtypes=[C.c_uint]
    return x


@pytest.mark.parametrize('mode',['rounding','ftz','daz'])
def test_numerical_environment_rejected(small,library,fenv,mode):
    with TotalSession(small[1],library) as s:
        old=fenv.get();mx=fenv.mxget()
        try:
            if mode=='rounding':assert fenv.set(fenv.down())==0
            else:fenv.mxset(mx|(0x8000 if mode=='ftz' else 0x40))
            with pytest.raises(ValueError):s.predict_buffer(array('B',[0]))
            with pytest.raises(ValueError):s.scores(array('B',[0]))
        finally:fenv.set(old);fenv.mxset(mx)
