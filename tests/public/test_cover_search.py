"""Independent truth-table, ownership and finite resource contracts; no learned tests."""
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations,product
import ctypes as C
import os
from pathlib import Path
import random

import pytest
from data.cnf import CNF
from spectra.cnf.cover import (ChoiceProblem, PreparedCover, UnsupportedStructure,
                              build_cover_runtime, solve_cover)

@pytest.fixture(scope='session')
def cover_library(tmp_path_factory):
    override=os.environ.get('SPECTRA_COVER_LIBRARY')
    return Path(override).resolve() if override else build_cover_runtime(tmp_path_factory.mktemp('cover'))


def truth_cnf(p,w):
    return all(any((w[abs(v)-1] if v>0 else not w[-v-1]) for v in clause) for clause in p.clauses)


def truth_choice(p,w):
    return all(sum(w[v-1] for v in g)>=1 for g in p.covers) and all(sum(w[v-1] for v in g)<=1 for g in p.exclusive)


def expected(p,check):
    return any(check(p,w) for w in product((False,True),repeat=p.nvars))


def test_all_512_two_variable_formulas(cover_library):
    clauses=[(),(1,),(2,),(1,2),(-1,),(-2,),(-1,-2),(1,-1),(2,2)]
    for mask in range(512):
        p=CNF(2,tuple(c for i,c in enumerate(clauses) if mask>>i&1))
        r=solve_cover(p,cover_library)
        assert (r.status=='SAT_VERIFIED')==expected(p,truth_cnf),(mask,r)
        assert r.nodes<=r.max_nodes
        assert r.reason==('satisfied' if r.status=='SAT_VERIFIED' else 'exhausted')
        if r.status=='SAT_VERIFIED': assert truth_cnf(p,r.witness)


def test_500_random_noncanonical_formulas(cover_library):
    rng=random.Random(7181)
    for _ in range(500):
        n=rng.randrange(1,7);clauses=[]
        for __ in range(rng.randrange(16)):
            if rng.randrange(2):g=[rng.randrange(1,n+1) for _ in range(rng.randrange(6))]
            else:g=[-rng.randrange(1,n+1) for _ in range(rng.randrange(1,3))]
            if g and rng.randrange(3)==0:g+=g
            if rng.randrange(5)==0:g += [1,-1]  # a genuine mixed tautology
            rng.shuffle(g);clauses.append(tuple(g))
        p=CNF(n,tuple(clauses));r=solve_cover(p,cover_library)
        assert (r.status=='SAT_VERIFIED')==expected(p,truth_cnf)
        assert r.status in ('SAT_VERIFIED','UNKNOWN')


def test_500_random_choice_instances_and_cnf_correspondence(cover_library):
    rng=random.Random(20261007)
    for _ in range(500):
        n=rng.randrange(7)
        groups=[tuple(i+1 for i in range(n) if rng.randrange(2)) for _ in range(rng.randrange(10))]
        split=rng.randrange(len(groups)+1)
        p=ChoiceProblem(n,tuple(groups[:split]),tuple(groups[split:]))
        f=CNF(n,p.covers+tuple((-a,-b) for g in p.exclusive for a,b in combinations(g,2)))
        for w in product((False,True),repeat=n): assert truth_choice(p,w)==truth_cnf(f,w)
        r=solve_cover(p,cover_library);s=solve_cover(f,cover_library)
        assert (r.status=='SAT_VERIFIED')==expected(p,truth_choice)==(s.status=='SAT_VERIFIED')
        if r.status=='SAT_VERIFIED':assert truth_choice(p,r.witness)

@pytest.mark.parametrize('clause',[(1,-2),(-1,-2,-3),(-1,-2,-3,-1),(1,2,-3)])
def test_unsupported_clauses_fail_closed(cover_library,clause):
    with pytest.raises(UnsupportedStructure):solve_cover(CNF(3,(clause,)),cover_library)

@pytest.mark.parametrize('n,clauses',[(0,()),(0,((),)),(1,((1,),)),(1,((-1,),)),(3,((1,-1,-2,-3),)),(3,((-1,-2,-1,-2),))])
def test_edge_semantics(cover_library,n,clauses):
    p=CNF(n,clauses);r=solve_cover(p,cover_library)
    assert (r.status=='SAT_VERIFIED')==expected(p,truth_cnf)

@pytest.mark.parametrize('budget',[0,1,2,3,8,64])
def test_exact_node_budget_and_deterministic_replay(cover_library,budget):
    p=ChoiceProblem(4,((1,2),(3,4),(1,3),(2,4)),((1,2),(3,4),(1,3),(2,4)))
    with PreparedCover(p,cover_library) as index:
        a=index.solve(max_nodes=budget);b=index.solve(max_nodes=budget)
    assert a.nodes<=budget
    assert {k:v for k,v in a.record().items() if k!='elapsed_ns'}=={k:v for k,v in b.record().items() if k!='elapsed_ns'}
    if a.status=='SAT_VERIFIED': assert truth_choice(p,a.witness)

@pytest.mark.parametrize('bad',[True,False,-1,1.5,'1',None,2**64])
def test_strict_search_flags(cover_library,bad):
    p=CNF(1,((1,),))
    for field in ('max_nodes','max_state_bytes','max_index_bytes'):
        with pytest.raises(ValueError):solve_cover(p,cover_library,**{field:bad})


def test_index_cap_and_state_cap(cover_library):
    p=ChoiceProblem(4,((1,2),(3,4)),((1,2),(3,4)))
    with pytest.raises(MemoryError):solve_cover(p,cover_library,max_index_bytes=1)
    with pytest.raises(MemoryError):solve_cover(p,cover_library,max_state_bytes=1)
    with PreparedCover(p,cover_library) as index:
        # State payload = three uint64 words and two uint32 counts = 32 bytes.
        r=index.solve(max_state_bytes=32)
        assert r.status=='UNKNOWN' and r.reason=='state_budget'
        assert r.state_word_bytes_peak<=32
        r=index.solve(max_state_bytes=96)
        assert r.status=='SAT_VERIFIED' and r.state_word_bytes_peak<=96


def test_depth_above_python_recursion_limit(cover_library):
    p=ChoiceProblem(2200,tuple((2*i+1,2*i+2) for i in range(1100)),())
    r=solve_cover(p,cover_library,max_nodes=1200,max_state_bytes=16*1024*1024)
    assert r.status=='SAT_VERIFIED' and r.decisions==1100
    assert truth_choice(p,r.witness)


def test_concurrent_calls_own_state_and_close(cover_library):
    p=ChoiceProblem(4,((1,2),(3,4)),((1,3),(2,4)))
    index=PreparedCover(p,cover_library)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(lambda _:index.solve(),range(48)))
    assert all(r.witness==results[0].witness for r in results)
    index.close();index.close()
    with pytest.raises(RuntimeError):index.solve()
    with pytest.raises(RuntimeError):index.__enter__()


def test_original_input_checker_is_not_bypassed(cover_library,monkeypatch):
    p=CNF(1,((1,),))
    monkeypatch.setattr(CNF,'violated',lambda self,w:(0,))
    with pytest.raises(AssertionError):solve_cover(p,cover_library)

@pytest.mark.parametrize('n,c,e',[(True,(),()),(-1,(),()),(65537,(),()),(2,[[1]],()),(2,((1,1),),()),(2,((3,),),()),(2,(),((0,),)),(2,((True,),),())])
def test_invalid_choice_inputs(n,c,e):
    with pytest.raises((ValueError,TypeError)):ChoiceProblem(n,c,e)


def test_explicit_build_refuses_overwrite(tmp_path):
    (tmp_path/'spectra_cover.so').write_bytes(b'do not replace')
    with pytest.raises(FileExistsError):build_cover_runtime(tmp_path)
    assert (tmp_path/'spectra_cover.so').read_bytes()==b'do not replace'


def test_unavailable_compiler_has_no_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):build_cover_runtime(tmp_path,compiler='missing-spectra-compiler-112233')


def test_native_parser_rejects_bad_offsets_and_literals(cover_library):
    # Valid accessible buffers, invalid content; pointer sizes are caller's contract.
    with PreparedCover(CNF(1,()),cover_library) as index:
        lib=index._library
        for offsets,literals in [([1,1],[1]),([0,2],[1]),([0,1],[0]),([0,1],[-2147483648])]:
            o=(C.c_uint64*2)(*offsets);l=(C.c_int32*1)(*literals)
            out=C.c_void_p();err=C.create_string_buffer(512)
            code=lib.spectra_cover_create(1,1,o,l,1,1024,1,C.byref(out),err,512)
            assert code and not out.value and err.value


def test_dense_ablation_preserves_search_and_witness(cover_library):
    from experiments.cover_search.tasks import colouring,build
    for i in range(32):
        p=build(colouring(12,3,401+i,planted=bool(i%2)))
        a=solve_cover(p,cover_library);b=solve_cover(p,cover_library,incremental=False)
        for field in ('status','witness','unsatisfied','reason','nodes','decisions','propagations','backtracks','trace_fingerprint'):
            assert getattr(a,field)==getattr(b,field)


def test_incremental_flag_is_boolean(cover_library):
    with pytest.raises(TypeError):solve_cover(CNF(0,()),cover_library,incremental=1)


def test_contradiction_still_counts_allocated_root_state(cover_library):
    r=solve_cover(ChoiceProblem(1,((),),()),cover_library)
    assert r.status=='UNKNOWN' and r.reason=='exhausted'
    assert r.state_word_bytes_peak==28


def test_prepared_geometry_cannot_be_reassigned(cover_library):
    with PreparedCover(CNF(4,((1,2,3,4),)),cover_library) as index:
        with pytest.raises(AttributeError):index.problem=CNF(1,((1,),))
        assert len(index.solve().witness)==4


def test_mutated_frozen_input_cannot_shrink_native_output_buffer(cover_library):
    p=CNF(4,((1,2,3,4),))
    with PreparedCover(p,cover_library) as index:
        object.__setattr__(p,'nvars',1)  # deliberately bypass dataclass contract
        with pytest.raises(ValueError):index.solve()
