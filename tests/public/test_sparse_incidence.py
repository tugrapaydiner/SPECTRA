"""Truth-table, reversible-state, ownership and resource-contract tests."""
from itertools import combinations,product
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import random
import pytest
from spectra.cnf.sparse import SparseRuntime,build_sparse_runtime,valid_witness

@pytest.fixture(scope='session')
def sparse_runtime(tmp_path_factory):
    library=os.environ.get('SPECTRA_SPARSE_LIBRARY')
    if not library:library=build_sparse_runtime(tmp_path_factory.mktemp('sparse-native'))
    return SparseRuntime(library)


def original(n,covers,exclusive,witness,assumptions=()):
    return (all(sum(witness[v-1] for v in g)>=1 for g in covers)
            and all(sum(witness[v-1] for v in g)<=1 for g in exclusive)
            and all(witness[abs(v)-1]==(v>0) for v in assumptions))


def satisfiable(n,covers,exclusive,assumptions=()):
    return any(original(n,covers,exclusive,w,assumptions) for w in product((False,True),repeat=n))


def stable(r):return {k:v for k,v in r.record().items() if k not in ('elapsed_ns','state_payload_bytes')}


def test_exhaustive_4096_three_variable_constraint_systems(sparse_runtime):
    groups=[tuple(i+1 for i in range(3) if mask>>i&1) for mask in range(8)]
    excludes=[g for g in groups if len(g)>1]
    assert len(groups)+len(excludes)==12
    for mask in range(4096):
        c=tuple(g for i,g in enumerate(groups) if mask>>i&1)
        e=tuple(g for i,g in enumerate(excludes,start=8) if mask>>i&1)
        truth=satisfiable(3,c,e)
        a=sparse_runtime.solve(3,c,e,heap=False)
        b=sparse_runtime.solve(3,c,e,heap=True)
        assert (a.status=='SAT_VERIFIED')==truth,(mask,a)
        assert stable(a)==stable(b),(mask,a,b)
        if truth:assert original(3,c,e,a.witness)


def test_3000_random_systems_and_assumptions(sparse_runtime):
    rng=random.Random(48091)
    for i in range(3000):
        n=rng.randrange(9)
        rows=[tuple(v+1 for v in range(n) if rng.randrange(3)==0) for _ in range(rng.randrange(20))]
        split=rng.randrange(len(rows)+1);c=tuple(rows[:split]);e=tuple(rows[split:])
        assumptions=tuple((v+1)*rng.choice((-1,1)) for v in range(n) if rng.randrange(3)==0)
        if n and i%7==0:assumptions=(1,-1)
        a=sparse_runtime.solve(n,c,e,heap=bool(i%2),assumptions=assumptions)
        assert (a.status=='SAT_VERIFIED')==satisfiable(n,c,e,assumptions),(i,n,c,e,assumptions,a)
        assert a.reason in ('satisfied','exhausted')
        if a.status=='SAT_VERIFIED':assert original(n,c,e,a.witness,assumptions)


def test_rollback_across_repeated_queries(sparse_runtime):
    c=((1,2),(3,4),(1,3),(2,4));e=c
    with sparse_runtime.prepare(4,c,e) as p:
        for mode in (False,True):
            baseline=p.solve(heap=mode)
            for assumptions in ((1,),(-1,),(1,3),(1,-1),(-4,),()):
                r=p.solve(heap=mode,assumptions=assumptions)
                assert (r.status=='SAT_VERIFIED')==satisfiable(4,c,e,assumptions)
                assert stable(p.solve(heap=mode))==stable(baseline)

@pytest.mark.parametrize('limit',[0,1,2,3,4,7,16,100])
def test_budget_and_replay(sparse_runtime,limit):
    c=((1,2),(3,4),(1,3),(2,4));e=c
    a=sparse_runtime.solve(4,c,e,max_work=limit)
    b=sparse_runtime.solve(4,c,e,max_work=limit)
    assert a.work<=limit
    assert stable(a)==stable(b)
    if a.status=='SAT_VERIFIED':assert original(4,c,e,a.witness)

@pytest.mark.parametrize('bad',[True,False,-1,1.5,'8',None,2**65])
def test_exact_unsigned_configuration(sparse_runtime,bad):
    for field in ('max_work','max_state_bytes','max_build_bytes'):
        with pytest.raises(ValueError):sparse_runtime.solve(2,((1,2),),(),**{field:bad})

@pytest.mark.parametrize('n,c,e',[(True,(),()),(-1,(),()),(1000001,(),()),(2,[],()),
    (2,([1],),()),(2,((1,1),),()),(2,((0,),),()),(2,((3,),),()),(2,((True,),),()),(2,((1.0,),),()),
    (0,((1,),),()),(2,(),((1,1),)),(2,(),((1,-2),))])
def test_admission_refuses_bad_inputs(sparse_runtime,n,c,e):
    with pytest.raises(ValueError):sparse_runtime.solve(n,c,e)

@pytest.mark.parametrize('bad',[(True,),(0,),(3,),(-3,),(2**100,),[1],(1,2,1,2,1)])
def test_bad_assumptions(sparse_runtime,bad):
    with pytest.raises(ValueError):sparse_runtime.solve(2,((1,2),),(),assumptions=bad)


def test_linear_payload_cap(sparse_runtime):
    with pytest.raises(MemoryError):sparse_runtime.solve(2,((1,2),),(),max_build_bytes=1)
    with pytest.raises(MemoryError):sparse_runtime.solve(2,((1,2),),(),max_state_bytes=1)
    a=sparse_runtime.solve(1000,(tuple(range(1,1001)),),(tuple(range(1,1001)),))
    b=sparse_runtime.solve(2000,(tuple(range(1,2001)),),(tuple(range(1,2001)),))
    assert b.index_bytes<2*a.index_bytes
    assert b.state_payload_bytes<2*a.state_payload_bytes
    assert b.index_bytes<50000


def test_large_exclusion_does_not_form_dense_conflicts(sparse_runtime):
    group=tuple(range(1,40001))
    a=sparse_runtime.solve(40000,(group,),(group,),max_work=50000,max_build_bytes=2*1024*1024,max_state_bytes=2*1024*1024)
    assert a.status=='SAT_VERIFIED' and sum(a.witness)==1
    assert a.index_bytes<1024*1024


def test_deep_iterative_branching(sparse_runtime):
    c=tuple((2*i+1,2*i+2) for i in range(1500))
    a=sparse_runtime.solve(3000,c,c,heap=True)
    assert a.status=='SAT_VERIFIED' and a.nodes==1500


def test_concurrent_search_and_close(sparse_runtime):
    p=sparse_runtime.prepare(4,((1,2),(3,4)),((1,3),(2,4)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        r=list(pool.map(lambda _:p.solve(),range(64)))
    assert all(stable(x)==stable(r[0]) for x in r)
    p.close();p.close()
    with pytest.raises(RuntimeError):p.solve()
    with pytest.raises(RuntimeError):p.__enter__()


def test_original_checker_rejection_is_fatal(sparse_runtime,monkeypatch):
    import spectra.cnf.sparse as module
    monkeypatch.setattr(module,'valid_witness',lambda *_:False)
    with pytest.raises(AssertionError):sparse_runtime.solve(1,((1,),),())


def test_no_original_object_reference_needed(sparse_runtime):
    c=((1,2),(3,4));e=((1,3),(2,4))
    p=sparse_runtime.prepare(4,c,e)
    del c,e
    assert p.solve().status=='SAT_VERIFIED'


def test_invalid_heap_mode(sparse_runtime):
    with pytest.raises(ValueError):sparse_runtime.solve(0,(),(),heap=1)


def test_build_cannot_replace_outputs(tmp_path):
    (tmp_path/'build.json').write_text('do not overwrite')
    with pytest.raises(FileExistsError):build_sparse_runtime(tmp_path)
    assert (tmp_path/'build.json').read_text()=='do not overwrite'


def test_missing_compiler_has_no_hidden_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):build_sparse_runtime(tmp_path,compiler='nonexistent-sparse-compiler')


def test_active_and_xor_modes_preserve_exact_search(sparse_runtime):
    rng=random.Random(590022)
    for _ in range(1200):
        n=rng.randrange(1,14)
        rows=[tuple(v+1 for v in range(n) if rng.randrange(3)==0) for __ in range(rng.randrange(2,24))]
        k=rng.randrange(len(rows)+1);c=tuple(rows[:k]);e=tuple(rows[k:])
        ref=sparse_runtime.solve(n,c,e,xor_units=False)
        for options in ({'xor_units':True},{'active':True,'xor_units':False},
                        {'active':True,'xor_units':True},{'heap':True,'xor_units':True}):
            assert stable(ref)==stable(sparse_runtime.solve(n,c,e,**options))


def test_assumption_conversion_is_charged(sparse_runtime):
    with sparse_runtime.prepare(10,(),()) as p:
        empty=p.solve(max_work=0)
        r=p.solve(assumptions=(-1,-2,-3))
        assert r.state_payload_bytes==empty.state_payload_bytes+12
        with pytest.raises(MemoryError):p.solve(assumptions=(-1,-2,-3),max_state_bytes=empty.state_payload_bytes+11)


@pytest.mark.parametrize('field',['heap','active','xor_units'])
def test_strategy_flags_are_not_numeric_aliases(sparse_runtime,field):
    with pytest.raises(ValueError):sparse_runtime.solve(2,((1,2),),(),**{field:1})


def test_incompatible_priority_structures_refused(sparse_runtime):
    with pytest.raises(ValueError):sparse_runtime.solve(2,((1,2),),(),heap=True,active=True)


def test_valid_budget_exit_does_not_lose_an_independently_valid_incumbent(sparse_runtime):
    # Even without search, the all-false assignment satisfies this particular input.
    r=sparse_runtime.solve(4,(),((1,2,3,4),),max_work=0,assumptions=(-1,))
    assert r.status=='SAT_VERIFIED' and r.reason=='budget'
    assert valid_witness(4,(),((1,2,3,4),),r.witness,(-1,))


def test_degree_and_lcv_preserve_truth_and_priority_agreement(sparse_runtime):
    rng=random.Random(271003)
    for i in range(1500):
        n=rng.randrange(1,9)
        rows=[tuple(v+1 for v in range(n) if rng.randrange(3)==0) for _ in range(rng.randrange(3,22))]
        split=rng.randrange(len(rows)+1);c,e=tuple(rows[:split]),tuple(rows[split:])
        assumptions=tuple((v+1)*rng.choice((-1,1)) for v in range(n) if rng.randrange(4)==0)
        truth=satisfiable(n,c,e,assumptions)
        for degree,lcv in ((True,False),(False,True),(True,True)):
            a=sparse_runtime.solve(n,c,e,degree=degree,lcv=lcv,assumptions=assumptions)
            assert (a.status=='SAT_VERIFIED')==truth,(i,degree,lcv,a)
            # Data structures may not change the configured decision rule.
            for priority in ({'heap':True},{'active':True}):
                b=sparse_runtime.solve(n,c,e,degree=degree,lcv=lcv,assumptions=assumptions,**priority)
                assert stable(a)==stable(b),(i,degree,lcv,priority)


@pytest.mark.parametrize('field',['degree','lcv'])
def test_branching_flags_exact_bool(sparse_runtime,field):
    with pytest.raises(ValueError):sparse_runtime.solve(2,((1,2),),(),**{field:1})
