"""Exact truth-table and original-input contracts for the optional domain backend."""
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations,product
import os
import random
import pytest
from spectra.cnf.coloring import ColoringRuntime,build_coloring_runtime,check_coloring

@pytest.fixture(scope='session')
def coloring_runtime(tmp_path_factory):
    library=os.environ.get('SPECTRA_COLORING_LIBRARY')
    if not library:library=build_coloring_runtime(tmp_path_factory.mktemp('coloring'))
    return ColoringRuntime(library)


def possible(n,k,edges,masks=()):
    # Separate exhaustive observer; no native-state or library-helper dependence.
    return any(all(w[a]!=w[b] for a,b in edges) and all((d>>w[i])&1 for i,d in enumerate(masks))
               for w in product(range(k),repeat=n))


def stable(r):return {k:v for k,v in r.record().items() if k!='elapsed_ns'}


def test_exhaustive_graphs_and_all_modes(coloring_runtime):
    all_edges=tuple(combinations(range(5),2))
    for mask in range(1024):
        edges=tuple(e for i,e in enumerate(all_edges) if mask>>i&1)
        for k in (2,3):
            expected=possible(5,k,edges)
            for symmetry,binary in ((False,False),(True,False),(False,True),(True,True)):
                r=coloring_runtime.solve(5,k,edges,symmetry=symmetry,binary=binary)
                assert (r.status=='SAT_VERIFIED')==expected,(mask,k,symmetry,binary,r)
                if expected:assert check_coloring(5,k,edges,r.labels)


def test_2000_random_list_problems(coloring_runtime):
    rng=random.Random(761033)
    for i in range(2000):
        n=rng.randrange(1,8);k=rng.randrange(1,5)
        edges=tuple((a,b) for a,b in combinations(range(n),2) if rng.randrange(3)==0)
        masks=tuple(rng.randrange(1<<k) for _ in range(n))
        expected=possible(n,k,edges,masks)
        for binary in (False,True):
            r=coloring_runtime.solve(n,k,edges,masks=masks,binary=binary,symmetry=True)
            assert (r.status=='SAT_VERIFIED')==expected,(i,n,k,edges,masks,binary,r)
            if expected:assert check_coloring(n,k,edges,r.labels,masks)


def test_every_binary_list_on_triangle_and_path(coloring_runtime):
    for edges in (((0,1),(1,2)),((0,1),(1,2),(0,2))):
        for masks in product(range(8),repeat=3):
            for binary in (False,True):
                r=coloring_runtime.solve(3,3,edges,masks=masks,binary=binary)
                assert (r.status=='SAT_VERIFIED')==possible(3,3,edges,masks)

@pytest.mark.parametrize('n,k,edges,masks',[(0,0,(),()),(0,64,(),()),(1,0,(),()),(2,64,((0,1),),()),
    (2,64,((0,1),),(1<<63,1<<62)),(2,1,((0,1),),()),(1,2,((0,0),),()),
    (2,2,((0,1),(1,0),(0,1)),()),(3,3,(),(4,2,1))])
def test_edge_cases(coloring_runtime,n,k,edges,masks):
    r=coloring_runtime.solve(n,k,edges,masks=masks)
    expected=possible(n,k,edges,masks)
    assert (r.status=='SAT_VERIFIED')==expected

@pytest.mark.parametrize('cap',[0,1,2,3,4,5,10,30,100])
def test_strict_work_cap_and_replay(coloring_runtime,cap):
    edges=tuple(combinations(range(5),2))
    r=coloring_runtime.solve(5,4,edges,max_work=cap)
    assert r.work<=cap
    assert stable(r)==stable(coloring_runtime.solve(5,4,edges,max_work=cap))
    assert r.status=='UNKNOWN'

@pytest.mark.parametrize('bad',[True,False,-1,1.5,None,'12',1<<80])
def test_invalid_resource_flags(coloring_runtime,bad):
    for key in ('max_work','max_state_bytes','max_build_bytes'):
        with pytest.raises(ValueError):coloring_runtime.solve(3,3,((0,1),),**{key:bad})

@pytest.mark.parametrize('n,k,edges,masks',[(True,3,(),()),(3,True,(),()),(-1,3,(),()),(100001,3,(),()),
    (1,65,(),()),(1,1,[],()),(2,2,([0,1],),()),(2,2,((True,1),),()),(2,2,((0,2),),()),
    (2,2,((0,-1),),()),(2,2,((0,1,1),),()),(2,2,(),(1,)),(1,1,(),(True,)),
    (1,1,(),(2,)),(1,1,(),(-1,)),(1,64,(),(1<<64,))])
def test_invalid_graph_inputs(coloring_runtime,n,k,edges,masks):
    with pytest.raises(ValueError):coloring_runtime.solve(n,k,edges,masks=masks)


def test_resource_bound_refusal(coloring_runtime):
    with pytest.raises(MemoryError):coloring_runtime.solve(10,3,(),max_build_bytes=1)
    with pytest.raises(MemoryError):coloring_runtime.solve(10,3,(),max_state_bytes=1)


def test_binary_closure_is_used_and_has_exact_optout(coloring_runtime):
    edges=tuple((i,(i+1)%100) for i in range(100))
    r=coloring_runtime.solve(100,2,edges,binary=True)
    s=coloring_runtime.solve(100,2,edges,binary=False)
    assert r.status==s.status=='SAT_VERIFIED'
    assert r.binary_calls==r.binary_solutions==1 and s.binary_calls==0
    # An odd cycle is contradictory; neither engine falsely certifies UNSAT.
    e=tuple((i,(i+1)%101) for i in range(101))
    assert coloring_runtime.solve(101,2,e).status=='UNKNOWN'


def test_no_symmetry_pruning_under_asymmetric_lists(coloring_runtime):
    r=coloring_runtime.solve(3,3,((0,1),(1,2)),masks=(4,3,2),symmetry=True)
    assert r.status=='SAT_VERIFIED' and r.labels[0]==2


def test_repeated_prepared_queries_and_concurrent_close(coloring_runtime):
    edges=((0,1),(1,2),(2,3),(3,0))
    p=coloring_runtime.prepare(4,3,edges)
    expected=p.solve()
    with ThreadPoolExecutor(max_workers=6) as pool:
        runs=list(pool.map(lambda _:p.solve(),range(40)))
    assert all(stable(x)==stable(expected) for x in runs)
    assert p.solve(max_work=0).status=='UNKNOWN'
    assert stable(p.solve())==stable(expected)
    p.close();p.close()
    with pytest.raises(RuntimeError):p.solve()
    with pytest.raises(RuntimeError):p.__enter__()


def test_independent_check_rejection_stops_sat(coloring_runtime,monkeypatch):
    monkeypatch.setattr(coloring_runtime._module,'check',lambda *_:False)
    with pytest.raises(AssertionError):coloring_runtime.solve(1,1,())

@pytest.mark.parametrize('labels',[(False,True),[0,1],(0,0),(-1,0),(0,2),(0,1,2)])
def test_original_observers_reject_bad_witness(coloring_runtime,labels):
    assert not coloring_runtime.check(2,2,((0,1),),labels)
    assert not check_coloring(2,2,((0,1),),labels)


def test_native_observer_catches_last_edge_and_list(coloring_runtime):
    assert not coloring_runtime.check(3,3,((0,1),(1,2),(0,2)),(0,1,0))
    assert not coloring_runtime.check(3,3,(),(0,1,2),(1,2,1))


def test_iterative_depth_and_linear_state(coloring_runtime):
    r=coloring_runtime.solve(2000,3,(),binary=False)
    assert r.status=='SAT_VERIFIED' and r.branches==2000
    # The two static rank permutations intentionally add 8*n index bytes.
    assert r.index_payload_bytes==20*2000+4 and r.state_payload_bytes<200000


def test_search_switches_are_bool(coloring_runtime):
    for key in ('symmetry','binary','buckets'):
        with pytest.raises(ValueError):coloring_runtime.solve(1,1,(),**{key:1})


def test_explicit_build_does_not_overwrite(tmp_path):
    (tmp_path/'build.json').write_text('preserve')
    with pytest.raises(FileExistsError):build_coloring_runtime(tmp_path)
    assert (tmp_path/'build.json').read_text()=='preserve'


def test_priority_buckets_preserve_complete_paths(coloring_runtime):
    rng=random.Random(838611)
    for i in range(600):
        n=rng.randrange(1,21);k=rng.randrange(1,6)
        edges=tuple((a,b) for a,b in combinations(range(n),2) if rng.randrange(5)==0)
        masks=tuple(rng.randrange(1,1<<k) for _ in range(n)) if i%2 else ()
        a=coloring_runtime.solve(n,k,edges,masks=masks,buckets=True,max_work=20000)
        b=coloring_runtime.solve(n,k,edges,masks=masks,buckets=False,max_work=20000)
        for field in ('status','labels','reason','work','branches','backtracks','domain_reductions','binary_calls','binary_edges','binary_solutions','trail_peak','trace'):
            assert getattr(a,field)==getattr(b,field),(i,field,a,b)


def test_priority_hierarchy_across_word_and_block_boundaries(coloring_runtime):
    for n in (63,64,65,4095,4096,4097):
        # No edges isolates exact rank/size membership without an exponential search.
        a=coloring_runtime.solve(n,3,(),binary=False,buckets=True)
        b=coloring_runtime.solve(n,3,(),binary=False,buckets=False)
        assert a.labels==b.labels and a.trace==b.trace and a.work==b.work


def test_priority_bucket_sixty_four_colours(coloring_runtime):
    n=80;edges=tuple((v,(v+1)%n) for v in range(n))
    a=coloring_runtime.solve(n,64,edges,buckets=True)
    b=coloring_runtime.solve(n,64,edges,buckets=False)
    assert a.labels==b.labels and a.trace==b.trace


def test_record_has_independent_mutable_label_list(coloring_runtime):
    r=coloring_runtime.solve(3,3,((0,1),(1,2)))
    x=r.record();x['labels'][0]=123
    assert r.labels[0]!=123 and r.record()['labels'][0]!=123
