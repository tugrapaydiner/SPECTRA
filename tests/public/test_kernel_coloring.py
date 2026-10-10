"""Original-constraint truth tables for the two-choice/core decomposition."""
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations,product
import os,random
import pytest
from spectra.cnf.kernel_coloring import KernelColoringRuntime,build_kernel_coloring_runtime
from spectra.cnf.coloring import check_coloring

@pytest.fixture(scope='session')
def kernel_runtime(tmp_path_factory):
    path=os.environ.get('SPECTRA_KERNEL_COLORING_LIBRARY')
    if path is None:path=build_kernel_coloring_runtime(tmp_path_factory.mktemp('kernel'))
    return KernelColoringRuntime(path)


def possible(n,k,edges,masks):
    domains=[tuple(c for c in range(k) if (masks[v] if masks else (1<<k)-1)>>c&1) for v in range(n)]
    return any(all(labels[a]!=labels[b] for a,b in edges) for labels in product(*domains))


def stable(result):return {k:v for k,v in result.record().items() if k!='elapsed_ns'}


def test_all_five_vertex_graphs(kernel_runtime):
    bank=tuple(combinations(range(5),2))
    for mask in range(1024):
        edges=tuple(e for i,e in enumerate(bank) if mask>>i&1)
        for k in (2,3):
            expected=possible(5,k,edges,())
            for symmetry in (False,True):
                r=kernel_runtime.solve(5,k,edges,symmetry=symmetry)
                assert (r.status=='SAT_VERIFIED')==expected,(mask,k,symmetry,r)
                if expected:assert check_coloring(5,k,edges,r.labels)


def test_4000_mixed_list_graphs(kernel_runtime):
    rng=random.Random(309177)
    for i in range(4000):
        n=rng.randrange(1,9);k=rng.randrange(1,5)
        edges=tuple(e for e in combinations(range(n),2) if rng.randrange(3)==0)
        if edges and i%13==0:edges+=edges[:1]
        masks=tuple(rng.randrange(1,1<<k) for _ in range(n))
        if i%17==0:masks=(0,)+masks[1:]
        r=kernel_runtime.solve(n,k,edges,masks=masks)
        assert (r.status=='SAT_VERIFIED')==possible(n,k,edges,masks),(i,n,k,edges,masks,r)
        if r.status=='SAT_VERIFIED':assert check_coloring(n,k,edges,r.labels,masks)
        assert r.core_vertices==sum(d.bit_count()>2 for d in masks)
        assert r.binary_vertices==sum(0<d.bit_count()<=2 for d in masks)


def test_all_triangle_list_assignments(kernel_runtime):
    edges=((0,1),(1,2),(0,2))
    for masks in product(range(8),repeat=3):
        r=kernel_runtime.solve(3,3,edges,masks=masks)
        assert (r.status=='SAT_VERIFIED')==possible(3,3,edges,masks)


def test_component_completion_under_core_assumptions(kernel_runtime):
    # Different pairs form implication chains; the sole wide variable can impose
    # either feasible or contradictory units. The full graph remains the oracle.
    rng=random.Random(480387)
    for i in range(500):
        n=8;k=3
        pairs=(3,5,6)
        masks=(7,)+tuple(rng.choice(pairs) for _ in range(n-1))
        edges=tuple(e for e in combinations(range(n),2) if rng.randrange(4)==0)
        r=kernel_runtime.solve(n,k,edges,masks=masks)
        assert (r.status=='SAT_VERIFIED')==possible(n,k,edges,masks)
        assert r.core_vertices==1 and r.core_branches<=3
        if r.status=='SAT_VERIFIED':assert check_coloring(n,k,edges,r.labels,masks)


def test_binary_ring_and_contradiction(kernel_runtime):
    n=2000;edges=tuple((i,(i+1)%n) for i in range(n))
    r=kernel_runtime.solve(n,2,edges,max_search_work=0)
    assert r.status=='SAT_VERIFIED' and r.core_branches==0 and r.components==2
    assert r.condensation_arcs==0 and check_coloring(n,2,edges,r.labels)
    n=2001;edges=tuple((i,(i+1)%n) for i in range(n))
    r=kernel_runtime.solve(n,2,edges)
    assert r.status=='UNKNOWN' and r.reason=='kernel_contradiction'


def test_free_binary_components_do_not_create_search_branches(kernel_runtime):
    n=1200;edges=tuple((v,v+1) for v in range(1,n-1,2))
    masks=(7,)+(3,)*(n-1)
    r=kernel_runtime.solve(n,3,edges,masks=masks)
    assert r.status=='SAT_VERIFIED' and r.core_vertices==1 and r.core_branches==1
    assert check_coloring(n,3,edges,r.labels,masks)

@pytest.mark.parametrize('n,k,e,m',[(0,0,(),()),(1,0,(),()),(0,64,(),()),
    (1,64,(),(1<<63,)),(2,64,((0,1),),(1<<63,1<<62)),(1,2,((0,0),),()),
    (2,3,((0,1),(1,0),(0,1)),(3,7)),(3,4,(),(12,7,15))])
def test_boundary_cases(kernel_runtime,n,k,e,m):
    r=kernel_runtime.solve(n,k,e,masks=m)
    assert (r.status=='SAT_VERIFIED')==possible(n,k,e,m)

@pytest.mark.parametrize('cap',[0,1,2,3,4,5,10,30,100])
def test_search_budget_and_replay(kernel_runtime,cap):
    e=tuple(combinations(range(5),2))
    a=kernel_runtime.solve(5,4,e,max_search_work=cap)
    b=kernel_runtime.solve(5,4,e,max_search_work=cap)
    assert a.status=='UNKNOWN' and a.search_work<=cap and stable(a)==stable(b)

@pytest.mark.parametrize('bad',[True,False,-1,1.5,None,'12',1<<80])
def test_invalid_caps(kernel_runtime,bad):
    for key in ('max_search_work','max_build_bytes','max_state_bytes'):
        with pytest.raises(ValueError):kernel_runtime.solve(2,3,((0,1),),**{key:bad})

@pytest.mark.parametrize('n,k,e,m',[(True,3,(),()),(2,True,(),()),(-1,3,(),()),
    (100001,3,(),()),(1,65,(),()),(2,3,[],()),(2,3,([0,1],),()),
    (2,3,((0,True),),()),(2,3,((0,2),),()),(2,3,(),(1,)),(1,3,(),(8,)),
    (1,3,(),(True,)),(1,64,(),(1<<64,))])
def test_invalid_inputs(kernel_runtime,n,k,e,m):
    with pytest.raises(ValueError):kernel_runtime.solve(n,k,e,masks=m)


def test_payload_caps(kernel_runtime):
    with pytest.raises(MemoryError):kernel_runtime.solve(100,2,(),max_build_bytes=100)
    with pytest.raises(MemoryError):kernel_runtime.solve(100,3,(),max_state_bytes=100)
    a=kernel_runtime.solve(200,2,());b=kernel_runtime.solve(400,2,())
    assert b.index_payload_bytes<2*a.index_payload_bytes
    assert b.build_payload_bound<2*a.build_payload_bound


def test_native_checker_cannot_be_bypassed(kernel_runtime,monkeypatch):
    monkeypatch.setattr(kernel_runtime._module,'check',lambda *_:False)
    with pytest.raises(AssertionError):kernel_runtime.solve(2,2,((0,1),))


def test_repeated_prepared_and_close(kernel_runtime):
    e=((0,1),(1,2),(2,3),(3,0));m=(3,6,5,7)
    p=kernel_runtime.prepare(4,3,e,masks=m);expected=p.solve()
    with ThreadPoolExecutor(max_workers=6) as pool:r=list(pool.map(lambda _:p.solve(),range(48)))
    assert all(stable(x)==stable(expected) for x in r)
    assert stable(p.solve())==stable(expected)
    p.close();p.close()
    with pytest.raises(RuntimeError):p.solve()
    with pytest.raises(RuntimeError):p.__enter__()


def test_deep_core_uses_no_python_recursion(kernel_runtime):
    r=kernel_runtime.solve(1600,3,())
    assert r.status=='SAT_VERIFIED' and r.core_branches==1600


def test_flags_and_output_independence(kernel_runtime):
    with pytest.raises(ValueError):kernel_runtime.solve(1,3,(),symmetry=1)
    r=kernel_runtime.solve(2,2,((0,1),));d=r.record();d['labels'][0]=99
    assert r.labels[0]!=99
