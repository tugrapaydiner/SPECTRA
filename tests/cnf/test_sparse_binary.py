"""Independent checks for exact 2-CNF residual closure (not performance evidence)."""
from itertools import product
import random
import os
import pytest
from spectra.cnf.sparse import SparseRuntime, build_sparse_runtime

@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    return SparseRuntime(os.environ.get('SPECTRA_SPARSE_LIBRARY') or build_sparse_runtime(tmp_path_factory.mktemp('binary-sparse')))

def truth(n,c,e,a=()):
    return any(all(any(w[v-1] for v in g) for g in c)
        and all(sum(w[v-1] for v in g)<=1 for g in e)
        and all(w[abs(v)-1]==(v>0) for v in a)
        for w in product((False,True),repeat=n))

def test_all_4096_small_systems(runtime):
    bank=[tuple(v+1 for v in range(3) if mask>>v&1) for mask in range(8)]
    ex=[g for g in bank if len(g)>1]
    used=0
    for mask in range(4096):
        c=tuple(g for i,g in enumerate(bank) if mask>>i&1)
        e=tuple(g for i,g in enumerate(ex,start=8) if mask>>i&1)
        r=runtime.solve(3,c,e,binary=True)
        assert (r.status=='SAT_VERIFIED')==truth(3,c,e),(mask,r)
        used+=r.binary_calls
    assert used>100

def test_2000_random_residuals(runtime):
    rng=random.Random(150028)
    for i in range(2000):
        n=rng.randrange(1,9)
        groups=[tuple(rng.sample(range(1,n+1),rng.randrange(min(5,n)+1))) for _ in range(rng.randrange(18))]
        cut=rng.randrange(len(groups)+1);c=tuple(groups[:cut]);e=tuple(groups[cut:])
        a=tuple(v*rng.choice((-1,1)) for v in rng.sample(range(1,n+1),rng.randrange(n+1)))
        r=runtime.solve(n,c,e,assumptions=a,binary=True,degree=bool(i%2))
        assert (r.status=='SAT_VERIFIED')==truth(n,c,e,a),(i,c,e,a,r)
        assert r.work<=1000000
        assert r.state_payload_bytes<=64*1024*1024

def test_nonlocal_binary_contradiction(runtime):
    # An odd 2-colour cycle, with no initial units.
    c=((1,2),(3,4),(5,6));e=c+((1,3),(2,4),(3,5),(4,6),(5,1),(6,2))
    r=runtime.solve(6,c,e,binary=True)
    assert r.status=='UNKNOWN' and r.reason=='exhausted'
    assert r.nodes==0 and r.binary_calls==1 and r.binary_solved==0

def test_binary_sat_and_assumptions(runtime):
    c=((1,2),(3,4),(5,6));e=c+((1,3),(2,4))
    r=runtime.solve(6,c,e,binary=True,assumptions=(-1,))
    assert r.status=='SAT_VERIFIED' and not r.witness[0]
    assert r.binary_solved==1 and r.nodes==0

@pytest.mark.parametrize('work',[0,1,2,3,10,100])
def test_binary_work_caps(runtime,work):
    r=runtime.solve(6,((1,2),(3,4),(5,6)),(),binary=True,max_work=work)
    assert r.work<=work
    if work<13:assert r.status=='UNKNOWN' and r.reason=='budget'

def test_payload_refusal_preserves_ordinary_search(runtime):
    c=((1,2),(3,4));e=c
    ordinary=runtime.solve(4,c,e)
    r=runtime.solve(4,c,e,binary=True,max_state_bytes=ordinary.state_payload_bytes)
    assert r.status=='SAT_VERIFIED' and r.binary_calls==0
    assert r.witness==ordinary.witness

def test_wide_exclusion_is_not_silently_dropped(runtime):
    c=((1,2),(3,4));e=((1,2,3,4),)
    r=runtime.solve(4,c,e,binary=True)
    assert r.status=='UNKNOWN' and not truth(4,c,e)

@pytest.mark.parametrize('bad',[1,None,'true'])
def test_binary_option_requires_bool(runtime,bad):
    with pytest.raises(ValueError):runtime.solve(2,((1,2),),(),binary=bad)
