"""Within-search conflict priorities must not change supported input semantics."""
from itertools import product
import random
import os
import pytest
from spectra.cnf.sparse import SparseRuntime,build_sparse_runtime,valid_witness
@pytest.fixture(scope='module')
def rt(tmp_path_factory):return SparseRuntime(os.environ.get('SPECTRA_SPARSE_LIBRARY') or build_sparse_runtime(tmp_path_factory.mktemp('weighted-sparse')))

def test_4096_systems(rt):
    bank=[tuple(v+1 for v in range(3) if mask>>v&1) for mask in range(8)];ex=[g for g in bank if len(g)>1]
    for mask in range(4096):
        c=tuple(g for i,g in enumerate(bank) if mask>>i&1);e=tuple(g for i,g in enumerate(ex,start=8) if mask>>i&1)
        r=rt.solve(3,c,e,wdeg=True,compact_updates=True,native_check=True)
        assert (r.status=='SAT_VERIFIED')==any(valid_witness(3,c,e,w) for w in product((False,True),repeat=3))

def test_3000_random_systems_and_modes(rt):
    rng=random.Random(273143)
    for i in range(3000):
        n=rng.randrange(1,9);bank=[tuple(rng.sample(range(1,n+1),rng.randrange(n+1))) for _ in range(rng.randrange(25))]
        cut=rng.randrange(len(bank)+1);c,e=tuple(bank[:cut]),tuple(bank[cut:]);a=tuple(v*rng.choice((-1,1)) for v in range(1,n+1) if rng.randrange(4)==0)
        r=rt.solve(n,c,e,wdeg=True,heap=i%3==0,active=i%3==1,degree=bool(i%2),binary=bool(i%5),compact_updates=bool(i%7),assumptions=a)
        assert (r.status=='SAT_VERIFIED')==any(valid_witness(n,c,e,w,a) for w in product((False,True),repeat=n)),(i,c,e,a,r)

def test_query_state_does_not_leak(rt):
    c=((1,2),(2,3),(1,3));e=c
    with rt.prepare(3,c,e) as p:
        a=p.solve(wdeg=True)
        for opts in ({'assumptions':(1,)},{'max_work':1},{'heap':True}):p.solve(wdeg=True,**opts)
        b=p.solve(wdeg=True)
    assert {k:v for k,v in a.record().items() if k!='elapsed_ns'}=={k:v for k,v in b.record().items() if k!='elapsed_ns'}
