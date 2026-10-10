"""The compact update path must preserve complete old search traces."""
import random
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import os
import pytest
from spectra.cnf.sparse import SparseRuntime,build_sparse_runtime

@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    return SparseRuntime(os.environ.get('SPECTRA_SPARSE_LIBRARY') or build_sparse_runtime(tmp_path_factory.mktemp('compact-sparse')))

def stable(r):return {k:v for k,v in r.record().items() if k!='elapsed_ns'}

def test_3000_exact_path_comparisons(runtime):
    rng=random.Random(792020)
    for i in range(3000):
        n=rng.randrange(1,10)
        bank=[tuple(rng.sample(range(1,n+1),rng.randrange(n+1))) for _ in range(rng.randrange(25))]
        split=rng.randrange(len(bank)+1);c=tuple(bank[:split]);e=tuple(bank[split:])
        assumptions=tuple(v*rng.choice((-1,1)) for v in range(1,n+1) if rng.randrange(4)==0)
        opts={'max_work':rng.choice((0,1,2,5,30,100000)), 'assumptions':assumptions,
            'degree':bool(i%2),'lcv':bool(i%3),'xor_units':bool(i%5),
            'heap':i%3==0,'active':i%3==1,'binary':bool(i%7)}
        a=runtime.solve(n,c,e,**opts)
        b=runtime.solve(n,c,e,**opts,compact_updates=True)
        assert stable(a)==stable(b),(i,c,e,assumptions,a,b)

def test_all_4096_systems_exact_paths(runtime):
    bank=[tuple(v+1 for v in range(3) if mask>>v&1) for mask in range(8)]
    ex=[g for g in bank if len(g)>1]
    for mask in range(4096):
        c=tuple(g for i,g in enumerate(bank) if mask>>i&1)
        e=tuple(g for i,g in enumerate(ex,start=8) if mask>>i&1)
        a=runtime.solve(3,c,e);b=runtime.solve(3,c,e,compact_updates=True)
        assert stable(a)==stable(b),(mask,a,b)

def test_multiple_covering_true_assignments(runtime):
    # Covers need >=1 rather than exactly one. Different true choices may cover
    # one positive group; only the first covering assignment freezes its counts.
    c=((1,2,3,4),(1,5),(2,6),(3,7),(4,8));e=((5,6),(6,7),(7,8),(8,5))
    for assumptions in ((),(1,2),(1,2,3,4),(-1,-2),(-1,-2,-3,-4)):
        a=runtime.solve(8,c,e,assumptions=assumptions)
        b=runtime.solve(8,c,e,assumptions=assumptions,compact_updates=True)
        assert stable(a)==stable(b)

@pytest.mark.parametrize('bad',[1,None,'true'])
def test_compact_flag_is_boolean(runtime,bad):
    with pytest.raises(ValueError):runtime.solve(2,((1,2),),(),compact_updates=bad)
