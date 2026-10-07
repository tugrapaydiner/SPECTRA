"""Separate original-input observer against pure-Python truth conditions."""
from itertools import product
import random
import os
import pytest
from spectra.cnf.sparse import SparseRuntime,build_sparse_runtime,valid_witness

@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    return SparseRuntime(os.environ.get('SPECTRA_SPARSE_LIBRARY') or build_sparse_runtime(tmp_path_factory.mktemp('observer-native')))

def test_50000_independent_witnesses(runtime):
    rng=random.Random(910341)
    for i in range(50000):
        n=rng.randrange(9)
        rows=[tuple(v for v in range(1,n+1) if rng.randrange(3)==0) for _ in range(rng.randrange(12))]
        cut=rng.randrange(len(rows)+1);c=tuple(rows[:cut]);e=tuple(rows[cut:])
        w=tuple(bool(rng.randrange(2)) for _ in range(n))
        a=tuple(v*rng.choice((-1,1)) for v in range(1,n+1) if rng.randrange(4)==0)
        assert runtime._module.check(n,c,e,w,a)==valid_witness(n,c,e,w,a)

@pytest.mark.parametrize('n,c,e,w,a',[
    (True,(),(),(),()),(1,(),(),(1,),()),(1,(),(),(),()),(1,[],(),(True,),()),
    (1,([1],),(),(True,),()),(1,((True,),),(),(True,),()),(1,((0,),),(),(True,),()),
    (1,((2,),),(),(True,),()),(1,(),(),(True,),(0,)),(1,(),(),(True,),(2**100,)),
    (1,(),(),(True,),(True,)),(1,(),(),(True,),(-1,)),(1,(),(),[True],()),
    (1,(),(),(True,),[1]),(2,(),((1,2),),(True,True),()),(1,((),),(),(False,),())])
def test_malformed_or_false_witnesses_refused(runtime,n,c,e,w,a):
    assert runtime._module.check(n,c,e,w,a) is False

def test_solving_does_not_require_trusting_compiled_index(runtime):
    c=((1,2),(3,4));e=((1,3),(2,4))
    for assumptions in ((),(1,),(-1,),(-1,-2)):
        a=runtime.solve(4,c,e,assumptions=assumptions,native_check=True)
        b=runtime.solve(4,c,e,assumptions=assumptions,native_check=False)
        assert {k:v for k,v in a.record().items() if k!='elapsed_ns'}=={k:v for k,v in b.record().items() if k!='elapsed_ns'}
