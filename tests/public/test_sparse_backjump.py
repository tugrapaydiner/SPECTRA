"""Reason-directed jumps preserve the complete search space under all supported inputs."""
from itertools import product
import random,os
from pathlib import Path
import pytest
from spectra.cnf.sparse import SparseRuntime,build_sparse_runtime,valid_witness
@pytest.fixture(scope='module')
def rt(tmp_path_factory):
    p=os.environ.get('SPECTRA_SPARSE_LIBRARY') or build_sparse_runtime(tmp_path_factory.mktemp('reason-native'))
    return SparseRuntime(p)

def possible(n,c,e,a=()):return any(valid_witness(n,c,e,w,a) for w in product((False,True),repeat=n))

def test_all_4096_small_systems(rt):
    bank=[tuple(v+1 for v in range(3) if mask>>v&1) for mask in range(8)];ex=[g for g in bank if len(g)>1]
    for mask in range(4096):
        c=tuple(g for i,g in enumerate(bank) if mask>>i&1);e=tuple(g for i,g in enumerate(ex,start=8) if mask>>i&1)
        r=rt.solve(3,c,e,backjump=True,full_conflicts=True,compact_updates=True,native_check=True)
        assert (r.status=='SAT_VERIFIED')==possible(3,c,e),(mask,r)
        assert r.work<=1000000

@pytest.mark.parametrize('seed',[41820,41821,41822,41823])
def test_random_inputs_assumptions_and_policies(rt,seed):
    rng=random.Random(seed)
    for i in range(2500):
        n=rng.randrange(1,10)
        # Avoid mostly trivial empty-clause failures; exercise overlapping units
        # and positive covers permitting several simultaneous true members.
        c=tuple(tuple(rng.sample(range(1,n+1),rng.randrange(1,min(n,4)+1))) for _ in range(rng.randrange(1,16)))
        e=tuple(tuple(rng.sample(range(1,n+1),rng.randrange(min(n,5)+1))) for _ in range(rng.randrange(16)))
        a=tuple(v*rng.choice((-1,1)) for v in range(1,n+1) if rng.randrange(6)==0)
        if i%113==0:a=(1,-1)
        opts={'backjump':True,'full_conflicts':bool(i%2),'compact_updates':bool(i%2),'wdeg':bool(i%3),'degree':bool(i%5),'heap':i%7==0,'active':i%7==1,'lcv':bool(i%11),'binary':i%13==0}
        r=rt.solve(n,c,e,assumptions=a,**opts)
        assert (r.status=='SAT_VERIFIED')==possible(n,c,e,a),(seed,i,c,e,a,r,opts)
        assert r.reason in ('satisfied','exhausted')

@pytest.mark.parametrize('limit',[0,1,2,3,4,8,16,32,64,100])
def test_work_caps_and_repeated_query_restore(rt,limit):
    c=((1,2),(2,3),(1,3));e=c
    with rt.prepare(3,c,e) as p:
        r=p.solve(max_work=limit,backjump=True,wdeg=True,compact_updates=True)
        a=p.solve(backjump=True,wdeg=True,compact_updates=True)
        b=p.solve(backjump=True,wdeg=True,compact_updates=True)
    assert r.work<=limit
    assert {k:v for k,v in a.record().items() if k!='elapsed_ns'}=={k:v for k,v in b.record().items() if k!='elapsed_ns'}
    assert a.status=='UNKNOWN' and a.reason=='exhausted'

def test_bad_backjump_flag(rt):
    with pytest.raises(ValueError):rt.solve(1,((1,),),(),backjump=1)


def test_exact_payload_is_bounded_and_prefix_fallback_works(rt):
    n=8000;c=((1,2),(2,3),(1,3));e=c
    r=rt.solve(n,c,e,backjump=True,full_conflicts=True)
    assert r.status=='UNKNOWN' and r.reason=='exhausted'
    assert r.conflict_payload_bytes==0
    small=rt.solve(3,c,e,backjump=True,full_conflicts=True)
    assert small.conflict_payload_bytes==32
    assert small.status=='UNKNOWN' and small.reason=='exhausted'
    prefix=rt.solve(3,c,e,backjump=True)
    tight=rt.solve(3,c,e,backjump=True,full_conflicts=True,max_state_bytes=prefix.state_payload_bytes)
    assert tight.conflict_payload_bytes==0 and tight.status=='UNKNOWN'


def test_exact_sets_require_reason_mode(rt):
    with pytest.raises(ValueError):rt.solve(1,((1,),),(),full_conflicts=True)
