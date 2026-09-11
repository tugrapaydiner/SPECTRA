"""Exhaustive small-instance oracles, never evidence of learned capability."""
from itertools import product, combinations
import numpy as np
import pytest
from data.cnf import CNF, random_3sat
from experiments.residual_patch.runtime import State, W, F, native_scores, library


def check_count(problem, a):
    return [sum(a[abs(l)-1] == (l>0) for l in set(row)) for row in problem.clauses]

@pytest.mark.parametrize('seed', range(12))
def test_atomic_full_rescan_oracle(seed):
    rng=np.random.default_rng(seed)
    clauses=tuple(tuple(int(x) for x in rng.choice([-3,-2,-1,1,2,3],size=int(rng.integers(0,7)))) for _ in range(8))
    problem=CNF(3,clauses)
    for a in product((False,True),repeat=3):
        for patch in [(0,),(1,),(2,),(0,1),(0,2),(1,2)]:
            b=tuple(not x if i in patch else x for i,x in enumerate(a))
            old=check_count(problem,a);new=check_count(problem,b)
            expected=(sum(x==0 and y>0 for x,y in zip(old,new)),sum(x>0 and y==0 for x,y in zip(old,new)))
            with State(problem,a) as s:
                before=s.inspect();assert before['counts']==old
                assert s.patch(patch)==expected
                after=s.inspect();assert after['counts']==new and after['witness']==b
                assert after['unsatisfied']==sum(x==0 for x in new)
                assert s.patch(patch)==expected[::-1]
                assert s.inspect()['witness']==a and s.inspect()['counts']==old
                assert problem.satisfied(a)==(sum(x==0 for x in old)==0)

@pytest.mark.parametrize('bad', [(),(0,0),(-1,),(3,),(True,),(1.0,),[1],(0,1,2)])
def test_rejected_patch_is_transactional(bad):
    with State(CNF(3,((1,2,-3),)),(False,False,True)) as s:
        before=s.inspect()
        with pytest.raises(ValueError):s.patch(bad)
        assert s.inspect()==before

@pytest.mark.parametrize('mode',['probsat','walksat','random_patch','greedy_patch','learned_patch'])
def test_native_search_exact_replay_and_truth(mode):
    formula,_=random_3sat(24,100,129,planted=True)
    rows=[]
    for _ in range(2):
        with State(formula,(False,)*24) as s:
            row=s.run(777,mode=mode,moves=250,interval=8,weights=np.zeros(W) if mode=='learned_patch' else None)
            truth=formula.satisfied(row['witness'])
            assert truth==(row['native_status']==1)
            assert s.inspect()['counts']==check_count(formula,row['witness'])
            row.pop('native_ns');rows.append(row)
    assert rows[0]==rows[1]

@pytest.mark.parametrize('seed',range(8))
def test_candidate_pool_features_and_exact_interaction(seed):
    formula,_=random_3sat(12,45,seed)
    with State(formula,(False,)*12) as s:
        before=s.inspect()['witness'];pairs,features=s.pool(seed)
        assert len(pairs)<=24 and features.shape==(len(pairs),F)
        assert len(set(map(tuple,pairs)))==len(pairs)
        for pair,f in zip(pairs,features):
            patch=tuple(int(v) for v in pair if v>=0)
            with State(formula,before) as clone:
                old=clone.inspect()['unsatisfied'];mb=clone.patch(patch)
                assert tuple(f[[5,6]]*8)==mb
                assert clone.inspect()['unsatisfied']==old-mb[0]+mb[1]
        assert s.inspect()['witness']==before


def test_native_scores_match_independent_numpy():
    rng=np.random.default_rng(441)
    w=rng.normal(size=W);x=rng.normal(size=(100,F))
    expected=np.maximum(x@w[:256].reshape(16,16).T+w[256:272],0)@w[272:288]+w[-1]
    assert np.allclose(native_scores(x,w),expected,atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('formula,witness,solved',[(CNF(0,()),(),True),(CNF(0,((),)),(),False),
    (CNF(1,((1,-1),)),(True,),True),(CNF(1,((1,),(-1,))), (True,),False)])
def test_empty_unused_and_unsat(formula,witness,solved):
    with State(formula,witness) as s:
        row=s.run(0,moves=10)
        assert (row['native_status']==1)==solved
        assert formula.satisfied(row['witness'])==solved


def test_closed_and_owned_state():
    with State(CNF(1,((1,),)),(False,)) as s:
        result=s.inspect();result['counts'][0]=999
        assert s.inspect()['counts']==[0]
    with pytest.raises(RuntimeError):s.inspect()
    s.close()

@pytest.mark.parametrize('kwargs',[{'moves':-1},{'moves':True},{'mode':'invented'},{'time_ns':-1},
   {'mode':'learned_patch'},{'mode':'learned_patch','weights':np.zeros(2)},
   {'mode':'learned_patch','weights':np.full(W,np.nan)},{'interval':0},{'restart':-1},{'cb':float('nan')}])
def test_runtime_rejects_invalid_configuration(kwargs):
    with State(CNF(1,((1,),)),(False,)) as s:
        before=s.inspect()
        with pytest.raises(ValueError):s.run(1,**kwargs)
        assert s.inspect()==before


def test_two_flip_counterexample_to_summing_single_gains():
    p=CNF(2,((1,),(2,),(-1,2)))
    with State(p,(False,False)) as s:
        f=s.features((0,1))
        assert f[12]!=0
        before=s.inspect()['unsatisfied'];make,br=s.patch((0,1))
        assert s.inspect()['unsatisfied']==before-make+br==0
