import pytest
from eval.symmetry_schedule import predict_schedule


def test_early_identity_and_fresh_view_composition():
    assert predict_schedule(2,2,identity_cycles=8,view_limit=4,budget=20)=={'valid':True,'transitions':2,'answer_source':'identity_32'}
    assert predict_schedule(7,6,identity_cycles=8,view_limit=4,budget=20)=={'valid':True,'transitions':7,'answer_source':'identity_32'}
    assert predict_schedule(None,7,identity_cycles=8,view_limit=4,budget=20)=={'valid':True,'transitions':11,'answer_source':'dihedral_32'}
    assert predict_schedule(24,18,identity_cycles=8,view_limit=4,budget=20)=={'valid':False,'transitions':20,'answer_source':None}
    assert not predict_schedule(None,None,identity_cycles=8,view_limit=4,budget=20)['valid']


def test_refinement_does_not_claim_automatic_dominance_over_continuation():
    assert not predict_schedule(9,None,identity_cycles=8,view_limit=4,budget=20)['valid']


@pytest.mark.parametrize('a,d',[(2,3),(None,2),(2,None),(True,5),(0,None),(33,None),(1.5,None)])
def test_invalid_parent_observations_rejected(a,d):
    with pytest.raises(ValueError):predict_schedule(a,d,identity_cycles=8,view_limit=4,budget=20)


@pytest.mark.parametrize('p,v,b',[(3,4,20),(8,True,20),(8,4,21),(8,0,20),(8,9,20),(8,4,33),(9,4,8)])
def test_invalid_schedules_rejected(p,v,b):
    with pytest.raises(ValueError):predict_schedule(None,None,identity_cycles=p,view_limit=v,budget=b)


def test_exhaustive_small_first_success_composition_matches_explicit_policy():
    # 33x33 observations, including no success; compare to explicit call-order
    # simulation rather than the algebra in production. Neither uses model data.
    for a in [None,*range(1,33)]:
        for d in [None,*range(1,33)]:
            if (a is not None and a<=4) or (d is not None and d<=4):
                if a!=d:continue
            for prefix,budget,views in [(4,32,8),(8,20,4),(12,24,4),(16,28,4)]:
                schedule=[('identity',i) for i in range(1,prefix+1)]
                schedule += [('transformed',i) for i in range(5,5+budget-prefix)]
                solved=False
                for used,(kind,i) in enumerate(schedule,1):
                    if (kind=='identity' and i==a) or (kind=='transformed' and i==d):
                        solved=True;break
                observed=predict_schedule(a,d,identity_cycles=prefix,view_limit=views,budget=budget)
                assert (observed['valid'],observed['transitions'])==(solved,used)
