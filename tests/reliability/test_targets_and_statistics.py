import numpy as np
import pytest
import torch
from spectra_reliability.targets import TargetStateEvaluator, backbone_hash, POLICY
from spectra_reliability.statistics import probability_metrics, paired_bootstrap, pool_selection
from spectra_reliability.semantics import SemanticMismatch


@torch.inference_mode()
def test_first_hit_distribution_probability_and_monotonic_horizons():
    m=TargetStateEvaluator('first_hit','1'*64,seed=11).eval()
    x=torch.ones(4,16,dtype=torch.long);y=torch.randn(4,16,64);z=torch.randn_like(y)
    p=m.first_hit_probabilities(x,y,z,continuation_policy=POLICY)
    assert p.shape==(4,5) and torch.allclose(p.sum(-1),torch.ones(4))
    cdf=torch.stack([m.within_horizon(x,y,z,h,continuation_policy=POLICY) for h in range(4)],dim=-1)
    assert torch.all(cdf[:,1:]>=cdf[:,:-1]) and torch.all(cdf[:,-1]<=1)
    assert torch.equal(cdf[:,0],p[:,0])
    with pytest.raises(SemanticMismatch):m.within_horizon(x,y,z,4,continuation_policy=POLICY)
    with pytest.raises(ValueError):m.within_horizon(x,y,z,True,continuation_policy=POLICY)
    with pytest.raises(SemanticMismatch):m.within_horizon(x,y,z,1,continuation_policy='other')
    with pytest.raises(SemanticMismatch):m(x,y,z)


def test_same_backbone_initialization_despite_target_cardinality():
    models=[TargetStateEvaluator(t,'1'*64,seed=51) for t in ('improvement','validity','quality','first_hit')]
    assert len({backbone_hash(m) for m in models})==1
    counts=[sum(p.numel() for p in m.parameters()) for m in models]
    assert counts[0]==counts[1]==counts[2] and counts[3]-counts[0]==260
    x=torch.ones(1,16,dtype=torch.long);y=torch.zeros(1,16,64)
    with pytest.raises(SemanticMismatch):models[0].current_value(x,y,y,target='validity')


def test_rank_auc_ties_and_single_class():
    assert probability_metrics([0,1],[.1,.9])['roc_auc']==1
    assert probability_metrics([0,1],[.9,.1])['roc_auc']==0
    assert probability_metrics([0,1],[.5,.5])['roc_auc']==.5
    assert probability_metrics([1,1],[.3,.4])['roc_auc'] is None
    with pytest.raises(ValueError):probability_metrics([0,1],[.1,np.nan])


def test_puzzle_bootstrap_not_pseudoreplicated_seed_rows():
    a=np.ones((2,9));b=np.zeros((2,9));r=paired_bootstrap(a,b,replicates=100)
    assert r['unique_puzzles']==9 and r['fixed_cores']==2 and r['point']==1
    assert r['ci95']==[1.,1.] and r['training_seed_population_inference'] is False
    with pytest.raises(ValueError):paired_bootstrap(a,b,replicates=100,statistic='ratio')


def test_fixed_pool_decomposition_and_stable_ties():
    v=np.array([[True,False],[False,True],[False,False]],dtype=bool)
    r=pool_selection(v,[[.1,.9],[.4,.4],[.1,.2]])
    assert r['covered_puzzles']==2 and r['returned_valid']==0
    assert r['selection_failures_given_coverage']==2
    assert r['selected_indices']==[1,0,1]


def test_horizon_specific_contracts_do_not_mislabel_current_validity():
    from spectra_reliability.semantics import FirstHitFamilyContract, CheckedHorizonEvaluator, TargetKind
    family=FirstHitFamilyContract('sudoku','1'*64,'decode','checker','identity',3)
    queries=[]
    def predict(state,contract):
        queries.append(contract)
        return .5
    evaluator=CheckedHorizonEvaluator(family,family,lambda state:state,predict)
    assert evaluator(0)==.5 and queries[-1].target is TargetKind.CURRENT_VALIDITY
    assert evaluator(2)==.5 and queries[-1].target is TargetKind.BUDGETED_SUCCESS and queries[-1].horizon==2
    with pytest.raises(SemanticMismatch):evaluator(4)
    assert evaluator.query_counts=={0:1,2:1}


def test_calibration_bins_cover_boundaries_once():
    labels=np.zeros(11);probs=np.arange(11)/10
    metrics=probability_metrics(labels,probs)
    assert sum(b['n'] for b in metrics['bins'])==11
