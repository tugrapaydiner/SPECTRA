"""Focused contracts for the M14 primary controlled experiment."""
from __future__ import annotations

import numpy as np
import torch

from scripts.m14_primary_experiment import (
    CONFIRM_DATA_SEED,
    DEV_DATA_SEED,
    MODEL_SEEDS,
    RESERVE_CONFIRM_DATA_SEED,
    aggregate_config,
    choose_candidate,
    clamp_givens,
    paired_effect,
    practical_gate,
    symbolic_bundle,
)


def _rows(cid: str, *, success: int, latency: float, n_examples: int = 8):
    rows=[]
    for seed in MODEL_SEEDS:
        for i in range(n_examples):
            rows.append({
                "config_id":cid,"family":cid,"seed":seed,"example_id":f"e{i}","example_index":i,
                "semantic_success":success,"exact_reference_match":success,"blank_cell_accuracy":float(success),
                "latency_ms":latency,"split":"x",
            })
    return rows


def test_data_hierarchy_seeds_are_distinct_and_five_training_seeds():
    assert DEV_DATA_SEED != CONFIRM_DATA_SEED != RESERVE_CONFIRM_DATA_SEED
    assert DEV_DATA_SEED != RESERVE_CONFIRM_DATA_SEED
    assert len(MODEL_SEEDS) == 5 and len(set(MODEL_SEEDS)) == 5


def test_clamp_givens_preserves_observed_cells():
    x=torch.tensor([[1,0,0,4,0,4,1,0,0,1,4,0,4,0,0,1]])
    pred=torch.full_like(x,2)
    out=clamp_givens(x,pred)
    assert torch.equal(out[x!=0],x[x!=0])
    assert torch.equal(out[x==0],torch.full_like(out[x==0],2))


def test_symbolic_reference_solves_valid_4x4():
    x=torch.tensor([[1,0,0,4,0,4,1,0,0,1,4,0,4,0,0,1]])
    ans=symbolic_bundle().solve(x)
    assert ans.shape==x.shape
    assert torch.equal(ans[x!=0],x[x!=0])
    assert set(ans.flatten().tolist()) <= {1,2,3,4}


def test_candidate_selection_uses_validation_cost_eligibility():
    aggs=[
        {"config_id":"single_pass_fp","family":"single","n":40,"semantic_validity":0.70,"latency_median_ms":10.0},
        {"config_id":"fp_n1","family":"fp","n":40,"semantic_validity":0.72,"latency_median_ms":7.0},
        {"config_id":"fp_n2","family":"fp","n":40,"semantic_validity":0.90,"latency_median_ms":13.0},
    ]
    sel=choose_candidate(aggs)
    assert sel["selected_config_id"]=="fp_n1"
    assert sel["selection_used_development"] is False
    assert sel["selection_used_confirmation"] is False
    c={r["config_id"]:r for r in sel["all_candidate_validation"]}
    assert c["fp_n2"]["latency_match_status"]=="unmatched"


def test_practical_gate_quality_superiority_path():
    effect={"quality_difference":0.04,"quality_ci95":[0.01,0.07],"latency_ratio_candidate_over_baseline":1.05,"latency_ratio_ci95":[0.98,1.12]}
    g=practical_gate(effect)
    assert g["pass"] and g["path"]=="quality_superiority"


def test_practical_gate_predeclared_tradeoff_path():
    effect={"quality_difference":-0.01,"quality_ci95":[-0.02,0.01],"latency_ratio_candidate_over_baseline":0.55,"latency_ratio_ci95":[0.50,0.60]}
    g=practical_gate(effect)
    assert g["pass"] and g["path"]=="quality_cost_tradeoff"


def test_practical_gate_does_not_lower_target_for_fast_but_bad_model():
    effect={"quality_difference":-0.05,"quality_ci95":[-0.07,-0.03],"latency_ratio_candidate_over_baseline":0.20,"latency_ratio_ci95":[0.18,0.22]}
    assert practical_gate(effect)["pass"] is False


def test_paired_hierarchical_effect_recovers_exact_difference_and_ratio():
    cand=_rows("fp_n1",success=1,latency=5.0)
    base=_rows("single_pass_fp",success=0,latency=10.0)
    e=paired_effect(cand,base,bootstraps=200,seed=17)
    assert e["n_training_seeds"]==5
    assert e["quality_difference"]==1.0
    assert e["quality_ci95"]==[1.0,1.0]
    assert e["latency_ratio_candidate_over_baseline"]==0.5
    assert np.allclose(e["latency_ratio_ci95"],[0.5,0.5])


def test_aggregate_labels_latency_not_budget_equivalence():
    rows=_rows("fp_n1",success=1,latency=5.0)
    a=aggregate_config(rows,"fp_n1")
    assert a["semantic_validity"]==1.0
    assert a["latency_median_ms"]==5.0
    assert "iso" not in " ".join(a.keys()).lower()
