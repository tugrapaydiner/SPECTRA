"""Focused contracts for the preregistered M14 Amendment-A n=0 intervention."""
import torch
import scripts.m14_primary_experiment as m
import scripts.m14_n0_intervention as a


def _fake_points(c64=(0.84,0.60), c96=(0.86,0.72), base=(0.85,1.0)):
    def row(cid,q,l): return {"config_id":cid,"family":cid,"n":100,"semantic_validity":q,"latency_median_ms":l}
    return [row("n0_dim64",*c64),row("n0_dim96",*c96),row("single_pass_fp",*base)]


def test_n0_one_step_is_one_block_application():
    model=a.make_n0_model("n0_dim64",1401).eval(); model.N_sup=1
    x=torch.tensor([[1,0,0,4,0,4,1,0,0,1,4,0,4,0,0,1]],dtype=torch.long)
    state=model.init_execution_state(x,height=4,width=4)
    out=model.run_execution_step(state)
    assert model.n==0 and model.T==1 and out["work_delta"]["recursive_cycles"]==1
    assert out["work_delta"]["block_applications"]==1


def test_n0_candidates_are_smaller_than_primary_baseline():
    base=m.make_model("single_pass",1401)
    bp=m.model_params(base)
    assert m.model_params(a.make_n0_model("n0_dim64",1401)) < bp
    assert m.model_params(a.make_n0_model("n0_dim96",1401)) < bp


def test_validation_preferred_pool_rule():
    params={"n0_dim64":50000,"n0_dim96":70000,"single_pass":100000}
    sel=a.select_validation_candidate(_fake_points(),params)
    assert sel["selected_config_id"]=="n0_dim96"
    assert sel["selection_pool"]=="q_ge_-0.02_and_latency_le_0.75"


def test_validation_fallback_prefers_quality_deficit_then_latency_excess():
    params={"n0_dim64":50000,"n0_dim96":70000,"single_pass":100000}
    pts=_fake_points(c64=(0.80,0.55),c96=(0.84,0.90),base=(0.85,1.0))
    sel=a.select_validation_candidate(pts,params)
    assert sel["selected_config_id"]=="n0_dim96"
    assert sel["selection_pool"]=="preregistered_lexicographic_fallback"


def test_primary_gate_is_unchanged():
    eff={"quality_difference":-0.01,"quality_ci95":[-0.02,0.01],"latency_ratio_candidate_over_baseline":0.60,"latency_ratio_ci95":[0.58,0.70]}
    assert m.practical_gate(eff)["pass"] is True
    eff2=dict(eff,latency_ratio_candidate_over_baseline=0.70,latency_ratio_ci95=[0.68,0.72])
    assert m.practical_gate(eff2)["pass"] is False


def test_overlap_audit_detects_reuse():
    prior={"splits":{"train":{"examples":[{"fingerprint":"a"}]},"validation":{"examples":[]},"test":{"examples":[]}}}
    later={"splits":{"train":{"examples":[]},"validation":{"examples":[]},"test":{"examples":[{"fingerprint":"a"}]}}}
    assert a.audit_disjoint(prior,later,later_seed=1)["overlap_count"]==1
