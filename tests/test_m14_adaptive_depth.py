"""Focused contracts for M14 Amendment-B validator-gated adaptive depth."""
import torch
import scripts.m14_primary_experiment as m
import scripts.m14_adaptive_depth as b

def test_validator_true_stops_after_one_real_step():
    model=b.make_recursive_model(32,1401).eval();x=torch.randint(0,5,(1,16))
    pred,w=b.run_adaptive_solve(model,x,3,validator=lambda _x,_a,_box:torch.ones(1,dtype=torch.bool))
    assert pred.shape==x.shape and w['realized_steps']==1 and w['block_applications']==2 and w['semantic_validator_calls_inside_solver']==1

def test_validator_false_executes_full_budget_and_matches_truncated_reference():
    model=b.make_recursive_model(32,1401).eval();x=torch.randint(0,5,(1,16))
    pred,w=b.run_adaptive_solve(model,x,2,validator=lambda _x,_a,_box:torch.zeros(1,dtype=torch.bool))
    old=model.N_sup;model.N_sup=2
    with torch.inference_mode(): logits,_=model(x,height=4,width=4); ref=m.clamp_givens(x,logits.argmax(-1))
    model.N_sup=old
    assert torch.equal(pred,ref) and w['realized_steps']==2 and w['block_applications']==4

def test_candidate_widths_are_smaller_than_baseline():
    bp=m.model_params(m.make_model('single_pass',1401))
    assert all(m.model_params(b.make_recursive_model(w,1401))<bp for w in b.WIDTHS)

def _point(cid,q,l):return {'config_id':cid,'semantic_validity':q,'latency_median_ms':l,'family':cid,'n':1}
def test_selection_prefers_primary_point_eligible_highest_quality():
    ag=[_point('single_pass_fp',.80,1.0),_point('fp32_semstop2',.84,1.1),_point('fp32_semstop3',.85,1.14)]
    for w in (48,64):
        for d in (2,3):ag.append(_point(b.candidate_id(w,d),.70,2.0))
    tr=[]
    for kind,p in [('single_pass',100000),('fp_adaptive_w32',30000),('fp_adaptive_w48',40000),('fp_adaptive_w64',50000)]:tr.append({'kind':kind,'architecture':{'trainable_params':p}})
    s=b.select_candidate(ag,tr);assert s['selected_config_id']=='fp32_semstop3' and s['selection_pool']=='primary_point_threshold_eligible'

def test_primary_gate_unchanged():
    e={'quality_difference':.04,'quality_ci95':[.01,.07],'latency_ratio_candidate_over_baseline':1.1,'latency_ratio_ci95':[1.05,1.18]};assert m.practical_gate(e)['pass']

def test_manifest_union_deduplicates():
    a={'splits':{'train':{'examples':[{'fingerprint':'a'}]},'validation':{'examples':[]},'test':{'examples':[]}}};c={'splits':{'train':{'examples':[]},'validation':{'examples':[]},'test':{'examples':[{'fingerprint':'a'},{'fingerprint':'b'}]}}};assert b.manifest_union(a,c)=={'a','b'}