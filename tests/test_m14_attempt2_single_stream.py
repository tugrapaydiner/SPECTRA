import torch

from model.single_stream_trm import InputConditionedSingleStreamTRM
from model.system1_student import System1Student
from scripts.m14_attempt2_single_stream import (
    CANDIDATE_KIND,
    choose_candidate,
    make_baseline,
    make_candidate,
    model_params,
)
from scripts.m14_primary_experiment import practical_gate


def test_single_stream_first_step_is_input_conditioned_and_z_stays_zero():
    model = InputConditionedSingleStreamTRM(
        dim=16, num_tokens=5, seq_len=16, n_layers=1, T=1, N_sup=1,
        heads=4, alpha_y=0.1, max_grid_size=8,
    )
    # Make the recurrence algebra transparent: the update is exactly x_emb+y.
    model.f = lambda h: h
    x1 = torch.tensor([[1,0,0,4,0,4,1,0,0,1,4,0,4,0,0,1]])
    x2 = torch.tensor([[2,0,0,3,0,3,2,0,0,2,3,0,3,0,0,2]])
    s1 = model.init_execution_state(x1, height=4, width=4)
    s2 = model.init_execution_state(x2, height=4, width=4)
    z1 = s1.z.clone(); z2 = s2.z.clone()
    o1 = model.run_execution_step(s1)
    o2 = model.run_execution_step(s2)
    assert not torch.equal(o1["y"], o2["y"])
    assert torch.equal(o1["z"], z1)
    assert torch.equal(o2["z"], z2)


def test_single_stream_executes_one_block_application_per_supervision_step():
    torch.manual_seed(0)
    model = InputConditionedSingleStreamTRM(
        dim=16, num_tokens=5, seq_len=16, n_layers=1, T=1, N_sup=3,
        heads=4, alpha_y=0.1, max_grid_size=8,
    ).eval()
    x = torch.randint(0, 5, (1, 16))
    with torch.inference_mode():
        _, steps = model(x, height=4, width=4)
    assert len(steps) == 3
    assert steps[0]["work_cumulative"]["block_applications"] == 1
    assert steps[1]["work_cumulative"]["block_applications"] == 2
    assert steps[2]["work_cumulative"]["block_applications"] == 3
    assert steps[2]["work_cumulative"]["recursive_cycles"] == 3


def test_candidate_unused_z_parameters_are_not_optimizer_owned():
    model = make_candidate(1401)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert "alpha_z" not in trainable
    assert not any(name.startswith("norm_z.") for name in trainable)
    assert "alpha_y" in trainable


def test_attempt2_candidate_is_smaller_than_unchanged_primary_baseline():
    candidate = make_candidate(1401)
    baseline = make_baseline(1401)
    assert isinstance(candidate, InputConditionedSingleStreamTRM)
    assert isinstance(baseline, System1Student)
    assert model_params(candidate) < model_params(baseline)


def _agg(cid, q, lat):
    return {
        "config_id": cid, "family": cid, "n": 2560,
        "semantic_validity": q, "exact_reference_match": q,
        "blank_cell_accuracy": q, "latency_median_ms": lat,
        "latency_p95_ms": lat, "latency_n": 320,
        "seeds": ["1401","2402","3403","4404","5405"],
        "n_training_seeds": 5, "seed_semantic_validity": {},
    }


def test_validation_selection_prioritizes_predeclared_tradeoff_path_b():
    rows = [
        _agg("single_pass_fp", 0.84, 1.00),
        _agg("single_stream_n1", 0.83, 0.60),  # B eligible
        _agg("single_stream_n2", 0.90, 1.05),  # A eligible, but B has priority
        _agg("single_stream_n3", 0.95, 1.60),
        _agg("single_stream_n4", 0.98, 2.00),
    ]
    sel = choose_candidate(rows)
    assert sel["selected_config_id"] == "single_stream_n1"
    assert sel["selection_rule"] == "path_b_point_eligible_priority"
    assert sel["selected_validation"]["path_b_point_eligible"] is True


def test_validation_selection_does_not_call_fast_path_iso_latency():
    rows = [
        _agg("single_pass_fp", 0.84, 1.00),
        _agg("single_stream_n1", 0.83, 0.60),
        _agg("single_stream_n2", 0.82, 1.10),
        _agg("single_stream_n3", 0.85, 1.50),
        _agg("single_stream_n4", 0.86, 2.00),
    ]
    sel = choose_candidate(rows)
    assert sel["selected_validation"]["validation_latency_ratio_vs_baseline"] == 0.60
    assert sel["selected_validation"]["latency_match_status"] == "unmatched"


def test_attempt2_uses_original_m14_practical_gate_unchanged():
    path_b = practical_gate({
        "quality_difference": -0.02,
        "quality_ci95": [-0.03, 0.01],
        "latency_ratio_candidate_over_baseline": 0.65,
        "latency_ratio_ci95": [0.60, 0.75],
    })
    assert path_b["pass"] is True and path_b["path"] == "quality_cost_tradeoff"
    miss = practical_gate({
        "quality_difference": -0.0201,
        "quality_ci95": [-0.03, 0.01],
        "latency_ratio_candidate_over_baseline": 0.65,
        "latency_ratio_ci95": [0.60, 0.75],
    })
    assert miss["pass"] is False


def test_candidate_kind_is_explicit_not_literal_n_zero_trm_alias():
    model = make_candidate(2402)
    assert CANDIDATE_KIND == "input_conditioned_single_stream"
    assert model.n == 0
    assert model.SEMANTICS == "input_conditioned_single_stream_v1"
