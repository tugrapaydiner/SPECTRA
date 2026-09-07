import torch

import scripts.m14_primary_experiment as m14
from model.trm import TRM
from scripts.m14_attempt5_dual_stream_semantic_exit import (
    CANDIDATE_KIND,
    choose_candidate,
    dual_stream_semantic_exit_solve,
)


def tiny_dual() -> TRM:
    torch.manual_seed(5)
    return TRM(
        dim=64, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1, N_sup=4,
        heads=4, alpha_y=0.1, alpha_z=0.1, max_grid_size=8,
        ternary=False, act8=False,
    ).eval()


def puzzle() -> torch.Tensor:
    return torch.tensor([[1,0,0,4,0,4,1,0,0,1,4,0,4,0,0,1]])


def test_first_valid_executes_exactly_two_shared_block_calls_and_no_halter():
    model = tiny_dual(); x = puzzle(); blocks = 0; halt = 0; checks = 0
    def bh(_m, _i, _o):
        nonlocal blocks; blocks += 1
    def hh(_m, _i, _o):
        nonlocal halt; halt += 1
    def validator(_x, _a, _box):
        nonlocal checks; checks += 1
        return torch.tensor([True])
    hb = model.blocks[0].register_forward_hook(bh)
    hhnd = model.halt_head.register_forward_hook(hh)
    try:
        answer, work = dual_stream_semantic_exit_solve(model, x, 4, validator=validator)
    finally:
        hb.remove(); hhnd.remove()
    assert answer.shape == x.shape
    assert blocks == 2
    assert halt == 0
    assert checks == 1
    assert work["executed_steps"] == 1
    assert work["block_applications"] == 2
    assert work["semantic_checks"] == 1
    assert work["output_head_calls"] == 1
    assert work["halt_head_calls"] == 0
    assert work["stopped_on_valid"] is True


def test_second_step_only_runs_for_unsolved_tail():
    model = tiny_dual(); x = puzzle(); blocks = 0; checks = 0
    def bh(_m, _i, _o):
        nonlocal blocks; blocks += 1
    def validator(_x, _a, _box):
        nonlocal checks; checks += 1
        return torch.tensor([checks == 2])
    h = model.blocks[0].register_forward_hook(bh)
    try:
        _, work = dual_stream_semantic_exit_solve(model, x, 4, validator=validator)
    finally:
        h.remove()
    assert checks == 2
    assert blocks == 4
    assert work["executed_steps"] == 2
    assert work["block_applications"] == 4
    assert work["stop_reason"] == "semantic_valid"


def test_always_invalid_consumes_exact_selected_budget():
    model = tiny_dual(); x = puzzle(); blocks = 0
    def bh(_m, _i, _o):
        nonlocal blocks; blocks += 1
    h = model.blocks[0].register_forward_hook(bh)
    try:
        _, work = dual_stream_semantic_exit_solve(
            model, x, 3, validator=lambda _x, _a, _b: torch.tensor([False])
        )
    finally:
        h.remove()
    assert blocks == 6
    assert work["executed_steps"] == 3
    assert work["block_applications"] == 6
    assert work["semantic_checks"] == 3
    assert work["stop_reason"] == "budget_exhausted"


def test_runtime_rejects_wrong_architecture_or_batch():
    import pytest
    model = tiny_dual(); x = puzzle()
    with pytest.raises(ValueError):
        dual_stream_semantic_exit_solve(model, x.expand(2, -1), 2)
    with pytest.raises(ValueError):
        dual_stream_semantic_exit_solve(model, x, 0)
    wrong = TRM(
        dim=48, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1, N_sup=4,
        heads=4, max_grid_size=8, ternary=False, act8=False,
    ).eval()
    with pytest.raises(ValueError):
        dual_stream_semantic_exit_solve(wrong, x, 2)


def _agg(cid, q, lat, blocks=None):
    row = {
        "config_id": cid, "family": cid, "n": 2560,
        "semantic_validity": q, "exact_reference_match": q,
        "blank_cell_accuracy": q, "latency_median_ms": lat,
        "latency_p95_ms": lat, "latency_n": 320,
        "seeds": ["1401","2402","3403","4404","5405"],
        "n_training_seeds": 5, "seed_semantic_validity": {},
    }
    if blocks is not None:
        row["block_applications_mean"] = blocks
    return row


def test_validation_selection_can_choose_high_quality_early_exit_point():
    rows = [
        _agg("single_pass_fp", 0.84, 1.00, 2.0),
        _agg("dual_stream_exit_k1", 0.84, 1.05, 2.0),
        _agg("dual_stream_exit_k2", 0.94, 1.12, 2.35),  # Path A eligible
        _agg("dual_stream_exit_k3", 0.97, 1.30, 2.40),
        _agg("dual_stream_exit_k4", 0.98, 1.31, 2.42),
    ]
    sel = choose_candidate(rows)
    assert sel["selected_config_id"] == "dual_stream_exit_k2"
    assert sel["selected_validation"]["path_a_point_eligible"] is True
    assert sel["selection_rule"] == "path_a_point_eligible_fallback"


def test_attempt5_uses_original_later_weight_training_objective_and_gate():
    assert CANDIDATE_KIND == "fp_recursive_dim64"
    assert m14.LATER_WEIGHTS == [0.1, 0.2, 0.3, 0.4]
    good = m14.practical_gate({
        "quality_difference": 0.03,
        "quality_ci95": [0.001, 0.05],
        "latency_ratio_candidate_over_baseline": 1.15,
        "latency_ratio_ci95": [1.0, 1.20],
    })
    assert good["pass"] is True
    bad = m14.practical_gate({
        "quality_difference": 0.03,
        "quality_ci95": [0.001, 0.05],
        "latency_ratio_candidate_over_baseline": 1.1501,
        "latency_ratio_ci95": [1.0, 1.20],
    })
    assert bad["pass"] is False
