import torch

from model.single_stream_trm import InputConditionedSingleStreamTRM
from scripts.m14_attempt3_semantic_early_exit import choose_candidate, semantic_early_exit_solve
from scripts.m14_primary_experiment import practical_gate


def tiny_model() -> InputConditionedSingleStreamTRM:
    torch.manual_seed(0)
    return InputConditionedSingleStreamTRM(
        dim=16, num_tokens=5, seq_len=16, n_layers=1, T=1, N_sup=4,
        heads=4, alpha_y=0.1, max_grid_size=8,
    ).eval()


def puzzle() -> torch.Tensor:
    return torch.tensor([[1,0,0,4,0,4,1,0,0,1,4,0,4,0,0,1]])


def test_semantic_exit_stops_before_later_blocks():
    model = tiny_model(); x = puzzle()
    block_calls = 0; halt_calls = 0; validator_calls = 0

    def block_hook(_m, _i, _o):
        nonlocal block_calls; block_calls += 1

    def halt_hook(_m, _i, _o):
        nonlocal halt_calls; halt_calls += 1

    def validator(_x, _answer, box):
        nonlocal validator_calls
        assert box == 2
        validator_calls += 1
        return torch.tensor([validator_calls >= 2])

    h1 = model.blocks[0].register_forward_hook(block_hook)
    h2 = model.halt_head.register_forward_hook(halt_hook)
    try:
        answer, work = semantic_early_exit_solve(model, x, 4, validator=validator)
    finally:
        h1.remove(); h2.remove()
    assert answer.shape == x.shape
    assert work["executed_steps"] == 2
    assert work["block_applications"] == 2
    assert work["semantic_checks"] == 2
    assert work["stopped_on_valid"] is True
    assert work["stop_reason"] == "semantic_valid"
    assert block_calls == 2
    assert halt_calls == 0


def test_always_invalid_consumes_exact_max_budget():
    model = tiny_model(); x = puzzle(); calls = 0

    def validator(_x, _answer, _box):
        nonlocal calls; calls += 1
        return torch.tensor([False])

    _, work = semantic_early_exit_solve(model, x, 3, validator=validator)
    assert calls == 3
    assert work["executed_steps"] == 3
    assert work["block_applications"] == 3
    assert work["semantic_checks"] == 3
    assert work["output_head_calls"] == 3
    assert work["halt_head_calls"] == 0
    assert work["stopped_on_valid"] is False
    assert work["stop_reason"] == "budget_exhausted"


def test_first_valid_executes_one_block_only():
    model = tiny_model(); x = puzzle()
    _, work = semantic_early_exit_solve(
        model, x, 4, validator=lambda _x, _a, _b: torch.tensor([True])
    )
    assert work["executed_steps"] == 1
    assert work["block_applications"] == 1
    assert work["semantic_checks"] == 1


def test_solver_rejects_batching_and_invalid_budget():
    model = tiny_model(); x = puzzle()
    import pytest
    with pytest.raises(ValueError):
        semantic_early_exit_solve(model, x.expand(2, -1), 2)
    with pytest.raises(ValueError):
        semantic_early_exit_solve(model, x, 0)
    with pytest.raises(ValueError):
        semantic_early_exit_solve(model, x, 5)


def _agg(cid, q, lat, steps=None):
    row = {
        "config_id": cid, "family": cid, "n": 2560,
        "semantic_validity": q, "exact_reference_match": q,
        "blank_cell_accuracy": q, "latency_median_ms": lat,
        "latency_p95_ms": lat, "latency_n": 320,
        "seeds": ["1401","2402","3403","4404","5405"],
        "n_training_seeds": 5, "seed_semantic_validity": {},
    }
    if steps is not None:
        row["executed_steps_mean"] = steps
    return row


def test_validation_selection_prioritizes_gate_eligible_semantic_exit():
    rows = [
        _agg("single_pass_fp", 0.84, 1.00),
        _agg("semantic_exit_k1", 0.20, 0.55, 1.0),
        _agg("semantic_exit_k2", 0.83, 0.62, 1.7),  # Path B point eligible
        _agg("semantic_exit_k3", 0.88, 1.10, 2.0),  # Path A point eligible
        _agg("semantic_exit_k4", 0.91, 1.20, 2.1),
    ]
    sel = choose_candidate(rows)
    assert sel["selected_config_id"] == "semantic_exit_k2"
    assert sel["selection_rule"] == "path_b_point_eligible_priority"
    assert sel["selected_validation"]["path_b_point_eligible"] is True
    assert sel["selected_validation"]["latency_match_status"] == "unmatched"


def test_attempt3_keeps_original_practical_gate_exactly():
    passing_a = practical_gate({
        "quality_difference": 0.03, "quality_ci95": [0.001, 0.05],
        "latency_ratio_candidate_over_baseline": 1.15,
        "latency_ratio_ci95": [1.0, 1.20],
    })
    assert passing_a["pass"] is True
    passing_b = practical_gate({
        "quality_difference": -0.02, "quality_ci95": [-0.03, 0.0],
        "latency_ratio_candidate_over_baseline": 0.65,
        "latency_ratio_ci95": [0.60, 0.75],
    })
    assert passing_b["pass"] is True
    miss = practical_gate({
        "quality_difference": -0.02, "quality_ci95": [-0.0301, 0.0],
        "latency_ratio_candidate_over_baseline": 0.65,
        "latency_ratio_ci95": [0.60, 0.75],
    })
    assert miss["pass"] is False
