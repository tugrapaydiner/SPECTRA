"""Meaningful M14 gates: data protection, actual solve scope and paired inference."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from eval.controlled_comparison import (
    answer_loss, assert_hard_quantized, budget_label, convert_int8, decode,
    digest, full_solve, load_partition, make_model, outcome, paired_bounds,
    set_quant_strength, wilson, write_json,
)


def test_confirmation_cannot_be_loaded_without_successful_gate(tmp_path):
    with pytest.raises(RuntimeError, match="sealed"):
        load_partition(tmp_path, "confirmation", "confirmation_evaluation")
    with pytest.raises(ValueError, match="mismatch"):
        load_partition(tmp_path, "confirmation", "training")


def test_confirmation_requires_untampered_pass_and_is_single_use(tmp_path):
    (tmp_path / "data").mkdir()
    np.savez_compressed(tmp_path / "data/confirmation.npz", inputs=np.zeros((1, 81), dtype=int), targets=np.ones((1, 81), dtype=int))
    write_json(tmp_path / "data/manifest.json", {"partitions": {"confirmation": {
        "array_sha256": digest(tmp_path / "data/confirmation.npz"), "examples": [{"id": "independent"}]}}})
    gate = tmp_path / "gate.json"
    write_json(gate, {"passed": False})
    write_json(tmp_path / "confirmation_authorization.json", {"development_gate_path": "gate.json", "development_gate_sha256": digest(gate)})
    with pytest.raises(RuntimeError, match="invalid"):
        load_partition(tmp_path, "confirmation", "confirmation_evaluation")
    write_json(gate, {"passed": True})
    with pytest.raises(RuntimeError, match="invalid"):
        load_partition(tmp_path, "confirmation", "confirmation_evaluation")
    write_json(tmp_path / "confirmation_authorization.json", {"development_gate_path": "gate.json", "development_gate_sha256": digest(gate)})
    x, y, rows = load_partition(tmp_path, "confirmation", "confirmation_evaluation")
    assert x.shape == y.shape == (1, 81) and rows[0]["id"] == "independent"
    with pytest.raises(RuntimeError, match="consumed"):
        load_partition(tmp_path, "confirmation", "confirmation_evaluation")


def test_data_hash_protects_training_inputs(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train.npz").write_bytes(b"tampered")
    write_json(tmp_path / "data/manifest.json", {"partitions": {"train": {"array_sha256": "wrong"}}})
    with pytest.raises(ValueError, match="hash"):
        load_partition(tmp_path, "train", "training")


@pytest.mark.parametrize("ratio,label", [(1.0, "iso_latency"), (1.04, "iso_latency"),
    (.94, "lower_cost_unmatched"), (1.06, "higher_cost_unmatched")])
def test_cost_equality_does_not_expand_to_broad_flop_tolerances(ratio, label):
    assert budget_label(ratio, .05) == label


def test_paired_seed_and_example_statistics_keep_timing_out_of_sample_size():
    a = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 0]])
    b = np.zeros_like(a); cost = np.ones_like(a, dtype=float)
    r = paired_bounds(a, b, cost, cost * 2, alpha=.025, draws=1000, seed=7)
    assert r["accuracy_difference"] == .75
    assert r["latency_ratio"] == .5
    assert r["seeds"] == 3 and r["paired_examples"] == 4
    assert r["accuracy_lower_bound"] <= .75
    with pytest.raises(ValueError, match="three seeds"):
        paired_bounds(a[:2], b[:2], cost[:2], cost[:2], alpha=.025, draws=1000, seed=7)


def test_zero_solve_bootstrap_is_labelled_degenerate_not_population_certainty():
    a = np.zeros((3, 192)); cost = np.ones_like(a)
    r = paired_bounds(a, a, cost, cost, alpha=.025, draws=1000, seed=7)
    assert r["accuracy_lower_bound"] == 0
    assert r["empirical_accuracy_interval_degenerate"]
    assert wilson(0, 192)[1] > 0


def test_ternary_strength_roundtrip_is_explicit():
    model = make_model("ternary_recursive", 10)
    set_quant_strength(model, .8)
    with pytest.raises(ValueError, match="exactly 1"):
        assert_hard_quantized(model)
    set_quant_strength(model, 1.)
    assert_hard_quantized(model)


def test_dynamic_int8_executes_real_quantized_linears_and_keeps_attention_fp32():
    torch.set_num_threads(2)
    model, metadata = convert_int8(make_model("single_pass", 10))
    assert 0 < metadata["quantized_weight_fraction"] < 1
    assert isinstance(model.blocks[0].ff[0], torch.ao.nn.quantized.dynamic.Linear)
    assert isinstance(model.blocks[0].attn, torch.nn.MultiheadAttention)
    with torch.inference_mode():
        logits, _ = model(torch.zeros((1, 81), dtype=torch.long), height=9, width=9)
    assert logits.shape == (1, 81, 10) and torch.isfinite(logits).all()


def test_decoder_never_consults_solution_and_preserves_only_supplied_clues():
    x = torch.tensor([[7, 0, 0]])
    logits = torch.zeros(1, 3, 10); logits[..., 0] = 100; logits[..., 2] = 1
    assert decode(logits, x, True).tolist() == [[7, 2, 2]]
    assert decode(logits, x, False).tolist() == [[0, 0, 0]]


def test_complete_symbolic_solve_checks_clues_and_returns_real_cost():
    solved = np.array([[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]).reshape(-1)
    puzzle = solved.copy(); puzzle[::10] = 0
    pred, valid, ms = full_solve(None, puzzle, symbolic=True)
    assert valid and ms > 0 and np.array_equal(pred, solved)
    contradictory = puzzle.copy(); contradictory[0] = contradictory[1]
    assert not full_solve(None, contradictory, symbolic=True)[1]
    assert outcome(puzzle, solved, pred)["blank_correct"] == 9


def test_blank_only_objective_has_real_gradients():
    torch.set_num_threads(2)
    model = make_model("fp_recursive", 10)
    x = torch.zeros((2, 81), dtype=torch.long); y = torch.ones_like(x)
    loss = answer_loss(model, x, y, blank_only=True); loss.backward()
    assert torch.isfinite(loss) and model.out_head.weight.grad.abs().sum() > 0
    assert model.halt_head.weight.grad is None


def test_independent_report_checker_rejects_valid_grid_that_changes_clues():
    from scripts.render_m14_comparison import independently_check
    solved = [(r * 3 + r // 3 + c) % 9 + 1 for r in range(9) for c in range(9)]
    puzzle = solved.copy(); puzzle[0] = 0
    relabelled = [v % 9 + 1 for v in solved]
    result = independently_check(puzzle, solved, relabelled)
    assert result["success"] == 0 and result["clue_errors"] == 80
    assert independently_check(puzzle, solved, solved)["success"] == 1


def test_paired_bootstrap_preserves_identical_system_pairings():
    rng = np.random.default_rng(3)
    outcomes = rng.integers(2, size=(3, 80)); costs = rng.uniform(.2, 2, size=(3, 80))
    result = paired_bounds(outcomes, outcomes, costs, costs * 3, alpha=.025, draws=1000, seed=5)
    assert result["accuracy_lower_bound"] == 0
    assert result["latency_ratio_upper_bound"] == pytest.approx(1 / 3)


def test_confirmation_authorization_is_bound_to_phase_and_selection(tmp_path):
    from eval.controlled_comparison import verify_confirmation_freeze
    phase = tmp_path / "initial"; phase.mkdir()
    write_json(phase / "selection.json", {"selected": "candidate_A"})
    write_json(phase / "native_selection.json", {"native": False})
    write_json(phase / "development_freeze.json", {
        "selection_sha256": digest(phase / "selection.json"),
        "native_selection_sha256": digest(phase / "native_selection.json")})
    write_json(phase / "development_gate.json", {"passed": True, "freeze_sha256": digest(phase / "development_freeze.json")})
    write_json(tmp_path / "confirmation_authorization.json", {"phase": "initial",
        "development_gate_path": "initial/development_gate.json",
        "development_gate_sha256": digest(phase / "development_gate.json")})
    verify_confirmation_freeze(tmp_path, "initial")
    with pytest.raises(RuntimeError, match="phase"):
        verify_confirmation_freeze(tmp_path, "blank_only")
    write_json(phase / "selection.json", {"selected": "candidate_B"})
    with pytest.raises(RuntimeError, match="configuration changed"):
        verify_confirmation_freeze(tmp_path, "initial")


def test_missing_container_battery_sysfs_is_unavailable_instead_of_crash(monkeypatch):
    from types import SimpleNamespace
    from eval import telemetry
    def missing_sensor():
        raise FileNotFoundError("/sys/class/power_supply is not mounted")
    monkeypatch.setattr(telemetry, "psutil", SimpleNamespace(sensors_battery=missing_sensor))
    assert telemetry._read_battery() == (1.0, 1.0)  # documented compatibility defaults


@pytest.mark.parametrize("energy", [None, .1])
def test_missing_sampled_memory_stays_null_in_edge_report(energy):
    from eval.reports import edge_report
    report = edge_report(accuracy=.5, latency_ms=2., peak_ram_mb=None, model_size_mb=.2, joules_per_problem=energy)
    assert report["peak_ram_mb"] is None and report["accuracy_per_mb"] is None
    assert report.get("edge_reasoning_score") is None
    assert "unavailable" in report["memory_note"]
