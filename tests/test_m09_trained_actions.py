from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from eval.action_checkpoint import (
    ACTION_FORMAT,
    load_m09_action_checkpoint,
    save_m09_action_checkpoint,
)
from eval.action_search import StateConditionedLatentNativeMCTS
from eval.checkpoint_eval import EvaluationContractError
from eval.latent_mcts import _quantize_int8
from model.latent_action import LatentActionCodebook, StateConditionedLatentActionCodebook
from model.trm import TRM
from scripts.train_action_policy import benefit_gate, greedy_coverage_select, utility_table


class _ToyModel(torch.nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.dim = dim
        self.T = 1
        self.n = 1
        self.token_embed = torch.nn.Embedding(10, dim)

    def encode_positions(self, x, height, width):
        return torch.zeros(1, x.shape[1], self.dim, device=x.device)

    def recursive_cycle(self, x_emb, y, z):
        return y + 0.01 * (x_emb + z), z + 0.01 * x_emb


class _ToySelectionMCTS(StateConditionedLatentNativeMCTS):
    def _step(self, x_emb, node, action):
        self.last_search_stats["transition_calls"] = int(self.last_search_stats["transition_calls"]) + 1
        y = node.y.clone()
        y[..., 0] = float(action)
        codes, scale = _quantize_int8(node.latent())
        return y, codes, scale

    def _value(self, x, node):
        return float(node.action_from_parent or 0)


def _fake_core(dim=8):
    return SimpleNamespace(
        sha256="core-sha",
        task={"task": "sudoku"},
        architecture={"dim": dim, "num_tokens": 10, "seq_len": 81},
        device=torch.device("cpu"),
    )


def _provenance():
    return {
        "reference_target_used": False,
        "candidate_bank_seed": 2901,
        "candidate_bank_count": 24,
        "selected_candidate_indices": [1, 2, 3],
        "target_utility_table_sha256": "table-sha",
        "reasoner_tensor_state_sha256": "tensor-sha",
        "train_id_sha256": "train-sha",
        "development_id_sha256": "dev-sha",
        "test_id_sha256": "test-sha",
        "target_generation_work": {"state_action_equivalents": 24_000},
        "training_method": "v1_hard_ce",
    }


def test_legacy_search_is_inference_only_and_does_not_train_codebook():
    torch.manual_seed(0)
    model = _ToyModel()
    cb = LatentActionCodebook(dim=4, n_actions=4, scale=0.5)
    before_d = cb.directions.detach().clone()
    before_p = cb.prior_logits.detach().clone()
    mcts = _ToySelectionMCTS(model, None, cb, 2, 2, n_rollouts=2, max_depth=1)
    mcts.search(torch.randint(0, 10, (1, 4)))
    assert torch.equal(before_d, cb.directions.detach())
    assert torch.equal(before_p, cb.prior_logits.detach())
    assert cb.directions.grad is None and cb.prior_logits.grad is None


def test_m09_optimizer_gives_nonzero_gradients_and_changes_intended_parameters():
    torch.manual_seed(1)
    module = StateConditionedLatentActionCodebook(
        dim=8, num_tokens=10, n_actions=4, scale=0.5, hidden_dim=16
    )
    opt = torch.optim.AdamW(module.parameters(), lr=1e-2)
    x = torch.randint(0, 10, (12, 16))
    y = torch.randn(12, 16, 8)
    z = torch.randn(12, 16, 8)
    labels = torch.arange(12) % 4
    target_dirs = torch.randn(3, 8)
    target_dirs = target_dirs / target_dirs.norm(dim=1, keepdim=True)
    before_d = module.directions.detach().clone()
    before_h = module.policy_head.weight.detach().clone()
    logits = module.policy_logits_for_state(x, y, z)
    loss = F.cross_entropy(logits, labels) + 0.5 * F.mse_loss(module.directions, target_dirs)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    assert module.directions.grad is not None and module.directions.grad.norm() > 0
    assert module.policy_head.weight.grad is not None and module.policy_head.weight.grad.norm() > 0
    opt.step()
    assert not torch.equal(before_d, module.directions.detach())
    assert not torch.equal(before_h, module.policy_head.weight.detach())


def test_state_conditioning_changes_priors():
    torch.manual_seed(2)
    module = StateConditionedLatentActionCodebook(
        dim=8, num_tokens=10, n_actions=4, scale=0.5, hidden_dim=16
    ).eval()
    x1 = torch.zeros(1, 16, dtype=torch.long)
    x2 = torch.ones(1, 16, dtype=torch.long)
    y = torch.zeros(1, 16, 8)
    z = torch.zeros_like(y)
    p1 = module.priors_for_state(x1, y, z)
    p2 = module.priors_for_state(x2, y, z)
    assert p1.shape == p2.shape == (1, 4)
    assert torch.allclose(p1.sum(1), torch.ones(1))
    assert not torch.allclose(p1, p2)


def test_learned_direction_changes_actual_recursive_transition():
    torch.manual_seed(3)
    reasoner = TRM(dim=8, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1, N_sup=2, heads=2, max_grid_size=8)
    module = StateConditionedLatentActionCodebook(
        dim=8, num_tokens=5, n_actions=4, scale=0.5, hidden_dim=16
    )
    with torch.no_grad():
        module.directions[0].fill_(0.25)
    x = torch.randint(0, 5, (1, 16))
    x_emb = reasoner.token_embed(x) + reasoner.encode_positions(x, 4, 4)
    y = torch.zeros_like(x_emb); z = torch.zeros_like(x_emb)
    with torch.no_grad():
        y0, z0 = reasoner.recursive_cycle(x_emb, y, module.apply_action(z, 0))
        y1, z1 = reasoner.recursive_cycle(x_emb, y, module.apply_action(z, 1))
    assert not torch.allclose(module.apply_action(z, 0), module.apply_action(z, 1))
    assert not torch.allclose(y0, y1) or not torch.allclose(z0, z1)


def test_state_conditioned_prior_controls_deterministic_first_selection():
    torch.manual_seed(4)
    model = _ToyModel()
    module = StateConditionedLatentActionCodebook(
        dim=4, num_tokens=10, n_actions=4, scale=0.5, hidden_dim=8
    )
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
        module.policy_head.bias[:] = torch.tensor([-3.0, -2.0, 5.0, -1.0])
    mcts = _ToySelectionMCTS(model, None, module.eval(), 2, 2, n_rollouts=1, max_depth=1)
    best = mcts.search(torch.zeros(1, 4, dtype=torch.long))
    assert best.path == (2,)
    assert mcts.last_search_stats["state_conditioned_prior_calls"] == 1
    assert mcts.last_search_stats["policy_forward_calls"] == 1


def test_uniform_legacy_prior_keeps_lowest_action_tie_break():
    torch.manual_seed(5)
    model = _ToyModel()
    cb = LatentActionCodebook(dim=4, n_actions=4, scale=0.5)
    with torch.no_grad(): cb.prior_logits.zero_()
    mcts = _ToySelectionMCTS(model, None, cb.eval(), 2, 2, n_rollouts=1, max_depth=1)
    best = mcts.search(torch.zeros(1, 4, dtype=torch.long))
    assert best.path == (0,)
    assert mcts.last_search_stats["global_prior_calls"] == 1


def test_action_checkpoint_roundtrip_binds_core_and_optimizer_state(tmp_path: Path):
    torch.manual_seed(6)
    core = _fake_core()
    module = StateConditionedLatentActionCodebook(
        dim=8, num_tokens=10, n_actions=4, scale=0.5, hidden_dim=16
    )
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3)
    x = torch.randint(0, 10, (4, 81)); y = torch.randn(4, 81, 8); z = torch.randn(4, 81, 8)
    loss = module.policy_logits_for_state(x, y, z).square().mean() + module.directions.square().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    path = tmp_path / "action.pt"
    save_m09_action_checkpoint(
        module, path, core=core, optimizer=opt, trained_steps=1,
        fitted_version="v1_hard_ce", provenance=_provenance(),
    )
    loaded = load_m09_action_checkpoint(path, core)
    assert loaded.payload["format"] == ACTION_FORMAT
    assert loaded.payload["training"]["optimizer_class"] == "AdamW"
    assert loaded.payload["training"]["optimizer_state_dict"]["state"]
    for key, value in module.state_dict().items():
        assert torch.equal(value, loaded.module.state_dict()[key])


def test_action_checkpoint_rejects_incompatible_core(tmp_path: Path):
    core = _fake_core()
    module = StateConditionedLatentActionCodebook(
        dim=8, num_tokens=10, n_actions=4, scale=0.5, hidden_dim=16
    )
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3)
    path = tmp_path / "action.pt"
    save_m09_action_checkpoint(
        module, path, core=core, optimizer=opt, trained_steps=1,
        fitted_version="v1_hard_ce", provenance=_provenance(),
    )
    bad_core = _fake_core(); bad_core.sha256 = "different-core"
    with pytest.raises(EvaluationContractError, match="incompatible"):
        load_m09_action_checkpoint(path, bad_core)


def test_action_checkpoint_rejects_reference_leakage(tmp_path: Path):
    core = _fake_core()
    module = StateConditionedLatentActionCodebook(
        dim=8, num_tokens=10, n_actions=4, scale=0.5, hidden_dim=16
    )
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3)
    prov = _provenance(); prov["reference_target_used"] = True
    with pytest.raises(EvaluationContractError, match="reference_target_used=false"):
        save_m09_action_checkpoint(
            module, tmp_path / "bad.pt", core=core, optimizer=opt, trained_steps=1,
            fitted_version="v1_hard_ce", provenance=prov,
        )


def test_greedy_coverage_selection_is_deterministic_and_train_only_math():
    small = torch.tensor([
        [0.2, 0.8, 0.3, 0.2],
        [0.5, 0.4, 0.9, 0.5],
        [0.4, 0.4, 0.4, 0.7],
    ])
    utilities = torch.cat([small, torch.full((3, 21), -1.0)], dim=1)
    selected, report = greedy_coverage_select(utilities, 3)
    assert selected == [0, 1, 2]
    assert report["coverage_gain"] > 0


def test_target_utility_builder_has_no_reference_answer_argument():
    names = set(inspect.signature(utility_table).parameters)
    assert "target" not in names and "solution" not in names and "reference" not in names


def test_benefit_gate_requires_real_improvement_not_training_metadata():
    def base(mean, trans=4, evals=2):
        return {"mean_score": mean, "rows": [], "work_totals": {
            "transition_calls": trans, "recursive_cycle_calls": trans,
            "verifier_evaluations": evals, "oracle_evaluator_calls": evals,
        }}
    comp = {
        "baselines": {
            "trained": base(0.51), "unguided": base(0.50),
            "fixed_random": base(0.49), "identity": base(0.48, trans=1),
        },
        "paired_vs_unguided": {"mean_delta": 0.01, "wins": 6, "losses": 2},
    }
    passed, checks = benefit_gate(comp)
    assert passed and all(checks.values())
    comp["paired_vs_unguided"]["mean_delta"] = 0.001
    passed, checks = benefit_gate(comp)
    assert not passed and not checks["mean_delta_at_least_0_005"]
