"""Milestone 08: exact MCTS semantics, accounting, and reference-tree tests."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from eval.latent_mcts import LatentNativeMCTS
from eval.mcts_reference import run_frozen_reference


class TinyModel(nn.Module):
    """Deterministic non-neural-cost fixture implementing the native MCTS API."""

    def __init__(self):
        super().__init__()
        self.dim = 2
        self.T = 1
        self.N_sup = 2
        self.token_embed = nn.Embedding(5, 2)
        with torch.no_grad():
            self.token_embed.weight.zero_()
        self.out_head = nn.Linear(2, 5, bias=False)
        with torch.no_grad():
            self.out_head.weight.zero_()
            self.out_head.weight[1, 0] = 1.0
            self.out_head.weight[2, 1] = 1.0

    def encode_positions(self, x, height, width):
        return torch.zeros((1, x.shape[1], 2), dtype=torch.float32, device=x.device)

    def recursive_cycle(self, x_emb, y, z):
        # The transition depends on both state components and is deterministic.
        z2 = z + 0.25 + 0.05 * y
        y2 = y + 0.2 * z2
        return y2, z2

    def forward(self, x, height=2, width=2):
        x_emb = self.token_embed(x) + self.encode_positions(x, height, width)
        y = torch.zeros_like(x_emb)
        z = torch.zeros_like(x_emb)
        steps = []
        for _ in range(self.N_sup):
            y, z = self.recursive_cycle(x_emb, y, z)
            logits = self.out_head(y)
            steps.append({"y": y, "z": z, "logits": logits})
        return steps[-1]["logits"], steps


class TinyCodebook:
    n_actions = 2

    def apply_action(self, z, action):
        if action == 0:
            return z
        if action == 1:
            return z + 0.5
        raise ValueError(action)

    def priors(self):
        return torch.tensor([0.5, 0.5], dtype=torch.float32)


class TinyVerifier:
    def __init__(self, *, constant=False, fail=False):
        self.constant = constant
        self.fail = fail

    def value(self, x, z, width):
        if self.fail:
            raise RuntimeError("intentional verifier failure")
        if self.constant:
            return torch.zeros(z.shape[0], device=z.device)
        return z.float().mean(dim=(1, 2))


def make_search(n_rollouts, *, max_depth=3, verifier=None, c_puct=1.0):
    return LatentNativeMCTS(
        TinyModel(),
        verifier or TinyVerifier(),
        TinyCodebook(),
        height=2,
        width=2,
        n_rollouts=n_rollouts,
        c_puct=c_puct,
        max_depth=max_depth,
    )


def x1():
    return torch.zeros((1, 4), dtype=torch.long)


def node_stats(export):
    return {
        tuple(row["path"]): (row["visits"], row["value_sum"], row["q"])
        for row in export["nodes"]
    }


def test_frozen_reference_tree_matches_hand_computed_contract():
    out = run_frozen_reference()
    rows = {tuple(r["path"]): r for r in out["nodes"]}
    assert rows[()]["visits"] == 4
    assert rows[()]["value_sum"] == pytest.approx(2.8)
    assert rows[()]["q"] == pytest.approx(0.7)
    assert rows[(0,)]["visits"] == 1
    assert rows[(0,)]["value_sum"] == pytest.approx(0.2)
    assert rows[(1,)]["visits"] == 3
    assert rows[(1,)]["value_sum"] == pytest.approx(2.6)
    assert rows[(1,)]["q"] == pytest.approx(2.6 / 3.0)
    assert rows[(1, 0)]["visits"] == 2
    assert rows[(1, 0)]["value_sum"] == pytest.approx(1.8)
    assert rows[(1, 1)]["visits"] == 0
    assert out["best_path"] == [1, 0]
    assert out["work"]["expansion_calls"] == 3
    assert out["work"]["transition_calls"] == 6
    assert out["work"]["verifier_evaluations"] == 4


@pytest.mark.parametrize(
    "budget, expected_batches",
    [(0, []), (1, [1]), (3, [3]), (4, [4]), (5, [4, 1]), (8, [4, 4])],
)
def test_batched_rollout_budget_edges_are_exact(budget, expected_batches):
    search = make_search(budget)
    node = search.search_batched(x1(), leaf_batch=4)
    out = search.export_tree()
    work = out["work"]
    assert work["requested_rollouts"] == budget
    assert work["completed_rollouts"] == budget
    assert work["verifier_evaluations"] == budget
    assert work["batch_sizes"] == expected_batches
    assert work["leaf_batches"] == len(expected_batches)
    assert work["verifier_calls"] == len(expected_batches)
    assert work["virtual_loss_outstanding"] == 0
    assert work["virtual_loss_applications"] == work["virtual_loss_cleanups"]
    if budget == 0:
        assert work["greedy_forward_calls"] == 1
        assert work["transition_calls"] == 0
        assert work["initial_expansion_transition_calls"] == 0
        assert node.children == []
    else:
        assert work["greedy_forward_calls"] == 0
        assert work["initial_expansion_calls"] == 1
        assert work["initial_expansion_transition_calls"] == 2
        assert work["transition_calls"] >= 2


def test_zero_search_decodes_the_ordinary_greedy_final_state():
    search = make_search(0)
    node = search.search(x1())
    answer = search.decode(node)
    logits, steps = search.model(x1(), height=2, width=2)
    assert torch.equal(answer, logits.argmax(dim=-1))
    work = search.export_tree()["work"]
    assert work["decode_calls"] == 1
    assert work["greedy_forward_calls"] == 1
    assert work["verifier_calls"] == 0


def test_initial_expansion_and_all_transition_work_are_counted():
    search = make_search(2, max_depth=3)
    search.search(x1())
    work = search.export_tree()["work"]
    assert work["initial_expansion_calls"] == 1
    assert work["initial_expansion_transition_calls"] == 2
    assert work["expansion_calls"] == 3  # root + one selected leaf per rollout
    assert work["transition_calls"] == 6
    assert work["recursive_cycle_calls"] == 6  # TinyModel.T == 1
    assert work["verifier_calls"] == 2
    assert work["verifier_evaluations"] == 2


def test_horizon_prevents_deeper_children_and_bounds_work():
    search = make_search(9, max_depth=1)
    search.search(x1())
    out = search.export_tree()
    assert max(row["depth"] for row in out["nodes"]) == 1
    assert all(not row["children"] for row in out["nodes"] if row["depth"] == 1)
    work = out["work"]
    assert work["max_depth_reached"] == 1
    assert work["expansion_calls"] == 1
    assert work["transition_calls"] == 2
    assert work["completed_rollouts"] == 9


def test_exact_ties_choose_lowest_action_deterministically():
    search = make_search(2, verifier=TinyVerifier(constant=True), c_puct=0.0)
    search.search(x1())
    assert search.last_search_stats["evaluated_paths"] == [[0], [0, 0]]
    assert search.last_search_stats["best_path"] == [0]


def test_batch_size_one_matches_serial_tree_and_output():
    serial = make_search(7, max_depth=3)
    batched = make_search(7, max_depth=3)
    best_s = serial.search(x1())
    best_b = batched.search_batched(x1(), leaf_batch=1, virtual_loss=2.0)
    ex_s, ex_b = serial.export_tree(), batched.export_tree()
    assert best_s.path == best_b.path
    assert torch.equal(serial.decode(best_s), batched.decode(best_b))
    assert node_stats(ex_s) == pytest.approx(node_stats(ex_b))
    for key in (
        "completed_rollouts", "expansion_calls", "transition_calls",
        "recursive_cycle_calls", "verifier_calls", "verifier_evaluations",
        "max_depth_reached",
    ):
        assert ex_s["work"][key] == ex_b["work"][key]
    assert ex_b["work"]["virtual_loss_outstanding"] == 0


def test_larger_batch_claims_only_invariants_not_serial_trajectory_identity():
    search = make_search(7, max_depth=3)
    search.search_batched(x1(), leaf_batch=3, virtual_loss=1.0)
    out = search.export_tree()
    work = out["work"]
    assert work["completed_rollouts"] == 7
    assert work["batch_sizes"] == [3, 3, 1]
    assert work["verifier_calls"] == 3
    assert work["virtual_loss_outstanding"] == 0
    assert max(row["depth"] for row in out["nodes"]) <= 3
    # No assertion that evaluated_paths equals serial: scheduling can differ.


def test_virtual_loss_is_cleaned_when_batched_verifier_raises():
    search = make_search(3, verifier=TinyVerifier(fail=True))
    with pytest.raises(RuntimeError, match="intentional verifier failure"):
        search.search_batched(x1(), leaf_batch=3, virtual_loss=1.25)
    out = search.export_tree()
    work = out["work"]
    assert work["status"] == "error"
    assert work["completed_rollouts"] == 0
    assert work["virtual_loss_applications"] > 0
    assert work["virtual_loss_applications"] == work["virtual_loss_cleanups"]
    assert work["virtual_loss_outstanding"] == 0
    assert all(row["visits"] == 0 for row in out["nodes"])
    assert all(row["value_sum"] == pytest.approx(0.0) for row in out["nodes"])


def test_repeated_calls_reset_tree_and_stats():
    search = make_search(2)
    first = search.search_batched(x1(), leaf_batch=2)
    root1 = search.root
    assert search.export_tree()["work"]["completed_rollouts"] == 2
    search.n_rollouts = 1
    second = search.search_batched(x1(), leaf_batch=4)
    root2 = search.root
    out = search.export_tree()
    assert root1 is not root2
    assert first is not second
    assert out["work"]["requested_rollouts"] == 1
    assert out["work"]["completed_rollouts"] == 1
    assert out["work"]["batch_sizes"] == [1]


def test_tree_export_contains_stable_state_and_trajectory_metadata():
    search = make_search(3)
    best = search.search(x1())
    out = search.export_tree()
    assert out["nodes"][0]["path"] == []
    assert out["nodes"][0]["z_codes_dtype"] == "int8"
    assert out["work"]["best_path"] == list(best.path)
    assert len(out["work"]["evaluated_paths"]) == 3
    assert out["work"]["node_count"] == len(out["nodes"])


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="n_rollouts"):
        make_search(-1)
    with pytest.raises(ValueError, match="max_depth"):
        make_search(1, max_depth=0)
    with pytest.raises(ValueError, match="c_puct"):
        make_search(1, c_puct=float("nan"))
    search = make_search(1)
    with pytest.raises(ValueError, match="leaf_batch"):
        search.search_batched(x1(), leaf_batch=0)
    with pytest.raises(ValueError, match="virtual_loss"):
        search.search_batched(x1(), leaf_batch=1, virtual_loss=-1.0)
    with pytest.raises(ValueError, match="shape"):
        search.search(torch.zeros((2, 4), dtype=torch.long))
    search.n_rollouts = True
    with pytest.raises(ValueError, match="n_rollouts"):
        search.search(x1())
