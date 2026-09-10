from __future__ import annotations

from dataclasses import replace
import itertools
import json
from types import SimpleNamespace

import pytest

from data.ancestry import ArtifactAncestry, ExclusionIndex, ManifestSource, digest
from eval.verified_search import (BudgetedVerifiedSearch, CheckedIncumbent,
                                  ValueContract, ValueTarget, pool_metrics)

A, B, C = (digest(s.encode()) for s in ("core", "verifier", "child"))


def manifest_source(fingerprints=(A,), *, ids=None, role="training", group=None):
    ids = ids or [str(i) for i in range(len(fingerprints))]
    rows = [{"id": i, "fingerprint": fp, "group_id": group or fp}
            for i, fp in zip(ids, fingerprints)]
    data = json.dumps({"splits": {"train": {"count": len(rows), "examples": rows}}}).encode()
    return ManifestSource.from_bytes("fixture.json", data, role=role,
                                     expected_sha256=digest(data), splits=("train",))


def test_ancestor_closure_catches_same_content_different_ids():
    train = manifest_source(ids=["old-id"])
    holdout = manifest_source(ids=["new-id"], role="previous_confirmation")
    registry = {A: ArtifactAncestry(A, manifests=(train,)),
                B: ArtifactAncestry(B, parents=(A,)), C: ArtifactAncestry(C, parents=(B,))}
    index = ExclusionIndex.close([C], registry)
    assert index.artifacts == tuple(sorted([A, B, C]))
    assert index.audit(holdout)["exact_overlap_count"] == 1
    with pytest.raises(ValueError, match="ancestor"):
        index.require_disjoint(holdout)


def test_group_overlap_is_excluded_even_with_different_final_content():
    index = ExclusionIndex.close([A], {A: ArtifactAncestry(A, manifests=(manifest_source(),))})
    heldout = manifest_source((B,), group=A)
    assert index.audit(heldout)["group_overlap_count"] == 1


def test_manifest_hash_and_inventory_are_checked():
    raw = b'{"splits":{"train":{"count":1,"examples":[]}}}'
    with pytest.raises(ValueError, match="hash"):
        ManifestSource.from_bytes("bad", raw, role="training", expected_sha256=A, splits=("train",))
    with pytest.raises(ValueError, match="inventory"):
        ManifestSource.from_bytes("bad", raw, role="training", expected_sha256=digest(raw), splits=("train",))


def test_missing_split_cannot_silently_be_an_empty_consumed_dataset():
    raw = b'{"splits":{"test":{"count":0,"examples":[]}}}'
    with pytest.raises(ValueError, match="missing declared split"):
        ManifestSource.from_bytes("bad", raw, role="training", expected_sha256=digest(raw), splits=("train",))


def test_missing_parent_cycles_and_unknown_roots_fail_closed():
    with pytest.raises(ValueError, match="missing ancestor"):
        ExclusionIndex.close([B], {B: ArtifactAncestry(B, parents=(A,))})
    with pytest.raises(ValueError, match="cycle"):
        ExclusionIndex.close([A], {A: ArtifactAncestry(A, parents=(B,)), B: ArtifactAncestry(B, parents=(A,))})
    with pytest.raises(ValueError, match="roots"):
        ExclusionIndex.close([], {})
    with pytest.raises(ValueError, match="missing consumed"):
        ArtifactAncestry(A)


def test_data_free_ancestor_and_explicit_development_exclusion():
    index = ExclusionIndex.close([A], {A: ArtifactAncestry(A, data_free=True)},
                                additional=[manifest_source((B,), role="development")])
    assert index.forbidden == frozenset({B})
    assert index.require_disjoint(manifest_source((C,)))["disjoint"]
    with pytest.raises(ValueError, match="data_free"):
        ArtifactAncestry(A, parents=(B,), data_free=True)


def test_generator_excludes_ancestral_content_without_changing_legacy_recipe():
    from scripts._common import build_data_splits
    from scripts.m14_primary_experiment import make_cfg
    cfg = make_cfg(1601)
    _, original = build_data_splits(cfg, 0, 0, 4)
    _, same = build_data_splits(cfg, 0, 0, 4)
    assert original == same and "external_exclusion" not in original
    forbidden = frozenset(r["fingerprint"] for r in original["splits"]["test"]["examples"])
    _, new = build_data_splits(cfg, 0, 0, 4, forbidden_fingerprints=forbidden, require_unique_examples=True)
    assert not forbidden & {r["fingerprint"] for r in new["splits"]["test"]["examples"]}
    assert new["external_exclusion"]["fingerprint_count"] == 4


def contract(target=ValueTarget.TERMINAL):
    return ValueContract(target, A, B, "transition.v1")


def search(*, target=ValueTarget.TERMINAL, checker=lambda a: a == (1,), value=lambda s: 1.0,
           transition=lambda state, action: state + (action,)):
    return BudgetedVerifiedSearch(initial=lambda: (), actions=[0, 1], transition=transition,
                                 decode=lambda state: state, checker=checker, value=value,
                                 contract=contract(target), model_sha256=A, transition_id="transition.v1")


def test_improvement_cannot_be_silently_used_as_absolute_value():
    with pytest.raises(ValueError, match="improvement"):
        search(target=ValueTarget.IMPROVEMENT)


def test_contract_is_bound_to_source_and_transition():
    with pytest.raises(ValueError, match="mismatch"):
        contract().bind_selection(model_sha256=C, transition_id="transition.v1")
    with pytest.raises(ValueError, match="mismatch"):
        contract().bind_selection(model_sha256=A, transition_id="changed")


def test_budget_contract_requires_correct_policy_and_horizon():
    c = ValueContract(ValueTarget.BUDGET, A, B, "t", continuation_policy="identity.v1", maximum_horizon=4)
    c.bind_selection(model_sha256=A, transition_id="t", continuation_policy="identity.v1", horizon=4)
    for policy, horizon in [("mcts", 4), ("identity.v1", 5), ("identity.v1", None)]:
        with pytest.raises(ValueError, match="policy or horizon"):
            c.bind_selection(model_sha256=A, transition_id="t", continuation_policy=policy, horizon=horizon)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_nonfinite_or_misspecified_values_rejected(value):
    with pytest.raises(ValueError):
        search(checker=lambda a: False, value=lambda s: value).solve(max_transitions=1, max_depth=2)


def test_valid_answer_wins_even_when_every_invalid_proxy_is_maximum():
    r = search().solve(max_transitions=8, max_depth=2)
    assert r.valid and r.answer == (1,)
    assert r.work["transitions"] == r.work["checks"] == r.work["decodes"] == 2
    assert r.work["value_calls"] == 1
    assert r.work["stop_reason"] == "valid_answer"


def test_prefix_is_charged_and_edges_are_not_generated_twice():
    calls = []
    def transition(state, action):
        calls.append(state + (action,))
        return calls[-1]
    r = search(checker=lambda a: False, transition=transition).solve(max_transitions=5, max_depth=3, identity_prefix=3)
    assert len(calls) == len(set(calls)) == r.work["transitions"] == 5
    assert calls[:3] == [(0,), (0, 0), (0, 0, 0)]
    assert r.work["checks"] == 5 and not r.valid


def test_prefix_cannot_exceed_shared_budget():
    r = search(checker=lambda a: False).solve(max_transitions=1, max_depth=4, identity_prefix=4)
    assert r.work["transitions"] == 1


def test_deadline_exhausted_during_initialization_returns_no_unchecked_fallback(monkeypatch):
    import eval.verified_search as module
    clock = itertools.count(0, 2_000_000)
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: next(clock))
    r = search().solve(max_transitions=4, max_depth=2, deadline_ms=1.0)
    assert r.answer is None and not r.valid and r.work["transitions"] == 0
    assert r.work["stop_reason"] == "soft_deadline"


def test_checker_rejects_soft_scores_instead_of_coercing_to_true():
    with pytest.raises(TypeError, match="bool"):
        search(checker=lambda a: 0.99).solve(max_transitions=1, max_depth=2)


def test_incumbent_cannot_lose_validity_under_all_small_candidate_orders():
    for order in itertools.permutations([(0, False), (1, True), (2, False)]):
        inc = CheckedIncumbent(lambda answer: answer["id"] == 1)
        for ident, valid in order:
            answer = {"id": ident}
            inc.offer(answer, 0.0 if valid else 1.0, (ident,))
            answer["id"] = -1
        snap = inc.snapshot()
        assert snap.valid and snap.answer == {"id": 1}
        snap.answer["id"] = 99
        assert inc.snapshot().answer["id"] == 1


def test_fixed_pool_decomposition_and_zero_coverage():
    groups = [{"pool_id": str(i), "candidate_validity": flags, "selected_index": 0}
              for i, flags in enumerate([[True, False], [False, True], [False, False]])]
    r = pool_metrics(groups)
    assert r["covered"] == 2 and r["selection_failures"] == 1
    assert r["coverage"] * r["conditional_selection_reliability"] == r["success"]
    assert pool_metrics(groups[2:])["conditional_selection_reliability"] is None
    with pytest.raises(ValueError, match="duplicate"):
        pool_metrics(groups + groups)


def test_missing_battery_sysfs_uses_policy_default_without_claiming_measurement(monkeypatch):
    import eval.telemetry as telemetry
    def missing():
        raise FileNotFoundError("no power_supply mount")
    monkeypatch.setattr(telemetry, "psutil", SimpleNamespace(sensors_battery=missing))
    assert telemetry._read_battery() == (1.0, 1.0)


def test_paired_cost_statistic_uses_common_seed_and_example_indices():
    from scripts.m16_cpu_experiment import effect
    rows = []
    for seed in [1401, 2402]:
        for i in range(4):
            for arm, latency in [("reference", 2.0+i), ("native", (2.0+i)/2)]:
                for r in range(3):
                    rows.append({"seed": seed, "example_index": i, "arm": arm,
                                 "latency_ms": latency, "valid": True})
    result = effect(rows)
    assert result["paired_mean_ratio_after_round_medians"] == .5
    assert result["ci95"] == [.5, .5] and result["gate_pass"]


def test_fixed_candidate_pool_is_reference_free_and_has_shared_target_states():
    import numpy as np
    import torch
    from scripts.m14_primary_experiment import make_model
    from scripts.m16_target_experiment import candidate_pool
    from data import sudoku
    core = make_model("fp_recursive_dim64", 11).eval()
    for p in core.parameters(): p.requires_grad_(False)
    rng = np.random.default_rng(1)
    solution = sudoku.generate_solution(2, rng)
    puzzle = solution.copy(); puzzle.flat[::2] = 0
    # No targets attribute exists: using a reference answer would fail this test.
    class Inputs:
        inputs = np.stack([puzzle.reshape(-1)])
        def __len__(self): return len(self.inputs)
    pool = candidate_pool(core, Inputs())
    assert pool["y"].shape == (16, 16, 64)
    assert pool["labels"].shape == (16, 3)
    assert ((pool["labels"] >= 0) & (pool["labels"] <= 1)).all()
    assert pool["depth"].tolist() == [d for d in range(1,5) for _ in range(4)]
    assert pool["action"].tolist() == [0,1,2,3] * 4


def test_microbatch_preserves_deterministic_trm_mean_loss_and_gradients():
    import torch
    from copy import deepcopy
    from train.accumulation import backward_mean_loss
    from train.losses import deep_supervision_loss
    from model.trm import TRM
    torch.manual_seed(711)
    full = TRM(dim=16, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1,
               N_sup=2, heads=4, max_grid_size=8)
    micro = deepcopy(full)
    x = torch.randint(0, 5, (7, 16)); y = torch.randint(0, 5, (7, 16))
    loss = deep_supervision_loss(full(x, height=4, width=4)[1], y)
    loss.backward()
    got = backward_mean_loss(
        lambda xx, yy: deep_supervision_loss(micro(xx, height=4, width=4)[1], yy),
        x, y, 2)
    assert abs(got-float(loss.detach())) < 1e-6
    for a, b in zip(full.parameters(), micro.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=3e-6, rtol=1e-4)
    with pytest.raises(ValueError):
        backward_mean_loss(lambda a,b: a.sum(), x, y, 0)
