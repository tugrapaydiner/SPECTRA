"""Milestone 05 acceptance/adversarial tests for checkpoint-backed evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from common import load_config
from eval.checkpoint_eval import (
    EvaluationContractError,
    load_action_policy_checkpoint,
    load_latent_verifier_checkpoint,
    load_research_trm_checkpoint,
    require_learned_search_auxiliaries,
    save_auxiliary_checkpoint,
    validate_checkpoint_manifest_compatibility,
)
from eval.evaluation_manifest import (
    EvaluationManifestError,
    build_evaluation_manifest,
    canonical_manifest_sha256,
    load_evaluation_manifest,
    write_evaluation_manifest,
)
from eval.research_eval import (
    InferenceSetting,
    assert_unique_realized_settings,
    evaluate_setting,
    realized_setting,
)
from eval.scaling import run_checkpoint_scaling_grid, run_scaling_grid
from model.energy import LatentEnergyVerifier
from model.latent_action import LatentActionCodebook
from model.stability import quant_strength_state
from scripts._common import build_data_splits, build_seeded_training_components, task_contract_from
from train.trainer import Trainer


def _make_artifacts(tmp_path: Path, *, ternary: bool = False, act8: bool = False):
    cfg = load_config("config/m04_cpu_reference.yaml")
    model, train_ds, val_ds, tcfg, _ = build_seeded_training_components(
        cfg, 24, 12, ternary=ternary, act8=act8
    )
    trainer = Trainer(model, train_ds, val_ds, tcfg, run_config=cfg)
    ckpt = tmp_path / ("ternary.pt" if ternary else "teacher.pt")
    trainer.fit(stop_at_step=1, checkpoint_path=ckpt)

    datasets, source = build_data_splits(cfg, 8, 4, 2, seed=123456)
    contract = task_contract_from(cfg)
    payload = build_evaluation_manifest(
        datasets["test"],
        split="test",
        task_scope=contract.scope,
        official_benchmark=contract.official_benchmark,
        task_config=dict(cfg.data),
        source_manifest=source,
    )
    manifest_path = tmp_path / "test_manifest.json"
    write_evaluation_manifest(manifest_path, payload)
    manifest = load_evaluation_manifest(manifest_path)
    core = load_research_trm_checkpoint(ckpt, device="cpu", weight_identity="recorded")
    return cfg, ckpt, core, manifest


def _trained_auxiliaries(tmp_path: Path, core, manifest):
    dim = int(core.architecture["dim"])
    num_tokens = int(core.architecture["num_tokens"])
    max_grid = int(core.architecture["max_grid_size"])
    verifier = LatentEnergyVerifier(
        num_tokens=num_tokens, dim=dim, n_layers=1, heads=2,
        max_grid_size=max_grid, act_bits=8,
    )
    # A real optimizer step keeps the fixture honest: it is not a fresh random
    # module relabelled as trained, even though this one-step fixture has no
    # scientific quality claim.
    x = torch.from_numpy(manifest.dataset.inputs[:1])
    z = torch.zeros((1, manifest.dataset.height * manifest.dataset.width, dim))
    opt_v = torch.optim.SGD(verifier.parameters(), lr=1e-3)
    opt_v.zero_grad(); verifier(x, z, manifest.dataset.width).mean().backward(); opt_v.step()

    action = LatentActionCodebook(dim, n_actions=3, scale=0.5)
    opt_a = torch.optim.SGD(action.parameters(), lr=1e-3)
    opt_a.zero_grad(); (action.directions.pow(2).mean() + action.prior_logits.sum()).backward(); opt_a.step()

    vp = tmp_path / "verifier.pt"
    ap = tmp_path / "action.pt"
    save_auxiliary_checkpoint(
        verifier, vp, kind="latent_energy_verifier", core=core, trained_steps=1,
        architecture={
            "class": "LatentEnergyVerifier", "num_tokens": num_tokens, "dim": dim,
            "n_layers": 1, "heads": 2, "max_grid_size": max_grid, "act_bits": 8,
        },
    )
    save_auxiliary_checkpoint(
        action, ap, kind="latent_action_codebook", core=core, trained_steps=1,
        architecture={
            "class": "LatentActionCodebook", "dim": dim, "n_actions": 3, "scale": 0.5,
        },
    )
    return load_latent_verifier_checkpoint(vp, core), load_action_policy_checkpoint(ap, core), vp, ap


def test_research_checkpoint_is_required_and_legacy_weights_are_rejected(tmp_path: Path):
    with pytest.raises(EvaluationContractError, match="does not exist"):
        load_research_trm_checkpoint(tmp_path / "missing.pt")

    cfg = load_config("config/m04_cpu_reference.yaml")
    model, *_ = build_seeded_training_components(cfg, 4, 2, ternary=False, act8=False)
    legacy = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict()}, legacy)
    with pytest.raises(EvaluationContractError, match="invalid research checkpoint"):
        load_research_trm_checkpoint(legacy)


def test_zero_step_checkpoint_is_not_a_research_checkpoint(tmp_path: Path):
    cfg = load_config("config/m04_cpu_reference.yaml")
    model, train_ds, val_ds, tcfg, _ = build_seeded_training_components(
        cfg, 8, 4, ternary=False, act8=False
    )
    trainer = Trainer(model, train_ds, val_ds, tcfg, run_config=cfg)
    path = tmp_path / "zero.pt"
    trainer.save_checkpoint(path)
    with pytest.raises(EvaluationContractError, match="after training began"):
        load_research_trm_checkpoint(path)


def test_immutable_manifest_roundtrip_and_tamper_detection(tmp_path: Path):
    _, _, _, manifest = _make_artifacts(tmp_path)
    before = manifest.dataset.inputs.copy()
    raw = json.loads(manifest.path.read_text(encoding="utf-8"))
    raw["examples"][0]["input"][0] = (int(raw["examples"][0]["input"][0]) + 1) % int(raw["num_tokens"])
    manifest.path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationManifestError, match="digest mismatch"):
        load_evaluation_manifest(manifest.path)
    assert before.shape[1] == 16


def test_checkpoint_restores_recorded_family_flags_params_ema_quant_and_eval_mode(tmp_path: Path):
    _, _, core, manifest = _make_artifacts(tmp_path, ternary=True, act8=True)
    validate_checkpoint_manifest_compatibility(core, manifest)
    assert core.architecture["class"] == "TRM"
    assert core.architecture["ternary"] is True
    assert core.architecture["act8"] is True
    assert core.weight_identity == "ema"
    assert core.param_count == sum(p.numel() for p in core.model.parameters())
    assert not core.model.training

    # Every trainable parameter in an EMA-labelled result must actually be EMA.
    ema = core.payload["weights"]["ema"]
    trainable = {name: p for name, p in core.model.named_parameters() if p.requires_grad}
    assert set(ema) == set(trainable)
    for key, value in trainable.items():
        assert torch.equal(value.detach().cpu(), ema[key].cpu())

    # FakeBitLinear rho is non-persistent, so it must come from the explicit M04
    # quantization snapshot rather than accidentally resetting to constructor rho=1.
    expected_rho = core.payload["training"]["quantization"]["strengths"]
    assert quant_strength_state(core.model) == expected_rho
    assert set(expected_rho.values()) == {0.25}


def test_manifest_task_config_mismatch_is_rejected_even_with_valid_digest(tmp_path: Path):
    _, _, core, manifest = _make_artifacts(tmp_path)
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["task_config"]["min_clues"] = 7
    payload["manifest_sha256"] = canonical_manifest_sha256(payload)
    bad = tmp_path / "shifted.json"
    bad.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shifted = load_evaluation_manifest(bad)
    with pytest.raises(EvaluationContractError, match="task configuration differs"):
        validate_checkpoint_manifest_compatibility(core, shifted)


def test_random_init_scaling_requires_explicit_smoke_flag():
    x = torch.zeros((1, 16), dtype=torch.long)
    y = torch.ones((1, 16), dtype=torch.long)
    with pytest.raises(EvaluationContractError, match="smoke-test-only"):
        run_scaling_grid(
            [40_000], [1], [0], x, y, 4, 4, 5, 16,
            device="cpu", n_latency_runs=1,
        )


def test_learned_search_requires_both_trained_compatible_auxiliaries(tmp_path: Path):
    _, _, core, manifest = _make_artifacts(tmp_path)
    with pytest.raises(EvaluationContractError, match="requires both"):
        require_learned_search_auxiliaries(
            core, verifier_checkpoint=None, action_checkpoint=None
        )
    with pytest.raises(EvaluationContractError, match="action policy|learned search"):
        run_checkpoint_scaling_grid(
            [core], manifest, greedy_n_sup=[], search_rollouts=[1],
            auxiliary_pairs=[None], n_latency_runs=0,
        )


def test_incompatible_auxiliary_core_hash_is_rejected(tmp_path: Path):
    _, _, core, manifest = _make_artifacts(tmp_path)
    _, _, vp, _ = _trained_auxiliaries(tmp_path, core, manifest)
    payload = torch.load(vp, map_location="cpu", weights_only=True)
    payload["compatibility"]["core_checkpoint_sha256"] = "0" * 64
    bad = tmp_path / "wrong_core_verifier.pt"
    torch.save(payload, bad)
    with pytest.raises(EvaluationContractError, match="incompatible with core"):
        load_latent_verifier_checkpoint(bad, core)


def test_search_rejects_ignored_nsup_and_rng_seed_knobs(tmp_path: Path):
    _, _, core, manifest = _make_artifacts(tmp_path)
    _, action, _, _ = _trained_auxiliaries(tmp_path, core, manifest)
    with pytest.raises(EvaluationContractError, match="not a latent-MCTS transition knob"):
        InferenceSetting(mode="latent_mcts", ordinary_n_sup=2, mcts_rollouts=2).validate()
    with pytest.raises(EvaluationContractError, match="deterministic.*search_seed"):
        InferenceSetting(mode="latent_mcts", mcts_rollouts=2, search_seed=99).validate()

    setting = InferenceSetting(mode="latent_mcts", mcts_rollouts=2)
    realized = realized_setting(core, setting, action_policy=action)
    assert realized["N_sup_consumed_by_search_transition"] is False
    assert "checkpoint_N_sup" not in realized
    assert realized["search_stochastic"] is False
    assert realized["search_seed_consumed"] is None
    assert "transition_T_cycles" in realized and "transition_inner_n" in realized
    with pytest.raises(EvaluationContractError, match="same realized computation"):
        assert_unique_realized_settings([realized, dict(realized)])


def test_learned_search_emits_realized_counts_and_provenance_per_example(tmp_path: Path):
    _, _, core, manifest = _make_artifacts(tmp_path)
    verifier, action, _, _ = _trained_auxiliaries(tmp_path, core, manifest)
    setting = InferenceSetting(mode="latent_mcts", mcts_rollouts=2)
    records = evaluate_setting(
        core, manifest, setting, verifier=verifier, action_policy=action
    )
    assert len(records) == len(manifest.dataset) == 2
    ids = [r["data_id"] for r in records]
    assert ids == manifest.dataset.ids
    for row in records:
        assert row["checkpoint_sha256"] == core.sha256
        assert row["evaluation_manifest_sha256"] == manifest.canonical_sha256
        assert row["param_count"] == core.param_count
        assert row["weight_identity"] == "ema"
        assert row["eval_backend"] == "pytorch_eager"
        assert isinstance(row["prediction"], list) and len(row["prediction"]) == 16
        assert isinstance(row["correct"], bool)
        c = row["realized_compute"]
        # n_actions=3: root expansion + one leaf expansion per rollout.
        assert c["search_expansions"] == 3
        assert c["search_transition_calls"] == 9
        assert c["verifier_calls"] == 2
        assert c["rollouts_completed"] == 2
        assert c["stopping_reason"] == "rollout_budget_exhausted"
        assert row["search_example_seed"] is None
        assert row["realized_setting"]["search_stochastic"] is False


def test_greedy_fixed_manifest_changes_only_realized_nsup_compute(tmp_path: Path):
    _, _, core, manifest = _make_artifacts(tmp_path)
    rows, examples = run_checkpoint_scaling_grid(
        [core], manifest, greedy_n_sup=[1, 2], search_rollouts=[],
        auxiliary_pairs=[None], n_latency_runs=0,
    )
    assert len(rows) == 2
    assert len(examples) == 2 * len(manifest.dataset)
    by_setting = {}
    for row in examples:
        key = row["inference_setting"]["ordinary_n_sup"]
        by_setting.setdefault(key, []).append(row["data_id"])
    assert by_setting[1] == by_setting[2] == manifest.dataset.ids
    assert rows[0]["realized_setting"]["ordinary_recursive_cycle_calls_per_example"] == 1
    assert rows[1]["realized_setting"]["ordinary_recursive_cycle_calls_per_example"] == 2
