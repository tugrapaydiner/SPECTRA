"""Milestone 04: reproducible training/checkpoint contract."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from common import load_config
from scripts._common import build_seeded_training_components
from train.checkpoint import (
    CheckpointError,
    load_checkpoint_payload,
    load_weight_identity,
    select_weight_state,
)
from train.trainer import Trainer


M04_ATOL = 1e-7
M04_RTOL = 1e-6


def _components(*, ternary: bool = False):
    cfg = load_config("config/m04_cpu_reference.yaml")
    model, train_ds, val_ds, tcfg, streams = build_seeded_training_components(
        cfg, 24, 12, ternary=ternary, act8=False
    )
    return cfg, model, train_ds, val_ds, tcfg, streams


def _assert_state_close(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=M04_ATOL, rtol=M04_RTOL)


def test_same_seed_starts_from_identical_weights():
    _, model_a, train_a, val_a, _, streams_a = _components()
    # Deliberately perturb global RNGs between constructions.
    torch.rand(17)
    np.random.random(17)

    _, model_b, train_b, val_b, _, streams_b = _components()
    assert streams_a == streams_b
    _assert_state_close(model_a.state_dict(), model_b.state_dict())
    assert np.array_equal(train_a.inputs, train_b.inputs)
    assert np.array_equal(train_a.targets, train_b.targets)
    assert np.array_equal(val_a.inputs, val_b.inputs)
    assert np.array_equal(val_a.targets, val_b.targets)


def test_cpu_reference_precision_is_explicit_and_fp16_label_cannot_fake_cpu():
    cfg, model, train_ds, val_ds, tcfg, _ = _components()
    trainer = Trainer(model, train_ds, val_ds, tcfg, run_config=cfg)
    runtime = trainer.runtime_settings()
    assert runtime == {
        "backend": "pytorch_eager",
        "device": "cpu",
        "device_type": "cpu",
        "precision_mode": "fp32",
        "parameter_dtype": "float32",
        "autocast_dtype": None,
        "grad_scaler": False,
    }

    bad = replace(tcfg, precision="fp16_amp")
    _, model2, train2, val2, _, _ = _components()
    with pytest.raises(ValueError, match="requires CUDA|implemented only for CUDA"):
        Trainer(model2, train2, val2, bad, run_config=cfg)


def test_validation_is_bounded():
    cfg, model, train_ds, val_ds, tcfg, _ = _components()
    tcfg.eval_batches = 2
    trainer = Trainer(model, train_ds, val_ds, tcfg, run_config=cfg)
    ev = trainer.evaluate()
    assert ev["eval_batches"] == 2
    assert ev["eval_examples"] == min(len(val_ds), 2 * tcfg.batch_size)
    assert ev["weight_identity"] == "ema"


def test_checkpoint_schema_quant_state_and_atomic_write(tmp_path: Path):
    cfg, model, train_ds, val_ds, tcfg, _ = _components(ternary=True)
    tcfg.max_steps = 4
    tcfg.quant_warmup_steps = 4
    trainer = Trainer(model, train_ds, val_ds, tcfg, run_config=cfg)
    path = tmp_path / "student.pt"
    trainer.fit(stop_at_step=2, checkpoint_path=path)

    payload = load_checkpoint_payload(path, require_resume=True)
    assert payload["version"] == 1
    assert payload["weights"]["training_identity"] == "raw"
    assert payload["weights"]["evaluation_identity"] == "ema"
    assert payload["weights"]["raw"] is not None
    assert payload["weights"]["ema"] is not None
    assert payload["training"]["global_step"] == 2
    assert payload["training"]["optimizer"]
    assert payload["training"]["scheduler"]
    assert payload["training"]["quantization"]["enabled"] is True
    assert payload["training"]["quantization"]["warmup_steps"] == 4
    assert payload["training"]["quantization"]["strengths"]
    assert payload["rng"]["schema_version"] == 1
    assert payload["sampler"]["position"] >= 0
    assert payload["architecture"]["ternary"] is True
    assert payload["task"]["train_dataset_sha256"]
    assert not list(tmp_path.glob(".student.pt.*.tmp"))


def test_legacy_checkpoint_migrates_weights_only_and_never_fakes_ema(tmp_path: Path):
    cfg, model, train_ds, val_ds, tcfg, _ = _components()
    trainer = Trainer(model, train_ds, val_ds, tcfg, run_config=cfg)

    raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema = trainer.ema.state_dict()
    first_ema_key = next(iter(ema))
    ema[first_ema_key].zero_()

    path = tmp_path / "legacy.pt"
    torch.save({"model": raw, "ema": ema}, path)
    migrated = load_checkpoint_payload(path, allow_legacy=True)
    assert migrated["resume_capable"] is False
    with pytest.raises(CheckpointError, match="cannot deterministically resume"):
        load_checkpoint_payload(path, allow_legacy=True, require_resume=True)

    _, target, _, _, _, _ = _components()
    load_weight_identity(target, migrated, identity="ema")
    target_params = dict(target.named_parameters())
    assert torch.equal(target_params[first_ema_key], ema[first_ema_key])
    assert not torch.equal(target_params[first_ema_key], raw[first_ema_key])

    no_ema = tmp_path / "legacy_no_ema.pt"
    torch.save({"model": raw}, no_ema)
    migrated_no_ema = load_checkpoint_payload(no_ema, allow_legacy=True)
    with pytest.raises(CheckpointError, match="substitution is forbidden"):
        select_weight_state(migrated_no_ema, "ema")


def test_interrupted_resume_matches_uninterrupted_cpu_reference(tmp_path: Path):
    # Uninterrupted reference under the original 8-step schedule.
    cfg_a, model_a, train_a, val_a, tcfg_a, _ = _components()
    full = Trainer(model_a, train_a, val_a, tcfg_a, run_config=cfg_a)
    full_result = full.fit()
    full_raw = {k: v.detach().clone() for k, v in full.model.state_dict().items()}
    full_ema = full.ema.state_dict()
    full_sampler = full.train_sampler.state_dict()

    # Interrupt at step 4 without shortening the original scheduler horizon.
    cfg_b, model_b, train_b, val_b, tcfg_b, _ = _components()
    interrupted = Trainer(model_b, train_b, val_b, tcfg_b, run_config=cfg_b)
    ckpt = tmp_path / "resume.pt"
    interrupted.fit(stop_at_step=4, checkpoint_path=ckpt)
    assert interrupted.cfg.max_steps == 8
    assert interrupted.step == 4

    # Fresh process-equivalent objects; restore all training state and continue.
    cfg_c, model_c, train_c, val_c, tcfg_c, _ = _components()
    resumed = Trainer(model_c, train_c, val_c, tcfg_c, run_config=cfg_c)
    resumed.load_checkpoint(ckpt)
    assert resumed.step == 4
    resumed_result = resumed.fit()

    assert resumed.step == full.step == 8
    _assert_state_close(full_raw, resumed.model.state_dict())
    _assert_state_close(full_ema, resumed.ema.state_dict())
    assert resumed.scheduler.get_last_lr() == pytest.approx(
        full.scheduler.get_last_lr(), rel=M04_RTOL, abs=M04_ATOL
    )
    assert resumed.train_sampler.state_dict()["epoch"] == full_sampler["epoch"]
    assert resumed.train_sampler.state_dict()["position"] == full_sampler["position"]
    assert torch.equal(resumed.train_sampler.state_dict()["order"], full_sampler["order"])
    assert resumed_result["final"]["cell_acc"] == pytest.approx(
        full_result["final"]["cell_acc"], rel=M04_RTOL, abs=M04_ATOL
    )
    assert resumed_result["final"]["board_acc"] == pytest.approx(
        full_result["final"]["board_acc"], rel=M04_RTOL, abs=M04_ATOL
    )


def test_nonfinite_loss_fails_before_optimizer_step(monkeypatch):
    cfg, model, train_ds, val_ds, tcfg, _ = _components()
    tcfg.max_steps = 1
    trainer = Trainer(model, train_ds, val_ds, tcfg, run_config=cfg)

    def nan_loss(steps, target, *args, **kwargs):
        return steps[-1]["logits"].sum() * torch.tensor(float("nan"))

    monkeypatch.setattr("train.trainer.deep_supervision_loss", nan_loss)
    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        trainer.fit(stop_at_step=1)
