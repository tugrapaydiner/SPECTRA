"""Strict checkpoint restoration for research evaluation.

Research mode never guesses architecture flags or creates fresh learned-search
auxiliaries. Everything labelled learned is restored from validated metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from eval.evaluation_manifest import LoadedEvaluationManifest
from model.energy import LatentEnergyVerifier
from model.latent_action import LatentActionCodebook
from model.stability import load_quant_strength_state, quant_strength_state
from model.trm import TRM
from train.checkpoint import CheckpointError, atomic_torch_save, load_checkpoint_payload, select_weight_state

AUXILIARY_FORMAT = "spectra.learned_auxiliary"
AUXILIARY_VERSION = 1


class EvaluationContractError(ValueError):
    """Raised when a research evaluation input cannot satisfy its contract."""


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise EvaluationContractError(f"checkpoint does not exist: {p}")
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError(f"checkpoint metadata {name!r} must be a mapping")
    return {str(k): v for k, v in value.items()}


def _tensor_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and all(isinstance(k, str) for k in value)
        and all(torch.is_tensor(v) for v in value.values())
    )


def _load_evaluation_identity(model: nn.Module, payload: Mapping[str, Any], identity: str) -> None:
    """Load raw or EMA with the same semantics used by ``EMA.average_parameters``.

    EMA checkpoints contain trainable parameters only. For EMA evaluation the
    trainable parameter set must come entirely from EMA; only persistent
    non-parameter model state is taken from the raw model state. Non-persistent
    experiment state (for example quantization strength) is restored separately
    from explicit checkpoint provenance. This never substitutes raw trainable
    parameters for a requested EMA result.
    """
    if identity == "raw":
        model.load_state_dict(select_weight_state(payload, "raw"), strict=True)
        return
    if identity != "ema":
        raise EvaluationContractError("weight identity must be raw or ema")

    raw = dict(select_weight_state(payload, "raw"))
    ema = dict(select_weight_state(payload, "ema"))
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(ema) != trainable:
        missing = sorted(trainable - set(ema))
        extra = sorted(set(ema) - trainable)
        raise EvaluationContractError(
            f"EMA trainable-parameter contract mismatch: missing={missing}, extra={extra}"
        )
    parameter_names = set(dict(model.named_parameters()))
    combined: dict[str, torch.Tensor] = {}
    for key in model.state_dict():
        if key in ema:
            combined[key] = ema[key]
        elif key in parameter_names:
            raise EvaluationContractError(
                f"EMA result is missing model parameter {key!r}; raw substitution is forbidden"
            )
        elif key in raw:
            combined[key] = raw[key]
        else:
            raise EvaluationContractError(f"checkpoint is missing non-parameter state {key!r}")
    model.load_state_dict(combined, strict=True)


@dataclass
class LoadedTRMCheckpoint:
    path: Path
    sha256: str
    payload: dict[str, Any]
    model: TRM
    weight_identity: str
    param_count: int
    device: torch.device
    eval_backend: str

    @property
    def architecture(self) -> dict[str, Any]:
        return dict(self.payload["architecture"])

    @property
    def task(self) -> dict[str, Any]:
        return dict(self.payload["task"])

    def provenance(self) -> dict[str, Any]:
        first = next(self.model.parameters())
        return {
            "checkpoint_path": str(self.path),
            "checkpoint_sha256": self.sha256,
            "checkpoint_format": self.payload["format"],
            "checkpoint_version": self.payload["version"],
            "checkpoint_global_step": int(self.payload["training"]["global_step"]),
            "weight_identity": self.weight_identity,
            "ema_buffer_policy": (
                "ema_trainable_parameters_plus_raw_persistent_state_plus_explicit_quantization_state"
                if self.weight_identity == "ema" else None
            ),
            "model_family": self.architecture["class"],
            "param_count": int(self.param_count),
            "ternary": bool(self.architecture["ternary"]),
            "act8": bool(self.architecture["act8"]),
            "checkpoint_n": int(self.architecture["n"]),
            "checkpoint_T": int(self.architecture["T"]),
            "checkpoint_N_sup": int(self.architecture["N_sup"]),
            "eval_backend": self.eval_backend,
            "eval_device": str(self.device),
            "eval_parameter_dtype": str(first.dtype).replace("torch.", ""),
            "training_runtime": dict(self.payload["runtime"]),
        }


def load_research_trm_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    weight_identity: str = "recorded",
) -> LoadedTRMCheckpoint:
    """Restore a trained TRM from current versioned metadata only."""
    p = Path(path)
    digest = sha256_file(p)
    try:
        payload = load_checkpoint_payload(
            p, map_location="cpu", allow_legacy=False, require_resume=True
        )
    except (CheckpointError, OSError, RuntimeError, ValueError) as exc:
        raise EvaluationContractError(f"invalid research checkpoint {p}: {exc}") from exc
    if payload.get("checkpoint_kind") != "training_state":
        raise EvaluationContractError("research evaluation requires a training-state checkpoint")
    training = _mapping(payload.get("training"), "training")
    if int(training.get("global_step", 0)) <= 0:
        raise EvaluationContractError("research evaluation requires a checkpoint after training began")

    arch = _mapping(payload.get("architecture"), "architecture")
    if arch.get("class") != "TRM":
        raise EvaluationContractError(
            f"unsupported research model family {arch.get('class')!r}; expected TRM"
        )
    required = (
        "dim", "num_tokens", "seq_len", "n_layers", "n", "T", "N_sup",
        "max_grid_size", "ternary", "act8", "resolved_model_config",
    )
    missing = [k for k in required if k not in arch]
    if missing:
        raise EvaluationContractError(f"checkpoint architecture metadata missing {missing}")
    model_cfg = _mapping(arch["resolved_model_config"], "resolved_model_config")
    for key in ("heads", "alpha_y", "alpha_z"):
        if key not in model_cfg:
            raise EvaluationContractError(
                f"checkpoint cannot reconstruct TRM exactly: resolved model config missing {key!r}"
            )
    task = _mapping(payload.get("task"), "task")
    if int(task.get("num_tokens", -1)) != int(arch["num_tokens"]):
        raise EvaluationContractError("checkpoint architecture/task vocabulary disagree")
    if int(task.get("height", 0)) * int(task.get("width", 0)) != int(arch["seq_len"]):
        raise EvaluationContractError("checkpoint task shape and architecture seq_len disagree")

    model = TRM(
        dim=int(arch["dim"]), num_tokens=int(arch["num_tokens"]), seq_len=int(arch["seq_len"]),
        n_layers=int(arch["n_layers"]), n=int(arch["n"]), T=int(arch["T"]),
        N_sup=int(arch["N_sup"]), heads=int(model_cfg["heads"]),
        alpha_y=float(model_cfg["alpha_y"]), alpha_z=float(model_cfg["alpha_z"]),
        max_grid_size=int(arch["max_grid_size"]), ternary=bool(arch["ternary"]),
        act8=bool(arch["act8"]),
    )
    recorded = payload["weights"].get("evaluation_identity")
    identity = recorded if weight_identity == "recorded" else weight_identity
    if identity not in {"raw", "ema"}:
        raise EvaluationContractError(
            "weight_identity must be recorded/raw/ema and the recorded identity must exist"
        )
    try:
        _load_evaluation_identity(model, payload, str(identity))
    except (CheckpointError, RuntimeError, ValueError, EvaluationContractError) as exc:
        raise EvaluationContractError(
            f"checkpoint weights are incompatible with reconstructed model: {exc}"
        ) from exc

    # FakeBitLinear.quant_strength is deliberately a non-persistent buffer, so
    # model state_dict/EMA cannot carry it. M04 stores it explicitly. Research
    # evaluation must restore that exact experiment state before inference.
    quant = _mapping(training.get("quantization", {}), "quantization")
    if bool(arch["ternary"]):
        if not bool(quant.get("enabled")):
            raise EvaluationContractError("ternary checkpoint lacks enabled quantization provenance")
        expected = _mapping(quant.get("strengths", {}), "quantization strengths")
        if not expected:
            raise EvaluationContractError("ternary checkpoint lacks quantization strengths")
        try:
            load_quant_strength_state(model, expected)
        except ValueError as exc:
            raise EvaluationContractError(f"invalid quantization state: {exc}") from exc
        actual = quant_strength_state(model)
        if expected != actual:
            raise EvaluationContractError(
                f"checkpoint quantization restore mismatch: expected={expected}, actual={actual}"
            )
    elif bool(quant.get("enabled")) or quant.get("strengths"):
        raise EvaluationContractError("non-ternary checkpoint carries active quantization state")

    dev = torch.device(device)
    model = model.to(dev).eval()
    if model.training:
        raise EvaluationContractError("research model failed to enter inference/eval mode")
    return LoadedTRMCheckpoint(
        p, digest, payload, model, str(identity), count_parameters(model), dev, "pytorch_eager"
    )


def validate_checkpoint_manifest_compatibility(
    core: LoadedTRMCheckpoint, manifest: LoadedEvaluationManifest
) -> None:
    task, mp = core.task, manifest.payload
    checks = {
        "task": (str(task.get("task")), str(mp.get("task"))),
        "height": (int(task.get("height", -1)), int(mp.get("height", -2))),
        "width": (int(task.get("width", -1)), int(mp.get("width", -2))),
        "num_tokens": (int(task.get("num_tokens", -1)), int(mp.get("num_tokens", -2))),
        "seq_len": (int(core.architecture["seq_len"]), int(mp.get("seq_len", -2))),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise EvaluationContractError(f"checkpoint/evaluation manifest mismatch: {bad}")
    saved = task.get("resolved_task_config")
    if not isinstance(saved, Mapping):
        raise EvaluationContractError("checkpoint lacks resolved task configuration")
    if dict(saved) != dict(mp.get("task_config", {})):
        raise EvaluationContractError(
            "checkpoint/evaluation manifest task configuration differs; "
            "distribution-shift evaluation must use a separately declared contract"
        )


def auxiliary_payload(
    module: nn.Module,
    *,
    kind: str,
    architecture: Mapping[str, Any],
    core: LoadedTRMCheckpoint,
    trained_steps: int,
) -> dict[str, Any]:
    if kind not in {"latent_energy_verifier", "latent_action_codebook"}:
        raise EvaluationContractError(f"unsupported auxiliary kind {kind!r}")
    if int(trained_steps) <= 0:
        raise EvaluationContractError("learned auxiliary must record trained_steps > 0")
    return {
        "format": AUXILIARY_FORMAT,
        "version": AUXILIARY_VERSION,
        "kind": kind,
        "trained_steps": int(trained_steps),
        "architecture": dict(architecture),
        "compatibility": {
            "core_checkpoint_sha256": core.sha256,
            "task": str(core.task["task"]),
            "dim": int(core.architecture["dim"]),
            "num_tokens": int(core.architecture["num_tokens"]),
            "seq_len": int(core.architecture["seq_len"]),
        },
        "state_dict": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
    }


def save_auxiliary_checkpoint(
    module: nn.Module,
    path: str | Path,
    *,
    kind: str,
    architecture: Mapping[str, Any],
    core: LoadedTRMCheckpoint,
    trained_steps: int,
) -> None:
    atomic_torch_save(
        auxiliary_payload(
            module, kind=kind, architecture=architecture, core=core, trained_steps=trained_steps
        ),
        path,
    )


@dataclass
class LoadedAuxiliary:
    path: Path
    sha256: str
    kind: str
    module: nn.Module
    payload: dict[str, Any]


def _load_aux(path: str | Path) -> tuple[Path, str, dict[str, Any]]:
    p, digest = Path(path), sha256_file(path)
    try:
        payload = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(p, map_location="cpu")
    except (OSError, RuntimeError) as exc:
        raise EvaluationContractError(f"cannot load auxiliary checkpoint {p}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise EvaluationContractError("auxiliary checkpoint root must be a mapping")
    if payload.get("format") != AUXILIARY_FORMAT or payload.get("version") != AUXILIARY_VERSION:
        raise EvaluationContractError("unsupported learned auxiliary checkpoint format/version")
    if int(payload.get("trained_steps", 0)) <= 0:
        raise EvaluationContractError("learned auxiliary is not marked as trained")
    if not _tensor_state_dict(payload.get("state_dict")):
        raise EvaluationContractError("learned auxiliary state_dict is malformed")
    return p, digest, dict(payload)


def _check_aux_core(payload: Mapping[str, Any], core: LoadedTRMCheckpoint) -> None:
    comp = _mapping(payload.get("compatibility"), "auxiliary compatibility")
    expected = {
        "core_checkpoint_sha256": core.sha256,
        "task": str(core.task["task"]),
        "dim": int(core.architecture["dim"]),
        "num_tokens": int(core.architecture["num_tokens"]),
        "seq_len": int(core.architecture["seq_len"]),
    }
    bad = {k: (comp.get(k), v) for k, v in expected.items() if comp.get(k) != v}
    if bad:
        raise EvaluationContractError(f"learned auxiliary is incompatible with core: {bad}")


def load_latent_verifier_checkpoint(path: str | Path, core: LoadedTRMCheckpoint) -> LoadedAuxiliary:
    p, digest, payload = _load_aux(path)
    if payload.get("kind") != "latent_energy_verifier":
        raise EvaluationContractError("expected latent_energy_verifier auxiliary checkpoint")
    _check_aux_core(payload, core)
    a = _mapping(payload.get("architecture"), "verifier architecture")
    if a.get("class") != "LatentEnergyVerifier":
        raise EvaluationContractError("unsupported verifier family")
    for key in ("num_tokens", "dim", "n_layers", "heads", "max_grid_size", "act_bits"):
        if key not in a:
            raise EvaluationContractError("verifier checkpoint architecture metadata is incomplete")
    verifier = LatentEnergyVerifier(
        int(a["num_tokens"]), int(a["dim"]), int(a["n_layers"]), int(a["heads"]),
        int(a["max_grid_size"]), int(a["act_bits"]),
    )
    try:
        verifier.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise EvaluationContractError(f"verifier state_dict incompatible with metadata: {exc}") from exc
    return LoadedAuxiliary(p, digest, str(payload["kind"]), verifier.to(core.device).eval(), payload)


def load_action_policy_checkpoint(path: str |Path, core: LoadedTRMCheckpoint) -> LoadedAuxiliary:
    p, digest, payload = _load_aux(path)
    if payload.get("kind") != "latent_action_codebook":
        raise EvaluationContractError("expected latent_action_codebook auxiliary checkpoint")
    _check_aux_core(payload, core)
    a = _mapping(payload.get("architecture"), "action architecture")
    if a.get("class") != "LatentActionCodebook":
        raise EvaluationContractError("unsupported action-policy family")
    for key in ("dim", "n_actions", "scale"):
        if key not in a:
            raise EvaluationContractError("action checkpoint architecture metadata is incomplete")
    codebook = LatentActionCodebook(int(a["dim"]), int(a["n_actions"]), float(a["scale"]))
    try:
        codebook.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise EvaluationContractError(f"action state_dict incompatible with metadata: {exc}") from exc
    return LoadedAuxiliary(p, digest, str(payload["kind"]), codebook.to(core.device).eval(), payload)


def require_learned_search_auxiliaries(
    core: LoadedTRMCheckpoint,
    *,
    verifier_checkpoint: str | Path | None,
    action_checkpoint: str | Path | None,
) -> tuple[LoadedAuxiliary, LoadedAuxiliary]:
    if verifier_checkpoint is None or action_checkpoint is None:
        raise EvaluationContractError(
            "learned latent MCTS requires both a compatible trained verifier checkpoint "
            "and a compatible trained action-policy checkpoint; fresh random auxiliaries are forbidden"
        )
    return (
        load_latent_verifier_checkpoint(verifier_checkpoint, core),
        load_action_policy_checkpoint(action_checkpoint, core),
    )
