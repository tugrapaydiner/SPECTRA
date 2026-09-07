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


def _plain_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError(f"checkpoint metadata {name!r} must be a mapping")
    return {str(k): v for k, v in value.items()}


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
    training = _plain_mapping(payload.get("training"), "training")
    if int(training.get("global_step", 0)) <= 0:
        raise EvaluationContractError("research evaluation requires a checkpoint after training began")

    arch = _plain_mapping(payload.get("architecture"), "architecture")
    if arch.get("class") != "TRM":
        raise EvaluationContractError(
            f"unsupported research model family {arch.get('class')!r}; expected TRM"
        )
    required_arch = (
        "dim", "num_tokens", "seq_len", "n_layers", "n", "T", "N_sup",
        "max_grid_size", "ternary", "act8", "resolved_model_config",
    )
    missing = [k for k in required_arch if k not in arch]
    if missing:
        raise EvaluationContractError(f"checkpoint architecture metadata missing {missing}")
    model_cfg = _plain_mapping(arch["resolved_model_config"], "resolved_model_config")
    for key in ("heads", "alpha_y", "alpha_z"):
        if key not in model_cfg:
            raise EvaluationContractError(
                f"checkpoint cannot reconstruct TRM exactly: resolved model config missing {key!r}"
            )

    task = _plain_mapping(payload.get("task"), "task")
    if task.get("num_tokens") is not None and int(task["num_tokens"]) != int(arch["num_tokens"]):
        raise EvaluationContractError("checkpoint architecture/task vocabulary disagree")
    if int(task.get("height", 0)) * int(task.get("width", 0)) != int(arch["seq_len"]):
        raise EvaluationContractError("checkpoint task shape and architecture seq_len disagree")

    model = TRM(
        dim=int(arch["dim"]),
        num_tokens=int(arch["num_tokens"]),
        seq_len=int(arch["seq_len"]),
        n_layers=int(arch["n_layers"]),
        n=int(arch["n"]),
        T=int(arch["T"]),
        N_sup=int(arch["N_sup"]),
        heads=int(model_cfg["heads"]),
        alpha_y=float(model_cfg["alpha_y"]),
        alpha_z=float(model_cfg["alpha_z"]),
        max_grid_size=int(arch["max_grid_size"]),
        ternary=bool(arch["ternary"]),
        act8=bool(arch["act8"]),
    )

    recorded = payload["weights"].get("evaluation_identity")
    identity = recorded if weight_identity == "recorded" else weight_identity
    if identity not in {"raw", "ema"}:
        raise EvaluationContractError(
            "weight_identity must be recorded/raw/ema and the recorded identity must exist"
        )
    try:
        state = select_weight_state(payload, str(identity))
        model.load_state_dict(state, strict=True)
    except (CheckpointError, RuntimeError, ValueError) as exc:
        raise EvaluationContractError(
            f"checkpoint weights are incompatible with reconstructed model: {exc}"
        ) from exc

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    if model.training:
        raise EvaluationContractError("research model failed to enter inference/eval mode")
    return LoadedTRMCheckpoint(
        path=p,
        sha256=digest,
        payload=payload,
        model=model,
        weight_identity=str(identity),
        param_count=count_parameters(model),
        device=dev,
        eval_backend="pytorch_eager",
    )


def validate_checkpoint_manifest_compatibility(
    core: LoadedTRMCheckpoint,
    manifest: LoadedEvaluationManifest,
) -> None:
    """Require task/config compatibility before evaluating a research checkpoint."""
    task = core.task
    mp = manifest.payload
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
    saved_task_cfg = task.get("resolved_task_config")
    if not isinstance(saved_task_cfg, Mapping):
        raise EvaluationContractError("checkpoint lacks resolved task configuration")
    if dict(saved_task_cfg) != dict(mp.get("task_config", {})):
        raise EvaluationContractError(
            "checkpoint/evaluation manifest task configuration differs; "
            "distribution-shift evaluation must use a separately declared contract"
        )


def _tensor_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and all(isinstance(k, str) for k in value)
        and all(torch.is_tensor(v) for v in value.values())
    )


def auxiliary_payload(
    module: nn.Module,
    *,
    kind: str,
    architecture: Mapping[str, Any],
    core: LoadedTRMCheckpoint,
    trained_steps: int,
) -> dict[str, Any]:
    """Build strict metadata for a learned-search auxiliary artifact."""
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
            module,
            kind=kind,
            architecture=architecture,
            core=core,
            trained_steps=trained_steps,
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


def _load_auxiliary_payload(path: str | Path) -> tuple[Path, str, dict[str, Any]]:
    p = Path(path)
    digest = sha256_file(p)
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


def _validate_aux_compatibility(payload: Mapping[str, Any], core: LoadedTRMCheckpoint) -> None:
    comp = _plain_mapping(payload.get("compatibility"), "auxiliary compatibility")
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


def load_latent_verifier_checkpoint(
    path: str | Path,
    core: LoadedTRMCheckpoint,
) -> LoadedAuxiliary:
    p, digest, payload = _load_auxiliary_payload(path)
    if payload.get("kind") != "latent_energy_verifier":
        raise EvaluationContractError("expected latent_energy_verifier auxiliary checkpoint")
    _validate_aux_compatibility(payload, core)
    a = _plain_mapping(payload.get("architecture"), "verifier architecture")
    if a.get("class") != "LatentEnergyVerifier":
        raise EvaluationContractError("unsupported verifier family")
    required = ("num_tokens", "dim", "n_layers", "heads", "max_grid_size", "act_bits")
    if any(k not in a for k in required):
        raise EvaluationContractError("verifier checkpoint architecture metadata is incomplete")
    verifier = LatentEnergyVerifier(
        num_tokens=int(a["num_tokens"]), dim=int(a["dim"]), n_layers=int(a["n_layers"]),
        heads=int(a["heads"]), max_grid_size=int(a["max_grid_size"]), act_bits=int(a["act_bits"]),
    )
    try:
        verifier.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise EvaluationContractError(f"verifier state_dict incompatible with metadata: {exc}") from exc
    verifier = verifier.to(core.device).eval()
    return LoadedAuxiliary(p, digest, str(payload["kind"]), verifier, payload)


def load_action_policy_checkpoint(
    path: str | Path,
    core: LoadedTRMCheckpoint,
) -> LoadedAuxiliary:
    p, digest, payload = _load_auxiliary_payload(path)
    if payload.get("kind") != "latent_action_codebook":
        raise EvaluationContractError("expected latent_action_codebook auxiliary checkpoint")
    _validate_aux_compatibility(payload, core)
    a = _plain_mapping(payload.get("architecture"), "action architecture")
    if a.get("class") != "LatentActionCodebook":
        raise EvaluationContractError("unsupported action-policy family")
    for key in ("dim", "n_actions", "scale"):
        if key not in a:
            raise EvaluationContractError("action checkpoint architecture metadata is incomplete")
    codebook = LatentActionCodebook(
        dim=int(a["dim"]), n_actions=int(a["n_actions"]), scale=float(a["scale"])
    )
    try:
        codebook.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise EvaluationContractError(f"action state_dict incompatible with metadata: {exc}") from exc
    codebook = codebook.to(core.device).eval()
    return LoadedAuxiliary(p, digest, str(payload["kind"]), codebook, payload)


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
