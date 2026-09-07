"""Strict M09 state-conditioned action checkpoint save/load helpers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from eval.checkpoint_eval import EvaluationContractError, LoadedTRMCheckpoint
from model.latent_action import (
    BudgetAlignedChallengerCodebook,
    StateConditionedLatentActionCodebook,
)
from train.checkpoint import atomic_torch_save

ACTION_FORMAT = "spectra.m09_action_policy"
ACTION_VERSION = 1
ACTION_KIND = "state_conditioned_latent_action"
SUPPORTED_FAMILIES = {
    "StateConditionedLatentActionCodebook": StateConditionedLatentActionCodebook,
    "BudgetAlignedChallengerCodebook": BudgetAlignedChallengerCodebook,
}


def _sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError(f"M09 checkpoint {name} must be a mapping")
    return {str(k): v for k, v in value.items()}


def action_architecture(module) -> dict[str, Any]:
    return {
        "class": type(module).__name__,
        "dim": int(module.dim),
        "num_tokens": int(module.num_tokens),
        "n_actions": int(module.n_actions),
        "scale": float(module.scale),
        "hidden_dim": int(module.hidden_dim),
        "state_representation": module.STATE_REPRESENTATION,
        "prior_semantics": module.PRIOR_SEMANTICS,
    }


def save_m09_action_checkpoint(
    module,
    path: str | Path,
    *,
    core: LoadedTRMCheckpoint,
    optimizer: torch.optim.Optimizer,
    trained_steps: int,
    fitted_version: str,
    provenance: Mapping[str, Any],
) -> None:
    if type(module).__name__ not in SUPPORTED_FAMILIES:
        raise EvaluationContractError("unsupported M09 action-policy family")
    if int(trained_steps) <= 0:
        raise EvaluationContractError("M09 action checkpoint requires trained_steps > 0")
    if not fitted_version:
        raise EvaluationContractError("M09 action checkpoint requires fitted_version")
    prov = dict(provenance)
    if prov.get("reference_target_used") is not False:
        raise EvaluationContractError("M09 action provenance must record reference_target_used=false")
    payload = {
        "format": ACTION_FORMAT,
        "version": ACTION_VERSION,
        "kind": ACTION_KIND,
        "architecture": action_architecture(module),
        "compatibility": {
            "core_checkpoint_sha256": core.sha256,
            "task": str(core.task["task"]),
            "dim": int(core.architecture["dim"]),
            "num_tokens": int(core.architecture["num_tokens"]),
            "seq_len": int(core.architecture["seq_len"]),
        },
        "training": {
            "trained_steps": int(trained_steps),
            "fitted_version": str(fitted_version),
            "optimizer_class": type(optimizer).__name__,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "provenance": prov,
        "state_dict": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
    }
    atomic_torch_save(payload, path)


@dataclass(frozen=True)
class LoadedM09ActionPolicy:
    path: Path
    sha256: str
    module: torch.nn.Module
    payload: dict[str, Any]


def load_m09_action_checkpoint(path: str | Path, core: LoadedTRMCheckpoint) -> LoadedM09ActionPolicy:
    p = Path(path)
    if not p.is_file():
        raise EvaluationContractError(f"M09 action checkpoint does not exist: {p}")
    try:
        payload = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(p, map_location="cpu")
    except (OSError, RuntimeError) as exc:
        raise EvaluationContractError(f"cannot load M09 action checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise EvaluationContractError("M09 action checkpoint root must be a mapping")
    payload = dict(payload)
    if payload.get("format") != ACTION_FORMAT or payload.get("version") != ACTION_VERSION:
        raise EvaluationContractError("unsupported M09 action checkpoint format/version")
    if payload.get("kind") != ACTION_KIND:
        raise EvaluationContractError("wrong M09 action checkpoint kind")

    comp = _plain_mapping(payload.get("compatibility"), "compatibility")
    expected = {
        "core_checkpoint_sha256": core.sha256,
        "task": str(core.task["task"]),
        "dim": int(core.architecture["dim"]),
        "num_tokens": int(core.architecture["num_tokens"]),
        "seq_len": int(core.architecture["seq_len"]),
    }
    bad = {k: (comp.get(k), v) for k, v in expected.items() if comp.get(k) != v}
    if bad:
        raise EvaluationContractError(f"M09 action checkpoint incompatible with core: {bad}")

    arch = _plain_mapping(payload.get("architecture"), "architecture")
    required = {"class", "dim", "num_tokens", "n_actions", "scale", "hidden_dim", "state_representation", "prior_semantics"}
    missing = sorted(required - set(arch))
    if missing:
        raise EvaluationContractError(f"M09 action architecture metadata missing {missing}")
    family = SUPPORTED_FAMILIES.get(str(arch["class"]))
    if family is None:
        raise EvaluationContractError("unsupported M09 action-policy family")
    if arch["state_representation"] != family.STATE_REPRESENTATION:
        raise EvaluationContractError("unsupported M09 search-state representation")
    if arch["prior_semantics"] != family.PRIOR_SEMANTICS:
        raise EvaluationContractError("unsupported M09 prior semantics")

    training = _plain_mapping(payload.get("training"), "training")
    if int(training.get("trained_steps", 0)) <= 0 or not training.get("fitted_version"):
        raise EvaluationContractError("M09 action checkpoint lacks trained version metadata")
    prov = _plain_mapping(payload.get("provenance"), "provenance")
    required_prov = {
        "reference_target_used", "candidate_bank_seed", "candidate_bank_count",
        "selected_candidate_indices", "target_utility_table_sha256", "reasoner_tensor_state_sha256",
        "train_id_sha256", "development_id_sha256", "test_id_sha256", "target_generation_work", "training_method",
    }
    missing_prov = sorted(required_prov - set(prov))
    if missing_prov:
        raise EvaluationContractError(f"M09 action provenance missing {missing_prov}")
    if prov["reference_target_used"] is not False:
        raise EvaluationContractError("M09 action checkpoint target provenance is leaky")

    module = family(
        dim=int(arch["dim"]), num_tokens=int(arch["num_tokens"]), n_actions=int(arch["n_actions"]),
        scale=float(arch["scale"]), hidden_dim=int(arch["hidden_dim"]),
    )
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not all(torch.is_tensor(v) for v in state.values()):
        raise EvaluationContractError("M09 action state_dict is malformed")
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise EvaluationContractError(f"M09 action state_dict incompatible with metadata: {exc}") from exc
    return LoadedM09ActionPolicy(p, _sha256_file(p), module.to(core.device).eval(), payload)
