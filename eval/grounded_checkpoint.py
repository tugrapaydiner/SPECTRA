"""Strict M07 grounded-verifier auxiliary checkpoints.

The file format intentionally reuses ``spectra.learned_auxiliary`` v1 so core
compatibility semantics remain aligned with M05, while adding explicit grounding
and state-representation metadata required by M07.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from eval.checkpoint_eval import (
    AUXILIARY_FORMAT,
    AUXILIARY_VERSION,
    EvaluationContractError,
    LoadedTRMCheckpoint,
    sha256_file,
)
from model.grounded_verifier import (
    GROUNDED_TARGET_ID,
    STATE_REPRESENTATION_VERSION,
    EnsembleGroundedStateVerifier,
    GroundedStateVerifier,
    grounded_architecture_metadata,
)
from train.checkpoint import atomic_torch_save

GROUNDED_AUX_KIND = "grounded_state_verifier"


@dataclass
class LoadedGroundedVerifier:
    path: Path
    sha256: str
    module: torch.nn.Module
    payload: dict[str, Any]


def _tensor_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and all(isinstance(k, str) for k in value)
        and all(torch.is_tensor(v) for v in value.values())
    )


def grounded_verifier_payload(
    module: torch.nn.Module,
    *,
    core: LoadedTRMCheckpoint,
    trained_steps: int,
    grounding: Mapping[str, Any],
    member_seeds: list[int],
    data_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if int(trained_steps) <= 0:
        raise EvaluationContractError("grounded verifier must record trained_steps > 0")
    arch = grounded_architecture_metadata(module)
    if not bool(arch.get("include_y")):
        # z-only models may be saved as diagnostic files, but not under the accepted kind.
        raise EvaluationContractError(
            "accepted grounded_state_verifier must include y; z-only ablations use a separate diagnostic format"
        )
    if arch.get("state_representation") != STATE_REPRESENTATION_VERSION:
        raise EvaluationContractError("grounded verifier representation version mismatch")

    g = dict(grounding)
    required = {
        "target_id": GROUNDED_TARGET_ID,
        "state_representation": STATE_REPRESENTATION_VERSION,
        "oracle": "model.verifier.sudoku_score",
        "reference_target_used": False,
        "bootstrapped": False,
    }
    bad = {k: (g.get(k), v) for k, v in required.items() if g.get(k) != v}
    if bad:
        raise EvaluationContractError(f"grounded verifier grounding metadata invalid: {bad}")

    return {
        "format": AUXILIARY_FORMAT,
        "version": AUXILIARY_VERSION,
        "kind": GROUNDED_AUX_KIND,
        "trained_steps": int(trained_steps),
        "architecture": arch,
        "compatibility": {
            "core_checkpoint_sha256": core.sha256,
            "task": str(core.task["task"]),
            "dim": int(core.architecture["dim"]),
            "num_tokens": int(core.architecture["num_tokens"]),
            "seq_len": int(core.architecture["seq_len"]),
        },
        "grounding": g,
        "member_seeds": [int(s) for s in member_seeds],
        "data_provenance": dict(data_provenance),
        "uncertainty": {
            "member_disagreement": "heuristic",
            "mcts_uncertainty_penalty_enabled_by_m07": False,
        },
        "state_dict": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
    }


def save_grounded_verifier_checkpoint(
    module: torch.nn.Module,
    path: str | Path,
    *,
    core: LoadedTRMCheckpoint,
    trained_steps: int,
    grounding: Mapping[str, Any],
    member_seeds: list[int],
    data_provenance: Mapping[str, Any],
) -> None:
    atomic_torch_save(
        grounded_verifier_payload(
            module,
            core=core,
            trained_steps=trained_steps,
            grounding=grounding,
            member_seeds=member_seeds,
            data_provenance=data_provenance,
        ),
        path,
    )


def save_z_only_ablation_checkpoint(
    module: torch.nn.Module,
    path: str | Path,
    *,
    core: LoadedTRMCheckpoint,
    trained_steps: int,
    grounding: Mapping[str, Any],
    member_seeds: list[int],
) -> None:
    """Save a diagnostic z-only model that can never masquerade as accepted M07."""
    arch = grounded_architecture_metadata(module)
    if bool(arch.get("include_y")):
        raise EvaluationContractError("z-only ablation checkpoint unexpectedly includes y")
    atomic_torch_save(
        {
            "format": "spectra.m07_z_only_ablation",
            "version": 1,
            "kind": "diagnostic_z_only_verifier",
            "trained_steps": int(trained_steps),
            "architecture": arch,
            "compatibility": {
                "core_checkpoint_sha256": core.sha256,
                "task": str(core.task["task"]),
                "dim": int(core.architecture["dim"]),
                "num_tokens": int(core.architecture["num_tokens"]),
                "seq_len": int(core.architecture["seq_len"]),
            },
            "grounding": dict(grounding),
            "member_seeds": [int(s) for s in member_seeds],
            "accepted_m07_verifier": False,
            "state_dict": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
        },
        path,
    )


def _check_core(payload: Mapping[str, Any], core: LoadedTRMCheckpoint) -> None:
    comp = payload.get("compatibility")
    if not isinstance(comp, Mapping):
        raise EvaluationContractError("grounded verifier lacks compatibility metadata")
    expected = {
        "core_checkpoint_sha256": core.sha256,
        "task": str(core.task["task"]),
        "dim": int(core.architecture["dim"]),
        "num_tokens": int(core.architecture["num_tokens"]),
        "seq_len": int(core.architecture["seq_len"]),
    }
    bad = {k: (comp.get(k), v) for k, v in expected.items() if comp.get(k) != v}
    if bad:
        raise EvaluationContractError(f"grounded verifier is incompatible with frozen core: {bad}")


def _construct_from_arch(a: Mapping[str, Any]) -> torch.nn.Module:
    required = (
        "class", "num_tokens", "dim", "n_members", "n_layers", "heads",
        "max_grid_size", "act_bits", "include_y", "state_representation",
    )
    if any(k not in a for k in required):
        raise EvaluationContractError("grounded verifier architecture metadata incomplete")
    if a["state_representation"] != STATE_REPRESENTATION_VERSION or not bool(a["include_y"]):
        raise EvaluationContractError("accepted grounded verifier must use search_state_xyz_v1 with y")
    kwargs = dict(
        num_tokens=int(a["num_tokens"]),
        dim=int(a["dim"]),
        n_layers=int(a["n_layers"]),
        heads=int(a["heads"]),
        max_grid_size=int(a["max_grid_size"]),
        act_bits=int(a["act_bits"]),
        include_y=True,
        state_representation=STATE_REPRESENTATION_VERSION,
    )
    if a["class"] == "EnsembleGroundedStateVerifier":
        return EnsembleGroundedStateVerifier(n_members=int(a["n_members"]), **kwargs)
    if a["class"] == "GroundedStateVerifier" and int(a["n_members"]) == 1:
        return GroundedStateVerifier(**kwargs)
    raise EvaluationContractError(f"unsupported grounded verifier family {a['class']!r}")


def load_grounded_verifier_checkpoint(
    path: str | Path,
    core: LoadedTRMCheckpoint,
) -> LoadedGroundedVerifier:
    p = Path(path)
    digest = sha256_file(p)
    try:
        payload = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(p, map_location="cpu")
    except (OSError, RuntimeError) as exc:
        raise EvaluationContractError(f"cannot load grounded verifier checkpoint {p}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise EvaluationContractError("grounded verifier checkpoint root must be a mapping")
    if payload.get("format") != AUXILIARY_FORMAT or payload.get("version") != AUXILIARY_VERSION:
        raise EvaluationContractError("unsupported grounded verifier auxiliary format/version")
    if payload.get("kind") != GROUNDED_AUX_KIND:
        raise EvaluationContractError("expected grounded_state_verifier checkpoint")
    if int(payload.get("trained_steps", 0)) <= 0 or not _tensor_state_dict(payload.get("state_dict")):
        raise EvaluationContractError("grounded verifier checkpoint is not a trained tensor state")
    _check_core(payload, core)

    grounding = payload.get("grounding")
    if not isinstance(grounding, Mapping):
        raise EvaluationContractError("grounded verifier lacks grounding metadata")
    required_grounding = {
        "target_id": GROUNDED_TARGET_ID,
        "state_representation": STATE_REPRESENTATION_VERSION,
        "oracle": "model.verifier.sudoku_score",
        "reference_target_used": False,
        "bootstrapped": False,
    }
    bad = {k: (grounding.get(k), v) for k, v in required_grounding.items() if grounding.get(k) != v}
    if bad:
        raise EvaluationContractError(f"grounded verifier provenance mismatch: {bad}")

    a = payload.get("architecture")
    if not isinstance(a, Mapping):
        raise EvaluationContractError("grounded verifier lacks architecture metadata")
    module = _construct_from_arch(a)
    try:
        module.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise EvaluationContractError(f"grounded verifier state_dict incompatible: {exc}") from exc
    module = module.to(core.device).eval()
    return LoadedGroundedVerifier(p, digest, module, dict(payload))
