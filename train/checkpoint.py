"""Versioned SPECTRA training checkpoints and explicit legacy migration."""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
import torch.nn as nn


CHECKPOINT_FORMAT = "spectra.training"
CHECKPOINT_VERSION = 1
LEGACY_FORMAT = "spectra.legacy_weights"


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot satisfy the requested contract."""


def _is_tensor_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and all(isinstance(k, str) for k in value)
        and all(torch.is_tensor(v) for v in value.values())
    )


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Atomically write a checkpoint in the destination directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        try:
            dir_fd = os.open(str(target.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def migrate_legacy_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate/migrate historical ``{"model": ..., "ema": ...}`` files.

    Legacy files remain explicitly weights-only. They cannot deterministically
    resume because optimizer/scheduler/RNG/sampler state was never recorded.
    """
    raw = payload.get("model")
    ema = payload.get("ema")
    if not _is_tensor_state_dict(raw):
        raise CheckpointError("legacy checkpoint is missing a valid 'model' state_dict")
    if ema is not None and not _is_tensor_state_dict(ema):
        raise CheckpointError("legacy checkpoint 'ema' is not a valid state_dict")

    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "checkpoint_kind": "legacy_weights",
        "migrated_from": LEGACY_FORMAT,
        "resume_capable": False,
        "weights": {
            "raw": dict(raw),
            "ema": None if ema is None else dict(ema),
            "training_identity": "raw",
            "evaluation_identity": "ema" if ema is not None else None,
        },
        "architecture": None,
        "task": None,
        "runtime": None,
        "training": None,
        "rng": None,
        "sampler": None,
    }


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    require_resume: bool = False,
) -> dict[str, Any]:
    """Validate a current-format checkpoint and return it as a plain dict."""
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint root must be a mapping")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"unsupported checkpoint format {payload.get('format')!r}"
        )
    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint version {version!r}; expected {CHECKPOINT_VERSION}"
        )

    weights = payload.get("weights")
    if not isinstance(weights, Mapping) or not _is_tensor_state_dict(weights.get("raw")):
        raise CheckpointError("checkpoint is missing raw model weights")
    ema = weights.get("ema")
    if ema is not None and not _is_tensor_state_dict(ema):
        raise CheckpointError("checkpoint EMA weights are malformed")

    eval_identity = weights.get("evaluation_identity")
    if eval_identity not in (None, "raw", "ema"):
        raise CheckpointError("invalid evaluation weight identity")
    if eval_identity == "ema" and ema is None:
        raise CheckpointError(
            "checkpoint declares EMA evaluation identity but contains no EMA weights"
        )
    if weights.get("training_identity") != "raw":
        raise CheckpointError("training checkpoint identity must be raw weights")

    resume_capable = bool(payload.get("resume_capable", False))
    if require_resume and not resume_capable:
        source = payload.get("migrated_from", payload.get("checkpoint_kind", "checkpoint"))
        raise CheckpointError(
            f"{source} is weights-only and cannot deterministically resume training"
        )
    if resume_capable:
        for key in ("architecture", "task", "runtime", "training", "rng", "sampler"):
            if payload.get(key) is None:
                raise CheckpointError(
                    f"resume-capable checkpoint is missing required field {key!r}"
                )
        training = payload["training"]
        for key in (
            "global_step",
            "optimizer",
            "scheduler",
            "schedule_signature",
            "quantization",
        ):
            if key not in training:
                raise CheckpointError(f"checkpoint training state is missing {key!r}")

    return dict(payload)


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    allow_legacy: bool = True,
    require_resume: bool = False,
) -> dict[str, Any]:
    """Load, explicitly migrate supported legacy weights, and validate."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)

    if isinstance(payload, Mapping) and "format" not in payload:
        if allow_legacy and "model" in payload:
            payload = migrate_legacy_checkpoint(payload)
        else:
            raise CheckpointError("unversioned checkpoint is not a supported legacy format")
    return validate_checkpoint_payload(payload, require_resume=require_resume)


def select_weight_state(
    payload: Mapping[str, Any],
    identity: str,
) -> Mapping[str, torch.Tensor]:
    """Return explicitly requested raw or EMA weights; never substitute."""
    checked = validate_checkpoint_payload(payload)
    if identity not in {"raw", "ema"}:
        raise CheckpointError("weight identity must be 'raw' or 'ema'")
    state = checked["weights"].get(identity)
    if state is None:
        raise CheckpointError(
            f"checkpoint does not contain requested {identity!r} weights; "
            "raw/EMA substitution is forbidden"
        )
    return state


def load_weight_identity(
    model: nn.Module,
    payload: Mapping[str, Any],
    *,
    identity: str,
    strict: bool = True,
) -> None:
    """Load exactly the selected raw/EMA identity into ``model``."""
    state = select_weight_state(payload, identity)
    model.load_state_dict(state, strict=strict)
