"""Strict task/decode/target/core/data-bound value checkpoints for M17.

Only locally produced or independently hash-verified state dictionaries are
accepted. weights_only=True reduces unpickling exposure; it is not a sandbox or
protection against every malformed tensor. No unsafe pickle fallback is used.
"""
from __future__ import annotations

import io
from pathlib import Path
import torch

from data.ancestry import digest, require_sha
from eval.checkable_tasks import TaskSpec
from eval.verified_search import ValueContract, ValueTarget
from model.typed_value import TypedStateValue

FORMAT = "spectra.task_value.v2"


def architecture(spec: TaskSpec) -> dict:
    return {"num_tokens": spec.num_tokens, "dim": spec.dim, "n_layers": 1,
            "heads": 4, "max_grid_size": spec.max_grid_size, "act_bits": 8, "include_y": True}


def ancestry(core_sha256: str, manifest_sha256: str) -> dict:
    return {"parent_checkpoints": [require_sha(core_sha256)],
            "consumed_manifests": [{"sha256": require_sha(manifest_sha256),
                                    "splits": ["train"], "role": "training"}]}


def save_task_value(path: Path, model: TypedStateValue, *, spec: TaskSpec,
                    core_sha256: str, training_manifest_sha256: str) -> str:
    if path.exists():
        raise FileExistsError("refusing to overwrite a fitted value checkpoint")
    if model.target is ValueTarget.BUDGET:
        raise ValueError("this experiment has no budget-conditioned target")
    payload = {"format": FORMAT, "target": model.target.value, "task": spec.metadata(),
               "architecture": architecture(spec), "model_state": model.state_dict(),
               "data_ancestry": ancestry(core_sha256, training_manifest_sha256)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return digest(path.read_bytes())


def load_task_value(path: Path, *, expected_sha256: str, expected_core_sha256: str,
                    expected_training_manifest_sha256: str, spec: TaskSpec,
                    diagnostic_improvement: bool = False):
    raw = path.read_bytes()
    if digest(raw) != require_sha(expected_sha256):
        raise ValueError("task value checkpoint content hash mismatch")
    p = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    keys = {"format", "target", "task", "architecture", "model_state", "data_ancestry"}
    if not isinstance(p, dict) or set(p) != keys or p["format"] != FORMAT:
        raise ValueError("unsupported task value checkpoint format/inventory")
    if p["task"] != spec.metadata() or p["architecture"] != architecture(spec):
        raise ValueError("value task/decode/architecture binding mismatch")
    if p["data_ancestry"] != ancestry(expected_core_sha256, expected_training_manifest_sha256):
        raise ValueError("value reasoner or consumed-data ancestry mismatch")
    target = ValueTarget(p["target"])
    if target is ValueTarget.BUDGET:
        raise ValueError("task value v2 does not define a budget-conditioned target")
    if target is ValueTarget.IMPROVEMENT and not diagnostic_improvement:
        raise ValueError("improvement values require explicit diagnostic opt-in")
    state = p["model_state"]
    if not isinstance(state, dict) or any(not isinstance(t, torch.Tensor) or
        t.layout != torch.strided or (t.is_floating_point() and not bool(torch.isfinite(t).all()))
        for t in state.values()):
        raise ValueError("invalid or non-finite value tensor state")
    model = TypedStateValue(target, **architecture(spec))
    expected_state = model.state_dict()
    if set(state) != set(expected_state) or any(
        state[k].dtype != expected_state[k].dtype or state[k].shape != expected_state[k].shape
        for k in expected_state
    ):
        raise ValueError("value tensor inventory/shape/dtype mismatch")
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    contract = ValueContract(target, expected_core_sha256, expected_sha256,
                             spec.transition_id, state_schema=spec.state_schema)
    return model, contract
