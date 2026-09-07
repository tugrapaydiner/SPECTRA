"""Immutable checkpoint-backed evaluation snapshots for SPECTRA.

Research evaluation must consume the exact examples recorded here; it must not
silently regenerate a validation/test set from a seed at evaluation time.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from data.datasets import GridDataset

EVAL_MANIFEST_FORMAT = "spectra.eval_manifest"
EVAL_MANIFEST_VERSION = 1


class EvaluationManifestError(ValueError):
    """Raised when an evaluation snapshot is malformed or has changed."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    body = {str(k): _plain(v) for k, v in payload.items() if k != "manifest_sha256"}
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_manifest_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_manifest_sha256(source_manifest: Mapping[str, Any] | None) -> str | None:
    if source_manifest is None:
        return None
    data = json.dumps(
        _plain(source_manifest), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_evaluation_manifest(
    dataset: GridDataset,
    *,
    split: str,
    task_scope: str,
    official_benchmark: bool,
    task_config: Mapping[str, Any],
    source_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze a ``GridDataset`` into a self-contained evaluation snapshot."""
    if split not in {"train", "validation", "test"}:
        raise EvaluationManifestError("split must be train, validation, or test")
    if len(dataset.ids) != len(dataset) or len(dataset.group_ids) != len(dataset):
        raise EvaluationManifestError("evaluation dataset requires stable ids/group_ids")
    if len(set(dataset.ids)) != len(dataset.ids):
        raise EvaluationManifestError("evaluation data ids must be unique")

    examples: list[dict[str, Any]] = []
    for i in range(len(dataset)):
        examples.append({
            "id": str(dataset.ids[i]),
            "group_id": str(dataset.group_ids[i]),
            "input": dataset.inputs[i].astype(np.int64, copy=False).tolist(),
            "target": dataset.targets[i].astype(np.int64, copy=False).tolist(),
            "input_mask": dataset.input_mask[i].astype(bool, copy=False).tolist(),
            "target_mask": dataset.target_mask[i].astype(bool, copy=False).tolist(),
            "metadata": _plain(dataset.metadata[i]),
        })

    payload: dict[str, Any] = {
        "format": EVAL_MANIFEST_FORMAT,
        "version": EVAL_MANIFEST_VERSION,
        "split": split,
        "task": str(dataset.task),
        "task_scope": str(task_scope),
        "official_benchmark": bool(official_benchmark),
        "height": int(dataset.height),
        "width": int(dataset.width),
        "seq_len": int(dataset.height * dataset.width),
        "num_tokens": None if dataset.num_tokens is None else int(dataset.num_tokens),
        "pad_token": None if dataset.pad_token is None else int(dataset.pad_token),
        "task_config": _plain(task_config),
        "source_data_manifest_sha256": _source_manifest_sha256(source_manifest),
        "count": len(dataset),
        "examples": examples,
    }
    payload["manifest_sha256"] = canonical_manifest_sha256(payload)
    return payload


def validate_evaluation_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise EvaluationManifestError("evaluation manifest root must be a mapping")
    if payload.get("format") != EVAL_MANIFEST_FORMAT:
        raise EvaluationManifestError(
            f"unsupported evaluation manifest format {payload.get('format')!r}"
        )
    if payload.get("version") != EVAL_MANIFEST_VERSION:
        raise EvaluationManifestError(
            f"unsupported evaluation manifest version {payload.get('version')!r}"
        )
    expected_digest = payload.get("manifest_sha256")
    actual_digest = canonical_manifest_sha256(payload)
    if expected_digest != actual_digest:
        raise EvaluationManifestError(
            "evaluation manifest digest mismatch; the immutable snapshot changed"
        )

    for key in ("task", "height", "width", "seq_len", "num_tokens", "task_config", "examples"):
        if key not in payload:
            raise EvaluationManifestError(f"evaluation manifest is missing {key!r}")
    h, w = int(payload["height"]), int(payload["width"])
    seq_len = int(payload["seq_len"])
    num_tokens = int(payload["num_tokens"])
    if h <= 0 or w <= 0 or seq_len != h * w or num_tokens <= 0:
        raise EvaluationManifestError("invalid evaluation shape/vocabulary contract")

    examples = payload["examples"]
    if not isinstance(examples, list) or int(payload.get("count", -1)) != len(examples):
        raise EvaluationManifestError("evaluation manifest count/examples disagree")
    ids: list[str] = []
    for row in examples:
        if not isinstance(row, Mapping):
            raise EvaluationManifestError("evaluation example must be a mapping")
        for key in ("id", "group_id", "input", "target", "input_mask", "target_mask"):
            if key not in row:
                raise EvaluationManifestError(f"evaluation example is missing {key!r}")
        ids.append(str(row["id"]))
        x = np.asarray(row["input"])
        y = np.asarray(row["target"])
        im = np.asarray(row["input_mask"])
        tm = np.asarray(row["target_mask"])
        if x.shape != (seq_len,) or y.shape != (seq_len,):
            raise EvaluationManifestError("evaluation example sequence shape mismatch")
        if im.shape != (seq_len,) or tm.shape != (seq_len,):
            raise EvaluationManifestError("evaluation example mask shape mismatch")
        if not np.issubdtype(x.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
            raise EvaluationManifestError("evaluation tokens must be integral")
        if np.any(x < 0) or np.any(x >= num_tokens) or np.any(y < 0) or np.any(y >= num_tokens):
            raise EvaluationManifestError("evaluation token outside declared vocabulary")
    if len(set(ids)) != len(ids):
        raise EvaluationManifestError("evaluation example ids are not unique")
    return dict(payload)


def dataset_from_evaluation_manifest(payload: Mapping[str, Any]) -> GridDataset:
    checked = validate_evaluation_manifest(payload)
    n = len(checked["examples"])
    seq_len = int(checked["seq_len"])
    if n:
        x = np.asarray([r["input"] for r in checked["examples"]], dtype=np.int64)
        y = np.asarray([r["target"] for r in checked["examples"]], dtype=np.int64)
        im = np.asarray([r["input_mask"] for r in checked["examples"]], dtype=bool)
        tm = np.asarray([r["target_mask"] for r in checked["examples"]], dtype=bool)
    else:
        x = np.empty((0, seq_len), dtype=np.int64)
        y = np.empty_like(x)
        im = np.empty((0, seq_len), dtype=bool)
        tm = np.empty((0, seq_len), dtype=bool)
    return GridDataset(
        x,
        y,
        int(checked["height"]),
        int(checked["width"]),
        task=str(checked["task"]),
        num_tokens=int(checked["num_tokens"]),
        pad_token=checked.get("pad_token"),
        input_mask=im,
        target_mask=tm,
        ids=[str(r["id"]) for r in checked["examples"]],
        group_ids=[str(r["group_id"]) for r in checked["examples"]],
        metadata=[dict(r.get("metadata", {})) for r in checked["examples"]],
    )


@dataclass(frozen=True)
class LoadedEvaluationManifest:
    path: Path
    payload: dict[str, Any]
    dataset: GridDataset
    canonical_sha256: str
    file_sha256: str


def load_evaluation_manifest(path: str | Path) -> LoadedEvaluationManifest:
    p = Path(path)
    if not p.is_file():
        raise EvaluationManifestError(f"evaluation manifest does not exist: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationManifestError(f"cannot read evaluation manifest {p}: {exc}") from exc
    checked = validate_evaluation_manifest(payload)
    return LoadedEvaluationManifest(
        path=p,
        payload=checked,
        dataset=dataset_from_evaluation_manifest(checked),
        canonical_sha256=str(checked["manifest_sha256"]),
        file_sha256=file_sha256(p),
    )


def write_evaluation_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    checked = validate_evaluation_manifest(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(checked, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
