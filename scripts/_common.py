"""Shared helpers for task construction and reproducible training entry points."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DotDict  # noqa: E402
from common.seed import SeedStreams, make_seed_streams, set_seed  # noqa: E402
from data.datasets import GridDataset  # noqa: E402
from data.splits import build_reproducible_splits, write_manifest  # noqa: E402
from data.task_contracts import TaskContract, contract_from_config  # noqa: E402
from model.trm import TRM  # noqa: E402
from train.trainer import TrainConfig  # noqa: E402


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def id_hash(ids: Iterable[str]) -> str:
    h = hashlib.sha256()
    for item in ids:
        b = str(item).encode("utf-8")
        h.update(len(b).to_bytes(8, "little")); h.update(b)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_plain(row), sort_keys=True) + "\n")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def environment_record() -> dict[str, Any]:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    return {
        "git_sha": git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "experiment_device": "cpu",
    }


def build_trm(cfg: DotDict, ternary: bool = False, act8: bool = False) -> TRM:
    m = cfg.model
    return TRM(
        dim=m.dim,
        num_tokens=cfg.data.num_tokens,
        seq_len=cfg.data.seq_len,
        n_layers=m.n_layers,
        n=m.n,
        T=m.T,
        N_sup=m.N_sup,
        heads=m.heads,
        alpha_y=m.alpha_y,
        alpha_z=m.alpha_z,
        max_grid_size=int(cfg.data.get("max_grid_size", 32)),
        ternary=ternary,
        act8=act8,
    )


def task_contract_from(cfg: DotDict) -> TaskContract:
    """Validate YAML task/data and return its normalized executable contract."""
    return contract_from_config(str(cfg.task), cfg.data)


def build_data_splits(
    cfg: DotDict,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int | None = None,
    manifest_path: str | Path | None = None,
    *,
    forbidden_fingerprints: frozenset[str] | None = None,
    require_unique_examples: bool = False,
    grouping: str = "exact",
    require_unique_groups: bool = False,
) -> tuple[dict[str, GridDataset], dict]:
    """Build reproducible grouped train/validation/test data and manifest."""
    contract = task_contract_from(cfg)
    datasets, manifest = build_reproducible_splits(
        contract.task,
        {"train": n_train, "validation": n_val, "test": n_test},
        int(cfg.seed if seed is None else seed),
        generator_kwargs=dict(contract.generator_kwargs),
        task_scope=contract.scope,
        official_benchmark=contract.official_benchmark,
        forbidden_fingerprints=forbidden_fingerprints,
        require_unique_examples=require_unique_examples,
        grouping=grouping,
        require_unique_groups=require_unique_groups,
    )
    if manifest_path is not None:
        write_manifest(manifest_path, manifest)
    return datasets, manifest


def build_datasets(
    cfg: DotDict, n_train: int, n_val: int, seed: int | None = None
) -> tuple[GridDataset, GridDataset]:
    """Train/validation wrapper using independent split RNG streams."""
    datasets, _ = build_data_splits(cfg, n_train, n_val, 0, seed=seed)
    return datasets["train"], datasets["validation"]


def train_config_from(
    cfg: DotDict,
    max_steps: int | None = None,
    *,
    precision: str | None = None,
) -> TrainConfig:
    tcfg = TrainConfig.from_config(cfg)
    if max_steps is not None:
        tcfg.max_steps = int(max_steps)
    if precision is not None:
        tcfg.precision = str(precision)
    return tcfg


def build_seeded_training_components(
    cfg: DotDict,
    n_train: int,
    n_val: int,
    *,
    ternary: bool,
    act8: bool,
    max_steps: int | None = None,
    precision: str | None = None,
) -> tuple[TRM, GridDataset, GridDataset, TrainConfig, SeedStreams]:
    """Construct data/model under independent streams in a fixed order.

    Data gets its own seed and never advances model-initialization randomness.
    Model initialization is reseeded explicitly after data construction. Trainer
    then starts the separate training stream; validation uses the eval stream.
    """
    streams = make_seed_streams(int(cfg.seed))
    deterministic = bool(cfg.get("train", {}).get("deterministic", False))

    # Seed before data construction.
    set_seed(streams.data, deterministic=deterministic)
    train_ds, val_ds = build_datasets(cfg, n_train, n_val, seed=streams.data)

    # Independently seed before model initialization.
    set_seed(streams.model, deterministic=deterministic)
    model = build_trm(cfg, ternary=ternary, act8=act8)

    tcfg = train_config_from(cfg, max_steps=max_steps, precision=precision)
    tcfg.train_seed = streams.train
    tcfg.eval_seed = streams.eval
    return model, train_ds, val_ds, tcfg, streams
