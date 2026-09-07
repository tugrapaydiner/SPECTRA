"""Shared helpers for entry-point scripts.

Milestone 03 removes the old "Sudoku else Maze" dataset shortcut. Every retained
task is now resolved through one executable task contract before any data is
constructed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DotDict  # noqa: E402
from data.datasets import GridDataset  # noqa: E402
from data.splits import build_reproducible_splits, write_manifest  # noqa: E402
from data.task_contracts import TaskContract, contract_from_config  # noqa: E402
from model.trm import TRM  # noqa: E402
from train.trainer import TrainConfig  # noqa: E402


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
    """Validate the YAML task/data block and return its normalized contract."""
    return contract_from_config(str(cfg.task), cfg.data)


def build_data_splits(
    cfg: DotDict,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[dict[str, GridDataset], dict]:
    """Build reproducible grouped train/validation/test data and its manifest."""
    contract = task_contract_from(cfg)
    datasets, manifest = build_reproducible_splits(
        contract.task,
        {"train": n_train, "validation": n_val, "test": n_test},
        int(cfg.seed if seed is None else seed),
        generator_kwargs=dict(contract.generator_kwargs),
        task_scope=contract.scope,
        official_benchmark=contract.official_benchmark,
    )
    if manifest_path is not None:
        write_manifest(manifest_path, manifest)
    return datasets, manifest


def build_datasets(
    cfg: DotDict, n_train: int, n_val: int, seed: int | None = None
) -> tuple[GridDataset, GridDataset]:
    """Backward-compatible train/validation wrapper using separate RNG streams."""
    datasets, _ = build_data_splits(cfg, n_train, n_val, 0, seed=seed)
    return datasets["train"], datasets["validation"]


def train_config_from(cfg: DotDict, max_steps: int | None = None) -> TrainConfig:
    t = cfg.train
    return TrainConfig(
        lr=float(t.lr),
        weight_decay=float(t.weight_decay),
        batch_size=int(t.batch_size),
        max_steps=int(max_steps if max_steps is not None else t.max_steps),
        lr_warmup_steps=int(t.get("warmup_steps", 100)),
        quant_warmup_steps=int(t.get("quant_warmup_steps", 500)),
        clip_grad_norm=float(t.clip_grad_norm),
        ema_decay=float(t.ema_decay),
        lambda_h=float(t.lambda_h),
        lambda_improve=float(t.lambda_improve),
        margin=float(t.margin),
        log_every=int(t.get("log_every", 50)),
        eval_every=int(t.get("eval_every", 500)),
        seed=int(cfg.seed),
        device=str(cfg.device),
    )
