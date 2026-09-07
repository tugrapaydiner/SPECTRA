"""Shared helpers for task construction and reproducible training entry points."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DotDict  # noqa: E402
from common.seed import SeedStreams, make_seed_streams, set_seed  # noqa: E402
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
    """Validate YAML task/data and return its normalized executable contract."""
    return contract_from_config(str(cfg.task), cfg.data)


def build_data_splits(
    cfg: DotDict,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int | None = None,
    manifest_path: str | Path | None = None,
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
