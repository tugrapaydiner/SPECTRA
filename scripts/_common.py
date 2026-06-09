"""Shared helpers for the entry-point scripts (model/dataset construction)."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when a script is run as `python scripts/foo.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from common import DotDict  # noqa: E402
from data.datasets import GridDataset, build_dataset  # noqa: E402
from model.trm import TRM  # noqa: E402
from train.trainer import TrainConfig  # noqa: E402


def build_trm(cfg: DotDict, ternary: bool = False, act8: bool = False) -> TRM:
    """Construct a :class:`TRM` from a resolved config."""
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


def _sudoku_kwargs(cfg: DotDict) -> dict:
    data = cfg.data
    clues = int((data.get("min_clues", 30) + data.get("max_clues", 50)) // 2)
    return {"box": int(data.get("box", 3)), "num_clues": clues, "augment": True}


def _maze_kwargs(cfg: DotDict) -> dict:
    data = cfg.data
    return {
        "height": int(data.height),
        "width": int(data.width),
        "min_path_len": int(data.get("min_path_len", 0)),
        "augment": True,
    }


def build_datasets(
    cfg: DotDict, n_train: int, n_val: int, seed: int | None = None
) -> tuple[GridDataset, GridDataset]:
    """Build train/val :class:`GridDataset` for the config's task."""
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    kwargs = _sudoku_kwargs(cfg) if cfg.task == "sudoku" else _maze_kwargs(cfg)
    train_ds = build_dataset(cfg.task, n_train, rng, **kwargs)
    val_ds = build_dataset(cfg.task, n_val, rng, **kwargs)
    return train_ds, val_ds


def train_config_from(cfg: DotDict, max_steps: int | None = None) -> TrainConfig:
    """Map the YAML ``train`` block to a :class:`TrainConfig`."""
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
