#!/usr/bin/env python3
"""Build reproducible grouped train/validation/test manifests for one task config."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import load_config
from scripts._common import build_data_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-size", type=int, default=32)
    ap.add_argument("--val-size", type=int, default=8)
    ap.add_argument("--test-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    datasets, manifest = build_data_splits(
        cfg,
        args.train_size,
        args.val_size,
        args.test_size,
        seed=args.seed,
        manifest_path=Path(args.out),
    )
    print(
        f"task={cfg.task} generator={manifest['generator_version']} "
        f"train={len(datasets['train'])} val={len(datasets['validation'])} "
        f"test={len(datasets['test'])} out={args.out}"
    )
    print("duplicate_audit=", manifest["duplicate_audit"])


if __name__ == "__main__":
    main()
