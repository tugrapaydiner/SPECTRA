#!/usr/bin/env python3
"""Freeze one reproducible split into a self-contained M05 evaluation manifest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config  # noqa: E402
from eval.evaluation_manifest import (  # noqa: E402
    build_evaluation_manifest,
    write_evaluation_manifest,
)
from scripts._common import build_data_splits, task_contract_from  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an immutable evaluation snapshot.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-size", type=int, default=32)
    ap.add_argument("--val-size", type=int, default=8)
    ap.add_argument("--test-size", type=int, default=8)
    ap.add_argument("--split", choices=["train", "validation", "test"], default="test")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    contract = task_contract_from(cfg)
    datasets, source_manifest = build_data_splits(
        cfg,
        args.train_size,
        args.val_size,
        args.test_size,
        seed=args.seed,
    )
    snapshot = build_evaluation_manifest(
        datasets[args.split],
        split=args.split,
        task_scope=contract.scope,
        official_benchmark=contract.official_benchmark,
        task_config=dict(cfg.data),
        source_manifest=source_manifest,
    )
    write_evaluation_manifest(args.out, snapshot)
    print(
        f"task={snapshot['task']} split={snapshot['split']} count={snapshot['count']} "
        f"manifest_sha256={snapshot['manifest_sha256']} out={args.out}"
    )
    print("Evaluation uses the embedded input/target arrays; no regeneration occurs at eval time.")


if __name__ == "__main__":
    main()
