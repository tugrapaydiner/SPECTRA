"""Train the FP16 TRM teacher (System 2) on a symbolic task.

    python scripts/train_teacher.py --config config/sudoku.yaml --steps 5000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import torch  # noqa: E402

from common import get_logger, load_config  # noqa: E402
from common.logging_utils import MetricLogger  # noqa: E402
from scripts._common import build_datasets, build_trm, train_config_from  # noqa: E402
from train.trainer import Trainer  # noqa: E402

log = get_logger("train_teacher")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the FP16 TRM teacher.")
    ap.add_argument("--config", default="config/sudoku.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--out", default="outputs/teacher")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    log.info("Building %s datasets (%d train / %d val)", cfg.task, args.train_size, args.val_size)
    train_ds, val_ds = build_datasets(cfg, args.train_size, args.val_size)

    model = build_trm(cfg, ternary=False, act8=False)
    tcfg = train_config_from(cfg, max_steps=args.steps)
    out_dir = Path(args.out)
    logger = MetricLogger(out_dir / "metrics.jsonl")

    trainer = Trainer(model, train_ds, val_ds, tcfg, metric_logger=logger)
    log.info("Training teacher for %d steps on %s", tcfg.max_steps, trainer.device)
    result = trainer.fit()
    logger.close()

    final = result["final"]
    log.info("Final cell_acc=%.3f board_acc=%.3f", final["cell_acc"], final["board_acc"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "ema": trainer.ema.state_dict()}, out_dir / "teacher.pt")
    log.info("Saved checkpoint to %s", out_dir / "teacher.pt")


if __name__ == "__main__":
    main()
