"""Train the W1.58A8 ternary student with quantization warmup (Phases 4-6).

    python scripts/train_bit_student.py --config config/sudoku.yaml --steps 8000

The Trainer applies the soft->hard quant warmup automatically for ternary models.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import torch  # noqa: E402

from common import get_logger, load_config  # noqa: E402
from common.logging_utils import MetricLogger  # noqa: E402
from model.stability import ternary_report  # noqa: E402
from scripts._common import build_datasets, build_trm, train_config_from  # noqa: E402
from train.trainer import Trainer  # noqa: E402

log = get_logger("train_bit_student")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the W1.58A8 ternary student.")
    ap.add_argument("--config", default="config/sudoku.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--no-act8", action="store_true", help="ternary weights only (W1.58A16)")
    ap.add_argument("--out", default="outputs/bit_student")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    train_ds, val_ds = build_datasets(cfg, args.train_size, args.val_size)

    model = build_trm(cfg, ternary=True, act8=not args.no_act8)
    tcfg = train_config_from(cfg, max_steps=args.steps)
    out_dir = Path(args.out)
    logger = MetricLogger(out_dir / "metrics.jsonl")

    trainer = Trainer(model, train_ds, val_ds, tcfg, metric_logger=logger)
    log.info("Training W1.58A%s student for %d steps", 8 if not args.no_act8 else 16, tcfg.max_steps)
    result = trainer.fit()
    logger.close()

    final = result["final"]
    report = ternary_report(model)
    log.info("Final board_acc=%.3f | ternary zero%%=%.3f", final["board_acc"], report.get("zero", 0.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "ema": trainer.ema.state_dict()}, out_dir / "student.pt")
    log.info("Saved checkpoint to %s", out_dir / "student.pt")


if __name__ == "__main__":
    main()
