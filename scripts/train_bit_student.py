"""Train the W1.58 student with explicit precision and resumable state."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

from common import get_logger, load_config  # noqa: E402
from common.logging_utils import MetricLogger  # noqa: E402
from model.stability import ternary_report  # noqa: E402
from scripts._common import build_seeded_training_components  # noqa: E402
from train.trainer import Trainer  # noqa: E402

log = get_logger("train_bit_student")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train W1.58A8/A16 with explicit FP32 or CUDA FP16-AMP compute."
    )
    ap.add_argument("--config", default="config/sudoku.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--no-act8", action="store_true", help="ternary weights only (W1.58A16)")
    ap.add_argument(
        "--precision",
        choices=["fp32", "fp16_amp"],
        default=None,
        help="actual compute mode; fp16_amp requires CUDA",
    )
    ap.add_argument(
        "--resume",
        default=None,
        help="resume a versioned M04 checkpoint; schedule/config must match",
    )
    ap.add_argument("--out", default="outputs/bit_student")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    model, train_ds, val_ds, tcfg, streams = build_seeded_training_components(
        cfg,
        args.train_size,
        args.val_size,
        ternary=True,
        act8=not args.no_act8,
        max_steps=args.steps,
        precision=args.precision,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "student.pt"
    logger = MetricLogger(out_dir / "metrics.jsonl")
    try:
        trainer = Trainer(
            model, train_ds, val_ds, tcfg, metric_logger=logger, run_config=cfg
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
            log.info("Resumed %s at global_step=%d", args.resume, trainer.step)

        runtime = trainer.runtime_settings()
        log.info(
            "Training W1.58A%s task=%s steps=%d device=%s precision=%s backend=%s",
            8 if not args.no_act8 else 16,
            cfg.task,
            tcfg.max_steps,
            runtime["device"],
            runtime["precision_mode"],
            runtime["backend"],
        )
        log.info("Seed streams: %s", streams.as_dict())
        result = trainer.fit(checkpoint_path=checkpoint_path)
    finally:
        logger.close()

    final = result["final"]
    report = ternary_report(model)
    log.info(
        "Final EMA board_acc=%.3f | raw ternary zero%%=%.3f",
        final["board_acc"],
        report.get("zero", 0.0),
    )
    log.info(
        "Saved versioned checkpoint %s (raw training weights + EMA evaluation weights)",
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
