"""Checkpoint-backed scaling/evaluation sweep.

Research example:

    python scripts/eval_scaling_laws.py \
      --checkpoint outputs/teacher/teacher.pt \
      --manifest outputs/eval/sudoku_test.json \
      --greedy-n-sup 1 2 4 8 \
      --out outputs/scaling.csv

Random initialization is available only through ``--smoke-random-init`` and every
row is permanently marked ``research_result=false``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import get_logger, load_config, resolve_device  # noqa: E402
from eval.checkpoint_eval import (  # noqa: E402
    EvaluationContractError,
    load_research_trm_checkpoint,
    require_learned_search_auxiliaries,
)
from eval.evaluation_manifest import load_evaluation_manifest  # noqa: E402
from eval.scaling import run_checkpoint_scaling_grid, run_scaling_grid, to_dataframe  # noqa: E402
from scripts._common import build_datasets  # noqa: E402

log = get_logger("eval_scaling_laws")


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _research(args) -> None:
    if not args.checkpoint:
        raise EvaluationContractError("research scaling requires --checkpoint")
    if not args.manifest:
        raise EvaluationContractError("research scaling requires --manifest")
    if args.config or args.params or args.depths or args.rollouts:
        raise EvaluationContractError(
            "--config/--params/--depths/--rollouts are random-init smoke arguments; "
            "research architecture/task state comes from checkpoint + immutable manifest"
        )

    manifest = load_evaluation_manifest(args.manifest)
    device = resolve_device(args.device)
    cores = [
        load_research_trm_checkpoint(p, device=device, weight_identity=args.weights)
        for p in args.checkpoint
    ]

    search_rollouts = [int(v) for v in (args.search_rollouts or [])]
    auxiliary_pairs = []
    if search_rollouts:
        if len(args.verifier_checkpoint or []) != len(cores) or len(args.action_checkpoint or []) != len(cores):
            raise EvaluationContractError(
                "learned search requires one --verifier-checkpoint and one --action-checkpoint per core checkpoint"
            )
        for core, verifier_path, action_path in zip(
            cores, args.verifier_checkpoint, args.action_checkpoint
        ):
            auxiliary_pairs.append(
                require_learned_search_auxiliaries(
                    core,
                    verifier_checkpoint=verifier_path,
                    action_checkpoint=action_path,
                )
            )
    else:
        if args.verifier_checkpoint or args.action_checkpoint:
            raise EvaluationContractError(
                "auxiliary checkpoints were supplied but no --search-rollouts were requested"
            )
        auxiliary_pairs = [None] * len(cores)

    rows, examples = run_checkpoint_scaling_grid(
        cores,
        manifest,
        greedy_n_sup=args.greedy_n_sup,
        search_rollouts=search_rollouts,
        auxiliary_pairs=auxiliary_pairs,
        search_seed=int(args.search_seed),
        c_puct=float(args.c_puct),
        uncertainty_beta=float(args.uncertainty_beta),
        n_latency_runs=int(args.latency_runs),
    )
    df = to_dataframe(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    predictions = args.predictions_out or str(out.with_suffix(".predictions.jsonl"))
    _write_jsonl(predictions, examples)
    log.info(
        "research checkpoint grid: checkpoints=%d settings=%d fixed_examples=%d -> %s",
        len(cores), len(rows), len(manifest.dataset), out,
    )
    log.info("per-example predictions -> %s", predictions)


def _smoke(args) -> None:
    if not args.config:
        raise EvaluationContractError("--smoke-random-init requires --config")
    if args.checkpoint or args.manifest or args.verifier_checkpoint or args.action_checkpoint:
        raise EvaluationContractError("smoke random-init mode cannot accept research checkpoints/manifests")
    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg.device)
    _, val = build_datasets(cfg, n_train=1, n_val=int(args.val_size))
    x = torch.from_numpy(val.inputs).to(device)
    y = torch.from_numpy(val.targets).to(device)
    rows = run_scaling_grid(
        param_targets=args.params or [40_000],
        depths=args.depths or [1],
        rollouts_list=args.rollouts or [0],
        x=x,
        y=y,
        height=val.height,
        width=val.width,
        num_tokens=int(cfg.data.num_tokens),
        seq_len=int(cfg.data.seq_len),
        device=device,
        n_latency_runs=int(args.latency_runs),
        max_grid_size=int(cfg.data.get("max_grid_size", 32)),
        smoke_random_init=True,
    )
    df = to_dataframe(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.warning("SMOKE RANDOM INIT ONLY: %d rows -> %s; these are not research results", len(rows), out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpoint-backed scaling/evaluation sweep.")
    ap.add_argument("--checkpoint", nargs="*", default=[])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--weights", choices=["recorded", "raw", "ema"], default="recorded")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--greedy-n-sup", type=int, nargs="*", default=None,
                    help="ordinary forward N_sup values; this knob is not applied to MCTS transitions")
    ap.add_argument("--search-rollouts", type=int, nargs="*", default=[])
    ap.add_argument("--verifier-checkpoint", nargs="*", default=[])
    ap.add_argument("--action-checkpoint", nargs="*", default=[])
    ap.add_argument("--search-seed", type=int, default=20260907)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--uncertainty-beta", type=float, default=0.0)
    ap.add_argument("--latency-runs", type=int, default=3)
    ap.add_argument("--out", default="outputs/scaling.csv")
    ap.add_argument("--predictions-out", default=None)

    # Explicitly segregated historical/random-init plumbing mode.
    ap.add_argument("--smoke-random-init", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--params", type=int, nargs="*", default=[])
    ap.add_argument("--depths", type=int, nargs="*", default=[])
    ap.add_argument("--rollouts", type=int, nargs="*", default=[])
    ap.add_argument("--val-size", type=int, default=8)
    args = ap.parse_args()

    if args.smoke_random_init:
        _smoke(args)
    else:
        _research(args)


if __name__ == "__main__":
    main()
