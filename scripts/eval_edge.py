"""Checkpoint-backed edge evaluation on one immutable evaluation snapshot.

Research example:

    python scripts/eval_edge.py \
      --ckpt outputs/teacher/teacher.pt \
      --manifest outputs/eval/sudoku_test.json \
      --n-sup 1 2 4 8 \
      --out outputs/edge_report.json

A missing checkpoint is never interpreted as a research result. Random init exists
only behind ``--smoke-random-init`` and is permanently labelled as such.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import get_logger, load_config, resolve_device  # noqa: E402
from eval.benchmarks import benchmark  # noqa: E402
from eval.checkpoint_eval import EvaluationContractError, load_research_trm_checkpoint  # noqa: E402
from eval.evaluation_manifest import load_evaluation_manifest  # noqa: E402
from eval.memory import model_size_mb, process_rss_mb  # noqa: E402
from eval.reports import compute_optimal_frontier  # noqa: E402
from eval.scaling import run_checkpoint_scaling_grid  # noqa: E402
from scripts._common import build_datasets, build_trm  # noqa: E402

log = get_logger("eval_edge")


def _research(args) -> None:
    if not args.ckpt:
        raise EvaluationContractError("research edge evaluation requires --ckpt")
    if not args.manifest:
        raise EvaluationContractError("research edge evaluation requires --manifest")
    if args.config or args.ternary or args.act8:
        raise EvaluationContractError(
            "research model family/ternary/A8 settings are restored from checkpoint metadata; "
            "--config/--ternary/--act8 are smoke-only"
        )

    device = resolve_device(args.device)
    manifest = load_evaluation_manifest(args.manifest)
    core = load_research_trm_checkpoint(args.ckpt, device=device, weight_identity=args.weights)
    depths = args.n_sup or [int(core.model.N_sup)]
    summaries, examples = run_checkpoint_scaling_grid(
        [core],
        manifest,
        greedy_n_sup=[int(v) for v in depths],
        search_rollouts=[],
        auxiliary_pairs=[None],
        n_latency_runs=int(args.latency_runs),
    )

    frontier_input = [
        {
            "recursion_depth": int(row["inference_setting"]["ordinary_n_sup"]),
            "accuracy": float(row["metrics"]["exact_reference_match"]),
            "cell_acc": float(row["metrics"]["cell_accuracy"]),
            "latency_ms": row["latency_ms"],
        }
        for row in summaries
        if row["latency_ms"] is not None
    ]
    frontier = compute_optimal_frontier(
        frontier_input, cost_key="latency_ms", acc_key="accuracy"
    ) if frontier_input else []

    report = {
        "schema": "spectra.m05.edge_report.v1",
        "result_kind": "research_checkpoint",
        "research_result": True,
        "checkpoint": core.provenance(),
        "evaluation_manifest": {
            "path": str(manifest.path),
            "canonical_sha256": manifest.canonical_sha256,
            "file_sha256": manifest.file_sha256,
            "split": manifest.payload["split"],
            "count": len(manifest.dataset),
        },
        "actual_backend": core.eval_backend,
        "model_size_mb": model_size_mb(core.model),
        "process_rss_mb": process_rss_mb(),
        "settings": summaries,
        "frontier_pareto": frontier,
        "per_example": examples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info(
        "research edge evaluation checkpoint=%s examples=%d settings=%d -> %s",
        core.sha256[:12], len(manifest.dataset), len(summaries), out,
    )


def _smoke(args) -> None:
    if not args.config:
        raise EvaluationContractError("--smoke-random-init requires --config")
    if args.ckpt or args.manifest:
        raise EvaluationContractError("smoke-random-init mode does not accept research checkpoint/manifest")
    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg.device)
    _, val_ds = build_datasets(cfg, n_train=1, n_val=int(args.val_size))
    model = build_trm(cfg, ternary=bool(args.ternary), act8=bool(args.act8)).to(device).eval()
    x = torch.from_numpy(val_ds.inputs).to(device)
    y = torch.from_numpy(val_ds.targets).to(device)
    report = benchmark(
        model, x, y, val_ds.height, val_ds.width, n_latency_runs=int(args.latency_runs)
    )
    report.update({
        "schema": "spectra.m05.edge_smoke.v1",
        "result_kind": "smoke_random_init",
        "research_result": False,
        "warning": "UNTRAINED RANDOM INITIALIZATION; NOT A RESEARCH RESULT",
        "param_count": sum(p.numel() for p in model.parameters()),
        "ternary": bool(args.ternary),
        "act8": bool(args.act8),
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.warning("SMOKE RANDOM INIT ONLY -> %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpoint-backed edge evaluation.")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--weights", choices=["recorded", "raw", "ema"], default="recorded")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-sup", type=int, nargs="*", default=None,
                    help="ordinary forward N_sup values; each changes executed forward compute")
    ap.add_argument("--latency-runs", type=int, default=3)
    ap.add_argument("--out", default="outputs/edge_report.json")

    # Explicit random-init plumbing mode only.
    ap.add_argument("--smoke-random-init", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ternary", action="store_true")
    ap.add_argument("--act8", action="store_true")
    ap.add_argument("--val-size", type=int, default=8)
    args = ap.parse_args()

    if args.smoke_random_init:
        _smoke(args)
    else:
        _research(args)


if __name__ == "__main__":
    main()
