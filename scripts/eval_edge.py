"""Edge benchmark: accuracy + latency + memory (+ RAPL joules) and a frontier sweep.

    python scripts/eval_edge.py --config config/sudoku.yaml --ckpt outputs/teacher/teacher.pt

Run it wrapped in cgroups (see eval/edge_energy.cgroup_command) on the Linux eval
box to measure under the Catastrophic/Standard Edge profiles.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import torch  # noqa: E402

from common import get_logger, load_config, resolve_device  # noqa: E402
from eval.benchmarks import benchmark, sweep_recursion_depth  # noqa: E402
from eval.edge_energy import rapl_available  # noqa: E402
from eval.reports import compute_optimal_frontier, save_report  # noqa: E402
from scripts._common import build_datasets, build_trm  # noqa: E402

log = get_logger("eval_edge")


def main() -> None:
    ap = argparse.ArgumentParser(description="Edge benchmark + frontier sweep.")
    ap.add_argument("--config", default="config/sudoku.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--ckpt", default=None, help="model checkpoint (.pt); random init if omitted")
    ap.add_argument("--ternary", action="store_true")
    ap.add_argument("--act8", action="store_true")
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--out", default="outputs/edge_report.json")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    device = resolve_device(cfg.device)
    _, val_ds = build_datasets(cfg, n_train=1, n_val=args.val_size)

    model = build_trm(cfg, ternary=args.ternary, act8=args.act8).to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    x = torch.from_numpy(val_ds.inputs).to(device)
    y = torch.from_numpy(val_ds.targets).to(device)
    h, w = val_ds.height, val_ds.width

    log.info("RAPL available: %s (joules measured only on Linux x86)", rapl_available())
    report = benchmark(model, x, y, h, w)
    log.info(
        "board_acc=%.3f latency=%.2fms peak_ram=%.0fMB size=%.3fMB joules=%s",
        report["accuracy"], report["latency_ms"], report["peak_ram_mb"],
        report["model_size_mb"], report.get("joules_per_problem"),
    )

    frontier_points = sweep_recursion_depth(model, x, y, h, w, args.depths)
    frontier = compute_optimal_frontier(frontier_points, cost_key="latency_ms", acc_key="accuracy")
    log.info("compute-optimal frontier (depth, acc, latency_ms):")
    for p in frontier:
        log.info("  depth=%d acc=%.3f latency=%.2fms", p["recursion_depth"], p["accuracy"], p["latency_ms"])

    report["frontier_points"] = frontier_points
    report["frontier_pareto"] = frontier
    save_report(report, args.out)
    log.info("Saved edge report to %s", args.out)


if __name__ == "__main__":
    main()
