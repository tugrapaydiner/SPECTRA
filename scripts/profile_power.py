"""Physical power/energy profiling via RAPL (BLUEPRINT section 22).

    python scripts/profile_power.py --config config/sudoku.yaml --ckpt outputs/teacher/teacher.pt

Follows the section 22.4 protocol: warm up, record idle baseline, run N inference
problems, report net joules/problem. Requires Linux RAPL; prints an explicit
notice and exits cleanly on hosts without it (e.g. Windows).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import torch  # noqa: E402

from common import get_logger, load_config, resolve_device  # noqa: E402
from eval.edge_energy import idle_power_watts, measure_energy_joules, rapl_available  # noqa: E402
from scripts._common import build_datasets, build_trm  # noqa: E402

log = get_logger("profile_power")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAPL energy profiling.")
    ap.add_argument("--config", default="config/sudoku.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ternary", action="store_true")
    ap.add_argument("--act8", action="store_true")
    ap.add_argument("--problems", type=int, default=200)
    ap.add_argument("--idle-seconds", type=float, default=60.0)
    args = ap.parse_args()

    if not rapl_available():
        log.warning(
            "RAPL counters not available on this host (%s). Energy profiling "
            "requires Linux Intel/AMD RAPL; run on the legacy x86 eval box.", sys.platform
        )
        return

    cfg = load_config(args.config, args.override)
    device = resolve_device(cfg.device)
    _, val_ds = build_datasets(cfg, n_train=1, n_val=1)
    model = build_trm(cfg, ternary=args.ternary, act8=args.act8).to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    x = torch.from_numpy(val_ds.inputs[:1]).to(device)
    h, w = val_ds.height, val_ds.width

    log.info("Warming up..."); [model(x, height=h, width=w) for _ in range(10)]
    log.info("Recording idle baseline for %.0fs...", args.idle_seconds)
    idle_w = idle_power_watts(args.idle_seconds)

    with torch.no_grad():
        result = measure_energy_joules(lambda: model(x, height=h, width=w), n_runs=args.problems)
    idle_energy = (idle_w or 0.0) * result["wall_seconds"]
    net = max(0.0, result["energy_joules"] - idle_energy)
    log.info("idle=%.2fW  total=%.2fJ  net=%.2fJ  joules/problem=%.4f",
             idle_w or float("nan"), result["energy_joules"], net, net / args.problems)


if __name__ == "__main__":
    main()
