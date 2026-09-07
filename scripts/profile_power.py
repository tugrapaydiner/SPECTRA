"""Physical CPU-package energy profiling with M13 measurement semantics.

Idle package power is reported separately; it is not silently subtracted from a
short inference window. Invalid/partial counters are unavailable, never zero.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from common import get_logger, load_config, resolve_device  # noqa: E402
from eval.edge_energy import idle_power_watts, measure_energy_record  # noqa: E402
from scripts._common import build_datasets, build_trm  # noqa: E402

log = get_logger("profile_power")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validated CPU-package RAPL energy profiling.")
    ap.add_argument("--config", default="config/sudoku.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ternary", action="store_true")
    ap.add_argument("--act8", action="store_true")
    ap.add_argument("--problems", type=int, default=64)
    ap.add_argument("--idle-seconds", type=float, default=5.0)
    args = ap.parse_args()
    if args.problems <= 0: raise ValueError("--problems must be positive")

    cfg = load_config(args.config, args.override); device = resolve_device(cfg.device)
    _, val_ds = build_datasets(cfg, n_train=1, n_val=max(16, args.problems))
    model = build_trm(cfg, ternary=args.ternary, act8=args.act8).to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval(); h, w = val_ds.height, val_ds.width
    xs = torch.from_numpy(val_ds.inputs[:min(args.problems, len(val_ds))]).to(device)

    with torch.inference_mode():
        for i in range(min(4, xs.shape[0])):
            model(xs[i:i+1], height=h, width=w)
    idle_w = idle_power_watts(args.idle_seconds)

    def representative_pass():
        with torch.inference_mode():
            for i in range(xs.shape[0]):
                model(xs[i:i+1], height=h, width=w)

    result = measure_energy_record(representative_pass, n_runs=1)
    if not result["available"]:
        log.warning("CPU-package energy unavailable: %s", result["failure_reason"])
        log.info("idle_package_power_watts=%s (reported separately)", idle_w)
        return
    joules = float(result["energy_joules"])
    log.info("package_total=%.6fJ  solves=%d  package_joules/solve=%.6f  idle_package_power_watts=%s",
             joules, xs.shape[0], joules / xs.shape[0], idle_w)
    log.info("scope=%s domains=%s", result["scope"], result["package_domain_ids"])


if __name__ == "__main__": main()
