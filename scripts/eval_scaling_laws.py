"""Iso-joule compute-optimal scaling laws (BLUEPRINT sections 5.5, 27.0).

    python scripts/eval_scaling_laws.py --config config/sudoku.yaml \
        --params 1000000 7000000 14000000 --depths 1 2 4 8 --rollouts 0 8 32 \
        --out outputs/scaling.csv

Sweeps parameter budget x recursion depth x Latent-MCTS rollouts, logs measured
RAPL microjoules per inference (on the Linux x86 eval box), and writes a DataFrame
ready to plot Accuracy vs. Measured Joules -- the experiment that actually tests
whether learned test-time search substitutes for parameter count.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import torch  # noqa: E402

from common import get_logger, load_config, resolve_device  # noqa: E402
from eval.cpufreq import pinned_frequency  # noqa: E402
from eval.edge_energy import rapl_available  # noqa: E402
from eval.latent_mcts import LatentNativeMCTS  # noqa: E402
from eval.scaling import run_null_hypothesis, run_scaling_grid, to_dataframe  # noqa: E402
from model.energy import LatentEnergyVerifier  # noqa: E402
from model.latent_action import LatentActionCodebook  # noqa: E402
from scripts._common import build_datasets  # noqa: E402

log = get_logger("eval_scaling_laws")


def main() -> None:
    ap = argparse.ArgumentParser(description="Iso-joule scaling-law sweep.")
    ap.add_argument("--config", default="config/sudoku.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--params", type=int, nargs="*", default=[1_000_000, 7_000_000, 14_000_000])
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--rollouts", type=int, nargs="*", default=[0, 8, 32])
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--out", default="outputs/scaling.csv")
    ap.add_argument("--null-hypothesis", action="store_true",
                    help="also evaluate the iso-FLOP dense baseline (spectra vs dense)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    device = resolve_device(cfg.device)
    _, val = build_datasets(cfg, n_train=1, n_val=args.val_size)
    x = torch.from_numpy(val.inputs).to(device)
    y = torch.from_numpy(val.targets).to(device)
    h, w = val.height, val.width
    num_tokens, seq_len = int(cfg.data.num_tokens), int(cfg.data.seq_len)
    max_grid = int(cfg.data.get("max_grid_size", 32))

    def mcts_factory(model, rollouts):
        verifier = LatentEnergyVerifier(num_tokens, dim=model.dim, max_grid_size=max_grid).to(device)
        codebook = LatentActionCodebook(model.dim, n_actions=3).to(device)
        return LatentNativeMCTS(model, verifier, codebook, h, w, n_rollouts=rollouts)

    log.info("RAPL available: %s (microjoules logged only on Linux x86)", rapl_available())
    # Hold the CPU at a fixed operating point so AVX2 thermal throttling does not
    # corrupt the iso-joule curve (no-op + reported off Linux / without root).
    with pinned_frequency(disable_turbo=True) as freq:
        log.info("CPU frequency pinned: %s (%s)", freq["pinned"], freq.get("reason", ""))
        factory = mcts_factory if any(r > 0 for r in args.rollouts) else None
        if args.null_hypothesis:
            # Spectra (small + recursion + search) vs dense (more params) at iso-FLOP.
            rows = run_null_hypothesis(
                args.params[0], args.depths, args.rollouts, x, y, h, w, num_tokens, seq_len,
                device=device, max_grid_size=max_grid, mcts_factory=factory,
            )
        else:
            rows = run_scaling_grid(
                args.params, args.depths, args.rollouts, x, y, h, w, num_tokens, seq_len,
                device=device, max_grid_size=max_grid, mcts_factory=factory,
            )
    df = to_dataframe(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("Scaling grid (%d points) -> %s\n%s", len(df), args.out, df.to_string(index=False))


if __name__ == "__main__":
    main()
