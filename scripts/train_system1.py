"""Distill the System 1 student from a trained System 2 teacher (Phase 10).

    python scripts/train_system1.py --config config/sudoku.yaml --teacher outputs/teacher/teacher.pt

Optionally runs a round of the self-play flywheel to compile fresh System 2
solutions into System 1 training targets.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import get_logger, load_config, resolve_device, set_seed  # noqa: E402
from data.datasets import build_sudoku_arrays  # noqa: E402
from model.system1_student import System1Student  # noqa: E402
from scripts._common import build_trm  # noqa: E402
from train.distill import system1_distillation_loss  # noqa: E402

log = get_logger("train_system1")


def main() -> None:
    ap = argparse.ArgumentParser(description="Distill System 1 from the teacher.")
    ap.add_argument("--config", default="config/sudoku.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--teacher", required=True, help="teacher checkpoint (.pt)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default="outputs/system1")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    if cfg.task != "sudoku":
        raise SystemExit("train_system1 currently scripts the Sudoku task only")
    box = int(cfg.data.get("box", 3))
    h = w = int(cfg.data.height)

    teacher = build_trm(cfg, ternary=False, act8=False).to(device)
    teacher.load_state_dict(torch.load(args.teacher, map_location=device)["model"])
    teacher.eval()

    # Mixed-difficulty data so the confidence head is calibrated.
    rng = np.random.default_rng(cfg.seed)
    xs, ys = [], []
    n_cells = box ** 4
    for frac in (0.25, 0.4, 0.55, 0.7, 0.85):
        a, b, _, _ = build_sudoku_arrays(box, 256, int(frac * n_cells), rng, True, True)
        xs.append(a)
        ys.append(b)
    tx = torch.from_numpy(np.concatenate(xs)).to(device)
    ty = torch.from_numpy(np.concatenate(ys)).to(device)
    with torch.no_grad():
        teacher_logits = teacher(tx, height=h, width=w)[0]

    student = System1Student(
        dim=cfg.model.dim, num_tokens=cfg.data.num_tokens, seq_len=cfg.data.seq_len,
        max_grid_size=int(cfg.data.get("max_grid_size", 32)),
    ).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=float(cfg.train.lr))
    student.train()
    for step in range(args.steps):
        idx = torch.randint(0, tx.shape[0], (int(cfg.train.batch_size),), device=device)
        logits, conf = student(tx[idx], h, w)
        loss, comp = system1_distillation_loss(logits, conf, teacher_logits[idx], ty[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            log.info("step %d loss=%.3f (ce=%.3f kl=%.3f conf=%.3f)",
                     step, comp["total"], comp["ce"], comp["kl"], comp["conf_bce"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": student.state_dict()}, out_dir / "system1.pt")
    log.info("Saved System 1 to %s", out_dir / "system1.pt")


if __name__ == "__main__":
    main()
