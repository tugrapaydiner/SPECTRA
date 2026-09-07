"""M04 reproducibility evidence: initialization and interrupted-resume equivalence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import load_config
from scripts._common import build_seeded_training_components
from train.checkpoint import load_checkpoint_payload
from train.trainer import Trainer


ATOL = 1e-7
RTOL = 1e-6


def components(config: str, train_size: int, val_size: int):
    cfg = load_config(config)
    model, train_ds, val_ds, tcfg, streams = build_seeded_training_components(
        cfg, train_size, val_size, ternary=False, act8=False
    )
    return cfg, model, train_ds, val_ds, tcfg, streams


def max_abs_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    return max(
        (a[k].detach().cpu().float() - b[k].detach().cpu().float()).abs().max().item()
        for k in a
    )


def all_close(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return set(a) == set(b) and all(
        torch.allclose(a[k], b[k], atol=ATOL, rtol=RTOL) for k in a
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/m04_cpu_reference.yaml")
    ap.add_argument("--train-size", type=int, default=24)
    ap.add_argument("--val-size", type=int, default=12)
    ap.add_argument("--out", default=".m04/repro_audit.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    resume_path = out.parent / "resume_state.pt"

    cfg0, init_a, train_a, val_a, _, streams = components(
        args.config, args.train_size, args.val_size
    )
    _, init_b, train_b, val_b, _, _ = components(
        args.config, args.train_size, args.val_size
    )
    init_exact = all(
        torch.equal(init_a.state_dict()[k], init_b.state_dict()[k])
        for k in init_a.state_dict()
    )
    data_exact = (
        torch.equal(torch.from_numpy(train_a.inputs), torch.from_numpy(train_b.inputs))
        and torch.equal(torch.from_numpy(train_a.targets), torch.from_numpy(train_b.targets))
        and torch.equal(torch.from_numpy(val_a.inputs), torch.from_numpy(val_b.inputs))
        and torch.equal(torch.from_numpy(val_a.targets), torch.from_numpy(val_b.targets))
    )

    cfg1, model1, train1, val1, tcfg1, _ = components(
        args.config, args.train_size, args.val_size
    )
    full = Trainer(model1, train1, val1, tcfg1, run_config=cfg1)
    full_result = full.fit()
    full_raw = {k: v.detach().clone() for k, v in full.model.state_dict().items()}
    full_ema = full.ema.state_dict()

    interrupt_at = tcfg1.max_steps // 2
    cfg2, model2, train2, val2, tcfg2, _ = components(
        args.config, args.train_size, args.val_size
    )
    interrupted = Trainer(model2, train2, val2, tcfg2, run_config=cfg2)
    interrupted.fit(stop_at_step=interrupt_at, checkpoint_path=resume_path)

    cfg3, model3, train3, val3, tcfg3, _ = components(
        args.config, args.train_size, args.val_size
    )
    resumed = Trainer(model3, train3, val3, tcfg3, run_config=cfg3)
    resumed.load_checkpoint(resume_path)
    resumed_result = resumed.fit()
    resumed_raw = resumed.model.state_dict()
    resumed_ema = resumed.ema.state_dict()

    payload = load_checkpoint_payload(resume_path, require_resume=True)
    raw_diff = max_abs_diff(full_raw, resumed_raw)
    ema_diff = max_abs_diff(full_ema, resumed_ema)
    raw_ok = all_close(full_raw, resumed_raw)
    ema_ok = all_close(full_ema, resumed_ema)
    sampler_ok = (
        resumed.train_sampler.state_dict()["epoch"] == full.train_sampler.state_dict()["epoch"]
        and resumed.train_sampler.state_dict()["position"] == full.train_sampler.state_dict()["position"]
        and torch.equal(
            resumed.train_sampler.state_dict()["order"],
            full.train_sampler.state_dict()["order"],
        )
    )

    evidence = {
        "schema": "spectra-m04-repro-audit-v1",
        "config": args.config,
        "seed_streams": streams.as_dict(),
        "precision": full.runtime_settings(),
        "schedule_max_steps": tcfg1.max_steps,
        "interrupt_at_step": interrupt_at,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "same_seed_initial_weights_bitwise_equal": init_exact,
        "same_seed_data_bitwise_equal": data_exact,
        "resume_raw_max_abs_diff": raw_diff,
        "resume_ema_max_abs_diff": ema_diff,
        "resume_raw_within_tolerance": raw_ok,
        "resume_ema_within_tolerance": ema_ok,
        "resume_sampler_state_equal": sampler_ok,
        "full_final_cell_acc": full_result["final"]["cell_acc"],
        "resumed_final_cell_acc": resumed_result["final"]["cell_acc"],
        "full_final_board_acc": full_result["final"]["board_acc"],
        "resumed_final_board_acc": resumed_result["final"]["board_acc"],
        "checkpoint": {
            "format": payload["format"],
            "version": payload["version"],
            "resume_capable": payload["resume_capable"],
            "training_identity": payload["weights"]["training_identity"],
            "evaluation_identity": payload["weights"]["evaluation_identity"],
            "global_step": payload["training"]["global_step"],
            "has_optimizer": bool(payload["training"]["optimizer"]),
            "has_scheduler": bool(payload["training"]["scheduler"]),
            "has_rng": payload["rng"] is not None,
            "has_sampler": payload["sampler"] is not None,
            "runtime": payload["runtime"],
        },
    }
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))

    if not (init_exact and data_exact and raw_ok and ema_ok and sampler_ok):
        raise SystemExit("M04 reproducibility audit failed")


if __name__ == "__main__":
    main()
