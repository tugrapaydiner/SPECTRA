"""The central training loop that wires the Stability Shield together.

A :class:`Trainer` ties together the pieces required for stable ternary/recursive
training (BLUEPRINT section 11):

  * AdamW + linear-warmup/cosine LR schedule,
  * gradient clipping (section 11.4),
  * EMA of parameters for validation/checkpointing (section 11.3),
  * quantization warmup for ternary models (section 11.6),
  * a collapse dashboard computed each evaluation (section 11.7).

It is deliberately small and task-agnostic: it consumes any ``GridDataset`` and a
``TRM`` and reports per-step metrics + collapse signals.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader

from common.logging_utils import MetricLogger
from common.seed import resolve_device, set_seed
from data.datasets import GridDataset
from eval.metrics import board_accuracy, cell_accuracy, per_step_accuracy
from model.stability import QuantWarmup
from model.trm import TRM
from train.collapse_watch import collapse_dashboard, total_grad_norm
from train.ema import EMA
from train.losses import deep_supervision_loss
from train.schedules import warmup_cosine


@dataclass
class TrainConfig:
    """Hyperparameters for :class:`Trainer` (defaults mirror config/base.yaml)."""

    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 128
    max_steps: int = 2000
    lr_warmup_steps: int = 100
    quant_warmup_steps: int = 200
    clip_grad_norm: float = 1.0
    ema_decay: float = 0.999
    lambda_h: float = 0.5
    lambda_improve: float = 0.1
    margin: float = 0.01
    log_every: int = 50
    eval_every: int = 200
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_config(cls, cfg) -> "TrainConfig":
        """Build from a loaded YAML config (config/base.yaml), not hardcoded magic.

        Maps ``cfg.train.*`` / ``cfg.seed`` / ``cfg.device`` onto the dataclass,
        falling back to the field defaults for anything the config omits. This is
        the wiring the blueprint implies -- so schedules/lambdas live in the config,
        not as literals buried in the trainer.
        """
        d = cls()
        t = dict(cfg.get("train", {}))
        return cls(
            lr=float(t.get("lr", d.lr)),
            weight_decay=float(t.get("weight_decay", d.weight_decay)),
            batch_size=int(t.get("batch_size", d.batch_size)),
            max_steps=int(t.get("max_steps", d.max_steps)),
            lr_warmup_steps=int(t.get("lr_warmup_steps", t.get("warmup_steps", d.lr_warmup_steps))),
            quant_warmup_steps=int(t.get("quant_warmup_steps", d.quant_warmup_steps)),
            clip_grad_norm=float(t.get("clip_grad_norm", d.clip_grad_norm)),
            ema_decay=float(t.get("ema_decay", d.ema_decay)),
            lambda_h=float(t.get("lambda_h", d.lambda_h)),
            lambda_improve=float(t.get("lambda_improve", d.lambda_improve)),
            margin=float(t.get("margin", d.margin)),
            log_every=int(t.get("log_every", d.log_every)),
            eval_every=int(t.get("eval_every", d.eval_every)),
            seed=int(cfg.get("seed", d.seed)),
            device=str(cfg.get("device", d.device)),
        )


class Trainer:
    """Trains a :class:`TRM` on a ``GridDataset`` with the Stability Shield."""

    def __init__(
        self,
        model: TRM,
        train_ds: GridDataset,
        val_ds: GridDataset,
        cfg: TrainConfig,
        metric_logger: MetricLogger | None = None,
    ):
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)

        self.opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = warmup_cosine(self.opt, cfg.lr_warmup_steps, cfg.max_steps)
        self.ema = EMA(model, cfg.ema_decay)
        self.quant_warmup = QuantWarmup(cfg.quant_warmup_steps) if model.ternary else None

        # drop_last only when there is more than one full batch, otherwise a
        # small dataset would yield zero batches and stall the infinite iterator.
        drop_last = len(train_ds) > cfg.batch_size
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=drop_last
        )
        self.val_ds = val_ds
        self.height = train_ds.height
        self.width = train_ds.width
        self.logger = metric_logger
        self.step = 0

    def _infinite_batches(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            yield from self.train_loader

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """One optimisation step; returns scalar training metrics."""
        self.model.train()
        if self.quant_warmup is not None:
            self.quant_warmup.apply(self.model, self.step)

        _, steps = self.model(x, height=self.height, width=self.width)
        loss = deep_supervision_loss(
            steps, y, self.cfg.lambda_h, self.cfg.lambda_improve, self.cfg.margin
        )
        self.opt.zero_grad()
        loss.backward()
        grad_norm = total_grad_norm(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
        self.opt.step()
        self.scheduler.step()
        self.ema.update(self.model)

        return {
            "loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def evaluate(self, use_ema: bool = True) -> dict[str, Any]:
        """Evaluate on the full validation set (under EMA weights by default)."""
        self.model.eval()
        x = torch.from_numpy(self.val_ds.inputs).to(self.device)
        y = torch.from_numpy(self.val_ds.targets).to(self.device)

        ctx = self.ema.average_parameters(self.model) if use_ema else nullcontext()
        with ctx:
            logits, steps = self.model(x, height=self.height, width=self.width)

        pred = logits.argmax(-1)
        accs = per_step_accuracy(steps, y)
        dash = collapse_dashboard(steps, accs[-1], self.model)
        return {
            "cell_acc": cell_accuracy(pred, y),
            "board_acc": board_accuracy(pred, y),
            "per_step_acc": accs,
            "dashboard": dash,
        }

    def fit(self) -> dict[str, Any]:
        """Run ``max_steps`` of training, evaluating periodically."""
        history: list[dict[str, Any]] = []
        batches = self._infinite_batches()

        for self.step in range(1, self.cfg.max_steps + 1):
            x, y = next(batches)
            x, y = x.to(self.device), y.to(self.device)
            metrics = self.train_step(x, y)

            if self.logger and self.step % self.cfg.log_every == 0:
                self.logger.log(self.step, **metrics)

            if self.step % self.cfg.eval_every == 0:
                ev = self.evaluate()
                row = {
                    "step": self.step,
                    "cell_acc": ev["cell_acc"],
                    "board_acc": ev["board_acc"],
                    **{f"dash_{k}": v for k, v in ev["dashboard"].items()},
                }
                history.append(row)
                if self.logger:
                    self.logger.log(self.step, **row)

        return {"history": history, "final": self.evaluate()}
