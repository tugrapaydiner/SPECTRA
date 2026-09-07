"""Reproducible SPECTRA training loop with versioned resumable state."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.logging_utils import MetricLogger
from common.seed import (
    capture_rng_state,
    derive_seed,
    isolated_seed,
    make_seed_streams,
    resolve_device,
    restore_rng_state,
    set_seed,
)
from data.datasets import GridDataset
from model.stability import (
    QuantWarmup,
    load_quant_strength_state,
    quant_strength_state,
)
from model.trm import TRM
from train.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    CheckpointError,
    atomic_torch_save,
    load_checkpoint_payload,
)
from train.collapse_watch import collapse_dashboard, total_grad_norm
from train.ema import EMA
from train.losses import deep_supervision_loss
from train.sampler import StatefulBatchSampler
from train.schedules import warmup_cosine


@dataclass
class TrainConfig:
    """Training state that materially affects optimization/reproducibility."""

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
    eval_batches: int = 4
    ckpt_every: int = 1000
    seed: int = 42
    train_seed: int | None = None
    eval_seed: int | None = None
    device: str = "auto"
    precision: str = "fp32"  # fp32 | fp16_amp
    backend: str = "pytorch_eager"
    deterministic: bool = False

    @classmethod
    def from_config(cls, cfg) -> "TrainConfig":
        d = cls()
        t = dict(cfg.get("train", {}))
        streams = make_seed_streams(int(cfg.get("seed", d.seed)))
        return cls(
            lr=float(t.get("lr", d.lr)),
            weight_decay=float(t.get("weight_decay", d.weight_decay)),
            batch_size=int(t.get("batch_size", d.batch_size)),
            max_steps=int(t.get("max_steps", d.max_steps)),
            lr_warmup_steps=int(
                t.get("lr_warmup_steps", t.get("warmup_steps", d.lr_warmup_steps))
            ),
            quant_warmup_steps=int(t.get("quant_warmup_steps", d.quant_warmup_steps)),
            clip_grad_norm=float(t.get("clip_grad_norm", d.clip_grad_norm)),
            ema_decay=float(t.get("ema_decay", d.ema_decay)),
            lambda_h=float(t.get("lambda_h", d.lambda_h)),
            lambda_improve=float(t.get("lambda_improve", d.lambda_improve)),
            margin=float(t.get("margin", d.margin)),
            log_every=int(t.get("log_every", d.log_every)),
            eval_every=int(t.get("eval_every", d.eval_every)),
            eval_batches=int(t.get("eval_batches", d.eval_batches)),
            ckpt_every=int(t.get("ckpt_every", d.ckpt_every)),
            seed=int(cfg.get("seed", d.seed)),
            train_seed=int(t.get("train_seed", streams.train)),
            eval_seed=int(t.get("eval_seed", streams.eval)),
            device=str(cfg.get("device", d.device)),
            precision=str(t.get("precision", d.precision)),
            backend=str(t.get("backend", d.backend)),
            deterministic=bool(t.get("deterministic", d.deterministic)),
        )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dataset_fingerprint(ds: GridDataset) -> str:
    h = hashlib.sha256()
    for text in (
        str(ds.task),
        str(ds.height),
        str(ds.width),
        str(ds.num_tokens),
        str(ds.pad_token),
    ):
        b = text.encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    for arr in (ds.inputs, ds.targets, ds.input_mask, ds.target_mask):
        contiguous = np.ascontiguousarray(arr)
        h.update(str(contiguous.dtype).encode("ascii"))
        h.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        h.update(contiguous.tobytes())
    for seq in (ds.ids, ds.group_ids):
        for item in seq:
            b = str(item).encode("utf-8")
            h.update(len(b).to_bytes(8, "little"))
            h.update(b)
    return h.hexdigest()


class Trainer:
    """Train a TRM with explicit RNG, precision, sampler, EMA, and checkpoint state."""

    def __init__(
        self,
        model: TRM,
        train_ds: GridDataset,
        val_ds: GridDataset,
        cfg: TrainConfig,
        metric_logger: MetricLogger | None = None,
        *,
        run_config: Mapping[str, Any] | None = None,
    ):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.precision = str(cfg.precision)
        self.backend = str(cfg.backend)
        if self.backend != "pytorch_eager":
            raise ValueError(f"unsupported training backend {self.backend!r}")
        if self.precision not in {"fp32", "fp16_amp"}:
            raise ValueError(
                f"unsupported precision {self.precision!r}; use 'fp32' or 'fp16_amp'"
            )
        if self.precision == "fp16_amp" and self.device.type != "cuda":
            raise ValueError(
                "fp16_amp is implemented only for CUDA; the deterministic CPU reference is fp32"
            )
        if cfg.max_steps <= 0 or cfg.batch_size <= 0:
            raise ValueError("max_steps and batch_size must be positive")
        if cfg.eval_batches <= 0:
            raise ValueError("eval_batches must be positive")
        if len(train_ds) <= 0 or len(val_ds) <= 0:
            raise ValueError("training and validation datasets must be non-empty")

        streams = make_seed_streams(cfg.seed)
        self.train_seed = int(cfg.train_seed if cfg.train_seed is not None else streams.train)
        self.eval_seed = int(cfg.eval_seed if cfg.eval_seed is not None else streams.eval)

        # Model initialization must happen before Trainer construction in entry
        # points; this starts the independent training-randomness stream.
        set_seed(self.train_seed, deterministic=bool(cfg.deterministic))

        self.model = model.to(self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = warmup_cosine(self.opt, cfg.lr_warmup_steps, cfg.max_steps)
        self.ema = EMA(self.model, cfg.ema_decay)
        self.quant_warmup = (
            QuantWarmup(cfg.quant_warmup_steps) if self.model.ternary else None
        )
        self.use_amp = self.precision == "fp16_amp"
        self.scaler = self._make_grad_scaler() if self.use_amp else None

        drop_last = len(train_ds) > cfg.batch_size
        self.train_sampler = StatefulBatchSampler(
            len(train_ds), cfg.batch_size, self.train_seed, drop_last=drop_last
        )
        self.train_loader = DataLoader(
            train_ds, batch_sampler=self.train_sampler, num_workers=0
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
        )

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.height = train_ds.height
        self.width = train_ds.width
        if (val_ds.height, val_ds.width) != (self.height, self.width):
            raise ValueError("train/validation spatial dimensions differ")
        self.logger = metric_logger
        self.step = 0
        self.examples_seen = 0
        self.run_config = _plain(run_config or {})

    def _make_grad_scaler(self):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):  # pragma: no cover
            return torch.cuda.amp.GradScaler(enabled=True)

    def _precision_context(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _infinite_batches(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            yield from self.train_loader

    def _architecture_snapshot(self) -> dict[str, Any]:
        resolved_model = self.run_config.get("model", {}) if self.run_config else {}
        return {
            "class": type(self.model).__name__,
            "dim": int(self.model.dim),
            "num_tokens": int(self.model.num_tokens),
            "seq_len": int(self.model.seq_len),
            "n_layers": len(self.model.blocks),
            "n": int(self.model.n),
            "T": int(self.model.T),
            "N_sup": int(self.model.N_sup),
            "max_grid_size": int(self.model.max_grid_size),
            "ternary": bool(self.model.ternary),
            "act8": bool(self.model.act8),
            "resolved_model_config": _plain(resolved_model),
        }

    def _task_snapshot(self) -> dict[str, Any]:
        resolved_data = self.run_config.get("data", {}) if self.run_config else {}
        return {
            "task": self.train_ds.task,
            "height": int(self.height),
            "width": int(self.width),
            "num_tokens": self.train_ds.num_tokens,
            "pad_token": self.train_ds.pad_token,
            "resolved_task_config": _plain(resolved_data),
            "train_dataset_sha256": _dataset_fingerprint(self.train_ds),
            "validation_dataset_sha256": _dataset_fingerprint(self.val_ds),
        }

    def runtime_settings(self) -> dict[str, Any]:
        first = next(self.model.parameters())
        return {
            "backend": self.backend,
            "device": str(self.device),
            "device_type": self.device.type,
            "precision_mode": self.precision,
            "parameter_dtype": str(first.dtype).replace("torch.", ""),
            "autocast_dtype": "float16" if self.use_amp else None,
            "grad_scaler": bool(self.use_amp),
        }

    def _schedule_signature(self) -> dict[str, Any]:
        return {
            "lr": float(self.cfg.lr),
            "weight_decay": float(self.cfg.weight_decay),
            "batch_size": int(self.cfg.batch_size),
            "max_steps": int(self.cfg.max_steps),
            "lr_warmup_steps": int(self.cfg.lr_warmup_steps),
            "quant_warmup_steps": int(self.cfg.quant_warmup_steps),
            "clip_grad_norm": float(self.cfg.clip_grad_norm),
            "ema_decay": float(self.cfg.ema_decay),
            "lambda_h": float(self.cfg.lambda_h),
            "lambda_improve": float(self.cfg.lambda_improve),
            "margin": float(self.cfg.margin),
            "train_seed": self.train_seed,
            "precision": self.precision,
            "backend": self.backend,
        }

    def _quantization_snapshot(self) -> dict[str, Any]:
        strengths = quant_strength_state(self.model)
        return {
            "enabled": bool(self.model.ternary),
            "warmup_steps": (
                int(self.quant_warmup.warmup_steps)
                if self.quant_warmup is not None else None
            ),
            "strengths": strengths,
        }

    def checkpoint_payload(self) -> dict[str, Any]:
        """Build the complete versioned deterministic-resume payload."""
        return {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "checkpoint_kind": "training_state",
            "resume_capable": True,
            "weights": {
                "raw": {
                    name: tensor.detach().clone()
                    for name, tensor in self.model.state_dict().items()
                },
                "ema": self.ema.state_dict(),
                "training_identity": "raw",
                "evaluation_identity": "ema",
            },
            "architecture": self._architecture_snapshot(),
            "task": self._task_snapshot(),
            "runtime": self.runtime_settings(),
            "resolved_config": _plain(self.run_config),
            "training": {
                "global_step": int(self.step),
                "examples_seen": int(self.examples_seen),
                "optimizer": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "schedule_signature": self._schedule_signature(),
                "quantization": self._quantization_snapshot(),
                "grad_scaler": (
                    self.scaler.state_dict() if self.scaler is not None else None
                ),
            },
            "rng": capture_rng_state(),
            "sampler": self.train_sampler.state_dict(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        atomic_torch_save(self.checkpoint_payload(), path)

    def _optimizer_to_device(self) -> None:
        for state in self.opt.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _validate_resume_contract(self, payload: Mapping[str, Any]) -> None:
        comparisons = {
            "architecture": (payload["architecture"], self._architecture_snapshot()),
            "task": (payload["task"], self._task_snapshot()),
            "schedule": (
                payload["training"]["schedule_signature"], self._schedule_signature()
            ),
        }
        for name, (saved, current) in comparisons.items():
            if saved != current:
                raise CheckpointError(
                    f"{name} mismatch prevents deterministic resume: "
                    f"checkpoint={saved!r}, current={current!r}"
                )
        runtime = payload["runtime"]
        current_runtime = self.runtime_settings()
        for key in ("backend", "device_type", "precision_mode"):
            if runtime.get(key) != current_runtime.get(key):
                raise CheckpointError(
                    f"runtime {key} mismatch: checkpoint={runtime.get(key)!r}, "
                    f"current={current_runtime.get(key)!r}"
                )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Restore a full M04 checkpoint for deterministic resumption."""
        payload = load_checkpoint_payload(
            path, map_location="cpu", allow_legacy=True, require_resume=True
        )
        self._validate_resume_contract(payload)
        if payload["weights"].get("ema") is None:
            raise CheckpointError(
                "resume checkpoint is missing EMA weights; raw substitution is forbidden"
            )

        self.model.load_state_dict(payload["weights"]["raw"], strict=True)
        self.ema.load_state_dict(payload["weights"]["ema"])
        self.opt.load_state_dict(payload["training"]["optimizer"])
        self._optimizer_to_device()
        self.scheduler.load_state_dict(payload["training"]["scheduler"])

        quant = payload["training"]["quantization"]
        if bool(quant.get("enabled")) != bool(self.model.ternary):
            raise CheckpointError("quantization enabled-state mismatch")
        if self.model.ternary:
            if int(quant["warmup_steps"]) != int(self.quant_warmup.warmup_steps):
                raise CheckpointError("quantization warmup schedule mismatch")
            load_quant_strength_state(self.model, quant["strengths"])
        elif quant.get("strengths"):
            raise CheckpointError("non-ternary checkpoint contains quantization strengths")

        scaler_state = payload["training"].get("grad_scaler")
        if self.scaler is None:
            if scaler_state is not None:
                raise CheckpointError("FP32 run cannot load FP16 GradScaler state")
        else:
            if scaler_state is None:
                raise CheckpointError("FP16 AMP checkpoint is missing GradScaler state")
            self.scaler.load_state_dict(scaler_state)

        self.step = int(payload["training"]["global_step"])
        self.examples_seen = int(payload["training"].get("examples_seen", 0))
        if not 0 <= self.step <= self.cfg.max_steps:
            raise CheckpointError("checkpoint global_step is outside current schedule")
        self.train_sampler.load_state_dict(payload["sampler"])
        restore_rng_state(payload["rng"])
        return payload

    @staticmethod
    def _all_gradients_finite(model: torch.nn.Module) -> bool:
        return all(
            bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
            if p.grad is not None
        )

    @staticmethod
    def _all_parameters_finite(model: torch.nn.Module) -> bool:
        return all(bool(torch.isfinite(p).all()) for p in model.parameters())

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """One optimization step with hard finite loss/gradient checks."""
        self.model.train()
        rho = 0.0
        if self.quant_warmup is not None:
            rho = self.quant_warmup.apply(self.model, self.step)

        self.opt.zero_grad(set_to_none=True)
        with self._precision_context():
            logits, steps = self.model(x, height=self.height, width=self.width)
            loss = deep_supervision_loss(
                steps, y, self.cfg.lambda_h, self.cfg.lambda_improve, self.cfg.margin
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"non-finite training loss at step {self.step}: {float(loss.detach())}"
            )

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
        else:
            loss.backward()

        if not self._all_gradients_finite(self.model):
            raise FloatingPointError(f"non-finite gradient at step {self.step}")
        grad_norm = total_grad_norm(self.model)
        clipped_return = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.clip_grad_norm, error_if_nonfinite=True
        )
        clipped_grad_norm = total_grad_norm(self.model)

        if self.scaler is not None:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        if not self._all_parameters_finite(self.model):
            raise FloatingPointError(f"non-finite model parameter after step {self.step}")

        self.scheduler.step()
        self.ema.update(self.model)
        pred = logits.detach().argmax(-1)
        train_cell_acc = float((pred == y).float().mean())

        return {
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "grad_norm_clipped": float(clipped_grad_norm),
            "clip_return_norm": float(clipped_return),
            "lr": float(self.scheduler.get_last_lr()[0]),
            "quant_strength": float(rho),
            "train_cell_acc": train_cell_acc,
            "batch_examples": float(x.shape[0]),
            "examples_seen": float(self.examples_seen),
        }

    @torch.no_grad()
    def evaluate(self, use_ema: bool = True) -> dict[str, Any]:
        """Evaluate a bounded number of deterministic validation batches."""
        self.model.eval()
        eval_seed = derive_seed(self.eval_seed, f"validation-step-{self.step}")
        cell_correct = 0
        cell_total = 0
        board_correct = 0
        board_total = 0
        step_correct: list[int] | None = None
        dashboard_rows: list[dict[str, Any]] = []
        batches_used = 0

        ctx = self.ema.average_parameters(self.model) if use_ema else nullcontext()
        with isolated_seed(eval_seed), ctx:
            for batch_index, (x, y) in enumerate(self.val_loader):
                if batch_index >= self.cfg.eval_batches:
                    break
                x, y = x.to(self.device), y.to(self.device)
                with self._precision_context():
                    logits, steps = self.model(x, height=self.height, width=self.width)
                pred = logits.argmax(-1)
                matches = pred == y
                cell_correct += int(matches.sum().item())
                cell_total += int(matches.numel())
                board_correct += int(matches.all(dim=1).sum().item())
                board_total += int(y.shape[0])

                if step_correct is None:
                    step_correct = [0 for _ in steps]
                for i, state in enumerate(steps):
                    step_correct[i] += int(
                        (state["logits"].argmax(-1) == y).sum().item()
                    )
                batch_acc = float(matches.float().mean())
                dashboard_rows.append(collapse_dashboard(steps, batch_acc, self.model))
                batches_used += 1

        if batches_used == 0 or cell_total == 0 or board_total == 0:
            raise RuntimeError("validation produced no batches")

        dashboard: dict[str, Any] = {}
        for key in dashboard_rows[0]:
            vals = [row[key] for row in dashboard_rows]
            if all(isinstance(v, bool) for v in vals):
                dashboard[key] = any(vals)
            elif all(isinstance(v, (int, float)) for v in vals):
                dashboard[key] = float(sum(float(v) for v in vals) / len(vals))
            else:
                dashboard[key] = vals[-1]

        return {
            "cell_acc": cell_correct / cell_total,
            "board_acc": board_correct / board_total,
            "per_step_acc": [correct / cell_total for correct in (step_correct or [])],
            "dashboard": dashboard,
            "eval_batches": batches_used,
            "eval_examples": board_total,
            "weight_identity": "ema" if use_ema else "raw",
        }

    def fit(
        self,
        *,
        stop_at_step: int | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Train to the original schedule or a bounded interruption point.

        ``stop_at_step`` never changes scheduler ``max_steps``; the interrupted
        and resumed run therefore follows the same original LR/quant schedule.
        """
        target = self.cfg.max_steps if stop_at_step is None else int(stop_at_step)
        if not self.step <= target <= self.cfg.max_steps:
            raise ValueError(f"stop_at_step must be in [{self.step}, {self.cfg.max_steps}]")

        history: list[dict[str, Any]] = []
        batches = self._infinite_batches()
        while self.step < target:
            x, y = next(batches)
            self.step += 1
            self.examples_seen += int(x.shape[0])
            x, y = x.to(self.device), y.to(self.device)
            metrics = self.train_step(x, y)

            if self.logger and self.step % self.cfg.log_every == 0:
                self.logger.log(self.step, **metrics)

            if self.step % self.cfg.eval_every == 0:
                ev = self.evaluate(use_ema=True)
                row = {
                    "step": self.step,
                    "cell_acc": ev["cell_acc"],
                    "board_acc": ev["board_acc"],
                    "eval_batches": ev["eval_batches"],
                    "eval_examples": ev["eval_examples"],
                    "weight_identity": ev["weight_identity"],
                    **{f"dash_{k}": v for k, v in ev["dashboard"].items()},
                }
                history.append(row)
                if self.logger:
                    self.logger.log(self.step, **row)

            if (
                checkpoint_path is not None
                and self.cfg.ckpt_every > 0
                and self.step % self.cfg.ckpt_every == 0
            ):
                self.save_checkpoint(checkpoint_path)

        final = self.evaluate(use_ema=True)
        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)
        return {"history": history, "final": final}
