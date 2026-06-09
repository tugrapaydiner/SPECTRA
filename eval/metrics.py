"""Accuracy and distribution metrics for the recursive core.

These cover the Phase 3 diagnostics (per-step accuracy) and the collapse signals
of BLUEPRINT section 11.7 (prediction entropy, most-common-token ratio). All
functions accept torch tensors and return Python scalars / lists so they are easy
to log to JSONL (section 29).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch


def cell_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of individual cells predicted correctly."""
    return (pred == target).float().mean().item()


def board_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of examples whose every cell is correct (exact-match)."""
    return (pred == target).all(dim=1).float().mean().item()


def blank_accuracy(
    pred: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor
) -> float:
    """Accuracy restricted to cells that were blank (0) in the input.

    Isolates learned inference from trivially copying given clues.
    """
    blank = inputs == 0
    if blank.sum() == 0:
        return float("nan")
    return (pred[blank] == target[blank]).float().mean().item()


def per_step_accuracy(
    steps: Sequence[dict[str, Any]], target: torch.Tensor
) -> list[float]:
    """Cell accuracy of each deep-supervision step's decoded answer."""
    return [cell_accuracy(s["logits"].argmax(-1), target) for s in steps]


def prediction_entropy(logits: torch.Tensor) -> float:
    """Mean per-token predictive entropy (nats). Low entropy + wrong = collapse."""
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1)
    return entropy.mean().item()


def most_common_token_ratio(pred: torch.Tensor) -> float:
    """Fraction of predictions taken by the single most frequent token.

    BLUEPRINT section 11.7: a value > 0.90 indicates output collapse.
    """
    flat = pred.reshape(-1)
    counts = torch.bincount(flat)
    return (counts.max().float() / flat.numel()).item()


def unique_predicted_tokens(pred: torch.Tensor) -> int:
    """Number of distinct predicted tokens (a diversity sanity check)."""
    return int(pred.reshape(-1).unique().numel())


def active_density_per_step(steps: Sequence[dict[str, Any]]) -> list[float]:
    """Active-token fraction at each supervision step (requires a router)."""
    return [s["active_density"] for s in steps if "active_density" in s]


def active_token_density(steps: Sequence[dict[str, Any]]) -> float:
    """Mean active-token density across recursion steps (BLUEPRINT section 7.6)."""
    densities = active_density_per_step(steps)
    return sum(densities) / len(densities) if densities else float("nan")
