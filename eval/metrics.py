"""Task-aware evaluation metrics for SPECTRA.

Milestone 03 separates three different notions that were previously easy to
conflate:

* exact reference match: candidate equals the stored target cell-for-cell;
* semantic validity: candidate solves the symbolic task even if an alternative
  valid solution exists;
* inference-cell accuracy: accuracy on cells that actually had to be inferred
  (Sudoku blanks), plus padding-aware content accuracy for padded local tasks.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from model.verifier import maze_correct, sudoku_correct


def cell_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred == target).float().mean().item()


def board_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Legacy name for exact reference match."""
    return exact_reference_match(pred, target)


def exact_reference_match(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of examples exactly equal to the retained reference target."""
    if pred.shape != target.shape or pred.ndim < 2:
        raise ValueError("pred/target must have identical batch-first shapes")
    return (pred == target).reshape(pred.shape[0], -1).all(dim=1).float().mean().item()


def masked_cell_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Cell accuracy on an explicit boolean mask; NaN when the mask is empty."""
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError("pred/target/mask shapes must match")
    mask = mask.bool()
    if int(mask.sum().item()) == 0:
        return float("nan")
    return (pred[mask] == target[mask]).float().mean().item()


def blank_accuracy(pred: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor) -> float:
    """Sudoku inference accuracy restricted to cells blank in the input."""
    return masked_cell_accuracy(pred, target, inputs == 0)


def semantic_validity(
    task: str,
    inputs: torch.Tensor,
    pred: torch.Tensor,
    *,
    box: int | None = None,
    height: int | None = None,
    width: int | None = None,
    require_optimal: bool = True,
) -> float:
    """Mean strict symbolic success for tasks with an implemented exact validator."""
    if task == "sudoku":
        if box is None:
            raise ValueError("Sudoku semantic validity requires box")
        return sudoku_correct(inputs, pred, box).float().mean().item()
    if task == "maze":
        if height is None or width is None:
            raise ValueError("Maze semantic validity requires height and width")
        return maze_correct(
            inputs, pred, height, width, require_optimal=require_optimal
        ).float().mean().item()
    raise ValueError(
        f"No strict semantic-validity metric is claimed for task {task!r}; "
        "use exact reference match for this local synthetic task"
    )


def task_metrics(
    task: str,
    inputs: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    box: int | None = None,
    height: int | None = None,
    width: int | None = None,
    pad_token: int | None = None,
    require_optimal: bool = True,
) -> dict[str, float]:
    """Primary metrics for one retained task contract.

    ARC-style and BabyAI-style are local synthetic variants in this repository;
    they intentionally do not receive an "official benchmark" semantic score.
    """
    out = {
        "exact_reference_match": exact_reference_match(pred, target),
        "cell_accuracy": cell_accuracy(pred, target),
    }
    if task == "sudoku":
        if box is None:
            raise ValueError("Sudoku metrics require box")
        out["semantic_validity"] = semantic_validity(task, inputs, pred, box=box)
        out["blank_cell_accuracy"] = blank_accuracy(pred, target, inputs)
    elif task == "maze":
        if height is None or width is None:
            raise ValueError("Maze metrics require height/width")
        out["semantic_validity"] = semantic_validity(
            task, inputs, pred, height=height, width=width,
            require_optimal=require_optimal,
        )
    elif task == "arc":
        if pad_token is None:
            raise ValueError("ARC-style metrics require pad_token")
        out["content_cell_accuracy"] = masked_cell_accuracy(
            pred, target, target != pad_token
        )
    return out


def per_step_accuracy(steps: Sequence[dict[str, Any]], target: torch.Tensor) -> list[float]:
    return [cell_accuracy(s["logits"].argmax(-1), target) for s in steps]


def prediction_entropy(logits: torch.Tensor) -> float:
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return (-(p * logp).sum(dim=-1)).mean().item()


def most_common_token_ratio(pred: torch.Tensor) -> float:
    flat = pred.reshape(-1)
    counts = torch.bincount(flat)
    return (counts.max().float() / flat.numel()).item()


def unique_predicted_tokens(pred: torch.Tensor) -> int:
    return int(pred.reshape(-1).unique().numel())


def active_density_per_step(steps: Sequence[dict[str, Any]]) -> list[float]:
    return [s["active_density"] for s in steps if "active_density" in s]


def active_token_density(steps: Sequence[dict[str, Any]]) -> float:
    densities = active_density_per_step(steps)
    return sum(densities) / len(densities) if densities else float("nan")
