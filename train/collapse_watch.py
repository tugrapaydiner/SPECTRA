"""Collapse detectors and training diagnostics (BLUEPRINT section 11.7).

SPECTRA stacks two unstable regimes (deep shared-weight recursion + low-bit QAT),
so we instrument the failure modes from day one:

  * output collapse -- predictions dominated by one token,
  * fixed-point collapse -- latent states stop changing while accuracy is bad,
  * recursive explosion -- activation/logit norms blow up or go non-finite,
  * (ternary saturation lives in Phase 4 once ternary weights exist).

The helpers also expose gradient norms (total + per recursive block) and per-step
state-delta norms used both for logging (section 29) and the Phase 3 gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn

from eval.metrics import most_common_token_ratio


# --------------------------------------------------------------------------- #
# Gradient norms
# --------------------------------------------------------------------------- #
def total_grad_norm(model: nn.Module) -> float:
    """L2 norm of all parameter gradients (call after ``loss.backward()``)."""
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum())
    return sq**0.5


def per_block_grad_norm(blocks: nn.ModuleList) -> list[float]:
    """Per-recursive-block gradient L2 norm (detects vanishing/exploding depth)."""
    norms = []
    for block in blocks:
        sq = 0.0
        for p in block.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum())
        norms.append(sq**0.5)
    return norms


# --------------------------------------------------------------------------- #
# State-delta and activation norms across supervision steps
# --------------------------------------------------------------------------- #
def _mean_l2(t: torch.Tensor) -> float:
    """Mean over the batch of per-example L2 norms of ``t`` ([B, ...])."""
    return t.reshape(t.shape[0], -1).norm(dim=1).mean().item()


def state_delta_norms(
    steps: Sequence[dict[str, Any]]
) -> tuple[list[float], list[float]]:
    """Consecutive-step change norms ``(||dy||, ||dz||)`` (section 11.7)."""
    dy, dz = [], []
    for a, b in zip(steps[:-1], steps[1:]):
        dy.append(_mean_l2(b["y"] - a["y"]))
        dz.append(_mean_l2(b["z"] - a["z"]))
    return dy, dz


def activation_norms(
    steps: Sequence[dict[str, Any]]
) -> tuple[list[float], list[float]]:
    """Per-step activation norms ``(||y||, ||z||)`` for explosion monitoring."""
    return ([_mean_l2(s["y"]) for s in steps], [_mean_l2(s["z"]) for s in steps])


# --------------------------------------------------------------------------- #
# Collapse report
# --------------------------------------------------------------------------- #
@dataclass
class CollapseReport:
    """Aggregated collapse signals for one forward pass."""

    most_common_ratio: float
    final_entropy_ok: bool
    has_nan: bool
    delta_y: list[float] = field(default_factory=list)
    delta_z: list[float] = field(default_factory=list)
    y_norms: list[float] = field(default_factory=list)
    z_norms: list[float] = field(default_factory=list)
    output_collapse: bool = False
    fixed_point_collapse: bool = False
    explosion: bool = False
    dimensional_collapse: bool = False
    token_variance_ratio: float = 1.0

    @property
    def collapsed(self) -> bool:
        """True if any *hard* failure occurred (output/dim collapse, NaN, explosion)."""
        return (
            self.output_collapse
            or self.has_nan
            or self.explosion
            or self.dimensional_collapse
        )


def dimensional_collapse_ratio(z: torch.Tensor) -> float:
    """Variance of the latent across tokens, normalised by its magnitude.

    Near zero means every token vector has converged to the same point -- the
    dimensional collapse RMSNorm cannot prevent (it only fixes magnitude). This is
    the detector counterpart to the ``model/spectral.py`` anti-collapse penalty.
    """
    token_var = z.var(dim=1).mean()            # variance across the L tokens
    scale = z.pow(2).mean() + 1e-8
    return float(token_var / scale)


def check_collapse(
    steps: Sequence[dict[str, Any]],
    accuracy: float,
    mode_threshold: float = 0.90,
    fixed_point_eps: float = 1e-4,
    explosion_factor: float = 50.0,
    dim_collapse_eps: float = 1e-3,
) -> CollapseReport:
    """Compute collapse signals for a forward pass.

    Args:
        steps: Per-step outputs from ``TRM.forward``.
        accuracy: Final-step cell accuracy (used to qualify fixed-point collapse).
        mode_threshold: Most-common-token ratio above which output has collapsed.
        fixed_point_eps: Below this state-delta the latent state is "frozen".
        explosion_factor: Growth ratio of activation norm flagged as explosion.
        dim_collapse_eps: Below this token-variance ratio the latent has collapsed
            to a degenerate (rank-1) representation.
    """
    final_logits = steps[-1]["logits"]
    final_pred = final_logits.argmax(-1)

    ratio = most_common_token_ratio(final_pred)
    has_nan = not bool(torch.isfinite(final_logits).all())
    dy, dz = state_delta_norms(steps)
    y_norms, z_norms = activation_norms(steps)

    # Fixed-point collapse: latent states frozen while accuracy is still poor.
    frozen = bool(dy and dz and dy[-1] < fixed_point_eps and dz[-1] < fixed_point_eps)
    fixed_point = frozen and accuracy < 0.5

    # Explosion: non-finite, or activation norm grew by a large factor.
    grew = bool(
        y_norms and y_norms[0] > 0 and (y_norms[-1] / max(y_norms[0], 1e-8)) > explosion_factor
    )
    explosion = has_nan or grew

    # Dimensional collapse: token vectors converged to one point (rank collapse).
    var_ratio = dimensional_collapse_ratio(steps[-1]["z"])
    dim_collapse = var_ratio < dim_collapse_eps

    return CollapseReport(
        most_common_ratio=ratio,
        final_entropy_ok=ratio < mode_threshold,
        has_nan=has_nan,
        delta_y=dy,
        delta_z=dz,
        y_norms=y_norms,
        z_norms=z_norms,
        output_collapse=ratio > mode_threshold,
        fixed_point_collapse=fixed_point,
        explosion=explosion,
        dimensional_collapse=dim_collapse,
        token_variance_ratio=var_ratio,
    )


def collapse_dashboard(
    steps: Sequence[dict[str, Any]], accuracy: float, model: nn.Module | None = None
) -> dict[str, Any]:
    """Flatten all collapse / quant signals into one log-friendly dict.

    This is the "collapse dashboard" deliverable of Phase 6: a single call that
    summarises output/fixed-point/explosion signals plus (for a ternary model)
    the weight distribution, ready to write to JSONL each eval.
    """
    rep = check_collapse(steps, accuracy)
    dash: dict[str, Any] = {
        "accuracy": accuracy,
        "most_common_ratio": rep.most_common_ratio,
        "has_nan": rep.has_nan,
        "output_collapse": rep.output_collapse,
        "fixed_point_collapse": rep.fixed_point_collapse,
        "explosion": rep.explosion,
        "dimensional_collapse": rep.dimensional_collapse,
        "token_variance_ratio": rep.token_variance_ratio,
        "final_delta_y": rep.delta_y[-1] if rep.delta_y else 0.0,
        "final_delta_z": rep.delta_z[-1] if rep.delta_z else 0.0,
        "final_y_norm": rep.y_norms[-1] if rep.y_norms else 0.0,
        "final_z_norm": rep.z_norms[-1] if rep.z_norms else 0.0,
    }
    if model is not None:
        # Local import avoids a hard train->model.stability dependency at import time.
        from model.stability import ternary_report

        tr = ternary_report(model)
        if tr.get("num_ternary_layers", 0) > 0:
            dash.update({"w_zero": tr["zero"], "w_neg": tr["neg"], "w_pos": tr["pos"]})
    return dash
