"""Spectral / dimensional-collapse guards for the recursive latent (GT audit #2).

RMSNorm + residual scaling bound the *magnitude* of the latent but do nothing to
stop **dimensional collapse** (all token vectors converging to one direction) or
guarantee the per-step map's spectrum stays near isometry. This module adds the
explicit regularizers that do:

  * ``orthogonal_init_`` -- semi-orthogonal weight init (preserves norm/rank early,
    the dynamical-isometry starting point of Pennington et al.).
  * ``dimensional_collapse_penalty`` -- a VICReg-style variance + covariance term
    that keeps every feature dimension informative (per-dim variance >= 1) and
    decorrelated (off-diagonal covariance -> 0), so ``z`` cannot collapse to a
    degenerate low-rank state.
  * ``jacobian_isometry_penalty`` -- a Hutchinson estimate that pushes the step
    Jacobian's singular values toward 1 (dynamical isometry). NOTE: the target is
    isometry, NOT strict contraction (||J|| < 1) -- a strict contraction would make
    the recursion converge to a single fixed point and destroy expressivity. We
    want norm/rank preservation, not collapse-to-a-point.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def orthogonal_init_(module: nn.Module, gain: float = 1.0) -> None:
    """Apply (semi-)orthogonal init to all 2D Linear/FakeBitLinear master weights."""
    for m in module.modules():
        w = getattr(m, "weight", None)
        if isinstance(w, nn.Parameter) and w.dim() == 2 and not isinstance(m, nn.Embedding):
            with torch.no_grad():
                nn.init.orthogonal_(w, gain=gain)


def dimensional_collapse_penalty(
    z: torch.Tensor, gamma: float = 1.0, lambda_var: float = 1.0, lambda_cov: float = 0.04
) -> torch.Tensor:
    """VICReg variance + covariance anti-collapse penalty on a latent ``[B, L, D]``.

    Variance term hinges each feature's std up to ``gamma`` (prevents per-dim
    collapse); covariance term drives off-diagonal feature covariances to zero
    (prevents redundant / rank-deficient representations).
    """
    feats = z.reshape(-1, z.shape[-1])                 # [N, D]
    n, d = feats.shape
    feats = feats - feats.mean(dim=0, keepdim=True)

    std = torch.sqrt(feats.var(dim=0) + 1e-6)          # [D]
    var_term = torch.relu(gamma - std).mean()

    cov = (feats.T @ feats) / max(1, n - 1)            # [D, D]
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_term = off_diag.pow(2).sum() / d

    return lambda_var * var_term + lambda_cov * cov_term


def jacobian_isometry_penalty(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    n_samples: int = 1,
) -> torch.Tensor:
    """Push the per-step Jacobian toward isometry (singular values ~ 1).

    Hutchinson estimate: for random ``u``, ``J^T u`` (a vjp) should have the same
    norm as ``u`` if ``J`` is isometric. Differentiable (``create_graph``), so it
    can be added directly to the training loss.

    Args:
        step_fn: A function ``z -> z'`` (one recursion step's latent update).
        z: The input latent ``[B, L, D]``.
        n_samples: Number of random probes (variance reduction).
    """
    z = z.detach().requires_grad_(True)
    out = step_fn(z)
    penalty = z.new_zeros(())
    for _ in range(n_samples):
        u = torch.randn_like(out)
        (g,) = torch.autograd.grad(
            out, z, grad_outputs=u, create_graph=True, retain_graph=True
        )
        g_norm = g.flatten(1).norm(dim=1)
        u_norm = u.flatten(1).norm(dim=1)
        penalty = penalty + (g_norm - u_norm).pow(2).mean()
    return penalty / n_samples


def latent_stability_loss(
    steps: list,
    lambda_collapse: float = 1.0,
    lambda_jacobian: float = 0.0,
    step_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Aggregate anti-collapse penalty over the recursion's latent states.

    Add to the deep-supervision loss to guarantee the latent trajectory stays
    high-rank. The Jacobian term is optional (it needs a ``step_fn`` closure and a
    second backward) and is off by default for speed.
    """
    device = steps[-1]["z"].device
    total = torch.zeros((), device=device)
    for s in steps:
        total = total + lambda_collapse * dimensional_collapse_penalty(s["z"])
    if lambda_jacobian > 0.0 and step_fn is not None:
        total = total + lambda_jacobian * jacobian_isometry_penalty(step_fn, steps[0]["z"])
    return total / len(steps)
