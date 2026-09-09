"""Spectral / dimensional-collapse diagnostics and regularizers.

RMSNorm + residual scaling bound the *magnitude* of the latent but do not prove
that a recurrent representation remains full-rank or that a per-step Jacobian is
isometric.  This module provides training heuristics inspired by established
anti-collapse and dynamical-isometry ideas:

* ``orthogonal_init_`` applies semi-orthogonal initialization to eligible 2-D
  weights.  Orthogonal initialization can improve conditioning in appropriate
  architectures, but it does not guarantee dynamical isometry for an arbitrary
  nonlinear recurrent SPECTRA block.
* ``dimensional_collapse_penalty`` is a VICReg-style variance + covariance
  objective.  It penalizes low per-feature variance and off-diagonal covariance on
  the sampled latent batch; a low loss does not prove global rank preservation,
  independence, or preserved reasoning.
* ``jacobian_isometry_penalty`` is a stochastic norm-preservation penalty using
  vector-Jacobian products.  With finitely many random probes it is a training
  signal/diagnostic, not a certificate that all Jacobian singular values equal 1.

Milestone 15 makes these boundaries explicit: any rank/isometry claim requires
separate assumptions and direct spectral/rank evidence on the relevant trajectory
and state distribution.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def orthogonal_init_(module: nn.Module, gain: float = 1.0) -> None:
    """Apply (semi-)orthogonal initialization to eligible 2-D master weights."""
    for m in module.modules():
        w = getattr(m, "weight", None)
        if isinstance(w, nn.Parameter) and w.dim() == 2 and not isinstance(m, nn.Embedding):
            with torch.no_grad():
                nn.init.orthogonal_(w, gain=gain)


def dimensional_collapse_penalty(
    z: torch.Tensor, gamma: float = 1.0, lambda_var: float = 1.0, lambda_cov: float = 0.04
) -> torch.Tensor:
    """VICReg-style variance + covariance penalty on latent ``[B, L, D]``.

    The variance term penalizes sampled feature standard deviations below
    ``gamma``.  The covariance term penalizes sampled off-diagonal feature
    covariance.  These are finite-sample objectives: they can discourage observed
    collapse/redundancy but do not certify the representation's global rank.
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
    """Penalize sampled Jacobian norm distortion using random VJP probes.

    For a truly isometric Jacobian, ``||J^T u|| == ||u||`` for every probe ``u``.
    Here only finitely many random probes are evaluated, so a small empirical
    penalty is not a proof of dynamical isometry or of every singular value being
    near one.  The term is differentiable (``create_graph=True``) and may be used
    as a training regularizer.

    Args:
        step_fn: A function ``z -> z'`` for one recurrent latent update.
        z: Input latent ``[B, L, D]``.
        n_samples: Number of random probes used by this stochastic estimate.
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
    """Aggregate sampled anti-collapse/isometry penalties over latent states.

    This helper can discourage the measured failure modes represented by the
    component losses.  It does not guarantee that the full recurrent trajectory is
    high-rank, isometric, stable, or task-correct.  The optional Jacobian term
    requires a ``step_fn`` closure and an additional higher-order gradient path.
    """
    device = steps[-1]["z"].device
    total = torch.zeros((), device=device)
    for s in steps:
        total = total + lambda_collapse * dimensional_collapse_penalty(s["z"])
    if lambda_jacobian > 0.0 and step_fn is not None:
        total = total + lambda_jacobian * jacobian_isometry_penalty(step_fn, steps[0]["z"])
    return total / len(steps)
