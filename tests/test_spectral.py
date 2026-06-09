"""GT2 tests: spectral / dimensional-collapse guards.

Proves the penalties detect and actively reverse latent collapse, the Jacobian
isometry penalty is zero for an isometric map and positive for a contraction, and
orthogonal init produces orthonormal weights.
"""

import torch
import torch.nn as nn

from model.spectral import (
    dimensional_collapse_penalty,
    jacobian_isometry_penalty,
    orthogonal_init_,
)


def test_collapse_penalty_flags_degenerate_latent():
    torch.manual_seed(0)
    # Collapsed: (near-)constant across tokens and tiny variance per dim.
    collapsed = torch.randn(8, 1, 16).expand(8, 12, 16) * 0.01
    diverse = torch.randn(8, 12, 16)
    assert dimensional_collapse_penalty(collapsed) > dimensional_collapse_penalty(diverse) + 0.5


def test_collapse_penalty_actively_decollapses():
    torch.manual_seed(0)
    z = nn.Parameter(torch.randn(16, 8, 16) * 0.01)  # start collapsed
    opt = torch.optim.Adam([z], lr=0.05)
    var0 = z.reshape(-1, 16).var(dim=0).mean().item()
    for _ in range(100):
        loss = dimensional_collapse_penalty(z)
        opt.zero_grad()
        loss.backward()
        opt.step()
    var1 = z.reshape(-1, 16).var(dim=0).mean().item()
    assert var1 > 5 * var0  # variance restored -> latent decollapsed


def test_orthogonal_init_is_orthonormal():
    m = nn.Linear(32, 32, bias=False)
    orthogonal_init_(m)
    w = m.weight.detach()
    assert torch.allclose(w @ w.T, torch.eye(32), atol=1e-4)


def test_jacobian_isometry_zero_for_identity_positive_for_contraction():
    torch.manual_seed(0)
    z = torch.randn(4, 8, 16)
    assert jacobian_isometry_penalty(lambda zz: zz, z).item() < 1e-8        # isometric
    assert jacobian_isometry_penalty(lambda zz: 0.1 * zz, z).item() > 0.1   # contraction


def test_jacobian_penalty_is_differentiable():
    torch.manual_seed(0)
    z = torch.randn(4, 8, 16)
    lin = nn.Linear(16, 16, bias=False)
    pen = jacobian_isometry_penalty(lambda zz: lin(zz), z)
    pen.backward()
    assert lin.weight.grad is not None and torch.isfinite(pen)
