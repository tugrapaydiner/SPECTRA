"""Input-conditioned single-stream recurrent TRM used by M14 Attempt 2.

The historical TRM with ``n=1`` performs two shared-block applications per
supervision step: an input-conditioned z update followed by a y update.  A literal
``n=0`` is not a faithful one-block replacement because the inherited y update is
``f(y + z)`` and therefore ignores the puzzle at the zero initial state.

This module defines the falsifiable M14 Attempt-2 intervention instead:

    u_k       = f_theta(x_emb + y_k)
    y_{k+1}   = RMSNorm(y_k + alpha_y * u_k)

Exactly one shared transformer block stack is applied per recursive cycle.  ``z``
is retained only as a compatibility state for the common execution interface and
is never updated.
"""
from __future__ import annotations

import torch

from model.trm import TRM


class InputConditionedSingleStreamTRM(TRM):
    """One-stream shared-weight recurrence with explicit input conditioning.

    This is an FP32 M14 experimental architecture.  It subclasses :class:`TRM`
    only to reuse the verified embedding/output/execution-state machinery.  The
    unused z-normalization/scalar parameters are frozen so they do not inflate the
    trainable parameter count or receive optimizer ownership.
    """

    SEMANTICS = "input_conditioned_single_stream_v1"

    def __init__(
        self,
        *,
        dim: int,
        num_tokens: int,
        seq_len: int,
        n_layers: int = 1,
        T: int = 1,
        N_sup: int = 4,
        heads: int = 4,
        alpha_y: float = 0.1,
        max_grid_size: int = 8,
    ) -> None:
        super().__init__(
            dim=dim,
            num_tokens=num_tokens,
            seq_len=seq_len,
            n_layers=n_layers,
            n=0,
            T=T,
            N_sup=N_sup,
            heads=heads,
            alpha_y=alpha_y,
            alpha_z=0.0,
            max_grid_size=max_grid_size,
            ternary=False,
            act8=False,
        )
        # z is a compatibility zero-state only in this architecture.
        for p in self.norm_z.parameters():
            p.requires_grad_(False)
        self.alpha_z.requires_grad_(False)

    def recursive_cycle(
        self,
        x_emb: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One dense single-stream cycle; every cycle sees the encoded puzzle."""
        update_y = self.f(x_emb + y)
        y_new = self.act_quant(self.norm_y(y + self.alpha_y * update_y))
        if mask is None:
            y = y_new
        else:
            y = mask * y_new + (1.0 - mask) * y
        return y, z

    def recursive_cycle_adaptive(
        self,
        x_emb: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        work: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One adaptive cycle with exactly one shared block-stack application."""
        update_y = self._f_active(x_emb + y, mask, work)
        y = self._commit_active_update(
            y, update_y, self.alpha_y, self.norm_y, mask, work
        )
        return y, z

    @property
    def effective_depth(self) -> int:
        """Block applications per recursive cycle for this one-stream recurrence."""
        return self.T * len(self.blocks)
