"""Tiny Recursive Model (TRM): the FP16 recursive reasoning core.

This implements the recursion of BLUEPRINT sections 5 and 15. Three states are
carried through the loop:

  * ``x`` -- the encoded (and frozen) input problem,
  * ``y`` -- the current answer state (decoded by ``out_head``),
  * ``z`` -- a latent "scratchpad" reasoning state.

A single recursive *cycle* refines ``z`` for ``n`` inner steps, then updates the
answer ``y`` once (section 5.2). Each deep-supervision step runs ``T`` cycles,
decodes ``y`` to logits + a halting logit, records the step, and then *detaches*
``y``/``z`` so gradients do not flow across supervision steps (section 5.2). The
effective network depth is ``T * (n + 1) * n_layers`` (section 5.3).

Updates use residual scaling + RMSNorm for stability (section 5.6):
``z <- RMSNorm(z + alpha_z * dz)``, likewise for ``y``.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from model.bitlinear import FakeBitLinear
from model.fake_quant import FakeActQuant
from model.operators import SwapBlock
from model.spatial_encoder import SpatialEncoder

LinearFactory = Callable[..., nn.Module]


class TRM(nn.Module):
    """Recursive reasoning core with deep supervision and a halting head.

    Args:
        dim: Hidden dimension D.
        num_tokens: Vocabulary size for input embedding and output head.
        seq_len: Sequence length L (stored for reference; positions are derived
            from ``height``/``width`` at forward time).
        n_layers: Number of ``SwapBlock`` layers in the shared operator f_theta.
        n: Inner latent recursion steps per cycle.
        T: Deep recursion cycles per supervision step.
        N_sup: Number of deep-supervision steps.
        heads: Attention heads per block.
        alpha_y: Initial residual-scaling for ``y`` updates (section 5.6).
        alpha_z: Initial residual-scaling for ``z`` updates (section 5.6).
        max_grid_size: Max grid extent for row/col positional embeddings.
        linear_cls: Projection factory for the blocks' FFNs (``nn.Linear`` for
            FP16; ``FakeBitLinear`` to ternarize only the FFN).
        ternary: If True, build a fully ternary recursive core -- ``FakeBitLinear``
            FFN + ternary attention + ternary output head (W1.58). Embeddings,
            norms, and the halting head stay higher precision (section 11.5).
        act8: If True, fake-quantize the recurrent latent states y, z to INT8
            inside the loop (the A8 of W1.58A8, section 19).
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        seq_len: int,
        n_layers: int = 2,
        n: int = 6,
        T: int = 3,
        N_sup: int = 16,
        heads: int = 8,
        alpha_y: float = 0.1,
        alpha_z: float = 0.1,
        max_grid_size: int = 32,
        linear_cls: LinearFactory = nn.Linear,
        ternary: bool = False,
        act8: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.n = n
        self.T = T
        self.N_sup = N_sup
        self.max_grid_size = max_grid_size
        self.ternary = ternary
        self.act8 = act8

        # INT8 latent-state recursion: quantize y, z each step (section 19).
        self.act_quant: nn.Module = FakeActQuant() if act8 else nn.Identity()

        # A fully ternary core ternarizes FFN, attention, and the output head.
        if ternary:
            linear_cls = FakeBitLinear

        # Input embedding + 2D positional embeddings (row/col), section 15.2.
        self.token_embed = nn.Embedding(num_tokens, dim)
        self.pos_encoder = SpatialEncoder(dim, max_grid_size)

        # Shared recursive operator f_theta: a small stack of blocks reused at
        # every recursion step (weight sharing is what makes recursion cheap).
        self.blocks = nn.ModuleList(
            [
                SwapBlock(dim, heads=heads, linear_cls=linear_cls, ternary_attn=ternary)
                for _ in range(n_layers)
            ]
        )

        # Decoding heads. Output projection is ternarized in a ternary core
        # (section 11.5 stage 2); the halting head stays higher precision.
        self.out_head: nn.Module = (
            FakeBitLinear(dim, num_tokens, bias=True) if ternary else nn.Linear(dim, num_tokens)
        )
        self.halt_head = nn.Linear(dim, 1)

        # Per-state output norms for the residual update.
        self.norm_y = nn.RMSNorm(dim)
        self.norm_z = nn.RMSNorm(dim)

        # Learnable residual-scaling gates (start small for stability).
        self.alpha_y = nn.Parameter(torch.tensor(float(alpha_y)))
        self.alpha_z = nn.Parameter(torch.tensor(float(alpha_z)))

    def encode_positions(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Row+column positional embeddings for a flattened ``height x width`` grid.

        Args:
            x: Input token ids of shape ``[B, L]`` (used only for length/device).
            height: Grid height (kept for interface symmetry / validation).
            width: Grid width; row = index // width, col = index % width.

        Returns:
            Positional embedding of shape ``[1, L, D]`` (broadcasts over batch).
        """
        return self.pos_encoder(x.shape[1], width, x.device)

    def f(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the shared operator f_theta (the block stack) once."""
        for block in self.blocks:
            h = block(h)
        return h

    def recursive_cycle(
        self,
        x_emb: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recursive cycle: refine ``z`` ``n`` times, then update ``y`` once.

        Mirrors section 5.2 with residual scaling from 5.6. When ``mask`` (shape
        ``[B, L, 1]``, 1 = active) is given, only active tokens receive updates --
        frozen tokens keep their previous state (lazy routing, section 7.2).
        """
        for _ in range(self.n):
            update_z = self.f(x_emb + y + z)
            z_new = self.act_quant(self.norm_z(z + self.alpha_z * update_z))
            z = z_new if mask is None else mask * z_new + (1.0 - mask) * z

        update_y = self.f(y + z)
        y_new = self.act_quant(self.norm_y(y + self.alpha_y * update_y))
        y = y_new if mask is None else mask * y_new + (1.0 - mask) * y
        return y, z

    def forward(
        self,
        x: torch.Tensor,
        height: int = 9,
        width: int = 9,
        y_target: torch.Tensor | None = None,
        router: Any | None = None,
        device_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Run deep-supervised recursion over the input.

        Args:
            x: Input token ids ``[B, L]``.
            height: Grid height for positional encoding.
            width: Grid width for positional encoding.
            y_target: Reserved for interface symmetry; the supervision targets are
                applied by the loss over ``step_outputs``, not inside forward.
            router: Optional lazy-routing policy (section 7/20). When provided, it
                is queried once per supervision step to produce a freeze/compute
                ``mask``; frozen tokens are not updated that step. Adds
                ``mask``/``active_density``/``router_logprob`` to each step output.
            device_state: Optional device-state vector ``[B, d]`` passed to the
                router (hardware-aware routing, section 8.1).

        Returns:
            ``(final_logits, step_outputs)`` where ``final_logits`` is ``[B, L,
            num_tokens]`` and ``step_outputs`` is a list (one dict per supervision
            step) with keys ``logits``, ``halt_logit``, ``y``, ``z`` (plus routing
            keys when ``router`` is given).
        """
        x_emb = self.token_embed(x) + self.encode_positions(x, height, width)
        # Answer/latent states start at zero each forward pass.
        y = torch.zeros_like(x_emb)
        z = torch.zeros_like(x_emb)

        step_outputs: list[dict[str, Any]] = []

        for k in range(self.N_sup):
            mask = logprob = None
            if router is not None:
                # Route on the current answer distribution / latent state.
                route_logits = self.out_head(y)
                mask, logprob = router(k, y, z, route_logits, device_state)

            for _ in range(self.T):
                y, z = self.recursive_cycle(x_emb, y, z, mask)

            logits = self.out_head(y)
            halt_logit = self.halt_head(y.mean(dim=1)).squeeze(-1)

            out: dict[str, Any] = {
                "logits": logits, "halt_logit": halt_logit, "y": y, "z": z
            }
            if router is not None:
                out["mask"] = mask
                out["active_density"] = mask.mean().item()
                out["router_logprob"] = logprob
            step_outputs.append(out)

            # Detach between supervision steps: bounds backprop through the
            # recurrence (section 5.2) while still supervising every step.
            y = y.detach()
            z = z.detach()

        return step_outputs[-1]["logits"], step_outputs

    @property
    def effective_depth(self) -> int:
        """Effective network depth: ``T * (n + 1) * n_layers`` (section 5.3)."""
        return self.T * (self.n + 1) * len(self.blocks)
