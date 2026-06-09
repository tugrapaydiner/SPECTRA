"""System 1 student + dual-mode reasoning (BLUEPRINT sections 4, 10).

System 1 is a fast feed-forward "reflex": a single forward pass (no recursion)
that answers easy instances cheaply. It is distilled from the recursive System 2
teacher. A learned confidence head predicts whether its own answer is correct, so
the :class:`DualModeReasoner` can *escalate* only the uncertain instances to the
expensive System 2 -- realising the section 4 architecture (cheap reflex first,
deep recursion only when needed).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from model.operators import SwapBlock
from model.spatial_encoder import SpatialEncoder


class System1Student(nn.Module):
    """Feed-forward student that predicts the answer in one pass + a confidence.

    Args:
        dim: Hidden dimension.
        num_tokens: Vocabulary size.
        seq_len: Sequence length (stored for reference).
        n_layers: Number of blocks (a little deeper than one TRM cycle, since
            there is no recursion to add depth).
        heads: Attention heads.
        max_grid_size: Max grid extent for positional embeddings.
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        seq_len: int,
        n_layers: int = 4,
        heads: int = 8,
        max_grid_size: int = 32,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.max_grid_size = max_grid_size
        self.token_embed = nn.Embedding(num_tokens, dim)
        self.pos_encoder = SpatialEncoder(dim, max_grid_size)
        self.blocks = nn.ModuleList([SwapBlock(dim, heads=heads) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(dim)
        self.out_head = nn.Linear(dim, num_tokens)
        self.conf_head = nn.Linear(dim, 1)  # predicts P(answer correct)

    def forward(
        self, x: torch.Tensor, height: int = 9, width: int = 9
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits [B, L, V], confidence_logit [B])`` in a single pass."""
        h = self.token_embed(x) + self.pos_encoder(x.shape[1], width, x.device)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        logits = self.out_head(h)
        conf_logit = self.conf_head(h.mean(dim=1)).squeeze(-1)
        return logits, conf_logit

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, height: int = 9, width: int = 9
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(answer [B, L], confidence [B] in [0, 1])``."""
        logits, conf_logit = self.forward(x, height, width)
        return logits.argmax(dim=-1), torch.sigmoid(conf_logit)


class DualModeReasoner:
    """Cheap System 1 first; escalate uncertain instances to System 2.

    Args:
        system1: The fast feed-forward student.
        system2: A callable ``(x, height, width) -> answer [B, L]`` (e.g. the TRM
            teacher, optionally wrapped with best-of-N / MCTS).
        threshold: Escalate when System 1 confidence < ``threshold``.
        height / width: Grid dims for positional encoding.
    """

    def __init__(
        self,
        system1: System1Student,
        system2: Callable[[torch.Tensor, int, int], torch.Tensor],
        threshold: float,
        height: int,
        width: int,
    ):
        self.system1 = system1
        self.system2 = system2
        self.threshold = threshold
        self.height = height
        self.width = width

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(answer [B, L], escalated [B] bool)``.

        System 2 is run only on the escalated (low-confidence) rows.
        """
        answer, confidence = self.system1.predict(x, self.height, self.width)
        escalate = confidence < self.threshold
        if escalate.any():
            s2_answer = self.system2(x[escalate], self.height, self.width)
            answer = answer.clone()
            answer[escalate] = s2_answer
        return answer, escalate
