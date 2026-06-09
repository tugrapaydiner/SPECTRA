"""Best-of-N candidate selection (BLUEPRINT section 9.1).

Decode several candidate answers from the model's output distribution and keep
the one a verifier scores best. The greedy answer is always included, so
best-of-N can only match or beat single-candidate decoding when the verifier is
reliable.
"""

from __future__ import annotations

from typing import Callable

import torch


def decode_candidates(
    logits: torch.Tensor,
    n: int,
    temperature: float = 1.0,
    include_greedy: bool = True,
) -> torch.Tensor:
    """Decode ``n`` candidate token grids from ``logits`` ``[B, L, V]``.

    Returns a tensor ``[n, B, L]``. Candidate 0 is the greedy argmax (when
    ``include_greedy``); the rest are temperature samples.
    """
    b, length, vocab = logits.shape
    candidates: list[torch.Tensor] = []
    if include_greedy:
        candidates.append(logits.argmax(dim=-1))
    probs = torch.softmax(logits / temperature, dim=-1).reshape(b * length, vocab)
    while len(candidates) < n:
        sample = torch.multinomial(probs, num_samples=1).reshape(b, length)
        candidates.append(sample)
    return torch.stack(candidates, dim=0)


def select_best(
    candidates: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the highest-scoring candidate per example.

    Args:
        candidates: ``[n, B, L]`` candidate grids.
        score_fn: Maps a ``[B, L]`` candidate to a ``[B]`` score (higher better).

    Returns:
        ``(best [B, L], scores [n, B])``.
    """
    n, b, _ = candidates.shape
    scores = torch.stack([score_fn(candidates[i]) for i in range(n)], dim=0)  # [n, B]
    best_idx = scores.argmax(dim=0)  # [B]
    best = candidates[best_idx, torch.arange(b, device=candidates.device)]
    return best, scores


def best_of_n(
    logits: torch.Tensor,
    n: int,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Decode ``n`` candidates and return the verifier-selected best ``[B, L]``."""
    candidates = decode_candidates(logits, n, temperature=temperature)
    best, _ = select_best(candidates, score_fn)
    return best
