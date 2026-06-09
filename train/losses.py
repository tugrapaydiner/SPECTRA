"""Training losses for the recursive core.

Implements the deep-supervision loss (BLUEPRINT section 17) and the corrected
policy-improvement loss (section 12). The deep-supervision loss combines, at
every supervision step:

  * cross-entropy of the decoded answer against the target,
  * a halting BCE that learns to predict whether the step is already correct, and
  * an *improvement* hinge that pushes each step to assign at least as much
    probability to the target as the previous step did (so later recursion does
    not regress).

The improvement term deliberately detaches only the *previous* step's log-prob
(``prev.detach()``), keeping the gradient on the current step alive -- the broken
variant in section 12 detaches the whole advantage and kills the gradient.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F


def deep_supervision_loss(
    steps: Sequence[dict[str, Any]],
    y_target: torch.Tensor,
    lambda_h: float = 0.5,
    lambda_improve: float = 0.1,
    margin: float = 0.01,
) -> torch.Tensor:
    """Average per-step (CE + halting BCE + improvement hinge) loss.

    Args:
        steps: Per-supervision-step outputs from ``TRM.forward`` (each a dict with
            ``logits`` ``[B, L, V]`` and ``halt_logit`` ``[B]``).
        y_target: Target token ids ``[B, L]``.
        lambda_h: Weight on the halting BCE term.
        lambda_improve: Weight on the step-improvement hinge.
        margin: Required improvement margin in log-prob (section 12).

    Returns:
        Scalar loss tensor.
    """
    total = torch.zeros((), device=y_target.device)
    prev_target_logp = None

    for s in steps:
        logits = s["logits"]
        halt_logit = s["halt_logit"]

        # Task cross-entropy over every token.
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_target.reshape(-1),
        )

        # Halting target: 1 if the whole answer is currently correct.
        correct = (logits.argmax(-1) == y_target).all(dim=1).float()
        halt_bce = F.binary_cross_entropy_with_logits(halt_logit, correct)

        # Per-token log-prob assigned to the target class.
        logp = F.log_softmax(logits, dim=-1)
        target_logp = logp.gather(-1, y_target.unsqueeze(-1)).squeeze(-1)

        # Improvement hinge: current step should not assign less target mass than
        # the previous step did (only previous is detached -> live gradient here).
        improve = torch.zeros((), device=y_target.device)
        if prev_target_logp is not None:
            improve = F.relu(prev_target_logp.detach() + margin - target_logp).mean()
        prev_target_logp = target_logp

        total = total + ce + lambda_h * halt_bce + lambda_improve * improve

    return total / len(steps)


def policy_improvement_loss(
    step_logits: Sequence[torch.Tensor],
    y_target: torch.Tensor,
    margin: float = 0.01,
) -> torch.Tensor:
    """Correct policy-improvement loss over a sequence of step logits (section 12).

    Each step is pushed to (a) maximise target log-prob and (b) improve on the
    previous step by at least ``margin``. Only the previous step's log-prob is
    detached, preserving the gradient on the current step.

    Args:
        step_logits: Sequence of ``[B, L, V]`` logits, one per recursion step.
        y_target: Target token ids ``[B, L]``.
        margin: Required improvement margin in log-prob.

    Returns:
        Scalar loss tensor.
    """
    losses: list[torch.Tensor] = []
    prev_target_logp = None

    for logits in step_logits:
        logp = F.log_softmax(logits, dim=-1)
        target_logp = logp.gather(-1, y_target.unsqueeze(-1)).squeeze(-1)
        losses.append(-target_logp.mean())

        if prev_target_logp is not None:
            improve_loss = F.relu(
                prev_target_logp.detach() + margin - target_logp
            ).mean()
            losses.append(improve_loss)

        prev_target_logp = target_logp

    return torch.stack(losses).sum() / len(losses)
