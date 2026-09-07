"""Distillation and auxiliary-training losses.

This module contains both independently supervised losses and search-bootstrapped
losses.  M07 makes that provenance boundary explicit: MCTS backup values are
useful self-training targets, but they are not independent verifier ground truth.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from train.losses import deep_supervision_loss

MCTS_BOOTSTRAP_TARGET_KIND = "mcts_bootstrap_value_v1"
GROUNDED_VERIFIER_TARGET_KIND = "sudoku_one_cycle_improvement_v1"


def trajectory_distillation_loss(
    student_steps: Sequence[dict[str, Any]],
    teacher_steps: Sequence[dict[str, Any]],
    y_target: torch.Tensor,
    lambda_task: float = 1.0,
    lambda_logit: float = 1.0,
    lambda_state: float = 1.0,
    temperature: float = 2.0,
    lambda_h: float = 0.5,
    lambda_improve: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined task + logit + state distillation loss."""
    if len(student_steps) != len(teacher_steps):
        raise ValueError("student and teacher must have the same number of steps")

    task = deep_supervision_loss(
        student_steps, y_target, lambda_h=lambda_h, lambda_improve=lambda_improve
    )

    tau = temperature
    logit_loss = torch.zeros((), device=y_target.device)
    state_loss = torch.zeros((), device=y_target.device)
    for s, t in zip(student_steps, teacher_steps):
        s_logp = F.log_softmax(s["logits"] / tau, dim=-1)
        t_p = F.softmax(t["logits"].detach() / tau, dim=-1)
        kl = (t_p * (torch.log(t_p + 1e-9) - s_logp)).sum(dim=-1).mean()
        logit_loss = logit_loss + kl * (tau * tau)
        state_loss = state_loss + F.mse_loss(s["y"], t["y"].detach())
        state_loss = state_loss + F.mse_loss(s["z"], t["z"].detach())

    n = len(student_steps)
    logit_loss = logit_loss / n
    state_loss = state_loss / n

    total = lambda_task * task + lambda_logit * logit_loss + lambda_state * state_loss
    return total, {
        "task": float(task.detach()),
        "logit": float(logit_loss.detach()),
        "state": float(state_loss.detach()),
        "total": float(total.detach()),
    }


def grounded_improvement_bce_loss(
    verifier,
    x: torch.Tensor,
    y_state: torch.Tensor,
    z_state: torch.Tensor,
    labels: torch.Tensor,
    width: int = 9,
) -> tuple[torch.Tensor, dict[str, float]]:
    """BCE for the independently grounded M07 one-cycle improvement event.

    ``labels`` must be binary values produced by the M07 oracle target constructor;
    this loss never constructs labels from verifier predictions or MCTS backups.
    """
    labels = labels.to(dtype=torch.float32)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if not torch.logical_or(labels == 0, labels == 1).all():
        raise ValueError("grounded verifier labels must be binary 0/1")
    logits = verifier.forward_logits(x, y_state, z_state, width)
    loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits))
    p = torch.sigmoid(logits.detach())
    return loss, {
        "bce": float(loss.detach()),
        "mean_probability": float(p.mean()),
        "positive_rate": float(labels.mean()),
    }


def latent_prm_loss(
    verifier,
    x: torch.Tensor,
    z_states: torch.Tensor,
    value_targets: torch.Tensor,
    width: int = 9,
    visit_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Regress **MCTS-bootstrapped** per-node values.

    The targets are ``q=W/N`` values produced by the current search/verifier loop.
    They are self-training signals of kind ``mcts_bootstrap_value_v1`` and must not
    be presented as independent verifier ground truth.  M07 acceptance never uses
    this loss or these targets as its grounding evidence.
    """
    x_rep = x.expand(z_states.shape[0], -1)
    pred = verifier.value(x_rep, z_states, width)
    sq_err = (pred - value_targets.to(pred)) ** 2
    if visit_weights is not None:
        w = visit_weights.to(pred) / visit_weights.sum().clamp_min(1e-8)
        loss = (w * sq_err).sum()
    else:
        loss = sq_err.mean()
    return loss, {"prm_mse": float(loss.detach())}


def alphazero_distillation_loss(
    student_policy_logits: torch.Tensor,
    student_value: torch.Tensor,
    search_policy: torch.Tensor,
    search_value: torch.Tensor,
    answer_logits: torch.Tensor | None = None,
    y_target: torch.Tensor | None = None,
    lambda_policy: float = 1.0,
    lambda_value: float = 1.0,
    lambda_answer: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """AlphaZero-style distillation from search outputs."""
    logp = F.log_softmax(student_policy_logits, dim=-1)
    policy_loss = -(search_policy.detach() * logp).sum(dim=-1).mean()
    value_loss = F.mse_loss(student_value, search_value.detach())

    total = lambda_policy * policy_loss + lambda_value * value_loss
    comp = {"policy": float(policy_loss.detach()), "value": float(value_loss.detach())}
    if answer_logits is not None and y_target is not None:
        ce = F.cross_entropy(answer_logits.reshape(-1, answer_logits.size(-1)), y_target.reshape(-1))
        total = total + lambda_answer * ce
        comp["answer_ce"] = float(ce.detach())
    comp["total"] = float(total.detach())
    return total, comp


def system1_distillation_loss(
    student_logits: torch.Tensor,
    student_conf_logit: torch.Tensor,
    teacher_logits: torch.Tensor,
    y_target: torch.Tensor,
    lambda_kl: float = 1.0,
    lambda_conf: float = 0.5,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Distil the recursive teacher into the feed-forward System 1 student."""
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)), y_target.reshape(-1)
    )

    tau = temperature
    s_logp = F.log_softmax(student_logits / tau, dim=-1)
    t_p = F.softmax(teacher_logits.detach() / tau, dim=-1)
    kl = (t_p * (torch.log(t_p + 1e-9) - s_logp)).sum(dim=-1).mean() * (tau * tau)

    correct = (student_logits.argmax(-1) == y_target).all(dim=1).float()
    conf_bce = F.binary_cross_entropy_with_logits(student_conf_logit, correct)

    total = ce + lambda_kl * kl + lambda_conf * conf_bce
    return total, {
        "ce": float(ce.detach()),
        "kl": float(kl.detach()),
        "conf_bce": float(conf_bce.detach()),
        "total": float(total.detach()),
    }
