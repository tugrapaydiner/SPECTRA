"""Teacher -> student trajectory distillation (BLUEPRINT sections 11, 10.4).

The Stability Shield trains the ternary/quantized *student* to imitate a stable
FP16 *teacher* -- not just its final answer, but its whole recursive trajectory.
The loss has three parts, per supervision step:

  * task: standard deep-supervision loss against the ground truth,
  * logit distillation: temperature-softened KL from the teacher's logits,
  * state distillation: MSE between student and teacher latent states (y, z),
    i.e. the section 10.4 ``lambda_z * ||z_hat - z||^2`` term.

The teacher is frozen (its tensors are detached), so gradients only update the
student.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from train.losses import deep_supervision_loss


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
    """Combined task + logit + state distillation loss.

    Args:
        student_steps: Student ``TRM.forward`` step outputs (carry gradient).
        teacher_steps: Teacher step outputs (treated as constants / detached).
        y_target: Ground-truth token ids ``[B, L]``.
        lambda_task / lambda_logit / lambda_state: Term weights.
        temperature: Softmax temperature for logit distillation.
        lambda_h / lambda_improve: Passed through to the task loss.

    Returns:
        ``(total_loss, components)`` where ``components`` holds the scalar parts.
    """
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
        # Per-token KL(teacher || student), averaged, scaled by tau^2.
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


def latent_prm_loss(
    verifier,
    x: torch.Tensor,
    z_states: torch.Tensor,
    value_targets: torch.Tensor,
    width: int = 9,
    visit_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """MCTS-bootstrapped **unsupervised Process Reward Model** training (BC #2).

    OpenAI's o1 scales on a PRM that scores intermediate reasoning steps, trained
    with expensive human step labels. SPECTRA bootstraps the PRM autonomously: the
    Monte-Carlo backup already assigns a value ``q = W/N`` to every intermediate
    latent ``z_k`` it visits (``LatentNativeMCTS.prm_targets``). Regressing the
    energy verifier ``V_psi(x, z)`` onto those backed-up per-step values turns the
    outcome verifier (ORM) into a **process** verifier (PRM) with NO human labels:

        L_PRM = sum_k w_k * ( V_psi(x, z_k) - q_k )^2 ,    w_k ∝ N_k (visit count)

    More-visited nodes have lower-variance targets, so visit weighting focuses the
    PRM on the parts of the tree the search actually trusted.

    Args:
        verifier: A latent value model exposing ``value(x, z, width) -> [M]``.
        x: The problem tokens ``[1, L]`` (broadcast over the M latents).
        z_states: Intermediate latents ``[M, L, D]`` (from ``prm_targets``).
        value_targets: Backed-up MCTS values ``[M]`` (the process rewards).
        visit_weights: Optional per-node visit counts ``[M]`` for weighting.
    """
    x_rep = x.expand(z_states.shape[0], -1)
    pred = verifier.value(x_rep, z_states, width)  # [M]
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
    """AlphaZero-style distillation: internalise the FULL MCTS search distribution.

    Cross-entropy on System 2's final answer alone is behavioral cloning -- it
    throws away 99% of the search signal and caps out. True self-play trains the
    policy head on the MCTS visit distribution ``pi(a) ∝ N(s,a)^{1/tau}`` and the
    value head on the search value ``v``:

        L = lambda_p * [ -sum_a pi(a) log p_student(a) ]   # policy: -pi^T log p
          + lambda_v * (v_student - v_search)^2            # value
          + lambda_a * CE(answer, y*)                      # (optional) grounded answer

    Args:
        student_policy_logits: Student policy logits ``[B, A]`` over codebook actions.
        student_value: Student value prediction ``[B]``.
        search_policy: MCTS visit distribution ``[B, A]`` (sums to 1, detached target).
        search_value: MCTS root value ``[B]`` (detached target).
        answer_logits / y_target: Optional grounded-answer CE term.
    """
    logp = F.log_softmax(student_policy_logits, dim=-1)
    policy_loss = -(search_policy.detach() * logp).sum(dim=-1).mean()  # AlphaZero policy CE
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
    """Distil the recursive teacher into the feed-forward System 1 student.

    The student learns to (a) predict the answer (cross-entropy vs ground truth),
    (b) match the teacher's softened logits (KL), and (c) predict whether its own
    answer is correct (confidence BCE) so dual-mode routing can trust it
    (BLUEPRINT section 10.4: "System 1: predict final answer directly").

    Args:
        student_logits: Student answer logits ``[B, L, V]``.
        student_conf_logit: Student confidence logit ``[B]``.
        teacher_logits: Teacher (System 2) answer logits ``[B, L, V]`` (detached).
        y_target: Ground-truth tokens ``[B, L]``.
        lambda_kl / lambda_conf: Term weights.
        temperature: Softmax temperature for the KL term.

    Returns:
        ``(total_loss, components)``.
    """
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
