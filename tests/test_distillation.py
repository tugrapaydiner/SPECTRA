"""Phase 10 gate: System 1 student, dual-mode reasoning, self-play flywheel.

Fast: System 1 forward/predict, distillation gradient, dual-mode routing, and the
flywheel quality gate. Slow gate: a distilled System 1 solves easy instances
cheaply in one pass, while the dual-mode controller escalates mainly the uncertain
(hard) instances to System 2 (BLUEPRINT sections 4, 10).
"""

import numpy as np
import pytest
import torch

from data import sudoku as sk
from data.datasets import build_sudoku_arrays
from data.generate import SyntheticSudokuGenerator, self_play_flywheel
from model.system1_student import DualModeReasoner, System1Student
from model.verifier import sudoku_correct
from train.distill import system1_distillation_loss


# --------------------------------------------------------------------------- #
# Fast unit tests
# --------------------------------------------------------------------------- #
def test_system1_forward_and_predict():
    s1 = System1Student(dim=32, num_tokens=5, seq_len=16, n_layers=2, max_grid_size=8)
    x = torch.randint(0, 5, (4, 16))
    logits, conf_logit = s1(x, height=4, width=4)
    assert logits.shape == (4, 16, 5) and conf_logit.shape == (4,)
    ans, conf = s1.predict(x, 4, 4)
    assert ans.shape == (4, 16)
    assert ((conf >= 0) & (conf <= 1)).all()


def test_system1_distillation_gradient():
    torch.manual_seed(0)
    s1 = System1Student(dim=32, num_tokens=5, seq_len=16, n_layers=2, max_grid_size=8)
    x = torch.randint(0, 5, (4, 16))
    y = torch.randint(1, 5, (4, 16))
    logits, conf_logit = s1(x, 4, 4)
    teacher_logits = torch.randn(4, 16, 5)
    loss, comp = system1_distillation_loss(logits, conf_logit, teacher_logits, y)
    loss.backward()
    assert {"ce", "kl", "conf_bce", "total"} <= set(comp)
    assert s1.out_head.weight.grad is not None
    assert s1.conf_head.weight.grad is not None


def test_dual_mode_routes_by_confidence():
    """System 2 runs only on escalated (low-confidence) rows."""
    torch.manual_seed(0)
    s1 = System1Student(dim=32, num_tokens=5, seq_len=16, n_layers=2, max_grid_size=8)
    calls = {}

    def sys2(x, h, w):
        calls["n"] = x.shape[0]
        return torch.zeros(x.shape[0], 16, dtype=torch.long)

    x = torch.randint(0, 5, (8, 16))
    # threshold > 1: sigmoid confidence always < threshold -> escalate all.
    answer, esc = DualModeReasoner(s1, sys2, threshold=2.0, height=4, width=4)(x)
    assert esc.all() and calls["n"] == 8

    # threshold 0: confidence never < 0 -> escalate none, System 2 never runs.
    calls.clear()
    _, esc0 = DualModeReasoner(s1, sys2, threshold=0.0, height=4, width=4)(x)
    assert not esc0.any() and "n" not in calls


def test_flywheel_admits_only_verified():
    gen = SyntheticSudokuGenerator(box=2, min_clue_frac=0.4, max_clue_frac=0.6)
    rng = np.random.default_rng(0)

    def perfect_solver(puzzles):
        out = [sk.solve(p.reshape(4, 4), 2).reshape(-1) for p in puzzles.cpu().numpy()]
        return torch.from_numpy(np.stack(out))

    inp, tgt, stats = self_play_flywheel(gen, perfect_solver, 20, rng, difficulty=0.5)
    assert stats["admitted"] == 20 and stats["admit_rate"] == 1.0
    assert inp.shape == tgt.shape == (20, 16)

    def broken_solver(puzzles):  # returns all-ones -> never a valid board
        return torch.ones(puzzles.shape[0], 16, dtype=torch.long)

    _, _, stats_bad = self_play_flywheel(gen, broken_solver, 20, rng, difficulty=0.5)
    assert stats_bad["admitted"] == 0


# --------------------------------------------------------------------------- #
# Slow gate
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_system1_cheap_and_system2_on_uncertain(trained_sudoku_4x4):
    """System 1 solves easy puzzles cheaply; dual-mode escalates the hard ones."""
    d = trained_sudoku_4x4
    teacher, device, h, w = d["model"], d["device"], d["height"], d["width"]
    box = 2

    # Mixed-difficulty distillation set so System 1 learns calibrated confidence.
    rng = np.random.default_rng(1)
    xs, ys = [], []
    for clues in (4, 5, 6, 7, 8, 9, 10, 11, 12):
        a, b, _, _ = build_sudoku_arrays(box, 120, clues, rng, require_unique=True, augment=True)
        xs.append(a)
        ys.append(b)
    tx = torch.from_numpy(np.concatenate(xs)).to(device)
    ty = torch.from_numpy(np.concatenate(ys)).to(device)

    # Cache teacher (System 2) logits once -> fast distillation.
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(tx, height=h, width=w)[0]

    student = System1Student(dim=64, num_tokens=5, seq_len=16, n_layers=3, max_grid_size=8).to(device)
    sopt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    student.train()
    for _ in range(400):
        idx = torch.randint(0, tx.shape[0], (128,), device=device)
        logits, conf_logit = student(tx[idx], h, w)
        loss, _ = system1_distillation_loss(logits, conf_logit, teacher_logits[idx], ty[idx])
        sopt.zero_grad()
        loss.backward()
        sopt.step()
    student.eval()

    # Easy (many clues) vs hard (few clues) evaluation sets.
    ex = torch.from_numpy(build_sudoku_arrays(box, 128, 11, np.random.default_rng(7), True, False)[0]).to(device)
    hx = torch.from_numpy(build_sudoku_arrays(box, 128, 4, np.random.default_rng(8), True, False)[0]).to(device)

    threshold = 0.7
    with torch.no_grad():
        e_ans, e_conf = student.predict(ex, h, w)
        h_ans, h_conf = student.predict(hx, h, w)
    easy_board = sudoku_correct(ex, e_ans, box).float().mean().item()
    easy_esc = (e_conf < threshold).float().mean().item()
    hard_esc = (h_conf < threshold).float().mean().item()
    print(f"\nSystem1 easy_board={easy_board:.3f} easy_escalate={easy_esc:.3f} hard_escalate={hard_esc:.3f}")

    # System 1 solves easy instances cheaply (single feed-forward pass).
    assert easy_board > 0.8, easy_board
    # System 2 activates mainly on the uncertain (hard) instances.
    assert easy_esc < 0.25
    assert hard_esc > 0.4
    assert hard_esc > easy_esc + 0.25
