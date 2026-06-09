"""Fix 3 tests: open-ended self-play flywheel.

Proves the curriculum samples the *frontier of competence* (ZPD: System 1 fails,
System 2 solves) and that the frontier *advances* as System 1 improves -- i.e.
it is open-ended, not a static fixed-difficulty loop (BLUEPRINT section 10).
"""

import numpy as np
import pytest
import torch

from train.flywheel import (
    AdversarialValidator,
    CurriculumState,
    FlywheelBuffer,
    SelfPlayFlywheel,
)


def test_curriculum_targets_the_zpd_frontier():
    cs = CurriculumState([0.1, 0.3, 0.5, 0.7, 0.9], ema=0.5, floor=0.01)
    for _ in range(20):
        cs.update(0, s1_solved=1.0, s2_solved=1.0)  # mastered (S1 wins)
        cs.update(1, s1_solved=0.9, s2_solved=1.0)  # nearly mastered
        cs.update(2, s1_solved=0.1, s2_solved=1.0)  # ZPD: S1 fails, S2 solves
        cs.update(3, s1_solved=0.0, s2_solved=0.2)  # mostly unsolvable
        cs.update(4, s1_solved=0.0, s2_solved=0.0)  # unsolvable
    w = cs.frontier_weights()
    assert w.argmax() == 2                  # peak weight on the ZPD bucket
    assert w[0] < w[2] and w[4] < w[2]      # mastered & unsolvable down-weighted


def test_curriculum_advances_as_system1_improves():
    cs = CurriculumState([0.2, 0.5, 0.8], ema=0.5, floor=0.01)
    for _ in range(20):  # phase 1: S1 only solves easy
        cs.update(0, 1.0, 1.0)
        cs.update(1, 0.1, 1.0)
        cs.update(2, 0.0, 0.4)
    early = cs.mean_sampled_difficulty()
    for _ in range(20):  # phase 2: S1 now also masters mid
        cs.update(0, 1.0, 1.0)
        cs.update(1, 1.0, 1.0)
        cs.update(2, 0.2, 1.0)
    late = cs.mean_sampled_difficulty()
    assert late > early  # frontier moved toward harder tasks


def test_flywheel_buffer_reservoir_retains_easy_after_hard():
    """Reservoir sampling keeps old easy tasks instead of evicting them (FIFO ring
    would discard all of them) -- the anti-catastrophic-forgetting guarantee."""
    buf = FlywheelBuffer(capacity=40, seed=0)
    buf.add(np.zeros((100, 4), int), np.zeros((100, 4), int), difficulty=0.1)  # easy
    buf.add(np.ones((100, 4), int), np.ones((100, 4), int), difficulty=0.9)    # hard
    assert len(buf) == 40
    # A uniform sample of all 200 retains easy (0.1) examples; a ring buffer would
    # hold only the last 40 (all hard).
    assert min(buf.difficulties) < 0.5


def test_flywheel_buffer_stratified_sampling_mixes_difficulties():
    """A training batch spans difficulty bins, so System 1 keeps seeing easy tasks."""
    buf = FlywheelBuffer(capacity=200, seed=0)
    rng = np.random.default_rng(0)
    buf.add(np.full((50, 4), 1, int), np.full((50, 4), 1, int), difficulty=0.1)  # easy marker
    buf.add(np.full((50, 4), 9, int), np.full((50, 4), 9, int), difficulty=0.9)  # hard marker
    xb, _ = buf.sample(32, rng)
    markers = set(xb[:, 0].tolist())
    assert 1 in markers and 9 in markers  # batch mixes easy + hard


def test_adversarial_validator_catches_clever_hans():
    from data import sudoku as sk

    rng = np.random.default_rng(0)
    puzzles, sols = [], []
    for _ in range(16):
        p, s = sk.generate_pair(2, num_clues=8, rng=rng, require_unique=True)
        puzzles.append(p.reshape(-1))
        sols.append(s.reshape(-1))
    P = torch.from_numpy(np.stack(puzzles))
    S = torch.from_numpy(np.stack(sols))
    val = AdversarialValidator(box=2, device="cpu")

    def reasoner(p):  # genuine solver -> invariant to symmetry
        out = [sk.solve(row.numpy().reshape(4, 4), 2).reshape(-1) for row in p]
        return torch.from_numpy(np.stack(out))

    good = val.probe(P, S, reasoner, np.random.default_rng(1))
    assert good["invariance"] > 0.99 and good["n_counterexamples"] == 0

    # Clever Hans: memorizes the exact boards; fails on their symmetry images.
    memo = {tuple(P[i].tolist()): S[i] for i in range(len(P))}

    def hans(p):
        return torch.stack(
            [memo.get(tuple(r.tolist()), torch.zeros(16, dtype=torch.long)) for r in p]
        )

    bad = val.probe(P, S, hans, np.random.default_rng(1))
    assert bad["invariance"] < 0.5            # shortcut detected
    assert bad["n_counterexamples"] > 0       # symmetry images mined for training


@pytest.mark.slow
def test_flywheel_collects_verified_and_advances(monkeypatch):
    from data import sudoku as sk
    from data.generate import SyntheticSudokuGenerator
    from model.system1_student import System1Student
    from model.verifier import sudoku_correct

    torch.manual_seed(0)
    np.random.seed(0)
    box = 2
    gen = SyntheticSudokuGenerator(box=2, min_clue_frac=0.25, max_clue_frac=0.85)
    s1 = System1Student(dim=64, num_tokens=5, seq_len=16, n_layers=3, max_grid_size=8)

    def system2_solve(puzzles):  # oracle symbolic solver = strong System 2
        out = [sk.solve(p.reshape(4, 4), box).reshape(-1) for p in puzzles.cpu().numpy()]
        return torch.from_numpy(np.stack(out))

    cs = CurriculumState([0.2, 0.4, 0.6, 0.85], ema=0.7, floor=0.02)
    buf = FlywheelBuffer()
    fw = SelfPlayFlywheel(gen, s1, system2_solve, sudoku_correct, cs, buf, 4, 4, box, device="cpu")
    opt = torch.optim.AdamW(s1.parameters(), lr=2e-3)
    rng = np.random.default_rng(0)

    frontier = []
    for _ in range(80):
        stats = fw.step(opt, rng, n_tasks=32, batch=64)
        frontier.append(stats["frontier_difficulty"])

    assert len(buf) > 0  # verified System-2 trajectories were mined and stored
    # Open-endedness: the frontier difficulty rises as System 1 masters easy tasks.
    assert np.mean(frontier[-15:]) > np.mean(frontier[:15])
