"""Phase 8 gate: verifier-guided inference.

Fast: symbolic verifiers rank correct above wrong; energy head + contrastive loss
+ hard-negative mining work; best-of-N machinery selects correctly. Slow/training
gates: the neural energy verifier ranks correct > wrong > 80%, and best-of-N beats
single-candidate decoding (BLUEPRINT section 9).
"""

import numpy as np
import pytest
import torch

from common.seed import resolve_device, set_seed
from data import maze as mz
from data import sudoku as sk
from data.datasets import build_sudoku_arrays
from eval.best_of_n import decode_candidates, select_best
from eval.latent_mcts import LatentMCTS
from model.energy import EnergyVerifier, contrastive_energy_loss, mine_hard_negatives
from model.trm import TRM
from model.verifier import arc_soft_score, maze_score, sudoku_correct, sudoku_score
from train.losses import deep_supervision_loss


# --------------------------------------------------------------------------- #
# Symbolic verifiers
# --------------------------------------------------------------------------- #
def test_sudoku_verifier_ranks_correct_above_wrong():
    box, rng = 3, np.random.default_rng(0)
    puzzles, sols = [], []
    for _ in range(16):
        p, s = sk.generate_pair(box, num_clues=40, rng=rng, require_unique=False)
        puzzles.append(p.reshape(-1))
        sols.append(s.reshape(-1))
    puzzle = torch.from_numpy(np.stack(puzzles))
    sol = torch.from_numpy(np.stack(sols))
    neg = mine_hard_negatives(sol, num_tokens=10, n_changes=3)

    s_correct = sudoku_score(puzzle, sol, box)
    s_wrong = sudoku_score(puzzle, neg, box)
    assert (s_correct == 1.0).all()
    assert (s_correct > s_wrong).all()
    assert sudoku_correct(puzzle, sol, box).all()
    assert not sudoku_correct(puzzle, neg, box).any()


def test_maze_score_rewards_valid_overlay():
    rng = np.random.default_rng(0)
    grid, start, goal = mz.generate_maze(15, 15, rng)
    path = mz.shortest_path(grid, start, goal)
    x = torch.from_numpy(mz.make_input(grid, start, goal).reshape(1, -1))
    y = torch.from_numpy(mz.make_target(grid, start, goal, path).reshape(1, -1))

    good = maze_score(x, y, 15, 15)
    # Corrupt: paint a path token onto a wall cell -> lower score.
    bad = y.clone()
    wall_idx = (x[0] == mz.WALL).nonzero()[0, 0]
    bad[0, wall_idx] = mz.PATH
    worse = maze_score(x, bad, 15, 15)
    assert good.item() > worse.item()


def test_arc_soft_score_penalises_invalid_colors():
    valid = torch.randint(0, 10, (2, 16))
    score_valid = arc_soft_score(valid, pad_token=10)
    invalid = valid.clone()
    invalid[0, 0] = 99  # out of palette
    score_invalid = arc_soft_score(invalid, pad_token=10)
    assert score_valid.mean().item() >= score_invalid.mean().item()


# --------------------------------------------------------------------------- #
# Energy verifier + best-of-N machinery
# --------------------------------------------------------------------------- #
def test_energy_verifier_forward_and_contrastive_loss():
    ev = EnergyVerifier(num_tokens=5, dim=32, n_layers=1, max_grid_size=8)
    x = torch.randint(0, 5, (4, 16))
    y = torch.randint(1, 5, (4, 16))
    e = ev(x, y, height=4, width=4)
    assert e.shape == (4,)
    neg = mine_hard_negatives(y, num_tokens=5, n_changes=2)
    loss = contrastive_energy_loss(ev(x, y, 4, 4), ev(x, neg, 4, 4))
    loss.backward()
    assert ev.head.weight.grad is not None


def test_hard_negatives_never_blank_and_differ():
    ans = torch.randint(1, 5, (8, 16))
    neg = mine_hard_negatives(ans, num_tokens=5, n_changes=2)
    assert (neg >= 1).all()  # never produces the blank token
    assert (neg != ans).sum() > 0  # actually changed something


def test_select_best_picks_highest_scoring():
    cands = torch.tensor([[[1, 1]], [[5, 5]], [[3, 3]]])  # [n=3, B=1, L=2]
    best, scores = select_best(cands, lambda c: c.sum(dim=-1).float())
    assert torch.equal(best, cands[1])  # candidate 1 has the largest sum
    assert scores.shape == (3, 1)


def test_decode_candidates_includes_greedy():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 5)
    cands = decode_candidates(logits, n=3, include_greedy=True)
    assert cands.shape == (3, 2, 4)
    assert torch.equal(cands[0], logits.argmax(dim=-1))


# --------------------------------------------------------------------------- #
# Training gates
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_energy_verifier_ranks_above_80pct():
    set_seed(0)
    device = resolve_device("auto")
    box, num_tokens = 3, 10

    def gen(num, rng):
        p = np.empty((num, 81), np.int64)
        s = np.empty((num, 81), np.int64)
        for i in range(num):
            pi, si = sk.generate_pair(box, num_clues=40, rng=rng, require_unique=False)
            p[i], s[i] = pi.reshape(-1), si.reshape(-1)
        return torch.from_numpy(p).to(device), torch.from_numpy(s).to(device)

    rng = np.random.default_rng(0)
    p_tr, s_tr = gen(768, rng)
    p_va, s_va = gen(256, rng)

    ev = EnergyVerifier(num_tokens=num_tokens, dim=64, n_layers=2).to(device)
    opt = torch.optim.AdamW(ev.parameters(), lr=1e-3)
    ev.train()
    for _ in range(300):
        idx = torch.randint(0, p_tr.shape[0], (128,), device=device)
        x, y_pos = p_tr[idx], s_tr[idx]
        y_neg = mine_hard_negatives(y_pos, num_tokens, n_changes=3)
        loss = contrastive_energy_loss(ev(x, y_pos, 9, 9), ev(x, y_neg, 9, 9))
        opt.zero_grad()
        loss.backward()
        opt.step()

    ev.eval()
    with torch.no_grad():
        y_neg = mine_hard_negatives(s_va, num_tokens, n_changes=3)
        rank_acc = (ev(p_va, s_va, 9, 9) < ev(p_va, y_neg, 9, 9)).float().mean().item()
    print(f"\nenergy verifier ranking acc={rank_acc:.3f}")
    assert rank_acc > 0.8, f"ranking acc {rank_acc:.3f} below 0.8"


@pytest.mark.slow
def test_best_of_n_beats_greedy():
    set_seed(0)
    device = resolve_device("auto")
    box = 2
    rng = np.random.default_rng(0)
    tx, ty, h, w = build_sudoku_arrays(box, 768, 8, rng, require_unique=True, augment=True)
    vx_np, vy_np, _, _ = build_sudoku_arrays(box, 256, 8, rng, require_unique=True, augment=True)
    tx = torch.from_numpy(tx).to(device)
    ty = torch.from_numpy(ty).to(device)
    vx = torch.from_numpy(vx_np).to(device)

    # Train only moderately so greedy leaves headroom for best-of-N.
    model = TRM(dim=64, num_tokens=5, seq_len=16, N_sup=3, max_grid_size=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(120):
        idx = torch.randint(0, tx.shape[0], (128,), device=device)
        _, steps = model(tx[idx], height=h, width=w)
        loss = deep_supervision_loss(steps, ty[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(vx, height=h, width=w)
    greedy_board = sudoku_correct(vx, logits.argmax(-1), box).float().mean().item()

    cands = decode_candidates(logits, n=16, temperature=1.0, include_greedy=True)
    best, _ = select_best(cands, lambda c: sudoku_score(vx, c, box))
    bon_board = sudoku_correct(vx, best, box).float().mean().item()
    print(f"\nbest-of-N: greedy_board={greedy_board:.3f} best_of_16={bon_board:.3f}")

    assert bon_board > greedy_board + 0.03, (greedy_board, bon_board)


# --------------------------------------------------------------------------- #
# Latent MCTS (section 9.6)
# --------------------------------------------------------------------------- #
def test_latent_mcts_runs_and_never_worse_than_greedy():
    """Latent MCTS returns a valid answer at least as good as greedy decoding."""
    torch.manual_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=4, max_grid_size=8)
    x = torch.randint(0, 5, (1, 16))
    value_fn = lambda xx, ans: sudoku_score(xx, ans, box=2).item()

    mcts = LatentMCTS(model, value_fn, height=4, width=4, n_rollouts=12, n_children=3)
    answer = mcts.search(x)
    assert answer.shape == (1, 16)

    with torch.no_grad():
        greedy = model(x, height=4, width=4)[0].argmax(dim=-1)
    # Greedy trajectory is in the tree, so MCTS value can only match or beat it.
    assert value_fn(x, answer) >= value_fn(x, greedy) - 1e-6


@pytest.mark.slow
def test_latent_mcts_matches_or_beats_greedy_trained(trained_sudoku_4x4):
    d = trained_sudoku_4x4
    model, vx = d["model"], d["vx"][:16]
    value_fn = lambda xx, ans: sudoku_score(xx, ans, box=2).item()

    with torch.no_grad():
        greedy = model(vx, height=d["height"], width=d["width"])[0].argmax(dim=-1)
    greedy_board = sudoku_correct(vx, greedy, 2).float().mean().item()

    mcts = LatentMCTS(model, value_fn, height=d["height"], width=d["width"], n_rollouts=24)
    answers = torch.cat([mcts.search(vx[i : i + 1]) for i in range(vx.shape[0])], dim=0)
    mcts_board = sudoku_correct(vx, answers, 2).float().mean().item()
    print(f"\nlatent MCTS: greedy_board={greedy_board:.3f} mcts_board={mcts_board:.3f}")

    assert mcts_board >= greedy_board - 1e-6
