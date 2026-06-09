"""Fix 1 tests: GAE-lambda dense, verifier-bootstrapped actor-critic.

Proves the router/halter is trained with per-step temporal-difference credit
(GAE) bootstrapped from the verifier value -- not a single terminal REINFORCE
advantage (BLUEPRINT section 5.5).
"""

import torch

from model.lazy_router import LatentValueHead, RLTokenRouter
from model.trm import TRM
from train.rl import (
    compute_gae,
    dense_actor_critic_loss,
    dense_step_rewards,
    make_target_critic,
    rollout_router_gae,
    soft_update,
)


def test_gae_matches_hand_computation():
    # gamma=lam=1 -> GAE returns equal Monte-Carlo return-to-go (terminal at K-1).
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [1.0], [0.5], [0.0]])  # [K+1, B]
    adv, ret = compute_gae(rewards, values, gamma=1.0, lam=1.0, last_is_terminal=True)
    assert torch.allclose(adv, torch.tensor([[2.5], [1.0], [1.5]]), atol=1e-6)
    assert torch.allclose(ret, torch.tensor([[3.0], [2.0], [2.0]]), atol=1e-6)


def test_gae_advantages_are_per_step_not_constant():
    torch.manual_seed(0)
    rewards = torch.randn(6, 4)
    values = torch.randn(7, 4)
    adv, _ = compute_gae(rewards, values, gamma=0.99, lam=0.95)
    # Dense credit: advantages differ across steps (a terminal REINFORCE scalar
    # would make every step identical).
    assert adv.shape == (6, 4)
    assert adv.std(dim=0).mean() > 1e-3


def test_dense_step_rewards_include_verifier_delta_and_penalties():
    verifier_values = torch.tensor([[0.0], [0.5], [0.3], [0.8]])  # [K+1, B]
    density = torch.tensor([[0.5], [0.2], [0.1]])                 # [K, B]
    r = dense_step_rewards(verifier_values, density, lambda_tokens=0.1, lambda_step=0.01)
    expected = torch.tensor([[0.44], [-0.23], [0.48]])  # delta - 0.1*dens - 0.01
    assert torch.allclose(r, expected, atol=1e-6)


def test_actor_critic_value_head_learns_returns():
    torch.manual_seed(0)
    head = LatentValueHead(dim=16)
    z = torch.randn(3, 5, 16)  # one latent state, B=3... use as [K,B,D]? here K=3,B=5
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    returns = torch.randn(3, 5)
    losses = []
    for _ in range(50):
        critic = torch.stack([head(z[k]) for k in range(3)])  # [3, 5]
        adv = torch.zeros_like(returns)
        loss, comp = dense_actor_critic_loss(
            step_logprobs=torch.zeros(3, 5), step_entropies=torch.zeros(3, 5),
            advantages=adv, returns=returns, critic_values=critic,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(comp["value"])
    assert losses[-1] < losses[0] * 0.5  # critic regresses toward the returns


def test_dense_rollout_end_to_end_gives_per_step_grads():
    torch.manual_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=4, max_grid_size=8)
    router = RLTokenRouter(dim=32, device_dim=6)
    router.train()
    value_head = LatentValueHead(dim=32, device_dim=6)

    x = torch.randint(0, 5, (4, 16))
    calls = {"n": 0}

    def verifier_value_fn(xx, z):
        calls["n"] += 1
        return -z.pow(2).mean(dim=(1, 2))  # placeholder latent value (Task 2 plugs E_psi)

    loss, comp, tele = rollout_router_gae(
        model, router, value_head, x, height=4, width=4,
        verifier_value_fn=verifier_value_fn,
        terminal_reward=torch.ones(4),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert calls["n"] == model.N_sup + 1  # verifier valued every latent incl. bootstrap
    assert router.policy[0].weight.grad is not None      # actor gets gradient
    assert value_head.net[0].weight.grad is not None     # critic gets gradient
    # Per-step advantages with real spread (not a single terminal scalar).
    assert tele["advantages"].shape == (model.N_sup, 4)
    assert tele["advantages"].std(dim=0).mean() > 1e-4


def test_target_critic_is_frozen_and_polyak_tracks():
    online = LatentValueHead(dim=16)
    target = make_target_critic(online)
    # Target is a frozen copy.
    assert all(not p.requires_grad for p in target.parameters())
    w0 = next(target.parameters()).clone()
    # Move the online net, then Polyak-update the target a little toward it.
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    soft_update(target, online, tau=0.1)
    w1 = next(target.parameters())
    moved = (w1 - w0).abs().mean().item()
    assert 0.0 < moved < 1.0  # tracks online, but slowly (stationary target)


def test_dense_rl_is_stable_with_target_network():
    """Multi-step dense RL with a target critic does not diverge (non-stationary
    trap defused). The critic value loss stays finite and bounded."""
    torch.manual_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=4, max_grid_size=8)
    router = RLTokenRouter(dim=32, device_dim=6)
    router.train()
    value_head = LatentValueHead(dim=32, device_dim=6)
    target = make_target_critic(value_head)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(router.parameters()) + list(value_head.parameters()),
        lr=3e-3,
    )
    x = torch.randint(0, 5, (8, 16))

    def verifier_value_fn(xx, z):
        return -z.pow(2).mean(dim=(1, 2))

    value_losses = []
    for _ in range(40):
        loss, comp, _ = rollout_router_gae(
            model, router, value_head, x, 4, 4, verifier_value_fn,
            target_value_head=target, terminal_reward=torch.ones(8),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
        opt.step()
        soft_update(target, value_head, tau=0.02)
        value_losses.append(comp["value"])

    import math as _m

    assert all(_m.isfinite(v) for v in value_losses)         # never NaN/Inf
    assert max(value_losses) < 1e3                           # bounded (no blow-up)
    # The critic regresses: late value loss is no worse than the early peak.
    assert sum(value_losses[-5:]) / 5 <= max(value_losses[:5]) + 1e-6
