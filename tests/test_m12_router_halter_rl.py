"""M12 focused contracts for grounded router/halter RL."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from eval.adaptive_rl_checkpoint import (
    load_adaptive_rl_checkpoint,
    save_adaptive_rl_checkpoint,
)
from model.grounded_verifier import GroundedStateVerifier
from model.halting import DEVICE_DIM, HaltingPolicy, halting_episode
from model.lazy_router import LatentValueHead, RLTokenRouter
from model.trm import TRM
from scripts.train_adaptive_rl import ownership_audit
from train.rl import (
    compute_gae,
    grounded_actor_critic_loss,
    grounded_step_rewards,
    make_target_critic,
    masked_policy_objective,
    soft_update,
)


def test_terminal_gae_matches_hand_computation_and_zeros_bootstrap():
    rewards = torch.tensor([[1.0], [2.0]])
    # The absurd final value proves the terminal transition is not bootstrapped.
    values = torch.tensor([[0.5], [0.4], [9.0]])
    valid = torch.tensor([[1], [1]], dtype=torch.bool)
    terminated = torch.tensor([[0], [1]], dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    adv, ret = compute_gae(
        rewards, values, gamma=1.0, lam=1.0,
        valid_mask=valid, terminated=terminated, truncated=truncated,
    )
    assert torch.allclose(adv, torch.tensor([[2.5], [1.6]]), atol=1e-7)
    assert torch.allclose(ret, torch.tensor([[3.0], [2.0]]), atol=1e-7)


def test_time_limit_truncation_bootstraps_but_ends_trace():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.4], [0.3]])
    valid = torch.tensor([[1], [1]], dtype=torch.bool)
    terminated = torch.zeros_like(valid)
    truncated = torch.tensor([[0], [1]], dtype=torch.bool)
    adv, ret = compute_gae(
        rewards, values, gamma=1.0, lam=1.0,
        valid_mask=valid, terminated=terminated, truncated=truncated,
    )
    # k=1: 2 + 0.3 - 0.4 = 1.9.  Truncation keeps this bootstrap, but
    # trace continuation beyond k=1 is zero.  k=0 then includes that sampled step.
    assert torch.allclose(adv, torch.tensor([[2.8], [1.9]]), atol=1e-7)
    assert torch.allclose(ret, torch.tensor([[3.3], [2.3]]), atol=1e-7)


def test_mixed_batch_early_terminal_masks_post_episode_slots():
    rewards = torch.tensor([[1.0, 0.1], [99.0, 0.2]])
    values = torch.tensor([[0.5, 0.5], [7.0, 0.4], [8.0, 0.3]])
    valid = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
    terminated = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
    truncated = torch.tensor([[0, 0], [0, 1]], dtype=torch.bool)
    adv, ret = compute_gae(
        rewards, values, gamma=1.0, lam=1.0,
        valid_mask=valid, terminated=terminated, truncated=truncated,
    )
    # Example 0 ends immediately. Its giant invalid reward/value at k=1 is ignored.
    assert adv[0, 0].item() == torch.tensor(0.5).item()
    assert adv[1, 0].item() == 0.0
    assert ret[1, 0].item() == 0.0
    # Example 1: final truncation bootstraps 0.3.
    assert torch.allclose(adv[:, 1], torch.tensor([0.3, 0.1]), atol=1e-7)


def test_discounted_potential_shaping_terminal_and_truncation_treatment():
    phi = torch.tensor([[0.2], [0.5], [0.8]])
    dens = torch.zeros(2, 1)
    valid = torch.ones(2, 1, dtype=torch.bool)
    success = torch.zeros_like(valid)
    bad_halt = torch.zeros_like(valid)

    term = torch.tensor([[0], [1]], dtype=torch.bool)
    trunc = torch.zeros_like(term)
    reward, parts = grounded_step_rewards(
        phi, dens, valid, term, trunc, success, bad_halt,
        gamma=0.9, success_reward=0.0, halt_failure_penalty=0.0,
        lambda_step=0.0, lambda_token=0.0,
    )
    # 0.9*0.5-0.2 = 0.25; terminal next potential is exactly zero: 0-0.5=-0.5.
    assert torch.allclose(parts["potential_shaping"], torch.tensor([[0.25], [-0.5]]), atol=1e-7)
    assert torch.allclose(reward, parts["potential_shaping"], atol=1e-7)

    term2 = torch.zeros_like(term)
    trunc2 = torch.tensor([[0], [1]], dtype=torch.bool)
    _, parts2 = grounded_step_rewards(
        phi, dens, valid, term2, trunc2, success, bad_halt,
        gamma=0.9, success_reward=0.0, halt_failure_penalty=0.0,
        lambda_step=0.0, lambda_token=0.0,
    )
    # Truncation is not absorbing: 0.9*0.8-0.5 = 0.22.
    assert torch.allclose(parts2["potential_shaping"], torch.tensor([[0.25], [0.22]]), atol=1e-7)


def test_forced_action_logprob_has_zero_actor_credit():
    lp1 = torch.tensor([[100.0], [0.3]], requires_grad=True)
    lp2 = torch.tensor([[-700.0], [0.3]], requires_grad=True)
    entropy = torch.zeros_like(lp1)
    advantage = torch.tensor([[1000.0], [2.0]])
    decision = torch.tensor([[0], [1]], dtype=torch.bool)
    loss1, _, _ = masked_policy_objective(lp1, entropy, advantage, decision, normalize_adv=False)
    loss2, _, _ = masked_policy_objective(lp2, entropy, advantage, decision, normalize_adv=False)
    assert torch.equal(loss1, loss2)
    loss1.backward()
    assert lp1.grad is not None
    assert lp1.grad[0, 0].item() == 0.0
    assert lp1.grad[1, 0].item() != 0.0


def test_historical_halting_helper_does_not_credit_forced_final_stop():
    policy = HaltingPolicy(dim=8)
    with torch.no_grad():
        for p in policy.parameters(): p.zero_()  # p(halt)=0.5
    y_steps = torch.zeros(2, 1, 3, 8)
    d = torch.zeros(1, DEVICE_DIM)
    halt_step, logp = halting_episode(policy, y_steps, d, sample=False)
    assert halt_step.item() == 1
    # Only step 0 was a policy decision: log P(continue)=log(0.5).
    assert torch.allclose(logp, torch.tensor([torch.log(torch.tensor(0.5))]), atol=1e-7)


def test_optimizer_gradient_ownership_is_exact():
    reasoner = TRM(dim=8, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1, N_sup=2, heads=2, max_grid_size=8)
    verifier = GroundedStateVerifier(num_tokens=5, dim=8, n_layers=1, heads=2, max_grid_size=8)
    for p in reasoner.parameters(): p.requires_grad_(False)
    for p in verifier.parameters(): p.requires_grad_(False)
    reasoner.eval(); verifier.eval()
    router = RLTokenRouter(8)
    halter = HaltingPolicy(8)
    critic = LatentValueHead(8)
    target = make_target_critic(critic)
    opt = torch.optim.AdamW(
        list(router.parameters()) + list(halter.parameters()) + list(critic.parameters()), lr=1e-3
    )
    audit = ownership_audit(
        router=router, halter=halter, critic=critic, target=target,
        reasoner=reasoner, verifier=verifier, optimizer=opt,
    )
    assert audit["optimizer_exactly_intended"]
    assert audit["optimizer_forbidden_overlap"] == 0
    assert audit["target_requires_grad_false"]

    # Backprop a synthetic joint objective: no forbidden gradient may appear.
    k, b = 2, 3
    z = torch.randn(b, 4, 8)
    d = torch.zeros(b, DEVICE_DIM)
    router_dist = router.action_distribution(z, d)
    ra = router_dist.sample()
    rlp = router_dist.log_prob(ra).sum(1).repeat(k, 1)
    rent = router_dist.entropy().mean(1).repeat(k, 1)
    ha, hlp0, hent0 = halter.act(z, d, sample=True)
    hlp = hlp0.repeat(k, 1); hent = hent0.repeat(k, 1)
    critic_values = torch.stack([critic(z, d) for _ in range(k)])
    adv = torch.randn(k, b); ret = torch.randn(k, b); mask = torch.ones(k, b, dtype=torch.bool)
    loss, _ = grounded_actor_critic_loss(
        router_logprobs=rlp, router_entropies=rent, router_decisions=mask,
        halter_logprobs=hlp, halter_entropies=hent, halter_decisions=mask,
        advantages=adv, returns=ret, critic_values=critic_values, valid_mask=mask,
    )
    opt.zero_grad(set_to_none=True); loss.backward()
    assert any(p.grad is not None for p in router.parameters())
    assert any(p.grad is not None for p in halter.parameters())
    assert any(p.grad is not None for p in critic.parameters())
    assert all(p.grad is None for p in target.parameters())
    assert all(p.grad is None for p in reasoner.parameters())
    assert all(p.grad is None for p in verifier.parameters())


def test_target_critic_changes_only_via_polyak_and_remains_gradient_free():
    critic = LatentValueHead(8)
    target = make_target_critic(critic)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    with torch.no_grad():
        for p in critic.parameters(): p.add_(1.0)
    assert all(torch.equal(v, before[k]) for k, v in target.state_dict().items())
    soft_update(target, critic, tau=0.2)
    assert any(not torch.equal(v, before[k]) for k, v in target.state_dict().items())
    assert all(not p.requires_grad for p in target.parameters())
    assert all(p.grad is None for p in target.parameters())


def test_strict_m12_checkpoint_roundtrip(tmp_path):
    dim = 8
    core = SimpleNamespace(
        sha256="a" * 64,
        architecture={"dim": dim, "num_tokens": 5, "seq_len": 16},
        task={"task": "sudoku"},
        device=torch.device("cpu"),
    )
    verifier = SimpleNamespace(sha256="b" * 64)
    router = RLTokenRouter(dim); halter = HaltingPolicy(dim)
    critic = LatentValueHead(dim); target = make_target_critic(critic)
    path = tmp_path / "m12.pt"
    objective = {
        "cost_kind": "logical_step_token_proxy_v1",
        "measured_energy_used": False,
        "measured_energy_joules": None,
    }
    save_adaptive_rl_checkpoint(
        path,
        router=router, halter=halter, critic=critic, target_critic=target,
        core=core, verifier=verifier, trained_steps=3, seed=1,
        objective=objective, ownership={"test": True}, tensor_hashes={}, training_summary={},
    )
    loaded = load_adaptive_rl_checkpoint(path, core=core, verifier=verifier)
    assert loaded.payload["format"] == "spectra.adaptive_rl"
    assert loaded.payload["kind"] == "router_halter_actor_critic"
    assert loaded.payload["objective"]["measured_energy_used"] is False
    for a, b in zip(router.state_dict().values(), loaded.router.state_dict().values()):
        assert torch.equal(a, b)
