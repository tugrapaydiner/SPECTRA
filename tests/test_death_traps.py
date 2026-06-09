"""AGI death-trap fixes: bounded latent error, virtual-loss MCTS, visit-count
distillation, and CPU-frequency fidelity.
"""

import torch

from eval import cpufreq
from eval.latent_mcts import LatentNativeMCTS
from model.energy import LatentEnergyVerifier
from model.latent_action import LatentActionCodebook
from model.latent_vq import LatentVQ
from model.trm import TRM
from train.distill import alphazero_distillation_loss


# --------------------------------------------------------------------------- #
# DT1 -- Latent VQ bounds compounding recursion/INT8 error
# --------------------------------------------------------------------------- #
def test_latent_vq_snap_on_codebook_and_idempotent():
    torch.manual_seed(0)
    vq = LatentVQ(dim=16, codebook_size=64, decay=0.0)
    z = torch.randn(2, 5, 16)
    zq = vq.snap(z)
    assert torch.allclose(vq.snap(zq), zq)                # idempotent projection
    # every snapped token equals some codebook row
    flat = zq.reshape(-1, 16)
    nearest = vq.codebook[(torch.cdist(flat, vq.codebook)).argmin(1)]
    assert torch.allclose(flat, nearest, atol=1e-5)


def test_latent_vq_bounds_error_while_free_recursion_explodes():
    torch.manual_seed(0)
    vq = LatentVQ(dim=16, codebook_size=128, decay=0.0)
    A = 1.25 * torch.eye(16) + 0.1 * torch.randn(16, 16)  # spectral radius > 1 (expansive)

    z_free = torch.randn(1, 1, 16)
    z_snap = z_free.clone()
    for _ in range(30):
        noise = 0.05 * torch.randn(1, 1, 16)
        z_free = z_free @ A.T + noise                     # O(L^d): unbounded
        z_snap = vq.snap(z_snap @ A.T + noise)            # O(r): projected each step

    cap = vq.codebook.abs().max().item()
    assert z_snap.abs().max().item() <= cap + 1e-4        # bounded by codebook, any depth
    assert z_free.abs().max().item() > 50 * (cap + 1e-4)  # free trajectory exploded


# --------------------------------------------------------------------------- #
# DT2 -- Virtual-loss leaf-parallel MCTS (batched verifier payload)
# --------------------------------------------------------------------------- #
def test_search_batched_uses_virtual_loss_and_batches_the_verifier():
    torch.manual_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=3, T=1, max_grid_size=8)
    verifier = LatentEnergyVerifier(num_tokens=5, dim=32, n_layers=1, max_grid_size=8)
    codebook = LatentActionCodebook(dim=32, n_actions=3)

    counter = {"calls": 0, "max_batch": 0}
    orig_value = verifier.value

    def counting_value(x, z, width=9):
        counter["calls"] += 1
        counter["max_batch"] = max(counter["max_batch"], z.shape[0])
        return orig_value(x, z, width)

    verifier.value = counting_value
    mcts = LatentNativeMCTS(model, verifier, codebook, 4, 4, n_rollouts=16)

    node = mcts.search_batched(torch.randint(0, 5, (1, 16)), leaf_batch=8)
    assert mcts.decode(node).shape == (1, 16)
    assert counter["max_batch"] >= 8       # leaves evaluated as a single SIMD payload
    assert counter["calls"] <= 4           # batched -> far fewer than 16 serial calls


# --------------------------------------------------------------------------- #
# DT3 -- AlphaZero visit-count distillation (not behavioral cloning)
# --------------------------------------------------------------------------- #
def test_root_visit_policy_is_a_distribution():
    torch.manual_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=3, T=1, max_grid_size=8)
    verifier = LatentEnergyVerifier(num_tokens=5, dim=32, n_layers=1, max_grid_size=8)
    codebook = LatentActionCodebook(dim=32, n_actions=4)
    mcts = LatentNativeMCTS(model, verifier, codebook, 4, 4, n_rollouts=12)
    mcts.search(torch.randint(0, 5, (1, 16)))
    pi = mcts.root_visit_policy(temperature=1.0)
    assert pi.shape == (4,)
    assert torch.allclose(pi.sum(), torch.tensor(1.0), atol=1e-6)


def test_alphazero_loss_distils_full_visit_distribution():
    torch.manual_seed(0)
    target_policy = torch.tensor([[0.10, 0.60, 0.20, 0.10]])  # MCTS visit distribution
    target_value = torch.tensor([0.5])
    logits = torch.zeros(1, 4, requires_grad=True)
    value = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([logits, value], lr=0.1)
    for _ in range(300):
        loss, _ = alphazero_distillation_loss(logits, value, target_policy, target_value)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # The student internalises the WHOLE search distribution, not just the argmax.
    assert torch.allclose(torch.softmax(logits, -1), target_policy, atol=0.05)
    assert abs(value.item() - 0.5) < 0.05


# --------------------------------------------------------------------------- #
# DT4 -- CPU frequency pinning is graceful where unavailable
# --------------------------------------------------------------------------- #
def test_cpufreq_pinning_is_graceful():
    state = cpufreq.read_state()
    assert "available" in state
    if not cpufreq.cpufreq_available():           # e.g. Windows / no /sys
        assert state["available"] is False
        with cpufreq.pinned_frequency() as status:
            assert status["pinned"] is False
            assert "unavailable" in status["reason"]
        assert cpufreq.available_governors() == []
