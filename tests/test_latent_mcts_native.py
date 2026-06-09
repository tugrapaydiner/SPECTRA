"""Fix 2 tests: latent-native, cache-resident MCTS.

Proves: (a) the energy verifier scores the INT8 latent directly and is trainable;
(b) the action codebook is a principled discrete move set (not random noise);
(c) the MCTS never decodes during search (``out_head`` called 0 times in
``search``, exactly once in ``decode``) and keeps latents on the INT8 grid.
"""

import pytest
import torch
import torch.nn as nn

from eval.latent_mcts import LatentNativeMCTS
from model.energy import (
    EnsembleLatentEnergyVerifier,
    LatentEnergyVerifier,
    contrastive_energy_loss,
)
from model.latent_action import LatentActionCodebook
from model.trm import TRM


class _CallCounter(nn.Module):
    """Wrap a module to count forward calls (for the 'no decode in search' proof)."""

    def __init__(self, inner: nn.Module, counter: dict, key: str):
        super().__init__()
        self.inner = inner
        self.counter = counter
        self.key = key

    def forward(self, *args, **kwargs):
        self.counter[self.key] += 1
        return self.inner(*args, **kwargs)


def test_latent_energy_verifier_scores_latent_without_decoding():
    torch.manual_seed(0)
    ver = LatentEnergyVerifier(num_tokens=5, dim=32, n_layers=1, max_grid_size=8)
    x = torch.randint(0, 5, (4, 16))
    z = torch.randn(4, 16, 32)
    e = ver(x, z, width=4)
    assert e.shape == (4,)  # scalar energy per example, from the latent alone


def test_latent_energy_verifier_learns_to_rank_latents():
    torch.manual_seed(0)
    ver = LatentEnergyVerifier(num_tokens=5, dim=32, n_layers=2, max_grid_size=8)
    opt = torch.optim.Adam(ver.parameters(), lr=2e-3)
    x = torch.randint(0, 5, (32, 16))

    def good(b):  # spatially-constant latent = "structured" (low energy target)
        return (torch.randn(b, 1, 32) * 0.5).expand(b, 16, 32).contiguous()

    def bad(b):  # spatially-varying latent = "unstructured"
        return torch.randn(b, 16, 32) * 0.5

    ver.train()
    for _ in range(200):
        loss = contrastive_energy_loss(ver(x, good(32), 4), ver(x, bad(32), 4), margin=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    ver.eval()
    with torch.no_grad():
        rank = (ver(x, good(32), 4) < ver(x, bad(32), 4)).float().mean().item()
    assert rank > 0.8, rank  # learned to score structured latents as lower energy


def test_action_codebook_is_principled():
    cb = LatentActionCodebook(dim=16, n_actions=4, scale=0.5)
    z = torch.randn(2, 5, 16)
    assert torch.equal(cb.apply_action(z, 0), z)            # action 0 = identity
    assert not torch.equal(cb.apply_action(z, 1), z)        # action 1 actually moves z
    priors = cb.priors()
    assert priors.shape == (4,)
    assert torch.allclose(priors.sum(), torch.tensor(1.0), atol=1e-6)


def test_latent_mcts_never_decodes_during_search():
    torch.manual_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=4, T=2, max_grid_size=8)
    verifier = LatentEnergyVerifier(num_tokens=5, dim=32, n_layers=1, max_grid_size=8)
    codebook = LatentActionCodebook(dim=32, n_actions=3)

    counter = {"out_head": 0}
    model.out_head = _CallCounter(model.out_head, counter, "out_head")

    mcts = LatentNativeMCTS(model, verifier, codebook, height=4, width=4, n_rollouts=10)
    x = torch.randint(0, 5, (1, 16))

    best = mcts.search(x)
    assert counter["out_head"] == 0, "search must not decode (no out_head calls)"

    answer = mcts.decode(best)
    assert counter["out_head"] == 1, "decode is the single out_head call"
    assert answer.shape == (1, 16)


def test_latent_is_int8_while_search_stats_are_fp():
    """Literal INT8 latent storage; PUCT statistics (Q/N/prior) stay FP (no leak)."""
    torch.manual_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=3, T=1, max_grid_size=8)
    verifier = LatentEnergyVerifier(num_tokens=5, dim=32, n_layers=1, max_grid_size=8)
    codebook = LatentActionCodebook(dim=32, n_actions=3)
    mcts = LatentNativeMCTS(model, verifier, codebook, height=4, width=4, n_rollouts=6)

    root = mcts.search(torch.randint(0, 5, (1, 16)))
    node = root.children[0]

    # --- Environment dynamics: the latent is stored as literal INT8 codes. ---
    assert node.z_codes.dtype == torch.int8
    assert node.z_codes.abs().max().item() <= 127
    assert node.z_scale.dtype == torch.float32
    # Dequantized latent lands exactly on the INT8 grid.
    levels = node.latent() / node.z_scale
    assert torch.allclose(levels, levels.round(), atol=1e-3)

    # --- Search statistics: FP / int, never quantized -> no PUCT precision loss. ---
    assert isinstance(node.value_sum, float)
    assert isinstance(node.q, float)
    assert isinstance(node.prior, float)
    assert isinstance(node.visits, int)


def test_ensemble_exposes_epistemic_uncertainty():
    torch.manual_seed(0)
    ens = EnsembleLatentEnergyVerifier(num_tokens=5, dim=32, n_members=4, n_layers=1, max_grid_size=8)
    x = torch.randint(0, 5, (4, 16))
    z = torch.randn(4, 16, 32)
    assert ens(x, z, 4).shape == (4, 4)  # [n_members, B]
    value, std = ens.value_with_uncertainty(x, z, 4)
    assert value.shape == (4,) and std.shape == (4,)
    assert (std >= 0).all() and std.sum() > 0  # members disagree -> uncertainty signal


def test_mcts_lcb_penalizes_uncertain_branches():
    torch.manual_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=3, T=1, max_grid_size=8)
    ens = EnsembleLatentEnergyVerifier(num_tokens=5, dim=32, n_members=3, n_layers=1, max_grid_size=8)
    cb = LatentActionCodebook(dim=32, n_actions=3)

    pessimistic = LatentNativeMCTS(model, ens, cb, 4, 4, n_rollouts=6, uncertainty_beta=3.0)
    mean_only = LatentNativeMCTS(model, ens, cb, 4, 4, n_rollouts=6, uncertainty_beta=0.0)
    x = torch.randint(0, 5, (1, 16))

    node = mean_only.search(x).children[0]
    # The LCB value is strictly below the mean value when the ensemble disagrees.
    assert pessimistic._value(x, node) < mean_only._value(x, node)
    assert pessimistic.search_and_decode(x).shape == (1, 16)


@pytest.mark.slow
def test_ensemble_epistemic_uncertainty_is_higher_ood():
    """Members agree in-distribution and disagree out-of-distribution -> the LCB
    automatically penalises OOD latents the search could otherwise chase."""
    torch.manual_seed(0)
    ens = EnsembleLatentEnergyVerifier(num_tokens=5, dim=24, n_members=4, n_layers=1, max_grid_size=8)
    x = torch.randint(0, 5, (64, 16))
    opt = torch.optim.Adam(ens.parameters(), lr=3e-3)
    for _ in range(150):
        z = torch.randn(64, 16, 24) * 0.5            # in-distribution latents
        target = z.pow(2).mean(dim=(1, 2))
        loss = ((ens(x, z, 4) - target[None]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    ens.eval()
    with torch.no_grad():
        _, std_in = ens.value_with_uncertainty(x, torch.randn(64, 16, 24) * 0.5, 4)
        _, std_ood = ens.value_with_uncertainty(x, torch.randn(64, 16, 24) * 0.5 + 8.0, 4)
    assert std_ood.mean() > std_in.mean()  # epistemic uncertainty grows OOD
