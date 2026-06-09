"""Blank-Check #2: MCTS-bootstrapped unsupervised Latent PRM.

Proves the search assigns process-reward labels to every intermediate latent step
(no human labels), and that a fresh energy verifier can be *trained* on those
labels to score intermediate reasoning steps -- an autonomously self-taught
Process Reward Model.
"""

import torch

from eval.latent_mcts import LatentNativeMCTS
from model.energy import LatentEnergyVerifier
from model.latent_action import LatentActionCodebook
from model.trm import TRM
from train.distill import latent_prm_loss


def _build_mcts(seed=0):
    torch.manual_seed(seed)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=3, T=1, max_grid_size=8)
    verifier = LatentEnergyVerifier(num_tokens=5, dim=32, n_layers=1, max_grid_size=8)
    codebook = LatentActionCodebook(dim=32, n_actions=3)
    return LatentNativeMCTS(model, verifier, codebook, 4, 4, n_rollouts=20)


def test_prm_targets_cover_intermediate_steps():
    mcts = _build_mcts()
    mcts.search(torch.randint(0, 5, (1, 16)))
    z, values, visits = mcts.prm_targets()
    assert z.shape[0] >= 4 and z.shape[1:] == (16, 32)   # multiple intermediate latents
    assert values.shape == (z.shape[0],) and visits.shape == (z.shape[0],)
    assert (visits > 0).all()                            # only visited (credited) nodes


def test_unsupervised_prm_learns_to_score_steps():
    mcts = _build_mcts()
    x = torch.randint(0, 5, (1, 16))
    mcts.search(x)
    z, values, visits = mcts.prm_targets()  # process-reward labels from the search

    # Train a FRESH verifier (different init) on the search-bootstrapped labels.
    torch.manual_seed(1)
    prm = LatentEnergyVerifier(num_tokens=5, dim=32, n_layers=1, max_grid_size=8)
    opt = torch.optim.Adam(prm.parameters(), lr=5e-3)
    prm.train()
    first = None
    for _ in range(150):
        loss, comp = latent_prm_loss(prm, x, z, values, width=4, visit_weights=visits)
        first = first if first is not None else comp["prm_mse"]
        opt.zero_grad()
        loss.backward()
        opt.step()

    prm.eval()
    with torch.no_grad():
        pred = prm.value(x.expand(z.shape[0], -1), z, 4)
    # The PRM regressed the per-step values and now correlates with the search.
    assert comp["prm_mse"] < 0.5 * first
    corr = torch.corrcoef(torch.stack([pred, values]))[0, 1]
    assert corr.item() > 0.5
