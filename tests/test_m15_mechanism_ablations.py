import torch

from model.grounded_verifier import GroundedStateVerifier
from model.latent_action import LatentActionCodebook
from model.latent_vq import LatentVQ
from model.trm import TRM
import scripts.m15_mechanism_ablations as m15


def tiny_core():
    torch.manual_seed(17)
    model = TRM(
        dim=64, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1, N_sup=4,
        heads=4, alpha_y=0.1, alpha_z=0.1, max_grid_size=8,
        ternary=False, act8=False,
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def verifier():
    torch.manual_seed(19)
    return GroundedStateVerifier(
        num_tokens=5, dim=64, n_layers=1, heads=4, max_grid_size=8,
        act_bits=8, include_y=True,
    ).eval()


def codebook():
    cb = LatentActionCodebook(dim=64, n_actions=4, scale=0.5).eval()
    with torch.no_grad():
        cb.prior_logits.zero_()
    return cb


def test_frozen_m15_protocol_identity_and_budgets():
    assert m15.PROTOCOL_COMMIT == "8caea3844a1eb9f855c4a06afe58b8853d4316a7"
    assert m15.CORE_SEEDS == [1401, 2402]
    assert m15.SEARCH_ROLLOUTS == 12
    assert m15.SEARCH_DEPTHS == [1, 2, 4]
    assert m15.BETA_GRID == [0.0, 0.5, 1.0, 2.0]


def test_fp32_search_state_is_not_int8_and_roundtrips_zero_root():
    model = tiny_core(); v = verifier(); cb = codebook()
    s = m15.searcher_for(model, v, cb, depth=2, fp32=True)
    s._reset_search_state("test")
    x = torch.tensor([[1,0,0,4,0,4,1,0,0,1,4,0,4,0,0,1]])
    xemb = model.token_embed(x) + model.encode_positions(x, 4, 4)
    root = s._make_zero_root(xemb)
    assert root.z_codes.dtype == torch.float32
    assert torch.equal(root.latent(), torch.zeros_like(xemb))


def test_int8_search_state_retains_native_storage_contract():
    model = tiny_core(); v = verifier(); cb = codebook()
    s = m15.searcher_for(model, v, cb, depth=2, fp32=False)
    s._reset_search_state("test")
    x = torch.tensor([[1,0,0,4,0,4,1,0,0,1,4,0,4,0,0,1]])
    xemb = model.token_embed(x) + model.encode_positions(x, 4, 4)
    root = s._make_zero_root(xemb)
    assert root.z_codes.dtype == torch.int8


def test_terminal_guard_blocks_expansion_of_known_valid_node(monkeypatch):
    model = tiny_core(); v = verifier(); cb = codebook()
    s = m15.searcher_for(model, v, cb, depth=4, guard=True)
    s._reset_search_state("test")
    x = torch.tensor([[1,0,0,4,0,4,1,0,0,1,4,0,4,0,0,1]])
    xemb = model.token_embed(x) + model.encode_positions(x, 4, 4)
    root = s._make_zero_root(xemb)
    s._m09_search_x = x
    monkeypatch.setattr(s, "_is_valid", lambda _x, _node: True)
    before = s.last_search_stats["transition_calls"]
    assert s._expand(root, xemb, initial=True) is False
    assert s.last_search_stats["transition_calls"] == before
    assert s.last_search_stats["semantic_guard_terminal_nodes"] == 1


def test_beta_selection_uses_regression_then_success_then_latency():
    base = [
        {"core_seed": 1, "example_id": "a", "semantic_success": 1, "symbolic_score": 1.0, "latency_ms": 1.0},
        {"core_seed": 1, "example_id": "b", "semantic_success": 1, "symbolic_score": 1.0, "latency_ms": 1.0},
    ]
    b0 = [
        {"core_seed": 1, "example_id": "a", "semantic_success": 0, "symbolic_score": .5, "latency_ms": 1.0},
        {"core_seed": 1, "example_id": "b", "semantic_success": 1, "symbolic_score": 1.0, "latency_ms": 1.0},
    ]
    b1 = [
        {"core_seed": 1, "example_id": "a", "semantic_success": 1, "symbolic_score": 1.0, "latency_ms": 1.3},
        {"core_seed": 1, "example_id": "b", "semantic_success": 1, "symbolic_score": 1.0, "latency_ms": 1.3},
    ]
    assert m15.select_beta({0.0: b0, 1.0: b1}, base)["selected_beta"] == 1.0


def _summary(reg, sem=0.8, exploit=0.2):
    return {
        "search_induced_regressions": reg,
        "search_induced_regression_rate": reg / 100,
        "semantic_success": sem,
        "proxy_exploitation_rate": exploit,
    }


def test_h1_support_rule_is_strict_and_fp32_can_disconfirm():
    good = {
        m15.PRIMARY_SEARCH: _summary(20, .80, .20),
        m15.ORACLE_SEARCH: _summary(5, .90, 0.0),
        m15.GUARDED_SEARCH: _summary(6, .90, 0.0),
        m15.FP32_SEARCH: _summary(12, .82, .10),
    }
    r = m15.mechanism_support(good)
    assert r["h1_supported"] is True
    bad = dict(good)
    bad[m15.FP32_SEARCH] = _summary(4, .90, .0)
    assert m15.mechanism_support(bad)["h1_supported"] is False


def test_vq_module_no_longer_claims_arbitrary_depth_trajectory_bound():
    doc = (LatentVQ.__module__ and __import__(LatentVQ.__module__, fromlist=["x"]).__doc__) or ""
    low = doc.lower()
    assert "do **not** imply" in low
    assert "unquantized recursive trajectory" in low
    assert "topology survives arbitrarily deep" not in low


def test_vq_covering_radius_is_only_local_projection_error():
    torch.manual_seed(3)
    vq = LatentVQ(dim=4, codebook_size=4, decay=0.0).eval()
    point = vq.codebook[0].detach().reshape(1, 1, 4)
    assert vq.covering_radius(point) == 0.0
    assert torch.equal(vq.snap(point), point)


def test_ensemble_uncertainty_is_not_claimed_as_ood_guarantee():
    import model.energy as energy
    doc = energy.EnsembleLatentEnergyVerifier.__doc__.lower()
    assert "does **not** by itself identify out-of-distribution states" in doc
    assert "requires validation" in doc


def test_overall_acceptance_needs_confirmed_mechanism():
    h1 = {"h1_supported": True}
    leaf = {"1": {"improvement_auc": .8}, "4": {"improvement_auc": .5}}
    assert m15.overall_acceptance(h1, h1, leaf, leaf)["pass"] is True
    nope = {"h1_supported": False}
    no_collapse = {"1": {"improvement_auc": .55}, "4": {"improvement_auc": .54}}
    assert m15.overall_acceptance(nope, nope, no_collapse, no_collapse)["pass"] is False
