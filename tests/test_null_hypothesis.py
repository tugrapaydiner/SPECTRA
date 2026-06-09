"""GT5 tests: the null hypothesis (dense, non-recursive, at iso-FLOP).

Proves the harness builds a dense baseline whose single-pass FLOPs match each
SPECTRA operating point and evaluates BOTH -- so 'test-time search substitutes for
parameters' is tested against 'just use more parameters', not assumed.
"""

import torch

from eval.scaling import (
    build_dense_baseline_for_flops,
    make_param_budget_model,
    run_null_hypothesis,
    spectra_inference_flops,
)


def test_flop_estimator_is_monotonic():
    model, _ = make_param_budget_model(40_000, num_tokens=5, seq_len=16, max_grid_size=8)
    model.N_sup = 2
    f_depth2 = spectra_inference_flops(model, 16, mcts_rollouts=0)
    model.N_sup = 4
    f_depth4 = spectra_inference_flops(model, 16, mcts_rollouts=0)
    assert f_depth4 > f_depth2                      # deeper recursion -> more FLOPs
    model.N_sup = 2
    f_search = spectra_inference_flops(model, 16, mcts_rollouts=16)
    assert f_search > f_depth2                      # MCTS rollouts -> more FLOPs


def test_iso_flop_dense_baseline_matches_target():
    target = 5e7
    dense, flops = build_dense_baseline_for_flops(
        target, num_tokens=5, seq_len=16, max_grid_size=8, n_layers=3
    )
    assert 0.5 * target < flops < 2.0 * target      # matched to the FLOP budget


def test_null_hypothesis_emits_paired_iso_flop_rows():
    torch.manual_seed(0)
    x = torch.randint(0, 5, (4, 16))
    y = torch.randint(1, 5, (4, 16))
    rows = run_null_hypothesis(
        spectra_param_target=40_000, depths=[1, 2], rollouts_list=[0],
        x=x, y=y, height=4, width=4, num_tokens=5, seq_len=16,
        device="cpu", max_grid_size=8,
    )
    assert len(rows) == 2 * 1 * 2                    # (depths x rollouts) x {spectra, dense}
    assert {r["model"] for r in rows} == {"spectra", "dense"}

    spectra = [r for r in rows if r["model"] == "spectra"]
    dense = [r for r in rows if r["model"] == "dense"]
    for s, d in zip(spectra, dense):
        assert 0.4 < d["flops"] / s["flops"] < 2.5  # iso-FLOP pairing
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)
    # The dense baseline spent its FLOP budget on PARAMETERS; SPECTRA on recursion.
    assert dense[0]["param_count"] > spectra[0]["param_count"]
