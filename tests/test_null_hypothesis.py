"""GT5 compute-estimator tests plus the M05 random-init research boundary.

The FLOP helpers remain useful for smoke analysis. M05 intentionally forbids the
old random-initialized dense-vs-SPECTRA comparison from being emitted as a
research result; a scientific null-hypothesis comparison requires separately
trained, compatible checkpoints and an immutable evaluation manifest.
"""

import pytest

from eval.checkpoint_eval import EvaluationContractError
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
    assert f_depth4 > f_depth2
    model.N_sup = 2
    f_search = spectra_inference_flops(model, 16, mcts_rollouts=16)
    assert f_search > f_depth2


def test_iso_flop_dense_baseline_matches_target():
    target = 5e7
    _, flops = build_dense_baseline_for_flops(
        target, num_tokens=5, seq_len=16, max_grid_size=8, n_layers=3
    )
    assert 0.5 * target < flops < 2.0 * target


def test_random_init_null_hypothesis_cannot_be_a_research_result():
    with pytest.raises(EvaluationContractError, match="smoke-only"):
        run_null_hypothesis(
            spectra_param_target=40_000,
            depths=[1, 2],
            rollouts_list=[0],
            x=None,
            y=None,
            height=4,
            width=4,
            num_tokens=5,
            seq_len=16,
            device="cpu",
            max_grid_size=8,
        )

    # Even an explicit smoke request does not silently resurrect the historical
    # paired rows: that path was removed rather than risk accidental publication.
    with pytest.raises(EvaluationContractError, match="intentionally disabled"):
        run_null_hypothesis(smoke_random_init=True)
