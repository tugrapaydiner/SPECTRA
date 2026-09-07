"""Historical scaling plumbing tests, explicitly segregated as random-init smoke.

M05 research evaluation is covered separately by ``test_m05_checkpoint_eval.py``.
These tests retain parameter-grid/energy plumbing coverage but may not emit a
research-labelled result.
"""

import torch

from eval.edge_energy import rapl_available
from eval.latent_mcts import LatentNativeMCTS
from eval.scaling import count_params, make_param_budget_model, run_scaling_grid, to_dataframe
from model.energy import LatentEnergyVerifier
from model.latent_action import LatentActionCodebook


def test_param_budget_models_scale_with_target_smoke_only():
    counts = []
    for target in (40_000, 150_000, 400_000):
        model, count = make_param_budget_model(target, num_tokens=5, seq_len=16, max_grid_size=8)
        assert count == count_params(model)
        counts.append(count)
        assert abs(count - target) < 0.5 * target
    assert counts[0] < counts[1] < counts[2]


def test_smoke_scaling_grid_emits_nonresearch_energy_rows():
    torch.manual_seed(0)
    x = torch.randint(0, 5, (4, 16))
    y = torch.randint(1, 5, (4, 16))
    rows = run_scaling_grid(
        param_targets=[40_000, 150_000], depths=[1, 2], rollouts_list=[0],
        x=x, y=y, height=4, width=4, num_tokens=5, seq_len=16,
        device="cpu", n_latency_runs=4, max_grid_size=8,
        smoke_random_init=True,
    )
    assert len(rows) == 2 * 2 * 1
    assert all(r["result_kind"] == "smoke_random_init" and not r["research_result"] for r in rows)
    df = to_dataframe(rows)
    for col in ["param_count", "recursion_depth", "mcts_rollouts", "accuracy",
                "latency_ms", "microjoules_per_infer", "energy_measured"]:
        assert col in df.columns
    assert df["param_count"].nunique() == 2
    assert df["accuracy"].between(0, 1).all()
    assert (df["energy_measured"] == rapl_available()).all()
    if not rapl_available():
        assert df["microjoules_per_infer"].isna().all()


def test_smoke_grid_can_cross_random_aux_mcts_axis_without_research_label():
    torch.manual_seed(0)
    x = torch.randint(0, 5, (2, 16))
    y = torch.randint(1, 5, (2, 16))

    def mcts_factory(model, rollouts):
        ver = LatentEnergyVerifier(num_tokens=5, dim=model.dim, n_layers=1, max_grid_size=8)
        cb = LatentActionCodebook(model.dim, n_actions=3)
        return LatentNativeMCTS(model, ver, cb, height=4, width=4, n_rollouts=rollouts)

    rows = run_scaling_grid(
        param_targets=[40_000], depths=[1], rollouts_list=[0, 2],
        x=x, y=y, height=4, width=4, num_tokens=5, seq_len=16,
        device="cpu", n_latency_runs=3, mcts_factory=mcts_factory, max_grid_size=8,
        smoke_random_init=True,
    )
    assert {r["mcts_rollouts"] for r in rows} == {0, 2}
    assert all(r["result_kind"] == "smoke_random_init" for r in rows)
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)
