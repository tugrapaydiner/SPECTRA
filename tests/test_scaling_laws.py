"""Fix 5 tests: iso-joule scaling-law harness.

Proves the harness varies parameter count (the axis the old sweep ignored),
crosses it with recursion depth AND MCTS rollouts, and emits a DataFrame with the
measured-energy columns (RAPL graceful where unavailable) -- BLUEPRINT 5.5 / 27.0.
"""

import torch

from eval.edge_energy import rapl_available
from eval.latent_mcts import LatentNativeMCTS
from eval.scaling import count_params, make_param_budget_model, run_scaling_grid, to_dataframe
from model.energy import LatentEnergyVerifier
from model.latent_action import LatentActionCodebook


def test_param_budget_models_scale_with_target():
    counts = []
    for target in (40_000, 150_000, 400_000):
        model, count = make_param_budget_model(target, num_tokens=5, seq_len=16, max_grid_size=8)
        assert count == count_params(model)
        counts.append(count)
        assert abs(count - target) < 0.5 * target  # within 50% of the budget
    assert counts[0] < counts[1] < counts[2]  # parameter count genuinely varies


def test_scaling_grid_emits_dataframe_with_energy_columns():
    torch.manual_seed(0)
    x = torch.randint(0, 5, (4, 16))
    y = torch.randint(1, 5, (4, 16))
    rows = run_scaling_grid(
        param_targets=[40_000, 150_000], depths=[1, 2], rollouts_list=[0],
        x=x, y=y, height=4, width=4, num_tokens=5, seq_len=16,
        device="cpu", n_latency_runs=4, max_grid_size=8,
    )
    assert len(rows) == 2 * 2 * 1
    df = to_dataframe(rows)
    for col in ["param_count", "recursion_depth", "mcts_rollouts", "accuracy",
                "latency_ms", "microjoules_per_infer", "energy_measured"]:
        assert col in df.columns
    assert df["param_count"].nunique() == 2          # parameter axis is real
    assert df["accuracy"].between(0, 1).all()
    # Energy honesty: measured iff RAPL is present (False on this host).
    assert (df["energy_measured"] == rapl_available()).all()
    if not rapl_available():
        assert df["microjoules_per_infer"].isna().all()


def test_scaling_grid_crosses_mcts_rollout_axis():
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
    )
    assert {r["mcts_rollouts"] for r in rows} == {0, 2}  # search axis swept
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)
