"""Compute-optimal scaling-law harness (BLUEPRINT sections 5.5, 27.0).

The reviewer's killer point: the old sweep varied only recursion depth, never
parameter count, and never measured joules -- so the central hypothesis ("learned
test-time search can substitute for parameter count") was untested. This harness
fixes that. It instantiates models at target parameter budgets (~1M / 7M / 14M),
crosses them against recursion depth AND Latent-MCTS rollouts, and logs physical
microjoules per inference via RAPL (graceful where unavailable), emitting a
DataFrame ready to plot Accuracy vs. Measured Joules.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

from eval.edge_energy import measure_energy_joules, rapl_available
from eval.latency import measure_latency
from eval.metrics import board_accuracy
from model.trm import TRM


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_param_budget_model(
    target_params: int,
    num_tokens: int,
    seq_len: int,
    n_layers: int = 2,
    heads: int = 8,
    max_grid_size: int = 32,
    **trm_kwargs: Any,
) -> tuple[TRM, int]:
    """Build a :class:`TRM` whose parameter count is closest to ``target_params``.

    The hidden dim is the main knob (params ~ n_layers * 12 * dim^2). We seed from
    the closed-form estimate, then build a handful of nearby (heads-aligned) dims
    and pick the closest -- exact, since building a TRM only allocates parameters.

    Returns ``(model, actual_param_count)``.
    """
    approx_dim = int(math.sqrt(max(1, target_params) / (n_layers * 12)))
    approx_dim = max(heads, (approx_dim // heads) * heads)

    best_model: TRM | None = None
    best_count = -1
    for dim in range(max(heads, approx_dim - 6 * heads), approx_dim + 7 * heads, heads):
        model = TRM(
            dim=dim, num_tokens=num_tokens, seq_len=seq_len,
            n_layers=n_layers, heads=heads, max_grid_size=max_grid_size, **trm_kwargs,
        )
        count = count_params(model)
        if best_model is None or abs(count - target_params) < abs(best_count - target_params):
            best_model, best_count = model, count
    assert best_model is not None
    return best_model, best_count


# A factory that wraps a model into a search controller exposing
# ``search_and_decode(x[1,L]) -> answer[1,L]`` for a given rollout count.
MCTSFactory = Callable[[TRM, int], Any]


@torch.no_grad()
def _greedy_accuracy(model, x, y, height, width):
    return board_accuracy(model(x, height=height, width=width)[0].argmax(-1), y)


@torch.no_grad()
def _mcts_accuracy(controller, x, y):
    answers = torch.cat([controller.search_and_decode(x[i : i + 1]) for i in range(x.shape[0])])
    return board_accuracy(answers, y)


def run_scaling_grid(
    param_targets: list[int],
    depths: list[int],
    rollouts_list: list[int],
    x: torch.Tensor,
    y: torch.Tensor,
    height: int,
    width: int,
    num_tokens: int,
    seq_len: int,
    device: torch.device | str = "cpu",
    n_latency_runs: int = 10,
    mcts_factory: MCTSFactory | None = None,
    n_layers: int = 2,
    max_grid_size: int = 32,
) -> list[dict[str, Any]]:
    """Sweep parameter budget x recursion depth x MCTS rollouts; measure each point.

    Returns a list of row dicts (one operating point). ``rollouts == 0`` is greedy
    decoding; ``rollouts > 0`` requires ``mcts_factory`` to build a latent search
    controller for the model.
    """
    rows: list[dict[str, Any]] = []
    x, y = x.to(device), y.to(device)
    for target in param_targets:
        model, pcount = make_param_budget_model(
            target, num_tokens, seq_len, n_layers=n_layers, max_grid_size=max_grid_size
        )
        model = model.to(device).eval()
        for depth in depths:
            model.N_sup = depth
            for rollouts in rollouts_list:
                if rollouts == 0:
                    acc = _greedy_accuracy(model, x, y, height, width)
                    infer = lambda: model(x[:1], height=height, width=width)
                else:
                    if mcts_factory is None:
                        raise ValueError("rollouts > 0 requires an mcts_factory")
                    controller = mcts_factory(model, rollouts)
                    acc = _mcts_accuracy(controller, x, y)
                    infer = lambda c=controller: c.search_and_decode(x[:1])

                latency = measure_latency(infer, n_runs=n_latency_runs, warmup=3)
                energy = measure_energy_joules(infer, n_runs=n_latency_runs)
                micro_j = energy["joules_per_run"] * 1e6 if energy is not None else None
                rows.append({
                    "target_params": target,
                    "param_count": pcount,
                    "recursion_depth": depth,
                    "mcts_rollouts": rollouts,
                    "accuracy": acc,
                    "latency_ms": latency["latency_ms_mean"],
                    "microjoules_per_infer": micro_j,
                    "joules_per_problem": (micro_j / 1e6) if micro_j is not None else None,
                    "energy_measured": energy is not None,
                })
    return rows


def to_dataframe(rows: list[dict[str, Any]]):
    """Convert grid rows to a pandas DataFrame (Accuracy vs Measured Joules)."""
    import pandas as pd

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# THE NULL HYPOTHESIS (GT audit #5): does search beat just-add-parameters?
# --------------------------------------------------------------------------- #
def transformer_forward_flops(
    params: int, seq_len: int, dim: int, n_layers: int, applications: int = 1
) -> float:
    """Approximate forward FLOPs: ``applications`` * (2*N*L linear + 2*n*L^2*d attn).

    The same accounting is applied to BOTH models so the iso-FLOP comparison is
    fair. ``applications`` is the number of times the (shared) weights are applied
    -- 1 for a dense single pass, ``N_sup*T*(n+1)`` for the recursion.
    """
    linear = 2.0 * params * seq_len
    attn = 2.0 * n_layers * seq_len * seq_len * dim
    return (linear + attn) * applications


def spectra_inference_flops(
    model: TRM, seq_len: int, mcts_rollouts: int = 0, n_children: int = 3
) -> float:
    """Total test-time FLOPs of a SPECTRA config (recursion x MCTS expansions)."""
    applications = model.N_sup * model.T * (model.n + 1)
    base = transformer_forward_flops(
        count_params(model), seq_len, model.dim, len(model.blocks), applications
    )
    # Each rollout expands ``n_children`` latent steps (one recursive step each).
    search_factor = 1.0 + mcts_rollouts * n_children / max(1, model.N_sup)
    return base * search_factor


def build_dense_baseline_for_flops(
    target_flops: float,
    num_tokens: int,
    seq_len: int,
    max_grid_size: int = 32,
    n_layers: int = 3,
    heads: int = 8,
):
    """Build a DENSE, non-recursive transformer (System1Student) whose single-pass
    FLOPs match ``target_flops`` -- the null hypothesis "just use more parameters"."""
    from model.system1_student import System1Student

    approx_params = target_flops / (2.0 * seq_len)
    approx_dim = max(heads, int((approx_params / (n_layers * 12)) ** 0.5))
    approx_dim = max(heads, (approx_dim // heads) * heads)  # align to heads

    best = None
    best_flops = -1.0
    for dim in range(max(heads, approx_dim - 6 * heads), approx_dim + 7 * heads, heads):
        m = System1Student(dim=dim, num_tokens=num_tokens, seq_len=seq_len,
                           n_layers=n_layers, max_grid_size=max_grid_size)
        f = transformer_forward_flops(count_params(m), seq_len, dim, n_layers, 1)
        if best is None or abs(f - target_flops) < abs(best_flops - target_flops):
            best, best_flops = m, f
    return best, best_flops


def run_null_hypothesis(
    spectra_param_target: int,
    depths: list[int],
    rollouts_list: list[int],
    x: torch.Tensor,
    y: torch.Tensor,
    height: int,
    width: int,
    num_tokens: int,
    seq_len: int,
    device: torch.device | str = "cpu",
    mcts_factory: MCTSFactory | None = None,
    n_children: int = 3,
    n_layers: int = 2,
    max_grid_size: int = 32,
) -> list[dict[str, Any]]:
    """For every SPECTRA operating point, build the ISO-FLOP dense baseline and
    evaluate both -- the experiment that decides whether the paper lives or dies.

    Emits paired rows tagged ``model="spectra"`` / ``model="dense"`` at matched
    FLOPs, ready to plot Accuracy vs FLOPs with both curves overlaid.
    """
    x, y = x.to(device), y.to(device)
    rows: list[dict[str, Any]] = []
    for depth in depths:
        for rollouts in rollouts_list:
            spectra, _ = make_param_budget_model(
                spectra_param_target, num_tokens, seq_len,
                n_layers=n_layers, max_grid_size=max_grid_size,
            )
            spectra.N_sup = depth
            spectra = spectra.to(device).eval()
            sp_flops = spectra_inference_flops(spectra, seq_len, rollouts, n_children)
            if rollouts == 0:
                sp_acc = _greedy_accuracy(spectra, x, y, height, width)
            else:
                if mcts_factory is None:
                    raise ValueError("rollouts > 0 requires an mcts_factory")
                sp_acc = _mcts_accuracy(mcts_factory(spectra, rollouts), x, y)

            dense, dense_flops = build_dense_baseline_for_flops(
                sp_flops, num_tokens, seq_len, max_grid_size, n_layers=n_layers + 1
            )
            dense = dense.to(device).eval()
            with torch.no_grad():
                dense_ans, _ = dense.predict(x, height, width)
            dense_acc = board_accuracy(dense_ans, y)

            rows.append({
                "model": "spectra", "recursion_depth": depth, "mcts_rollouts": rollouts,
                "param_count": count_params(spectra), "flops": sp_flops, "accuracy": sp_acc,
            })
            rows.append({
                "model": "dense", "recursion_depth": 1, "mcts_rollouts": 0,
                "param_count": count_params(dense), "flops": dense_flops, "accuracy": dense_acc,
            })
    return rows
