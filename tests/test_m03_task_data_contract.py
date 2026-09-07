"""Milestone 03 acceptance/adversarial tests for task, data, split, and metrics contracts."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from common import load_config
from data import maze as mz
from data import sudoku as sk
from data.splits import build_reproducible_splits
from data.task_contracts import contract_from_config
from eval.metrics import exact_reference_match, semantic_validity, task_metrics
from model.verifier import maze_correct, sudoku_correct, sudoku_puzzle_valid
from scripts._common import build_data_splits


def _hash_rows(a: np.ndarray) -> set[str]:
    return {hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in a}


@pytest.mark.parametrize(
    "path",
    [
        "config/sudoku.yaml",
        "config/maze.yaml",
        "config/arc.yaml",
        "config/babyai.yaml",
        "config/smarthome.yaml",
    ],
)
def test_retained_configs_match_generated_shapes_vocab_padding_and_masks(path):
    cfg = load_config(path)
    contract = contract_from_config(str(cfg.task), cfg.data)
    datasets, manifest = build_data_splits(cfg, 3, 2, 2, seed=123)
    assert manifest["generator_version"]
    assert manifest["task_scope"] == contract.scope
    for split, ds in datasets.items():
        assert ds.height == contract.height and ds.width == contract.width
        assert ds.inputs.shape == ds.targets.shape == (len(ds), contract.seq_len)
        assert ds.input_mask.shape == ds.inputs.shape
        assert ds.target_mask.shape == ds.targets.shape
        if len(ds):
            assert int(ds.inputs.min()) >= 0 and int(ds.inputs.max()) < contract.num_tokens
            assert int(ds.targets.min()) >= 0 and int(ds.targets.max()) < contract.num_tokens
        if cfg.task == "arc":
            assert contract.official_benchmark is False
            assert "NOT official ARC" in contract.scope
            assert np.array_equal(ds.input_mask, ds.inputs != contract.pad_token)
            assert np.array_equal(ds.target_mask, ds.targets != contract.pad_token)
        elif cfg.task == "babyai":
            assert contract.official_benchmark is False
            assert "NOT official BabyAI" in contract.scope
        elif cfg.task == "smarthome":
            assert np.array_equal(ds.input_mask, ds.inputs != contract.pad_token)
            assert ds.target_mask.all()


def test_arc_config_no_longer_silently_builds_10x10():
    cfg = load_config("config/arc.yaml")
    datasets, _ = build_data_splits(cfg, 1, 1, 1, seed=7)
    for ds in datasets.values():
        assert ds.height == 30 and ds.width == 30
        assert ds.inputs.shape[1] == 900


def test_manifest_reproducibility_separate_streams_and_duplicate_audit():
    cfg = load_config("config/sudoku.yaml")
    _, m1 = build_data_splits(cfg, 4, 2, 2, seed=77)
    _, m2 = build_data_splits(cfg, 4, 2, 2, seed=77)
    assert m1 == m2
    spawn = m1["rng_streams"]
    keys = {
        tuple(spawn[s][kind])
        for s in ("train", "validation", "test")
        for kind in ("generation_spawn_key", "augmentation_spawn_key")
    }
    assert len(keys) == 6
    assert all(v == 0 for v in m1["duplicate_audit"]["cross_split_group_overlap"].values())
    assert all(v == 0 for v in m1["duplicate_audit"]["cross_split_exact_overlap"].values())
    groups = {
        s: {e["group_id"] for e in m1["splits"][s]["examples"]}
        for s in ("train", "validation", "test")
    }
    assert not (groups["train"] & groups["validation"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["validation"] & groups["test"])


def test_sudoku_numpy_validator_rejects_malformed_domain_and_contradictions():
    box = 2
    solved = sk.generate_solution(box, np.random.default_rng(1))
    assert sk.is_solved(solved, box)
    assert not sk.is_valid_grid(solved.reshape(-1), box)
    bad_hi = solved.copy(); bad_hi[0, 0] = 5
    bad_lo = solved.copy(); bad_lo[0, 0] = -1
    bad_dup = solved.copy(); bad_dup[0, 0] = bad_dup[0, 1]
    for bad in (bad_hi, bad_lo, bad_dup):
        assert not sk.is_valid_grid(bad, box)
        assert sk.count_solutions(bad, box) == 0
        assert sk.solve(bad, box) is None
    float_grid = solved.astype(np.float64)
    assert not sk.is_valid_grid(float_grid, box)


def test_sudoku_numpy_tensor_validator_crosscheck():
    rng = np.random.default_rng(2)
    puzzles = []
    solutions = []
    for _ in range(8):
        p, s = sk.generate_pair(2, 8, rng, require_unique=True)
        puzzles.append(p); solutions.append(s)
    p_np = np.stack(puzzles)
    s_np = np.stack(solutions)
    p = torch.from_numpy(p_np.reshape(8, -1))
    s = torch.from_numpy(s_np.reshape(8, -1))
    assert sudoku_puzzle_valid(p, 2).tolist() == [sk.is_valid_grid(x, 2) for x in p_np]
    assert sudoku_correct(p, s, 2).all()

    contradictory = p.clone()
    contradictory[0, 0] = contradictory[0, 1] = 1
    tensor_valid = sudoku_puzzle_valid(contradictory, 2)
    numpy_valid = [sk.is_valid_grid(x.reshape(4, 4).numpy(), 2) for x in contradictory]
    assert tensor_valid.tolist() == numpy_valid
    assert not sudoku_correct(contradictory, s, 2)[0]


def test_randomized_sudoku_generation_is_not_one_fixed_completed_board():
    rng = np.random.default_rng(5)
    boards = np.stack([sk.generate_solution(3, rng, method="random_backtracking") for _ in range(12)])
    assert all(sk.is_solved(b, 3) for b in boards)
    assert len(_hash_rows(boards.reshape(12, -1))) == 12


def _loop_maze():
    inp = np.zeros((5, 5), dtype=np.int64)
    inp[1:4, 1:4] = mz.OPEN
    inp[1, 1] = mz.START
    inp[1, 3] = mz.GOAL
    optimal = inp.copy(); optimal[1, 2] = mz.PATH
    longer = inp.copy()
    longer[2, 1] = mz.PATH; longer[2, 2] = mz.PATH; longer[2, 3] = mz.PATH
    return inp, optimal, longer


def test_maze_success_checks_connectivity_walls_and_optimality():
    inp, optimal, longer = _loop_maze()
    assert mz.candidate_success(inp, optimal, require_optimal=True)
    assert mz.candidate_success(inp, longer, require_optimal=False)
    assert not mz.candidate_success(inp, longer, require_optimal=True)
    assert not mz.candidate_success(inp, inp.copy(), require_optimal=True)

    wall_cross = optimal.copy(); wall_cross[0, 2] = mz.PATH
    assert not mz.candidate_success(inp, wall_cross, require_optimal=False)
    branch = longer.copy(); branch[1, 2] = mz.PATH
    assert not mz.candidate_success(inp, branch, require_optimal=False)


def test_generated_unsolved_maze_copy_is_not_semantic_success():
    x, y = mz.generate_pair(15, 15, np.random.default_rng(4), min_path_len=8)
    assert mz.candidate_success(x, y, 15, 15)
    assert not mz.candidate_success(x, x, 15, 15)
    xt = torch.from_numpy(x.reshape(1, -1))
    yt = torch.from_numpy(y.reshape(1, -1))
    assert maze_correct(xt, yt, 15, 15).item()
    assert not maze_correct(xt, xt, 15, 15).item()


def test_metrics_separate_reference_match_semantic_validity_and_blank_accuracy():
    # Empty 4x4 Sudoku admits multiple semantically valid solutions.
    puzzle = torch.zeros((1, 16), dtype=torch.int64)
    target_np = sk.generate_solution(2, np.random.default_rng(10))
    pred_np = sk.generate_solution(2, np.random.default_rng(11))
    target = torch.from_numpy(target_np.reshape(1, -1))
    pred = torch.from_numpy(pred_np.reshape(1, -1))
    assert sk.is_solved(target_np, 2) and sk.is_solved(pred_np, 2)
    if torch.equal(target, pred):
        pytest.skip("extremely unlikely identical randomized Sudoku draws")
    assert exact_reference_match(pred, target) == 0.0
    assert semantic_validity("sudoku", puzzle, pred, box=2) == 1.0
    metrics = task_metrics("sudoku", puzzle, pred, target, box=2)
    assert metrics["semantic_validity"] == 1.0
    assert metrics["exact_reference_match"] == 0.0
    assert 0.0 <= metrics["blank_cell_accuracy"] < 1.0
