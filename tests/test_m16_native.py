from pathlib import Path

import pytest
import torch

from deploy import m10_native
from deploy.m10_artifact import export_cpu_artifact, load_cpu_artifact
from deploy.m10_runtime import CPURecursiveRuntime
from deploy.validated_runtime import ValidatedCPURecursiveRuntime
from model.trm import TRM


def artifact(path: Path):
    torch.manual_seed(1600)
    m = TRM(dim=16, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1, N_sup=2,
            heads=4, max_grid_size=8, ternary=True, act8=True).eval()
    export_cpu_artifact(m, path, height=4, width=4, box=2,
                       source_checkpoint_sha256="fixture", source_checkpoint_tensor_sha256="fixture",
                       training_seed=1600, training_step=0, data_provenance={"fixture": True}, export_git_sha="fixture")
    return load_cpu_artifact(path)


def test_native_handle_bit_exact_to_checked_operator_and_owns_storage(tmp_path):
    a = artifact(tmp_path / "model.pt")
    e = a.payload["packed_linears"]["blocks.0.ff.0"]
    torch.manual_seed(1610)
    x = torch.randn(37, e["in_features"])
    h = m10_native.load_extension().ValidatedPackedLinear(
        e["packed"], e["scale"], e["bias"], e["out_features"], e["in_features"])
    expected = m10_native.dense_ternary_linear_fp32(x, e["packed"], e["scale"], e["bias"], e["out_features"])
    assert torch.equal(h.forward(x), expected)
    e["packed"].fill_(255)
    e["scale"].fill_(float("nan"))
    e["bias"].fill_(999)
    assert torch.equal(h.forward(x), expected)
    assert h.storage_bytes() > 0


@pytest.mark.parametrize("corrupt", ["reserved", "padding", "scale", "bias", "shape"])
def test_native_handle_rejects_malformed_weights(corrupt):
    packed = torch.tensor([1, 0], dtype=torch.uint8)
    scale = torch.ones(1)
    bias = torch.zeros(1)
    if corrupt == "reserved": packed[0] = 3
    if corrupt == "padding": packed[1] = 4
    if corrupt == "scale": scale[0] = -1
    if corrupt == "bias": bias[0] = float("nan")
    if corrupt == "shape": packed = packed[:1]
    with pytest.raises(RuntimeError):
        m10_native.load_extension().ValidatedPackedLinear(packed, scale, bias, 1, 5)


def test_native_handle_input_check_is_not_bypassed():
    h = m10_native.load_extension().ValidatedPackedLinear(
        torch.zeros(2, dtype=torch.uint8), torch.ones(1), torch.empty(0), 1, 5)
    for bad in [torch.ones(2, 4), torch.ones(0, 5), torch.ones(2, 5, dtype=torch.float64), torch.ones(5, 2).T]:
        with pytest.raises(RuntimeError): h.forward(bad)


def test_validated_whole_runtime_preserves_all_steps_and_work(tmp_path):
    a = artifact(tmp_path / "model.pt")
    checked = CPURecursiveRuntime(a)
    validated = ValidatedCPURecursiveRuntime(a)
    x = torch.tensor([[0,1,2,0,0,2,0,1,1,0,2,0,2,0,0,1]], dtype=torch.long)
    expected, got = checked.forward(x), validated.forward(x)
    assert torch.equal(expected.logits, got.logits)
    assert torch.equal(expected.answer, got.answer)
    for before, after in zip(expected.step_outputs, got.step_outputs):
        for key in before: assert torch.equal(before[key], after[key])
    assert checked._native_scalar_products == validated._native_scalar_products
    report = validated.backend_report()
    assert report["additional_owned_weight_bytes"] > 0 and report["vectorized"] is False


def test_native_checker_matches_independent_numpy_and_torch_on_mutations():
    import numpy as np
    from data import sudoku
    from model.verifier import sudoku_correct
    rng = np.random.default_rng(16001)
    for box in [1, 2, 3]:
        n = box * box
        solution = sudoku.generate_solution(box, rng)
        puzzle = solution.copy()
        puzzle.flat[::2] = 0
        x = torch.from_numpy(puzzle.reshape(1, -1))
        checker = m10_native.load_extension().SudokuProblem(x, box)
        for trial in range(64):
            candidate = solution.copy()
            if trial:
                i = int(rng.integers(n*n))
                candidate.flat[i] = int(rng.integers(-1, n+2))
            y = torch.from_numpy(candidate.reshape(1, -1))
            independent = sudoku.is_valid_grid(puzzle, box) and sudoku.is_solved(candidate, box) and sudoku.respects_clues(puzzle, candidate, box)
            assert checker.check(y) == independent == bool(sudoku_correct(x, y, box).item())


def test_native_checker_invalid_givens_cannot_be_hidden_by_clamping():
    for bad in [-1, 5, 1]:
        x = torch.zeros(1, 16, dtype=torch.long)
        x[0, 0] = bad
        x[0, 1] = 1
        checker = m10_native.load_extension().SudokuProblem(x, 2)
        y, valid = checker.decode(torch.zeros(1, 16, 5))
        assert not valid


def test_native_decode_matches_first_argmax_and_owns_puzzle():
    x = torch.zeros(1, 16, dtype=torch.long)
    x[0, 0] = 4
    checker = m10_native.load_extension().SudokuProblem(x, 2)
    x[0, 0] = 1
    logits = torch.zeros(1, 16, 5)
    logits[:, :, 2:4] = 1
    y, _ = checker.decode(logits)
    assert y[0, 0].item() == 4 and (y[0, 1:] == 2).all()
    for value in [float("nan"), float("inf")]:
        logits[0, 0, 0] = value
        with pytest.raises(RuntimeError, match="finite"): checker.decode(logits)


def test_native_semantic_exit_matches_original_step_and_answer():
    from deploy.semantic_exit import native_semantic_exit
    from scripts.m14_attempt5_dual_stream_semantic_exit import dual_stream_semantic_exit_solve
    torch.manual_seed(1604)
    model = TRM(dim=64, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1, N_sup=4,
                heads=4, max_grid_size=8).eval()
    x = torch.tensor([[0,1,2,0,0,2,0,1,1,0,2,0,2,0,0,1]], dtype=torch.long)
    for k in [1, 2, 4]:
        expected, before = dual_stream_semantic_exit_solve(model, x, k)
        got, after = native_semantic_exit(model, x, k)
        assert torch.equal(got, expected)
        assert after["executed_steps"] == before["executed_steps"]
        assert after["final_semantic"] == before["final_semantic"]
