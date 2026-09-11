"""Independent native/reference checks; none of these are M17 study seeds."""
import numpy as np
import pytest
import torch

from data import maze
from deploy.m10_native import load_extension
from eval.checkable_tasks import TaskSpec, action_directions, semantic_exit
from model.trm import TRM


@pytest.fixture(scope="module")
def native():
    return load_extension()


def pair(seed=311, h=9, w=9):
    x, y = maze.generate_pair(h, w, np.random.default_rng(seed))
    return torch.from_numpy(x.reshape(1, -1)), torch.from_numpy(y.reshape(1, -1))


def reference(x, y, h, w, optimal=True):
    return maze.candidate_success(x.numpy().reshape(h, w), y.numpy().reshape(h, w), h, w, require_optimal=optimal)


@pytest.mark.parametrize("h,w", [(3, 5), (5, 7), (9, 9), (11, 11)])
def test_native_generated_maze_parity(native, h, w):
    rng = np.random.default_rng(733)
    for seed in range(7):
        x, target = pair(seed+811, h, w)
        problem = native.MazeProblem(x, h, w, True)
        assert problem.check(target) is True
        shortest = problem.shortest_solution()
        assert torch.equal(shortest, target)
        assert reference(x, shortest, h, w)
        for _ in range(8):
            candidate = target.clone()
            for index in rng.integers(0, h*w, 4):
                candidate[0, index] = int(rng.integers(-1, 7))
            assert problem.check(candidate) == reference(x, candidate, h, w)


def test_native_maze_general_graph_optimality_is_not_assumed(native):
    x = torch.zeros((1, 25), dtype=torch.int64)
    for r, c in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]:
        x[0, r*5+c] = maze.OPEN
    x[0, 6], x[0, 8] = maze.START, maze.GOAL
    long_route = x.clone()
    for r, c in [(2, 1), (3, 1), (3, 2), (3, 3), (2, 3)]:
        long_route[0, r*5+c] = maze.PATH
    assert native.MazeProblem(x, 5, 5, False).check(long_route)
    assert reference(x, long_route, 5, 5, False)
    assert not native.MazeProblem(x, 5, 5, True).check(long_route)
    assert not reference(x, long_route, 5, 5, True)


def test_detached_cycle_is_not_a_valid_path(native):
    x = torch.zeros((1, 63), dtype=torch.int64)
    x[0, 10], x[0, 11], x[0, 12] = maze.START, maze.OPEN, maze.GOAL
    cycle = [4*9+5, 4*9+6, 5*9+5, 5*9+6]
    x[0, cycle] = maze.OPEN
    candidate = x.clone()
    candidate[0, [11]+cycle] = maze.PATH
    assert not reference(x, candidate, 7, 9, False)
    assert not native.MazeProblem(x, 7, 9, False).check(candidate)


def test_maze_problem_owns_input_and_returned_answers(native):
    x, target = pair()
    problem = native.MazeProblem(x, 9, 9, True)
    original = x.clone()
    x.zero_()
    assert problem.check(target)
    answer = problem.shortest_solution()
    answer.zero_()
    assert problem.check(problem.shortest_solution())
    decoded, valid = problem.decode(torch.zeros(1, 81, 5))
    assert torch.equal(decoded, original)
    assert valid == reference(original, decoded, 9, 9)


def test_maze_logits_restore_argmax_ties_and_check_parity(native):
    x, _ = pair()
    problem = native.MazeProblem(x, 9, 9, True)
    spec = TaskSpec("maze", 9, 9, 16, 16)
    generator = torch.Generator().manual_seed(807)
    for logits in [torch.zeros(1, 81, 5), torch.ones(1, 81, 5),
                   torch.randn(1, 81, 5, generator=generator)]:
        y, flag = problem.decode(logits)
        expected = spec.restore(x, logits.argmax(-1))
        assert torch.equal(y, expected)
        assert flag == reference(x, y, 9, 9)
    logits = torch.zeros(1, 81, 5)
    logits[..., 0], logits[..., 4] = 3.0, 2.0
    y, _ = problem.decode(logits)
    assert torch.equal(y, x)  # Not OPEN/PATH-only argmax.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_maze_decode_rejects_nonfinite_even_at_immutable_cells(native, bad):
    x, _ = pair()
    logits = torch.zeros(1, 81, 5)
    logits[0, 0, 4] = bad  # A wall cell is still part of the finite-logit contract.
    with pytest.raises(RuntimeError, match="finite"):
        native.MazeProblem(x, 9, 9, True).decode(logits)


@pytest.mark.parametrize("which", ["no_start", "two_starts", "bad_token", "unreachable"])
def test_malformed_or_unsatisfiable_maze_is_not_certified(native, which):
    x, target = pair()
    if which == "no_start":
        x[x == maze.START] = maze.OPEN
    elif which == "two_starts":
        x[0, int(torch.nonzero(x[0] == maze.OPEN)[0])] = maze.START
    elif which == "bad_token":
        x[0, 0] = 9
    else:
        x[x == maze.OPEN] = maze.WALL
    problem = native.MazeProblem(x, 9, 9, True)
    assert not problem.check(target)
    assert not problem.check(problem.shortest_solution())


def test_native_maze_rejects_malformed_tensor_boundaries(native):
    x, y = pair()
    with pytest.raises(RuntimeError):
        native.MazeProblem(x.float(), 9, 9, True)
    with pytest.raises(RuntimeError):
        native.MazeProblem(x, 0, 9, True)
    with pytest.raises(RuntimeError):
        native.MazeProblem(x, 9, 8, True)
    p = native.MazeProblem(x, 9, 9, True)
    for bad in [y.float(), y.flatten(), y[:, :80]]:
        with pytest.raises(RuntimeError):
            p.check(bad)
    with pytest.raises(RuntimeError):
        p.decode(torch.zeros(1, 81, 5, dtype=torch.float64))


def test_task_bound_native_and_reference_cycles_agree(native):
    spec = TaskSpec("maze", 5, 5, 16, 8)
    torch.manual_seed(727)
    model = TRM(dim=16, num_tokens=5, seq_len=25, n_layers=1, n=1, T=1,
                N_sup=4, heads=4, max_grid_size=8).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x, _ = pair(449, 5, 5)
    for cycles in (1, 4, 6):
        a, wa = semantic_exit(model, x, spec, cycles, native=True)
        b, wb = semantic_exit(model, x, spec, cycles, native=False)
        assert torch.equal(a, b)
        assert (wa["valid"], wa["transitions"]) == (wb["valid"], wb["transitions"])
        assert wa["depth_extrapolation"] == (cycles > 4)
        assert wa["checker_constructions"] == 1
    model.train()
    with pytest.raises(ValueError, match="frozen"):
        semantic_exit(model, x, spec)


def test_task_and_action_contract_boundaries():
    for args in [("other", 4, 4, 16, 8), ("sudoku", 4, 4, 16, 8, 3), ("maze", 9, 9, 16, 8)]:
        with pytest.raises(ValueError):
            TaskSpec(*args)
    d = action_directions(16)
    assert torch.equal(d[0], torch.zeros(16))
    assert torch.allclose(d[1:].norm(dim=1), torch.full((3,), .5))
    assert torch.equal(d, action_directions(16))


def test_degenerate_generator_is_rejected_before_emitting_a_bad_target():
    from data.task_contracts import contract_from_config
    with pytest.raises(ValueError, match="distinct"):
        maze.generate_pair(3, 3, np.random.default_rng(73))
    with pytest.raises(ValueError, match="distinct"):
        contract_from_config("maze", {"num_tokens": 5, "height": 3, "width": 3, "seq_len": 9})


def test_native_checker_supports_distinct_endpoints_in_a_small_general_grid(native):
    x = torch.tensor([[0, 0, 0, 2, 1, 3, 0, 0, 0]])
    target = x.clone(); target[0, 4] = 4
    assert native.MazeProblem(x, 3, 3, True).check(target)
    assert reference(x, target, 3, 3)


def test_exhaustive_3x3_fixed_endpoint_states_match_reference(native):
    from itertools import product
    # Every wall/open/path assignment of the other seven cells, under both
    # optimality contracts. 2 * 3**7 = 4,374 complete native/reference checks.
    free = list(range(1, 8))
    for values in product([maze.WALL, maze.OPEN, maze.PATH], repeat=7):
        answer = torch.tensor([[maze.START, *values, maze.GOAL]])
        x = answer.clone(); x[x == maze.PATH] = maze.OPEN
        for optimal in (False, True):
            assert native.MazeProblem(x, 3, 3, optimal).check(answer) == reference(x, answer, 3, 3, optimal)
