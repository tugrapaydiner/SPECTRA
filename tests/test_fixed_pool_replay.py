"""Independent replay contracts use small fixtures, not study confirmation seeds."""
from copy import deepcopy
import io
import json

import numpy as np
import pytest
import torch

from data import maze
from data.splits import example_fingerprint
from eval.checkable_tasks import MAZE11, SUDOKU_SHIFT
from eval.fixed_pool import TARGET_NAMES, score_pool
from eval.fixed_pool_replay import compare_records, decode_metrics, reconstruct_pool, replay_rows
from eval.verified_search import ValueTarget
from model.task_value import architecture
from model.typed_value import TypedStateValue
from scripts.m17_models import candidate_pool, core_architecture, freeze, TRM
from scripts.verify_fixed_pool_replay import frozen_inputs


def fixture_input(spec):
    if spec == SUDOKU_SHIFT:
        return torch.tensor([[1, 0, 0, 4, 0, 4, 1, 0, 0, 1, 4, 0, 4, 0, 0, 1]])
    grid, start, goal = maze.generate_maze(11, 11, np.random.default_rng(99121))
    return torch.from_numpy(maze.make_input(grid, start, goal).reshape(1, -1))


@pytest.mark.parametrize("spec", [SUDOKU_SHIFT, MAZE11])
def test_independent_pool_matches_production_tensor_by_tensor(spec):
    torch.manual_seed(71123)
    core = freeze(TRM(**core_architecture(spec)).cpu())
    x = fixture_input(spec).repeat(2, 1)
    saved = x.clone()
    independent, diagnostic = reconstruct_pool(core, x, spec, batch_size=1)
    production = candidate_pool(core, x, spec, batch_size=1)
    assert set(independent) == set(production)
    assert all(torch.equal(independent[k], production[k]) for k in production)
    assert torch.equal(x, saved)
    assert diagnostic["candidate_decodes"] == diagnostic["continuation_decodes"] == 32
    assert diagnostic["maze_restricted_score_failures"] == 0
    assert not any(p.grad is not None for p in core.parameters())
    values = {t: freeze(TypedStateValue(t, **architecture(spec)).cpu())
              for t in (ValueTarget.IMPROVEMENT, ValueTarget.TERMINAL, ValueTarget.QUALITY)}
    rows = replay_rows(71123, ["fixture-a", "fixture-b"], independent, values, spec, "development")
    assert compare_records(rows, deepcopy(rows))["head_scores_compared"] == 96
    assert all(r["reference_target_used"] is False for r in rows)
    with pytest.raises(ValueError, match="unique"):
        replay_rows(71123, ["a", "a"], independent, values, spec, "development")
    with pytest.raises(ValueError, match="missing or unexpected"):
        replay_rows(71123, ["a"], independent, values, spec, "development")
    values[ValueTarget.QUALITY].train()
    with pytest.raises(ValueError, match="frozen"):
        replay_rows(71123, ["a", "b"], independent, values, spec, "development")


def record():
    n = 16
    return score_pool({"core_seed": 19, "example_id": "fixture",
        "validity": [False]*15 + [True], "improvement_labels": [1.]+[0.]*15,
        "structural_quality": [.75]*15+[1.], "depths": [d for d in range(1, 5) for _ in range(4)],
        "actions": list(range(4))*4,
        "scores": {name: [float(i)/(n-1) for i in range(n)] for name in TARGET_NAMES},
        "surface": "development", "task": SUDOKU_SHIFT.metadata(), "reference_target_used": False})


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "label", "quality", "head",
                                      "score", "selection", "order", "task", "reference", "nan"])
def test_replay_never_accepts_corrupted_or_incomplete_records(mutation):
    old = [record()]
    new = deepcopy(old)
    if mutation == "missing": new = []
    elif mutation == "extra":
        r = record(); r["example_id"] = "extra"; new.append(r)
    elif mutation == "duplicate": new.append(deepcopy(new[0]))
    elif mutation == "label": new[0]["validity"][3] = True
    elif mutation == "quality": new[0]["structural_quality"][3] = .25
    elif mutation == "head": new[0]["scores"]["invented"] = [0.]*16
    elif mutation == "score": new[0]["scores"][TARGET_NAMES[0]][4] += .001
    elif mutation == "selection": new[0]["selected_indices"][TARGET_NAMES[0]] = 0
    elif mutation == "order": new[0]["actions"][0] = 3
    elif mutation == "task": new[0]["task"]["decode_id"] = "changed"
    elif mutation == "reference": new[0]["reference_target_used"] = True
    else: new[0]["scores"][TARGET_NAMES[0]][3] = float("nan")
    with pytest.raises((ValueError, KeyError)):
        compare_records(old, new)


def test_roundoff_allowance_never_permits_a_changed_tie_break():
    a = record()
    for name in TARGET_NAMES:
        a["scores"][name] = [.5]*16
    a = score_pool(a)
    b = deepcopy(a)
    b["scores"][TARGET_NAMES[0]][15] += 1e-7
    b = score_pool(b)
    with pytest.raises(ValueError, match="exact replay"):
        compare_records([a], [b])
    b = deepcopy(a)
    b["scores"][TARGET_NAMES[0]][0] += 1e-7
    b = score_pool(b)
    result = compare_records([a], [b])
    assert result["status"] == "PASS" and result["head_scores_exact"] == 47


@pytest.mark.parametrize("spec", [SUDOKU_SHIFT, MAZE11])
def test_numpy_decode_preserves_frozen_argmax_and_restoration(spec):
    x = fixture_input(spec)
    logits = torch.zeros(*x.shape, spec.num_tokens)
    answer, valid, quality = decode_metrics(x, logits, spec)
    expected = spec.restore(x, logits.argmax(-1))
    assert torch.equal(answer, expected)
    assert torch.equal(valid, spec.correct(x, expected))
    assert torch.equal(quality, spec.quality(x, expected))
    logits[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        decode_metrics(x, logits, spec)
    with pytest.raises(ValueError, match="FP32"):
        decode_metrics(x, torch.zeros(*x.shape, spec.num_tokens).double(), spec)


def test_maze_restoration_reduces_quality_to_three_levels():
    x = fixture_input(MAZE11)
    logits = torch.zeros(4, 121, 5)
    logits[:, :, maze.OPEN] = 1.
    opens = torch.nonzero(x[0] == maze.OPEN).flatten()
    logits[1, opens[0], maze.PATH] = 2.
    logits[2, opens, maze.PATH] = 2.
    target = MAZE11.symbolic_answer(x)
    logits[3, target[0] == maze.PATH, maze.PATH] = 2.
    answer, valid, quality = decode_metrics(x.repeat(4, 1), logits, MAZE11)
    nonempty = (answer == maze.PATH).any(-1)
    assert torch.equal(quality, .5 + .25*nonempty.float() + .25*valid.float())
    assert quality.tolist() == [.5, .75, .75, 1.]
    assert valid.tolist() == [False, False, False, True]


def members_fixture():
    x = fixture_input(SUDOKU_SHIFT).numpy()
    y = np.array([[1, 2, 3, 4, 3, 4, 1, 2, 2, 1, 4, 3, 4, 3, 2, 1]], dtype=np.int64)
    fp = example_fingerprint("sudoku", x[0], y[0], 4, 4)
    manifest = {"task": "sudoku", "seed": 71123,
        "splits": {"test": {"count": 1, "examples": [{"id": "fixture", "fingerprint": fp}]}}}
    stream = io.BytesIO()
    np.savez(stream, test_inputs=x, test_targets=y)
    prefix = "fixture/manifests/seed71123"
    return {prefix+".json": json.dumps(manifest).encode(), prefix+"_arrays.npz": stream.getvalue()}


def test_frozen_inputs_reject_changed_content_without_generating_data():
    members = members_fixture()
    x, ids, sha = frozen_inputs(members, "fixture/", 71123, SUDOKU_SHIFT)
    assert torch.equal(x, fixture_input(SUDOKU_SHIFT)) and ids == ["fixture"] and len(sha) == 64
    name = "fixture/manifests/seed71123_arrays.npz"
    with np.load(io.BytesIO(members[name])) as arrays:
        inp, target = arrays["test_inputs"].copy(), arrays["test_targets"].copy()
    target[0, 0] = 4
    stream = io.BytesIO(); np.savez(stream, test_inputs=inp, test_targets=target)
    members[name] = stream.getvalue()
    with pytest.raises(ValueError, match="fingerprint"):
        frozen_inputs(members, "fixture/", 71123, SUDOKU_SHIFT)
