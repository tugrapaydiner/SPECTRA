"""Small CPU fixtures exercise the runner without opening any study data seed."""
from pathlib import Path

import numpy as np
import pytest
import torch

from common import load_config
from data.ancestry import ManifestSource, digest
from eval.checkable_tasks import MAZE11, TaskSpec, semantic_exit
from eval.verified_search import ValueContract, ValueTarget
from scripts._common import build_data_splits
from scripts.m16_evidence import write_json
import scripts.m17_models as models
from scripts.m17_cross_task import FAMILIES, PROTOCOL_COMMIT, checked_search, closed_summary, symbolic_solve


def fixture_data(tmp_path):
    cfg = load_config("config/maze.yaml", overrides=["seed=99113", "device=cpu",
        "data.height=11", "data.width=11", "data.seq_len=121", "data.augment=false"])
    ds, manifest = build_data_splits(cfg, 8, 4, 4, seed=99113, require_unique_examples=True)
    path = tmp_path/"fixture_manifest.json"
    write_json(path, manifest)
    return ds, digest(path.read_bytes())


def test_frozen_m17_parameters_and_never_using_reserved_seeds_in_fixtures():
    assert PROTOCOL_COMMIT == "fe15894d1d57afb99a9235462c76acde3fa28d89"
    assert models.CORE_STEPS == 1800 and models.CORE_BATCH == 32
    assert models.VALUE_STEPS == 180 and models.VALUE_BATCH == 64
    assert models.MAZE_SEEDS == [1701, 2702]
    assert models.WEIGHTS == [.1, .2, .3, .4]
    assert FAMILIES["maze"]["sizes"] == (1024, 128, 128)
    assert FAMILIES["sudoku_shift"]["sizes"] == (0, 64, 128)
    declared_seeds = {v[k] for v in FAMILIES.values() for k in ("development_seed", "confirmation_seed")}
    assert 99113 not in declared_seeds


def test_small_core_fit_pool_targets_and_value_roundtrip(tmp_path, monkeypatch):
    # In-memory test-only counts; the CLI recipe has no shorter-training switch.
    monkeypatch.setattr(models, "CORE_STEPS", 2)
    monkeypatch.setattr(models, "VALUE_STEPS", 2)
    ds, sha = fixture_data(tmp_path)
    core, core_sha, record = models.train_maze_core(733, ds, sha, tmp_path/"core")
    assert record["optimizer_steps"] == 2 and record["final_checkpoint_only"]
    assert all(not p.requires_grad for p in core.parameters())
    x = torch.from_numpy(ds["train"].inputs[:2]).long()
    pool = models.candidate_pool(core, x, MAZE11, batch_size=1)
    assert pool["x"].shape == (32, 121)
    assert pool["labels"].shape == (32, 3)
    assert set(pool["example_index"].tolist()) == {0, 1}
    assert not pool["y"].is_inference()  # Safe to use as training inputs.
    assert not any(p.grad is not None for p in core.parameters())
    fits = []
    for i in range(3):
        _, contract, fit = models.fit_maze_value(pool, i, 733, core_sha, sha, tmp_path/"values")
        fits.append(fit)
        assert contract.model_sha256 == core_sha
    assert len({r["initial_state_sha256"] for r in fits}) == 1
    assert len({r["sampled_index_sha256"] for r in fits}) == 1
    assert len({r["training_pool_sha256"] for r in fits if "training_pool_sha256" in r}) == 0
    # Changing reference targets cannot influence the input-only pool builder.
    ds["train"].targets[:] = 0
    again = models.candidate_pool(core, x, MAZE11, batch_size=1)
    assert all(torch.equal(pool[k], again[k]) for k in pool)


def test_checked_search_and_bfs_are_charged_and_independently_correct(tmp_path):
    ds, _ = fixture_data(tmp_path)
    torch.manual_seed(333)
    core = models.freeze(models.TRM(**models.core_architecture(MAZE11)).cpu())
    x = torch.from_numpy(ds["validation"].inputs[:1]).long()
    core_sha = "4"*64
    contract = ValueContract(ValueTarget.QUALITY, core_sha, "5"*64, MAZE11.transition_id,
                             state_schema=MAZE11.state_schema)
    answer, work = checked_search(core, None, contract, x, MAZE11, core_sha)
    assert work["transitions"] <= 24
    assert work["checks"] == work["decodes"] == work["native_semantic_checks"]
    assert work["checker_constructions"] == work["input_embeddings"] == 1
    assert work["valid"] == MAZE11.independent_correct(x, answer)
    baseline, bw = semantic_exit(core, x, MAZE11, 4)
    assert not bw["valid"] or work["valid"]
    for native in (False, True):
        answer, sw = symbolic_solve(x, MAZE11, native=native)
        assert sw["valid"] and MAZE11.independent_correct(x, answer)
        assert sw["bfs_preprocessing_charged"] and sw["checker_constructions"] == 1


def timing(arm="native_k4", *, valid=True, repeat=0, example="a"):
    return {"core_seed": 1, "example_id": example, "round": repeat, "arm": arm,
            "valid": valid, "latency_ms": [1., 10., 2.][repeat], "answer": [1 if valid else 0],
            "work": {"transitions": 1, "value_calls": 0}}


def test_closed_summary_uses_three_round_medians_not_repeated_correctness_trials():
    rr = [timing(repeat=i) for i in range(3)]
    s = closed_summary(rr)["arms"]["native_k4"]
    assert s["model_example_pairs"] == 1 and s["timing_rows"] == 3
    assert s["mean_ms_after_round_medians"] == 2.
    assert s["valid_answers"] == 1


def test_closed_summary_rejects_unpaired_or_unstable_answers_and_regression():
    base = [timing(repeat=i) for i in range(3)]
    with pytest.raises(ValueError, match="three"):
        closed_summary(base[:2])
    bad = base.copy(); bad[-1] = timing(repeat=2, valid=False)
    with pytest.raises(ValueError, match="changed"):
        closed_summary(bad)
    with pytest.raises(ValueError, match="paired"):
        closed_summary(base+[timing("checked_quality", repeat=i, example="b") for i in range(3)])
    with pytest.raises(AssertionError, match="lost"):
        closed_summary(base+[timing("checked_quality", repeat=i, valid=False) for i in range(3)])
