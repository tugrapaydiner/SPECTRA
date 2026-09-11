"""Independent finite-group oracle and integration tests; no learned data gates."""
import itertools
import json

import numpy as np
import pytest

from data import sudoku
from data.ancestry import ArtifactAncestry, ExclusionIndex, ManifestSource, digest
from data.sudoku_symmetry import (POLICY, Sudoku4OrbitLookup, _spatial_maps,
                                  canonical_sudoku4, sudoku4_orbit_key)
from scripts._common import build_data_splits
from scripts.m14_primary_experiment import make_cfg

SOLUTION = np.array([1,2,3,4,3,4,1,2,2,1,4,3,4,3,2,1])
PUZZLE = SOLUTION.copy()
PUZZLE[[0,2,5,6,9,11,12,14]] = 0


def independent_spatial_group():
    """Closure of generators, not the production direct-product construction."""
    grid = np.arange(16).reshape(4, 4)
    generators = [grid.T.ravel()]
    for order in ([1,0,2,3], [0,1,3,2], [2,3,0,1]):
        generators.append(grid[order, :].ravel())
        generators.append(grid[:, order].ravel())
    pending = [tuple(range(16))]
    seen = set(pending)
    while pending:
        row = np.array(pending.pop())
        for generator in generators:
            child = tuple(row[generator])
            if child not in seen:
                seen.add(child)
                pending.append(child)
    return sorted(seen)


def test_spatial_group_equals_independent_generator_closure():
    oracle = independent_spatial_group()
    assert len(oracle) == 128
    assert set(oracle) == set(map(tuple, _spatial_maps()))
    for indices in oracle:
        assert sudoku.is_solved(SOLUTION[list(indices)].reshape(4, 4), 2)


def test_canonical_matches_all_3072_explicit_transformations():
    maps = independent_spatial_group()
    candidates = []
    for digits in itertools.permutations(range(1, 5)):
        labels = np.array([0, *digits], dtype=np.uint8)
        candidates.extend(labels[PUZZLE[list(indices)]].tobytes() for indices in maps)
    canonical = canonical_sudoku4(PUZZLE)
    assert bytes(canonical.board) == min(candidates)
    # Every spatial/digit group image has the same representative, not just a
    # convenient sample of rotations. These checks do not add pytest test counts.
    expected = canonical.key
    for raw in candidates:
        assert sudoku4_orbit_key(np.frombuffer(raw, dtype=np.uint8)) == expected


@pytest.mark.parametrize("board", [PUZZLE, SOLUTION, np.zeros(16, dtype=int),
                                    np.array([4] + [0]*15)])
def test_witness_is_complete_bijection_and_roundtrips(board):
    witness = canonical_sudoku4(board)
    assert sorted(witness.positions) == list(range(16))
    assert sorted(witness.labels) == list(range(5)) and witness.labels[0] == 0
    assert tuple(witness.transform(board)) == witness.board
    assert np.array_equal(witness.restore(witness.transform(SOLUTION)), SOLUTION)
    assert sudoku4_orbit_key(np.array(witness.board)) == witness.key


@pytest.mark.parametrize("board", [np.zeros(81, dtype=int), np.zeros((1,16), dtype=int),
    np.zeros(16), np.zeros(16, dtype=bool), np.full(16, -1), np.full(16, 5)])
def test_rejects_wrong_geometry_dtype_and_symbols(board):
    with pytest.raises(ValueError):
        canonical_sudoku4(board)


def test_different_clue_counts_do_not_collapse():
    assert sudoku4_orbit_key(SOLUTION) != sudoku4_orbit_key(PUZZLE)
    assert sudoku4_orbit_key(PUZZLE) != sudoku4_orbit_key(np.zeros(16, dtype=int))


def test_training_only_lookup_restores_solution_and_abstains():
    lookup = Sudoku4OrbitLookup(PUZZLE[None], SOLUTION[None])
    assert lookup.orbit_count == 1
    labels = np.array([0,3,1,4,2])
    inp = labels[PUZZLE.reshape(4, 4).T.ravel()]
    answer = lookup.solve(inp)
    assert answer is not None and sudoku.is_solved(answer.reshape(4, 4), 2)
    assert sudoku.respects_clues(inp.reshape(4, 4), answer.reshape(4, 4), 2)
    answer[:] = 0  # caller cannot mutate the table
    assert sudoku.is_solved(lookup.solve(inp).reshape(4, 4), 2)
    assert lookup.solve(np.zeros(16, dtype=int)) is None
    with pytest.raises(ValueError, match="valid"):
        Sudoku4OrbitLookup(PUZZLE[None], np.zeros((1,16), dtype=int))


def test_legacy_default_recipe_and_manifest_unchanged():
    _, default = build_data_splits(make_cfg(773), 16, 8, 8)
    _, explicit = build_data_splits(make_cfg(773), 16, 8, 8, grouping="exact")
    assert default == explicit
    assert "require_unique_groups" not in default["split_policy"]


def test_symmetry_grouped_splits_and_existing_ancestry_close_over_orbits():
    cfg = make_cfg(774)
    _, first = build_data_splits(cfg, 48, 12, 12, grouping=POLICY, require_unique_groups=True)
    raw = json.dumps(first).encode()
    source = ManifestSource.from_bytes("first", raw, role="training", expected_sha256=digest(raw))
    model_sha = digest(b"fixture-model")
    index = ExclusionIndex.close([model_sha], {model_sha: ArtifactAncestry(model_sha, manifests=(source,))})
    datasets, second = build_data_splits(cfg, 24, 8, 8, grouping=POLICY,
        require_unique_groups=True, forbidden_fingerprints=index.forbidden)
    groups = [r["group_id"] for split in second["splits"].values() for r in split["examples"]]
    assert len(groups) == len(set(groups)) == 40
    assert not set(groups) & source.groups
    for name, ds in datasets.items():
        assert [sudoku4_orbit_key(x) for x in ds.inputs] == ds.group_ids
    assert all(v == 0 for v in second["duplicate_audit"]["cross_split_group_overlap"].values())
    assert all(v == 0 for v in second["duplicate_audit"]["within_split_duplicate_groups"].values())
    raw2 = json.dumps(second).encode()
    candidate = ManifestSource.from_bytes("second", raw2, role="development", expected_sha256=digest(raw2))
    assert index.require_disjoint(candidate)["disjoint"]
    _, replay = build_data_splits(cfg, 24, 8, 8, grouping=POLICY,
        require_unique_groups=True, forbidden_fingerprints=index.forbidden)
    assert replay == second


def test_augmentation_stays_inside_declared_orbit():
    cfg = make_cfg(775)
    cfg.data.augment = True
    datasets, manifest = build_data_splits(cfg, 12, 4, 4, grouping=POLICY)
    for name, ds in datasets.items():
        assert [sudoku4_orbit_key(x) for x in ds.inputs] == ds.group_ids


def test_policy_rejects_unknown_or_unsupported_geometry():
    with pytest.raises(ValueError, match="unknown"):
        build_data_splits(make_cfg(774), 0, 0, 0, grouping="made_up")
    cfg = make_cfg(776)
    cfg.data.box = 3
    cfg.data.num_tokens = 10
    cfg.data.height = cfg.data.width = 9
    cfg.data.seq_len = 81
    with pytest.raises(ValueError, match="box=2"):
        build_data_splits(cfg, 0, 0, 0, grouping=POLICY)


def test_equivalent_resampling_cannot_cross_split_or_escape_retry_limit(monkeypatch):
    from data import splits
    calls = itertools.count()
    def generate(*args, **kwargs):
        labels = np.array([0,2,3,4,1]) if next(calls) % 2 else np.arange(5)
        return labels[PUZZLE], labels[SOLUTION], 4, 4, {}
    monkeypatch.setattr(splits, '_generate_base', generate)
    with pytest.raises(RuntimeError, match="non-leaking"):
        splits.build_reproducible_splits('sudoku', {'train':1, 'validation':1}, 5,
            generator_kwargs={'box':2, 'num_tokens':5, 'seq_len':16, 'augment':False},
            task_scope='fixture', official_benchmark=False, grouping=POLICY,
            max_duplicate_retries=2)


def test_wrong_augmentation_cannot_silently_change_declared_group(monkeypatch):
    from data import splits
    monkeypatch.setattr(splits, '_generate_base', lambda *a, **k: (PUZZLE, SOLUTION, 4, 4, {}))
    monkeypatch.setattr(splits, 'augment_example', lambda *a, **k: (np.zeros(16, dtype=int), SOLUTION))
    with pytest.raises(RuntimeError, match="escaped"):
        splits.build_reproducible_splits('sudoku', {'train':1}, 5,
            generator_kwargs={'box':2, 'num_tokens':5, 'seq_len':16, 'augment':True},
            task_scope='fixture', official_benchmark=False, grouping=POLICY)
