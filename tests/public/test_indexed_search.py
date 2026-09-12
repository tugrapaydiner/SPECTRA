from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import itertools
import random
import pytest
from data.cnf import CNF, random_3sat
from spectra.cnf import PreparedCNF, solve, solve_indexed
from spectra.cnf.indexed import _BreakState, _RankedCompactState, _solve_ranked
from spectra.cnf.ranked import RankedSet


def semantic(result):
    return {k: v for k, v in result.record().items() if k != 'elapsed_ns'}


@pytest.mark.parametrize('size', range(10))
def test_every_small_ranked_subset(size):
    for flags in itertools.product((False, True), repeat=size):
        expected = [i for i, present in enumerate(flags) if present]
        state = RankedSet(size, expected + expected)
        assert list(state) == expected
        assert len(state) == len(expected)
        assert [state.select(i) for i in range(len(state))] == expected
        for bad in (-1, len(state), True, 1.0):
            with pytest.raises(IndexError):
                state.select(bad)


@pytest.mark.parametrize('size', [0, 1, 255, 256, 257, 511, 512, 513, 4097, 65537])
def test_ranked_dynamic_updates_and_boundaries(size):
    rng = random.Random(7931 + size)
    reference = set(range(size))
    ranked = RankedSet(size, reference)
    for _ in range(300):
        if not size:
            assert not ranked and list(ranked) == []
            continue
        value = rng.randrange(size)
        if rng.randrange(2):
            ranked.add(value); reference.add(value)
        else:
            ranked.discard(value); reference.discard(value)
        expected = sorted(reference)
        assert len(ranked) == len(reference)
        assert list(ranked) == expected
        if expected:
            for rank in {0, len(expected)-1, rng.randrange(len(expected))}:
                assert ranked.select(rank) == expected[rank]
        assert (value in ranked) == (value in reference)
    for value in list(reference):
        ranked.remove(value)
    assert not ranked
    if size:
        with pytest.raises(KeyError):
            ranked.remove(0)


@pytest.mark.parametrize('bad', [-1, True, 1.0, '2', None])
def test_ranked_invalid_universe(bad):
    with pytest.raises(ValueError):
        RankedSet(bad)


def test_ranked_rejected_mutation_is_atomic():
    ranked = RankedSet(7, [1, 4, 6])
    for value in [-1, 7, True, 1.0, '1', None]:
        for method in [ranked.add, ranked.remove, ranked.discard]:
            with pytest.raises(ValueError):
                method(value)
            assert list(ranked) == [1, 4, 6]
        assert value not in ranked


def random_general_cnf(seed):
    rng = random.Random(seed)
    n = rng.randrange(1, 12)
    clauses = []
    for _ in range(rng.randrange(1, 40)):
        clause = tuple(rng.choice((-1, 1)) * rng.randrange(1, n+1)
                       for _ in range(rng.randrange(0, 7)))
        clauses.append(clause)
        if rng.randrange(8) == 0:
            clauses.append(clause)
    return CNF(n, tuple(clauses))


@pytest.mark.parametrize('formula_seed', range(24))
def test_break_only_cache_against_original_finite_differences(formula_seed):
    problem = random_general_cnf(formula_seed)
    index = PreparedCNF(problem)
    rng = random.Random(formula_seed + 1000)
    state = _BreakState(index, tuple(bool(rng.randrange(2)) for _ in range(problem.nvars)))
    for _ in range(16):
        witness = state.witness
        residuals = set(problem.violated(witness))
        assert list(state.residuals) == sorted(residuals)
        for v in range(problem.nvars):
            changed = tuple(not x if i == v else x for i, x in enumerate(witness))
            newly_violated = set(problem.violated(changed)) - residuals
            assert state.break_count(v) == len(newly_violated)
            assert 0 <= state.break_count(v) < len(index.weights)
        state.flip(rng.randrange(problem.nvars))


@pytest.mark.parametrize('formula_seed', range(32))
def test_complete_seeded_paths_equal_on_general_and_3sat_inputs(formula_seed):
    problems = [random_general_cnf(formula_seed),
                random_3sat(32, 134, formula_seed+74000, planted=bool(formula_seed % 2))[0]]
    for problem in problems:
        prepared = PreparedCNF(problem)
        for seed in (0, 17001, 2**64-1):
            for cap in (0, 1, 23, 128):
                expected = semantic(solve(problem, seed=seed, max_flips=cap))
                for result in (solve_indexed(problem, seed=seed, max_flips=cap),
                               prepared.solve(seed=seed, max_flips=cap),
                               _solve_ranked(problem, seed=seed, max_flips=cap)):
                    assert semantic(result) == expected
                    assert result.elapsed_ns > 0
                    assert problem.violated(result.witness) == result.unsatisfied


@pytest.mark.parametrize('problem', [CNF(0, ()), CNF(0, ((),)), CNF(2, ()),
    CNF(2, ((1, -1), (2, -2))), CNF(2, ((1, 1), (-1,), (2, -2), (1, 1))),
    CNF(3, ((1, 2, 3),)), CNF(4, ((4,), (-4,), (4, -4), ()))])
def test_degenerate_and_nonempty_xor_zero_inputs(problem):
    for seed in range(10):
        for cap in (0, 100):
            expected = semantic(solve(problem, seed=seed, max_flips=cap))
            assert semantic(solve_indexed(problem, seed=seed, max_flips=cap)) == expected
            assert semantic(PreparedCNF(problem).solve(seed=seed, max_flips=cap)) == expected


def test_index_immutable_shared_preparation_not_shared_mutation():
    problem = random_3sat(128, 537, 9427)[0]
    index = PreparedCNF(problem)
    first = _BreakState(index, (True,) * 128)
    second = _BreakState(index, (True,) * 128)
    first.flip(3)
    assert second.witness == (True,) * 128
    assert first.index is second.index
    for name in ('assignment', 'counts', 'xors', 'breaks', 'residuals'):
        assert getattr(first, name) is not getattr(second, name)
    with pytest.raises(FrozenInstanceError):
        index.problem = CNF(0, ())
    assert all(type(o) is tuple for o in index.occurrences)
    seeds = list(range(8))
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda s: semantic(index.solve(seed=s, max_flips=512)), seeds))
    assert actual == [semantic(solve(problem, seed=s, max_flips=512)) for s in seeds]


def test_ranked_full_cache_joint_patch_interface_stays_exact():
    problem = CNF(3, ((1, 2, 3), (-1, -2), (3,), (), (2, 2)))
    state = _RankedCompactState(problem, (True, True, True))
    for patch in [(0, 1), (1, 2), (0, 1, 2), (2,), (0, 2)]:
        before = set(problem.violated(state.witness))
        future = tuple(not x if i in patch else x for i, x in enumerate(state.witness))
        after = set(problem.violated(future))
        assert state.make_break_patch(patch) == (len(before-after), len(after-before))
        state.flip_patch(patch)
        assert state.unsatisfied == tuple(sorted(after))
        assert list(state.residuals) == list(state.unsatisfied)


@pytest.mark.parametrize('seed, cap', [(-1, 2), (2**64, 2), (True, 2), (1.0, 2),
                                     (0, -1), (0, True), (0, 1.0)])
def test_invalid_search_settings(seed, cap):
    p = CNF(1, ((1,),))
    for fn in [lambda: solve_indexed(p, seed=seed, max_flips=cap),
               lambda: PreparedCNF(p).solve(seed=seed, max_flips=cap),
               lambda: _solve_ranked(p, seed=seed, max_flips=cap)]:
        with pytest.raises(ValueError):
            fn()


def test_invalid_problem():
    for bad in [None, {}, 'p cnf 0 0', True]:
        with pytest.raises(TypeError):
            PreparedCNF(bad)
        with pytest.raises(TypeError):
            solve_indexed(bad)


def test_indexed_cli_witness_and_no_overwrite(tmp_path, capsys):
    from spectra.cli import main
    source = tmp_path / 'x.cnf'; source.write_text('p cnf 1 1\n1 0\n')
    out = tmp_path / 'out.json'
    assert main(['cnf', 'solve', str(source), '--backend', 'indexed', '--out', str(out)]) == 0
    assert main(['cnf', 'check', str(source), str(out)]) == 0
    assert main(['cnf', 'solve', str(source), '--backend', 'indexed', '--out', str(out)]) == 2
