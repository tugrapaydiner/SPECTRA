"""Independent cache, budget and witness checks for the opt-in policy study."""
import itertools
import json
import random

import pytest

from spectra.cnf import CNF, SplitMix64
from spectra.cnf.indexed import PreparedCNF
from spectra.cnf.focused import _DenseResiduals, _pick, _state, solve_focused


def violated(problem, witness):
    return tuple(i for i, clause in enumerate(problem.clauses)
                 if not any(witness[abs(lit)-1] == (lit > 0) for lit in clause))


def test_dense_pool_random_mutations_against_set():
    pool, reference, rng = _DenseResiduals(80, ()), set(), random.Random(7521)
    for _ in range(3000):
        value = rng.randrange(80)
        if value in reference and rng.randrange(2):
            pool.remove(value)
            reference.remove(value)
        else:
            pool.add(value)
            reference.add(value)
        assert len(pool) == len(reference)
        assert set(pool) == reference
        assert {pool.select(i) for i in range(len(pool))} == reference
    with pytest.raises(KeyError):
        pool.remove(next(v for v in range(80) if v not in reference))


def test_dense_break_state_against_literal_rescan_after_every_flip():
    rng = random.Random(202610024333)
    for _ in range(60):
        p = CNF(7, tuple(tuple(rng.choice(tuple(range(1, 8))+tuple(range(-7, 0)))
                              for _ in range(rng.randrange(6))) for _ in range(20)))
        state = _state(PreparedCNF(p), SplitMix64(rng.randrange(10000)))
        for _ in range(80):
            w = state.witness
            bad = set(violated(p, w))
            assert set(state.residuals) == bad
            for v in range(7):
                flipped = w[:v]+(not w[v],)+w[v+1:]
                assert state.breaks[v] == len(set(violated(p, flipped))-bad)
            state.flip(rng.randrange(7))


@pytest.mark.parametrize('policy', ['poly', 'freebie', 'minbreak', 'sharp', 'anti_reverse', 'novelty_break'])
def test_exhaustive_small_formulas_have_valid_witnesses_and_budget(policy):
    clauses = [(), (1,), (-1,), (2,), (-2,), (1, 2), (1, -2), (-1, 2), (-1, -2)]
    for mask in range(1 << len(clauses)):
        p = CNF(2, tuple(c for i, c in enumerate(clauses) if mask & (1 << i)))
        r = solve_focused(p, seed=42, max_flips=64, policy=policy, restart_interval=7)
        assert r.unsatisfied == violated(p, r.witness)
        assert (r.status == 'SAT_VERIFIED') == (not r.unsatisfied)
        assert 0 <= r.flips <= 64 and r.restarts <= 9
        if r.status == 'SAT_VERIFIED':
            assert any(not violated(p, w) for w in itertools.product((False, True), repeat=2))


def test_freebie_and_minbreak_never_ignore_zero_break():
    for policy in ('freebie', 'minbreak'):
        for seed in range(100):
            assert _pick((0, 1, 2), [2, 0, 3], (1, .2, .1, .05), SplitMix64(seed), policy) == 1


def test_zero_budget_tautologies_and_empty_clause():
    for p in (CNF(0, ()), CNF(0, ((),)), CNF(2, ((1, -1), (2, -2))), CNF(2, ((), (1,)))):
        r = solve_focused(p, max_flips=0)
        assert r.flips == r.restarts == 0
        assert r.unsatisfied == violated(p, r.witness)
    assert solve_focused(CNF(2, ((), (1,))), max_flips=100).flips == 0


def test_restart_accounting_and_determinism():
    p = CNF(1, ((1,), (-1,)))
    first = solve_focused(p, max_flips=31, restart_interval=5).record()
    second = solve_focused(p, max_flips=31, restart_interval=5).record()
    assert first['flips'] == 31 and first['restarts'] == 6
    first.pop('elapsed_ns'); second.pop('elapsed_ns')
    assert first == second
    assert first['learned'] is False and first['restart_interval'] == 5


@pytest.mark.parametrize('settings', [dict(seed=True), dict(seed=-1), dict(seed=2**64),
                                    dict(max_flips=True), dict(max_flips=-1), dict(policy='bad'),
                                    dict(restart_interval=True), dict(restart_interval=-1)])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        solve_focused(CNF(1, ((1,),)), **settings)


def test_default_is_the_selected_age_policy_and_cli_checks_original(tmp_path, capsys):
    from spectra.cnf import solve_focused as public
    from spectra.cli import main
    p = CNF(3, ((1, 2, 3), (-1, -2, 3)))
    result = public(p, seed=7, max_flips=2048)
    assert result.policy == 'novelty_break' and result.restart_interval == 0
    source, output = tmp_path/'input.cnf', tmp_path/'answer.json'
    source.write_text('p cnf 3 2\n1 2 3 0\n-1 -2 3 0\n')
    assert main(['cnf', 'solve', str(source), '--backend', 'focused', '--seed', '7', '--out', str(output)]) == 0
    record = json.loads(output.read_text())
    assert record['status'] == 'SAT_VERIFIED' and record['policy'] == 'novelty_break'
    assert main(['cnf', 'check', str(source), str(output)]) == 0
