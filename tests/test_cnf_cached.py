"""Independent truth-table, differential, and atomicity contracts."""
import itertools
import random

import pytest

from data.cnf import CNF
from eval.cnf_cached import CachedCNFRepairState
from eval.cnf_repair import CNFRepairState


def flipped(witness, patch):
    return tuple(not value if i in patch else value for i, value in enumerate(witness))


def exact_score(problem, witness, patch):
    before = set(problem.violated(witness))
    after = set(problem.violated(flipped(witness, patch)))
    return len(before - after), len(after - before)


def check_state(problem, state):
    assert state.unsatisfied == problem.violated(state.witness)
    assert state.valid == problem.satisfied(state.witness)
    for variable in range(problem.nvars):
        assert state.make_break(variable) == exact_score(problem, state.witness, (variable,))


@pytest.mark.parametrize('n', range(4))
def test_exhaustive_single_clauses_and_patches(n):
    literals = tuple(range(1, n + 1)) + tuple(range(-n, 0))
    clauses = [()]
    for width in range(1, 4):
        clauses.extend(itertools.product(literals, repeat=width))
    for clause in clauses:
        # Duplicate clauses keep separate contributions; repeat/tautological
        # literals are checked against ORIGINAL, unnormalized CNF semantics.
        problem = CNF(n, (clause, clause, ()))
        for witness in itertools.product((False, True), repeat=n):
            state = CachedCNFRepairState(problem, witness)
            check_state(problem, state)
            for mask in range(1, 1 << n):
                patch = tuple(v for v in range(n) if mask & (1 << v))
                assert state.make_break_patch(patch) == exact_score(problem, witness, patch)
                if len(patch) == 1:
                    fast = CachedCNFRepairState(problem, witness)
                    fast.flip(patch[0])
                    assert fast.witness == flipped(witness, patch)
                    check_state(problem, fast)
                    fast.flip(patch[0])
                    check_state(problem, fast)
                    assert fast.witness == witness
                state.flip_patch(patch)
                assert state.witness == flipped(witness, patch)
                check_state(problem, state)
                state.flip_patch(patch)
                assert state.witness == witness
                check_state(problem, state)


@pytest.mark.parametrize('seed', range(12))
def test_long_random_atomic_sequences(seed):
    rng = random.Random(seed)
    n = 16
    clauses = tuple(tuple(rng.choice((-1, 1)) * rng.randint(1, n)
                          for _ in range(rng.randint(0, 18))) for _ in range(40))
    problem = CNF(n, clauses)
    witness = tuple(bool(rng.randrange(2)) for _ in range(n))
    state = CachedCNFRepairState(problem, witness)
    reference = CNFRepairState(problem, witness)
    for _ in range(150):
        patch = tuple(rng.sample(range(n), rng.randint(1, 6)))
        before = state.witness
        assert state.make_break_patch(patch) == exact_score(problem, before, patch)
        state.flip_patch(patch)
        for variable in patch:
            reference.flip(variable)
        assert state.witness == reference.witness
        check_state(problem, state)


@pytest.mark.parametrize('bad', [(), (0, 0), (True,), (1.0,), (-1,), (3,), (0, 3), [0], None])
def test_rejected_patch_has_no_mutation(bad):
    state = CachedCNFRepairState(CNF(3, ((1, 2), (-1, -2), (3,))), (False,) * 3)
    # Include counters: rejection cannot be hidden work or partial state changes.
    import copy
    before = copy.deepcopy(state.__dict__)
    for method in (state.flip_patch, state.make_break_patch):
        with pytest.raises(ValueError):
            method(bad)
        assert state.__dict__ == before


@pytest.mark.parametrize('bad', [True, 1.0, -1, 3, '0', None])
def test_invalid_single_variable(bad):
    state = CachedCNFRepairState(CNF(3, ()), (False,) * 3)
    with pytest.raises(ValueError):
        state.flip(bad)
    with pytest.raises(ValueError):
        state.make_break(bad)
    assert state.flips == 0
    assert state.score_queries == 0


def test_coupled_patch_is_not_sum_of_single_scores():
    # Both variables jointly break a clause; no individual variable does so.
    state = CachedCNFRepairState(CNF(2, ((1, 2),)), (True, True))
    assert state.make_break(0) == state.make_break(1) == (0, 0)
    assert state.make_break_patch((0, 1)) == (0, 1)
    state.flip_patch((0, 1))
    assert not state.valid
    assert state.make_break_patch((0, 1)) == (1, 0)


def test_zero_variables_and_empty_formulas():
    assert CachedCNFRepairState(CNF(0, ()), ()).valid
    assert not CachedCNFRepairState(CNF(0, ((),)), ()).valid


def test_large_variable_index_and_tautologies():
    p = CNF(4096, ((4096, 4096), (-4096, 4096), (-1, 4096)))
    s = CachedCNFRepairState(p, (False,) * 4096)
    assert s.make_break(4095) == (1, 0)
    s.flip(4095)
    assert s.valid
    assert s.make_break(4095) == (0, 1)
    s.flip_patch((0, 4095))
    assert s.unsatisfied == (0, 2)


def test_query_cache_owns_inputs_and_charges_actual_work():
    p = CNF(3, ((1, 1, 2), (-1, 3)))
    s = CachedCNFRepairState(p, (False,) * 3)
    old = s.witness
    for _ in range(100):
        assert s.make_break(0) == (1, 1)
    assert s.feature_literal_visits == 0
    assert s.score_queries == 100
    s.flip(0)
    assert old == (False,) * 3
    assert s.literal_updates == 2
    assert s.cache_literal_visits == 4
    assert s.flips == s.patches == 1


@pytest.mark.parametrize('problem,witness', [(None, ()), (CNF(1, ()), [False]),
    (CNF(1, ()), (0,)), (CNF(1, ()), ()), (CNF(1, ()), (True, False))])
def test_strict_constructor(problem, witness):
    with pytest.raises((ValueError, TypeError)):
        CachedCNFRepairState(problem, witness)
