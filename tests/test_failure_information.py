from copy import deepcopy
from itertools import permutations

import numpy as np
import pytest

from eval.failure_information import (Observation, LinearSelector, MODES, features,
    first_success, simulate, fixed_order, coverage, fit_selector, best_static_order,
    input_random_order)


def case(successes=(), identity_time=None, token=1):
    def steps(n, time=None):
        return [{'answer': [token]*16, 'valid': i+1 == time} for i in range(n)]
    return {'input': [0]*16, 'identity': steps(32, identity_time),
            'views': [steps(4, 2 if a in successes else None) for a in range(7)]}


def test_budget_oracle_and_regression_are_distinct():
    cases = [case([6]), case(identity_time=22), case(identity_time=3), case()]
    result = coverage(cases)
    assert result['prefix_plus_all_views_oracle_valid'] == 2
    assert result['all_trajectories_union_valid'] == 3
    assert result['oracle_new_vs_original'] == 1
    assert result['oracle_regressions_vs_identity32'] == 1
    assert result['unsolved_by_every_trajectory'] == 1


def test_every_schedule_matches_independent_stopping_oracle():
    c = case([2, 6])
    for order in permutations(range(7), 3):
        for budget in (12, 16, 20):
            available = order[:(budget-8)//4]
            first = next((i for i, a in enumerate(available) if a in (2, 6)), None)
            r = simulate(c, fixed_order(order), budget)
            assert r['valid'] == (first is not None)
            assert r['transitions'] == (budget if first is None else 10+4*first)
            assert r['actions'] == list(available if first is None else available[:first+1])


def test_success_stops_before_selector_or_future_answers():
    c = case(identity_time=3)
    def forbidden(_):
        raise AssertionError('selector must not be invoked')
    assert simulate(c, forbidden)['transitions'] == 3


def test_selector_observes_only_previously_failed_answers():
    c = case([1])
    c['views'][0][3]['answer'] = [3]*16
    observations = []
    def choose(obs):
        observations.append(obs)
        assert not hasattr(obs, 'views') and not hasattr(obs, 'example_id')
        return len(obs.failures)
    result = simulate(c, choose)
    assert result['valid'] and result['transitions'] == 14
    assert observations[0].failures == ()
    assert observations[1].failures == ((0, (3,)*16),)


@pytest.mark.parametrize('mode', MODES)
def test_feature_dimensions_and_finite_coefficients(mode):
    o = Observation((0,)*16, (1,)*16)
    d = {'input_only': 89, 'prefix_only': 169, 'failure_aware': 249}[mode]
    assert features(o, mode).shape == (d,)
    assert LinearSelector(mode, np.zeros((7, d))).choose(o) == 0
    with pytest.raises(ValueError):
        LinearSelector(mode, np.full((7, d), np.nan))


def test_information_ablations_hide_correct_fields():
    a = Observation((0,)*16, (1,)*16, ((0, (2,)*16),))
    b = Observation((0,)*16, (4,)*16, ((0, (3,)*16),))
    c = Observation((0,)*16, (1,)*16, ((0, (3,)*16),))
    np.testing.assert_array_equal(features(a, 'input_only'), features(b, 'input_only'))
    np.testing.assert_array_equal(features(a, 'prefix_only'), features(c, 'prefix_only'))
    assert not np.array_equal(features(a, 'failure_aware'), features(c, 'failure_aware'))


def test_only_failure_bits_compile_to_a_fixed_order():
    # For fixed input and no feedback content, the all-failure path determines
    # the order. Successful execution is a prefix of that path.
    weights = np.random.default_rng(5).normal(size=(7, 89))
    policy = LinearSelector('input_only', weights)
    order = simulate(case(), policy.choose)['actions']
    for success in range(7):
        c = case([success], token=4)
        assert simulate(c, policy.choose) == simulate(c, fixed_order(order))


@pytest.mark.parametrize('bad', [True, -1, 7, '1'])
def test_illegal_selector_output_rejected(bad):
    with pytest.raises(ValueError):
        simulate(case(), lambda _: bad)


def test_repeated_selector_output_rejected():
    with pytest.raises(ValueError):
        simulate(case(), lambda _: 0)


@pytest.mark.parametrize('budget', [True, 0, 8, 13, 32])
def test_unsupported_budget_rejected(budget):
    with pytest.raises(ValueError):
        simulate(case(), fixed_order(range(7)), budget)


def test_fitting_learns_signal_without_modifying_cases():
    cases = [case([0], token=1), case([1], token=4)]*3
    before = deepcopy(cases)
    policy = fit_selector(cases, 'prefix_only', ridge=.001)
    assert policy.choose(Observation((0,)*16, (1,)*16)) == 0
    assert policy.choose(Observation((0,)*16, (4,)*16)) == 1
    assert cases == before


def test_no_residual_training_examples_has_explicit_constant_fallback():
    policy = fit_selector([case(identity_time=2)], 'failure_aware')
    assert np.count_nonzero(policy.weights) == 0


def test_static_order_uses_quality_then_cost_then_lexicographic_ties():
    cases = [case([5]), case([3]), case([3])]
    assert best_static_order(cases) == (3, 5, 0)


def test_random_control_is_deterministic_input_only_permutation():
    order = input_random_order([0]*16)
    assert order == input_random_order([0]*16)
    assert sorted(order) == list(range(7))


def test_observation_rejects_repeated_actions_and_noninteger_symbols():
    with pytest.raises(ValueError):
        Observation((0,)*16, (1,)*16, ((0, (2,)*16), (0, (3,)*16)))
    with pytest.raises(ValueError):
        Observation((False,)*16, (1,)*16)


def test_crossfit_holds_entire_groups_across_models():
    from scripts.failure_information_pilot import crossfit_partition
    cases = [{'core_seed': seed, 'group': group, 'fold': fold}
             for seed in (1,2) for group,fold in [('a',0),('b',1),('c',0)]]
    train, test = crossfit_partition(cases,0)
    assert {c['group'] for c in train} == {'b'}
    assert {c['group'] for c in test} == {'a','c'}
    assert len(train)==2 and len(test)==4
    cases[0]['fold']=1
    with pytest.raises(ValueError):
        crossfit_partition(cases,0)


def test_maze_group_includes_both_endpoint_labelings_and_spatial_maps():
    import torch
    from eval.checkable_tasks import MAZE11
    from eval.symmetry_search import grid_views
    from scripts.failure_information_pilot import group_key
    x = torch.zeros((1,121), dtype=torch.int64)
    x[0,12], x[0,108], x[0,41] = 2,3,1
    key = group_key(x,MAZE11)
    swap = torch.tensor([0,1,3,2,4])
    for view in grid_views(MAZE11):
        transformed = view.apply(x)
        assert group_key(transformed,MAZE11) == key
        assert group_key(swap[transformed],MAZE11) == key


def test_extended_view_creates_candidates_outside_old_four_cycle_ceiling():
    from scripts.transpose_continuation_pilot import predict
    c = case()
    extended = [{'answer':[2]*16, 'valid':i==8} for i in range(32)]
    assert coverage([c])['prefix_plus_all_views_oracle_valid']==0
    assert not predict(c,extended,'prefix_transpose_12')['valid']
    result = predict(c,extended,'prefix_transpose_20')
    assert result['valid'] and result['transitions']==17
    assert predict(c,extended,'transpose_20')['transitions']==9


def test_extended_view_preserves_prefix_and_respects_depth_caps():
    from scripts.transpose_continuation_pilot import predict
    c = case(identity_time=5)
    extended = [{'answer':[2]*16, 'valid':i==23} for i in range(32)]
    assert predict(c,extended,'prefix_transpose_20')['transitions']==5
    assert not predict(c,extended,'transpose_20')['valid']
    with pytest.raises(ValueError):
        predict(c,extended,'unknown')


def test_bound_evidence_reader_checks_digest_and_count(tmp_path):
    import gzip
    from scripts.verify_failure_information import read_bound_rows
    from scripts.failure_information_pilot import sha
    raw = b'{"valid":false}\n'
    path = tmp_path/'rows.gz'
    path.write_bytes(gzip.compress(raw))
    assert read_bound_rows(path,sha(raw),1)==[{'valid':False}]
    with pytest.raises(ValueError):
        read_bound_rows(path,'0'*64,1)
    with pytest.raises(ValueError):
        read_bound_rows(path,sha(raw),2)


def test_evidence_analysis_rejects_incomplete_or_foreign_inventory():
    from scripts.failure_information_pilot import timed_analysis
    from scripts.transpose_continuation_pilot import analyze
    with pytest.raises(ValueError):
        timed_analysis([],[],{})
    with pytest.raises(ValueError):
        analyze([],{})
