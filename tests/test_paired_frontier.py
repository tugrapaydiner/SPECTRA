from copy import deepcopy
import math

import numpy as np
import pytest

from eval.paired_frontier import FrontierGate, paired_frontier, quality_headroom


def records(candidate=((True, False), (True, True)), comparator=((False, True), (False, True))):
    rows = []
    for ai, (arm, outcomes) in enumerate((('new', candidate), ('base', comparator))):
        for mi, model in enumerate((11, 22)):
            for ei, example in enumerate(('a', 'b')):
                valid = outcomes[mi][ei]
                for r, timing in enumerate((1., 100., 3.)):
                    rows.append({'family': 'test', 'core_seed': model, 'example_id': example,
                        'arm': arm, 'round': r, 'valid': valid, 'answer': [1 if valid else 0],
                        'work': {'valid': valid, 'transitions': 1},
                        'latency_ms': timing * (1 if ai == 0 else 2)})
    return rows


def analyze(rows, **kwargs):
    return paired_frontier(rows, family='test', candidate='new', comparator='base',
        model_seeds=(11, 22), example_ids=('a', 'b'), replicates=200, **kwargs)


def test_known_discordant_pairs_round_medians_and_paired_cost():
    result = analyze(records())
    assert result['model_example_pairs'] == 4
    assert result['unique_examples'] == 2
    assert result['candidate_valid'] == 3 and result['comparator_valid'] == 2
    assert result['new_solves'] == 2 and result['regressions'] == 1
    assert result['endpoints']['success_gain']['point'] == .25
    assert result['endpoints']['mean_latency_ratio']['point'] == .5
    assert result['endpoints']['p95_latency_ratio']['point'] == .5
    assert result == analyze(list(reversed(records())))
    assert not result['bootstrap']['timing_rounds_resampled_as_examples']


def test_shared_example_and_model_resampling_has_expected_cluster_variance():
    # Only one model has any gains: uncertainty must retain the zero-gain model,
    # not act as if 12 timing rows were independent positive observations.
    result = analyze(records(((True, True), (False, False)), ((False, False), (False, False))))
    assert result['endpoints']['success_gain']['crossed_percentile_95'] == [0., 1.]
    assert not result['descriptive_gate_conditions']['positive_quality_interval_lower']


def test_identical_arms_have_no_quality_gain_or_invented_significance():
    rows = records(((True, False), (True, True)), ((True, False), (True, True)))
    result = analyze(rows)
    assert result['endpoints']['success_gain'] == {'point': 0., 'crossed_percentile_95': [0., 0.]}
    assert not result['descriptive_gate_pass']


@pytest.mark.parametrize('n,good,gain,required,possible', [
    (20, 19, .05, 1, True), (100, 96, .05, 5, False),
    (256, 241, .05, 13, True), (2250, 2249, .05, 113, False),
    (100, 100, 0., 0, True), (100, 94, .07, 7, False),
])
def test_exact_integer_ceiling_without_float_rounding(n, good, gain, required, possible):
    report = quality_headroom(good, n, gain)
    assert report['required_net_additional_solves'] == required
    assert report['point_gain_attainable_on_this_inventory'] == possible
    assert report['available_additional_solves'] == n-good


@pytest.mark.parametrize('mutation', ['missing', 'missing_model', 'duplicate', 'foreign_model',
    'foreign_example', 'foreign_family', 'foreign_arm', 'bool_seed', 'bool_round',
    'round_outside', 'changed_answer', 'changed_work', 'changed_valid', 'no_answer', 'work_bool'])
def test_malformed_or_unpaired_observations_fail(mutation):
    rows = deepcopy(records())
    if mutation == 'missing': rows.pop()
    elif mutation == 'missing_model': rows = [r for r in rows if r['core_seed'] != 22]
    elif mutation == 'duplicate': rows[-1] = deepcopy(rows[0])
    elif mutation == 'foreign_model': rows[0]['core_seed'] = 33
    elif mutation == 'foreign_example': rows[0]['example_id'] = 'c'
    elif mutation == 'foreign_family': rows[0]['family'] = 'other'
    elif mutation == 'foreign_arm': rows[0]['arm'] = 'other'
    elif mutation == 'bool_seed': rows[0]['core_seed'] = True
    elif mutation == 'bool_round': rows[0]['round'] = False
    elif mutation == 'round_outside': rows[0]['round'] = 3
    elif mutation == 'changed_answer': rows[0]['answer'] = [9]
    elif mutation == 'changed_work': rows[0]['work']['transitions'] = 2
    elif mutation == 'changed_valid': rows[0]['valid'] = False
    elif mutation == 'no_answer': rows[0]['answer'] = None
    elif mutation == 'work_bool': rows[0]['work']['valid'] = 1
    with pytest.raises(ValueError): analyze(rows)


@pytest.mark.parametrize('value', [0., -1., True, math.nan, math.inf, -math.inf, '1'])
def test_invalid_timing_rejected(value):
    rows = records(); rows[0]['latency_ms'] = value
    with pytest.raises(ValueError, match='latency'): analyze(rows)


@pytest.mark.parametrize('gain', [True, math.inf, math.nan, -.1, 1.1])
def test_invalid_thresholds_rejected(gain):
    with pytest.raises(ValueError): FrontierGate(min_success_gain=gain)


def test_tail_cost_can_fail_even_when_mean_cost_passes():
    rows = records(((True, True), (True, True)), ((False, False), (False, False)))
    for r in rows:
        r['latency_ms'] = (3. if r['core_seed'] == 22 and r['example_id'] == 'b' else .1) if r['arm'] == 'new' else 1.
    result = analyze(rows)
    assert result['descriptive_gate_conditions']['mean_latency_point_ratio']
    assert not result['descriptive_gate_conditions']['p95_latency_point_ratio']
    assert not result['descriptive_gate_pass']


def test_independent_draw_oracle_for_crossed_confidence_interval():
    # Enumerate the seed/example draws from a fresh RNG and compute statistics
    # with Python lists, including multiplicities of sampled clusters.
    rng = np.random.default_rng(2026091110)
    delta = [[1, -1], [1, 0]]
    draws = []
    for _ in range(200):
        models = rng.integers(2, size=2).tolist()
        examples = rng.integers(2, size=2).tolist()
        draws.append(sum(delta[m][e] for m in models for e in examples) / 4)
    expected = np.quantile(draws, [.025, .975]).tolist()
    assert analyze(records())['endpoints']['success_gain']['crossed_percentile_95'] == expected
