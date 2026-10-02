"""Synthetic malformed receipts must not enter the trained-FP timing analysis."""
import random

import pytest

from spectra.fp_evidence import ARMS, ORDER_SEED, ROUNDS, analyze


@pytest.fixture
def records():
    board = [1, 2, 3, 4, 3, 4, 1, 2, 2, 1, 4, 3, 4, 3, 2, 1]
    cases = [{'case_id': f'synthetic:{seed}:0', 'family': 'synthetic', 'seed': seed,
              'problem_id': 'synthetic:0', 'input': board.copy(), 'blocks': 1}
             for seed in (1, 2)]
    rng = random.Random(ORDER_SEED)
    rows = []
    for case in cases:
        for repeat in range(ROUNDS):
            order = list(ARMS)
            rng.shuffle(order)
            for position, arm in enumerate(order):
                steps = 0 if arm == 'symbolic' else 1
                rows.append({'case_id': case['case_id'], 'round': repeat,
                             'order': position, 'arm': arm,
                             'elapsed_ns': 50 if arm == 'prepared' else 100,
                             'answer': board.copy(), 'valid': True, 'steps': steps,
                             'work': {'executed_steps': steps, 'semantic_checks': 1,
                                      'block_applications': 2 * steps,
                                      'checker_constructions': 1, 'final_semantic': True,
                                      'target_used': False,
                                      'stop_reason': 'symbolic_complete' if arm == 'symbolic' else 'semantic_valid'}})
    return cases, rows


@pytest.mark.parametrize('field', ['round', 'order'])
@pytest.mark.parametrize('value', [False, 0.])
def test_schedule_indices_must_be_genuine_integers(records, field, value):
    next(row for row in records[1] if row[field] == 0)[field] = value
    with pytest.raises(ValueError):
        analyze(*records)


@pytest.mark.parametrize('field,value', [('seed', True), ('seed', 1.),
                                        ('blocks', True), ('blocks', 1.),
                                        ('blocks', 0), ('blocks', -1)])
def test_model_identity_cannot_be_boolean_fractional_or_empty(records, field, value):
    case = records[0][0]
    case[field] = value
    if field == 'blocks':
        for row in records[1]:
            if row['case_id'] == case['case_id']:
                row['work']['block_applications'] = int(row['steps'] * 2 * value)
    with pytest.raises(ValueError):
        analyze(*records)


def test_one_problem_identity_cannot_refer_to_different_inputs(records):
    # Both answers remain independently valid, but resampling these distinct
    # original inputs as one shared problem would be the wrong cluster.
    records[0][1]['input'][0] = 0
    with pytest.raises(ValueError):
        analyze(*records)


def fail_neural_rows(records, steps):
    for row in records[1]:
        if row['arm'] != 'symbolic':
            row['answer'] = [0] * 16
            row['valid'] = False
            row['steps'] = steps
            row['work'].update(executed_steps=steps, semantic_checks=steps,
                               block_applications=2 * steps, final_semantic=False,
                               stop_reason='budget_exhausted')


@pytest.mark.parametrize('steps', [1, 2, 3])
def test_budget_exhaustion_requires_the_complete_four_step_budget(records, steps):
    fail_neural_rows(records, steps)
    with pytest.raises(ValueError):
        analyze(*records)


def test_complete_failed_attempts_remain_in_the_analysis(records):
    fail_neural_rows(records, 4)
    result = analyze(*records)
    assert result['gate'] == 'FAIL'
    assert result['arms']['prepared']['distinct_valid'] == 0
    assert result['arms']['prepared']['observations'] == 14
    assert result['arms']['prepared']['charged_ns_per_verified_answer'] is None


def test_consistent_two_model_problem_cluster_is_preserved(records):
    result = analyze(*records)
    assert result['gate'] == 'PASS'
    assert result['cases'] == 2 and result['observations'] == 56
    assert result['primary_ratio'] == .5
    assert result['problem_bootstrap_ci95'] == [.5, .5]
