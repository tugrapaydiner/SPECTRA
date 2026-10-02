"""Recorded tensor identities must agree with their reported differences.

These are synthetic accounting records, not measured compiler observations.
"""
import copy
import hashlib
import struct

import pytest

from spectra.compiler_evidence import ARMS, analyze, schedule


def tensor_hash(value):
    return hashlib.sha256(struct.pack('<d', value)).hexdigest()


@pytest.fixture
def records():
    board = [1, 2, 3, 4, 3, 4, 1, 2, 2, 1, 4, 3, 4, 3, 2, 1]
    cases = [{'case_id': 'synthetic:1:0', 'family': 'synthetic', 'seed': 1,
              'problem_id': 'synthetic:0', 'input': board, 'blocks': 1}]
    work = {'executed_steps': 1, 'block_applications': 2, 'semantic_checks': 1,
            'checker_constructions': 1, 'final_semantic': True, 'target_used': False,
            'checker': 'native_exact_sudoku_v1', 'stop_reason': 'semantic_valid'}
    rows = [{'case_id': c['case_id'], 'round': r, 'order': position, 'arm': arm,
             'elapsed_ns': 100, 'valid': True, 'answer': board.copy(),
             'work': copy.deepcopy(work)} for c, r, position, arm in schedule(cases)]
    step = {'finite': True, 'bitwise': True,
            **{name + '_sha256': tensor_hash(0.) for name in ('y', 'z', 'logits')},
            **{'max_abs_' + name: 0. for name in ('y', 'z', 'logits')}}
    traces = [{'case_id': cases[0]['case_id'], 'arm': arm,
               'embedding_sha256': tensor_hash(0.), 'embedding_bitwise': True,
               'steps': [copy.deepcopy(step) for _ in range(4)]} for arm in ARMS]
    return cases, rows, traces


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('tensor', ['y', 'z', 'logits'])
def test_identical_tensor_cannot_have_positive_difference(records, arm, tensor):
    traces = records[2]
    trace = next(t for t in traces if t['arm'] == arm)
    trace['steps'][0]['max_abs_' + tensor] = .25
    with pytest.raises(ValueError):
        analyze(*records)


@pytest.mark.parametrize('arm', ['inductor_default', 'inductor_frozen'])
def test_identical_finite_reference_cannot_be_reported_nonfinite(records, arm):
    step = next(t for t in records[2] if t['arm'] == arm)['steps'][0]
    step['finite'] = False
    step.update({name: None for name in ('max_abs_y', 'max_abs_z', 'max_abs_logits')})
    with pytest.raises(ValueError):
        analyze(*records)


@pytest.mark.parametrize('tensor', ['y', 'z'])
def test_changed_logits_do_not_hide_contradictory_unchanged_tensors(records, tensor):
    step = records[2][-1]['steps'][0]
    step.update(bitwise=False, logits_sha256=tensor_hash(.5), max_abs_logits=.5)
    step['max_abs_' + tensor] = .25
    with pytest.raises(ValueError):
        analyze(*records)


def test_consistent_exact_record_remains_valid(records):
    result = analyze(*records)
    assert all(arm['exact_track_pass'] for arm in result['arms'].values())


@pytest.mark.parametrize('value,difference', [(-0., 0.), (.5, .5)])
def test_nonbitwise_finite_difference_is_preserved(records, value, difference):
    step = records[2][-1]['steps'][0]
    step.update(bitwise=False, logits_sha256=tensor_hash(value), max_abs_logits=difference)
    result = analyze(*records)['arms']['inductor_frozen']
    assert not result['exact_track_pass'] and result['bounded_task_track_pass']
    assert result['nonbitwise_cases'] == 1 and result['max_abs_logits'] == difference


def test_actual_nonfinite_difference_remains_a_reported_failure(records):
    step = records[2][-1]['steps'][0]
    step.update(bitwise=False, finite=False, logits_sha256=tensor_hash(float('inf')))
    # The producer records all differences as null if any tensor is nonfinite.
    step.update({name: None for name in ('max_abs_y', 'max_abs_z', 'max_abs_logits')})
    result = analyze(*records)['arms']['inductor_frozen']
    assert result['nonfinite_cases'] == 1
    assert not result['exact_track_pass'] and not result['bounded_task_track_pass']
