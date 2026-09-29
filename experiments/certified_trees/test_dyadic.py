"""Differential source reconstruction, not empirical calibration of new bounds."""
import copy
import json
import math
import random
import struct
import zlib

import pytest

from . import dyadic, packed
from .reference import certificate_oracle as oracle
from .session import VerifiedCompact


def source(seed=0, *, binary=False, exponent=0, scale=1.0):
    rng = random.Random(seed)
    d, c = rng.randint(1, 5), 1 if binary else rng.randint(2, 7)
    trees = []
    for _ in range(rng.randint(1, 8)):
        depth = rng.randint(0, 3)
        splits = [{'split_type': 'FloatFeature', 'float_feature_index': rng.randrange(d),
                   'border': rng.choice([-2.0, 0.0, 0.5, 1.5, 4.0])} for _ in range(depth)]
        values = [math.ldexp(rng.randrange(-2**18, 2**18) / 2**17, exponent)
                  for _ in range((1 << depth) * c)]
        # Include cancellation, class-shared offsets and exactly zero leaves.
        for k in range(0, len(values), 13):
            values[k] = -0.0
        trees.append({'splits': splits, 'leaf_values': values})
    return {'features_info': {'float_features': [{'feature_index': i,
                    'flat_feature_index': i} for i in range(d)]},
            'scale_and_bias': [scale, [math.ldexp(rng.uniform(-1, 1), exponent)
                                      for _ in range(c)]],
            'oblivious_trees': trees}


def compare(doc, *, bits, pairwise):
    raw = json.dumps(doc, allow_nan=False).encode()
    kw = {'bits': bits, 'pairwise': pairwise}
    try:
        expected = oracle.compile_source(raw, 3, **kw)
    except ValueError:
        with pytest.raises(ValueError):
            dyadic.compile_source(raw, 3, **kw)
        return
    actual = dyadic.compile_source(raw, 3, **kw)
    assert actual == expected
    assert oracle.canonical(actual) == oracle.canonical(expected)
    assert packed.pack(actual) == packed.pack(expected)
    for backend in ('reference', 'dyadic'):
        assert packed.verify(raw, packed.pack(expected), backend=backend)['status'] == 'PASS'


@pytest.mark.parametrize('seed', range(32))
@pytest.mark.parametrize('bits,pairwise', [(8, False), (16, False), (8, True), (16, True)])
def test_differential_models(seed, bits, pairwise):
    # Same source for each pair of backend comparisons; all combinations retained.
    doc = source(seed, binary=seed % 5 == 0,
                 exponent=(-600, -30, -8, 0, 30, 500)[seed % 6],
                 scale=(1.0, 0.3, 3.0, 2.0**-100, 2.0**100)[seed % 5])
    compare(doc, bits=bits, pairwise=pairwise)


@pytest.mark.parametrize('bits', [8, 16])
@pytest.mark.parametrize('pairwise', [False, True])
def test_subnormal_mixture_and_nondyadic_bias(bits, pairwise):
    doc = source(17, scale=0.3)
    doc['oblivious_trees'] = [{'splits': [], 'leaf_values': [
        5e-324, -5e-324, 1.0, math.nextafter(1.0, math.inf), -0.0]}]
    doc['scale_and_bias'][1] = [1e-300, -1e-300, 0.7, -0.4, 0.0]
    compare(doc, bits=bits, pairwise=pairwise)


@pytest.mark.parametrize('pairwise', [False, True])
def test_round_ties_even_for_negative_and_positive_values(pairwise):
    values = [-1.5, -0.5, 0.5, 1.5, 127.0, 0.0, 100.0, 126.0]
    doc = source(19)
    doc['scale_and_bias'] = [1.0, [0.0, -1.5]]
    doc['oblivious_trees'] = [{'splits': [
        {'split_type': 'FloatFeature', 'float_feature_index': 0, 'border': b}
        for b in (-0.5, 0.5, 1.5)], 'leaf_values': [x for v in values for x in (0.0, v)]}]
    raw = json.dumps(doc).encode()
    result = dyadic.compile_source(raw, 3, bits=8, pairwise=pairwise)
    assert result['quantization']['step_exponent'] == 0
    assert [row[1] for row in result['trees'][0]['leaves']] == [-2, 0, 0, 2, 127, 0, 100, 126]
    assert result == oracle.compile_source(raw, 3, bits=8, pairwise=pairwise)


@pytest.mark.parametrize('bias', [0.0, 2.0**-850, -2.0**-850, 1.0, 2.0**850])
def test_zero_leaves_bias_only_negative_shift(bias):
    doc = source(1)
    doc['scale_and_bias'] = [3.0, [0.0, bias]]
    doc['oblivious_trees'] = [{'splits': [], 'leaf_values': [0.0, -0.0]}]
    compare(doc, bits=16, pairwise=True)


@pytest.mark.parametrize('exponent', [-1074, -1000, 899, 900, 1000])
def test_extreme_domain_acceptance_or_rejection(exponent):
    doc = source(1)
    doc['scale_and_bias'] = [1.0, [0.0, 0.0]]
    doc['oblivious_trees'] = [{'splits': [], 'leaf_values': [0.0, 2.0**exponent]}]
    compare(doc, bits=16, pairwise=True)


@pytest.mark.parametrize('bad', ['bool_leaf', 'string_leaf', 'nan_leaf', 'leaf_inventory',
    'bad_border', 'wide_border', 'categorical', 'unknown_feature', 'bool_feature',
    'duplicate_flat', 'negative_scale', 'zero_scale', 'bad_scale', 'bad_bias',
    'no_trees', 'asymmetric', 'too_deep', 'not_object', 'missing_features'])
def test_source_rejections_match_reference(bad):
    doc = source(25)
    tree = doc['oblivious_trees'][0]
    if bad == 'bool_leaf': tree['leaf_values'][0] = True
    elif bad == 'string_leaf': tree['leaf_values'][0] = '0'
    elif bad == 'nan_leaf': tree['leaf_values'][0] = math.nan
    elif bad == 'leaf_inventory': tree['leaf_values'].append(0.0)
    elif bad == 'bad_border':
        tree['splits'] = [{'split_type': 'FloatFeature', 'float_feature_index': 0, 'border': 0.1}]
    elif bad == 'wide_border':
        tree['splits'] = [{'split_type': 'FloatFeature', 'float_feature_index': 0, 'border': 1e100}]
    elif bad == 'categorical': doc['features_info']['categorical_features'] = [{}]
    elif bad == 'unknown_feature':
        tree['splits'] = [{'split_type': 'FloatFeature', 'float_feature_index': 255, 'border': 0.5}]
    elif bad == 'bool_feature': doc['features_info']['float_features'][0]['feature_index'] = True
    elif bad == 'duplicate_flat':
        doc['features_info']['float_features'].append({'feature_index': 100, 'flat_feature_index': 0})
    elif bad in ('negative_scale', 'zero_scale', 'bad_scale'):
        doc['scale_and_bias'][0] = {'negative_scale': -1.0, 'zero_scale': 0.0, 'bad_scale': True}[bad]
    elif bad == 'bad_bias': doc['scale_and_bias'][1] = []
    elif bad == 'no_trees': doc['oblivious_trees'] = []
    elif bad == 'asymmetric': doc['trees'] = [{}]
    elif bad == 'too_deep': tree['splits'] = [{}] * 13
    elif bad == 'not_object': doc = []
    else: del doc['features_info']
    raw = json.dumps(doc).encode()
    for fn in (oracle.compile_source, dyadic.compile_source):
        with pytest.raises(ValueError): fn(raw, 3)


@pytest.mark.parametrize('raw', [b'{}', b'{"a":0,"a":1}', b'\xff', b'[[[', b'{"a":1e10000}'])
def test_strict_json_rejection(raw):
    for fn in (oracle.compile_source, dyadic.compile_source):
        with pytest.raises(ValueError): fn(raw, 3)


@pytest.mark.parametrize('kwargs', [{'bits': True}, {'bits': 32}, {'pairwise': 1},
                                    {'features': True}, {'features': 0}, {'features': 257}])
def test_settings_rejected(kwargs):
    raw = json.dumps(source(5)).encode()
    for fn in (oracle.compile_source, dyadic.compile_source):
        with pytest.raises(ValueError): fn(raw, 3, **kwargs)


@pytest.mark.parametrize('bad', ['leaf', 'bias', 'lower', 'upper', 'predicate', 'tree',
                                 'oracle_digest', 'pair_bound'])
def test_self_consistent_forged_certificates_are_rejected(bad):
    raw = json.dumps(source(12)).encode()
    original = oracle.compile_source(raw, 3, bits=16, pairwise=True)
    forged = copy.deepcopy(original)
    if bad == 'leaf': forged['trees'][0]['leaves'][0][-1] += 1
    elif bad == 'bias': forged['bias'][-1] += 1
    elif bad == 'lower': forged['error_low'][-1] += 1
    elif bad == 'upper': forged['error_high'][-1] -= 1
    elif bad == 'predicate': forged['trees'][-1]['thresholds'][0] += 1
    elif bad == 'tree': forged['trees'] = list(reversed(forged['trees']))
    elif bad == 'pair_bound': forged['pair_upper'][0][-1] -= 1
    packed_raw = bytearray(packed.pack(forged))
    if bad == 'oracle_digest': packed_raw[packed.HEADER.size - 1] ^= 1
    # pack() recomputed both CRC and the canonical oracle digest for semantic
    # mutations. Hash closure cannot substitute for original-source proof.
    for backend in ('reference', 'dyadic'):
        with pytest.raises(ValueError): packed.verify(raw, bytes(packed_raw), backend=backend)


def test_no_reference_leaf_compilation_or_parser_is_called(monkeypatch):
    raw = json.dumps(source(9)).encode()
    expected = oracle.compile_source(raw, 3)
    def fail(*args, **kwargs): raise AssertionError('reference implementation invoked')
    monkeypatch.setattr(oracle, 'compile_source', fail)
    monkeypatch.setattr(oracle, 'parse_source', fail)
    assert dyadic.compile_source(raw, 3) == expected
    obj = VerifiedCompact(raw, packed.pack(expected))
    assert obj.info['verification_backend'] == 'dyadic'


@pytest.mark.parametrize('backend', [None, [], True, 'trusted', 'skip'])
def test_no_verification_bypass(backend):
    raw = json.dumps(source(2)).encode()
    comp = packed.pack(oracle.compile_source(raw, 3))
    with pytest.raises(ValueError): VerifiedCompact(raw, comp, backend=backend)
