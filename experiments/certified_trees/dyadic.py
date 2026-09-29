"""Exact dyadic implementation of the preserved certificate-compiler policy.

Every finite stored binary64 leaf is an integer times a common power of two.
Leaf contrasts, rounding and residual extrema therefore need no Fraction objects.
Only class-sized scale/bias and roundoff expressions use general rationals.

This is not approximate verification, empirical calibration, cached trust or a
new certificate. The complete output must match reference.certificate_oracle.
The original oracle remains unchanged. No numerical framework or native import.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import struct
from typing import Any

from .reference import certificate_oracle as reference


@dataclass(slots=True)
class _Tree:
    features: list[int]
    thresholds: list[int]
    values: list[int] | list[float]


@dataclass(slots=True)
class _Source:
    digest: str
    features: int
    maximum: int
    classes: int
    binary: bool
    scale: Fraction
    bias: tuple[Fraction, ...]
    denominator_bits: int
    trees: list[_Tree]
    mass: list[int]
    amplitude: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any) -> float:
    _require(type(value) in (int, float), 'finite scalar required')
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError('finite binary64 scalar required') from exc
    _require(math.isfinite(value), 'finite scalar required')
    return value


def _integer(value: float, denominator_bits: int) -> int:
    numerator, denominator = value.as_integer_ratio()
    shift = denominator_bits - (denominator.bit_length() - 1)
    # frexp-derived grid is conservative, including for subnormal inputs.
    if shift < 0:
        raise AssertionError('internal dyadic grid is too coarse')
    return numerator << shift


def _parse(raw: bytes, maximum: int, features: int | None) -> _Source:
    """Mirror the reference's supported schema without constructing leaf Fractions.

    Strict byte-limited JSON decoding is deliberately shared. Numeric extraction,
    dyadic conversion, residual calculation and roundoff computation are separate.
    """
    _require(type(maximum) is int and 1 <= maximum <= 255, 'maximum must be 1..255')
    doc = reference.loads(raw)
    _require(type(doc) is dict, 'source object required')
    info = doc.get('features_info', {})
    _require(type(info) is dict, 'invalid feature metadata')
    for key in ('categorical_features', 'text_features', 'embedding_features', 'ctrs'):
        _require(not info.get(key), 'only numeric source features are supported')
    fs = info.get('float_features')
    _require(type(fs) is list and 1 <= len(fs) <= 256, 'numeric feature inventory required')
    mapping: dict[int, int] = {}
    for feature in fs:
        _require(type(feature) is dict, 'invalid feature entry')
        index, flat = feature.get('feature_index'), feature.get('flat_feature_index')
        _require(type(index) is int and type(flat) is int and 0 <= index < 256 and
                 0 <= flat < 256 and index not in mapping, 'invalid feature indices')
        mapping[index] = flat
    _require(len(set(mapping.values())) == len(mapping), 'duplicate flat feature index')
    inferred = max(mapping.values()) + 1
    width = inferred if features is None else features
    _require(type(width) is int and inferred <= width <= 256, 'invalid declared feature count')
    sb = doc.get('scale_and_bias')
    _require(type(sb) is list and len(sb) == 2 and type(sb[1]) is list and
             1 <= len(sb[1]) <= 64, 'explicit scale/bias metadata required')
    scale = Fraction.from_float(_number(sb[0]))
    bias = tuple(Fraction.from_float(_number(x)) for x in sb[1])
    _require(scale > 0, 'only positive common output scales are supported')
    binary = len(bias) == 1
    if binary:
        bias = (Fraction(0), bias[0])
    classes = len(bias)
    raw_trees = doc.get('oblivious_trees')
    _require(type(raw_trees) is list and 1 <= len(raw_trees) <= 4096,
             'bounded oblivious-tree inventory required')
    _require(not doc.get('trees'), 'asymmetric trees not supported')
    trees: list[_Tree] = []
    scalars = denominator_bits = 0
    for tree in raw_trees:
        _require(type(tree) is dict, 'invalid tree entry')
        splits = tree.get('splits') or []
        _require(type(splits) is list and len(splits) <= 12, 'tree depth limit')
        fidx, cutoffs = [], []
        for split in splits:
            _require(type(split) is dict and split.get('split_type') == 'FloatFeature',
                     'only FloatFeature splits are supported')
            idx = split.get('float_feature_index')
            _require(type(idx) is int and idx in mapping, 'unknown split feature')
            border = _number(split.get('border'))
            try:
                round32 = struct.unpack('<f', struct.pack('<f', border))[0]
            except (OverflowError, struct.error) as exc:
                raise ValueError('unrepresentable split border') from exc
            _require(round32 == border, 'border must identify an exact binary32 threshold')
            fidx.append(mapping[idx])
            cutoffs.append(min(maximum, max(-1, math.floor(border))))
        values = tree.get('leaf_values')
        leaf_count = 1 << len(splits)
        expected = leaf_count * (1 if binary else classes)
        _require(type(values) is list and len(values) == expected, 'leaf-value inventory')
        scalars += leaf_count * classes
        _require(scalars <= reference.MAX_SCALARS, 'reference-oracle leaf-scalar budget exceeded')
        floats = []
        for original in values:
            value = _number(original)
            if binary:
                floats.append(0.0)
            floats.append(value)
            if value:
                denominator_bits = max(denominator_bits, 53 - math.frexp(value)[1])
        trees.append(_Tree(fidx, cutoffs, floats))
    # Release the decoded JSON before allocating the common-grid integer bank.
    del doc, raw_trees, values, tree, floats
    mass = [0] * classes
    amplitude = 0
    for tree in trees:
        integers = [_integer(v, denominator_bits) for v in tree.values]
        tree.values = integers
        maxima = [0] * classes
        for offset in range(0, len(integers), classes):
            base = integers[offset]
            for c in range(classes):
                value = integers[offset + c]
                magnitude = abs(value)
                if magnitude > maxima[c]:
                    maxima[c] = magnitude
                contrast = abs(value - base)
                if contrast > amplitude:
                    amplitude = contrast
        mass = [a + b for a, b in zip(mass, maxima)]
    return _Source(reference.sha(raw), width, maximum, classes, binary, scale, bias,
                   denominator_bits, trees, mass, amplitude)


def _nearest_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(2 * remainder > denominator or
                          (2 * remainder == denominator and quotient & 1))


def _roundoff(source: _Source) -> list[Fraction]:
    """Same source-operation envelope, independently evaluated with exact rationals."""
    count = len(source.trees) + 4
    u = Fraction(1, 2**53)
    eta = Fraction(1, 2**1074)
    denominator = 1 - count * u
    _require(denominator > 0, 'arithmetic depth exceeds roundoff argument')
    gamma = count * u / denominator
    out = []
    for c, integer_mass in enumerate(source.mass):
        mass = Fraction(integer_mass, 1 << source.denominator_bits)
        total = source.scale * mass + abs(source.bias[c])
        _require(mass < 2**900 and total < 2**900,
                 'source exceeds conservative no-overflow envelope')
        error = gamma * total + count * eta * max(Fraction(1), source.scale) / denominator
        out.append(error / source.scale)
    return out


def compile_source(raw: bytes, maximum: int, *, features: int | None = None,
                   bits: int = 16, pairwise: bool = False) -> dict[str, Any]:
    """Construct exactly the original certificate, including its canonical digest."""
    _require(type(bits) is int and bits in (8, 16), 'bits must be 8 or 16')
    _require(type(pairwise) is bool, 'pairwise flag must be boolean')
    source = _parse(raw, maximum, features)
    c = source.classes
    if pairwise:
        work = sum(len(t.values) for t in source.trees) * c
        _require(work <= reference.MAX_PAIR_WORK, 'exact pairwise reference-work budget exceeded')
    bias = [(v - source.bias[0]) / source.scale for v in source.bias]
    amplitude = max([Fraction(source.amplitude, 1 << source.denominator_bits)] +
                    [abs(v) for v in bias])
    limit = (1 << (bits - 1)) - 1
    exponent = reference.ceil_log2(amplitude / limit) if amplitude else 0
    _require(-900 <= exponent <= 900, 'quantization exponent outside reference scope')
    step = reference.power2(exponent)
    shift = source.denominator_bits + exponent
    denominator = 1 << max(0, shift)
    fine_bits = 20
    fine = 1 << fine_bits
    normalized_bias = [v / step for v in bias]
    qb = [_nearest_even(v.numerator, v.denominator) for v in normalized_bias]
    bias_residual = [v - q for v, q in zip(normalized_bias, qb)]
    minimum_sum, maximum_sum = [0] * c, [0] * c
    pair_sum = [[0] * c for _ in range(c)] if pairwise else None
    compiled_trees = []
    for tree in source.trees:
        minimum, maximum_residual = [None] * c, [None] * c
        rows = []
        residuals = [] if pairwise else None
        for offset in range(0, len(tree.values), c):
            base = tree.values[offset]
            integers, residual = [], []
            for j in range(c):
                numerator = tree.values[offset + j] - base
                if shift < 0:
                    numerator <<= -shift
                q = _nearest_even(numerator, denominator)
                _require(-limit <= q <= limit, 'quantizer range failure')
                r = numerator - q * denominator
                integers.append(q)
                if pairwise:
                    residual.append(r)
                if minimum[j] is None or r < minimum[j]:
                    minimum[j] = r
                if maximum_residual[j] is None or r > maximum_residual[j]:
                    maximum_residual[j] = r
            rows.append(integers)
            if pairwise:
                residuals.append(residual)
        minimum_sum = [a + b for a, b in zip(minimum_sum, minimum)]
        maximum_sum = [a + b for a, b in zip(maximum_sum, maximum_residual)]
        if pairwise:
            for winner in range(c):
                for other in range(c):
                    if winner != other:
                        pair_sum[winner][other] += max(r[other] - r[winner] for r in residuals)
        compiled_trees.append({'features': tree.features, 'thresholds': tree.thresholds,
                               'leaves': rows})
        # Integer input leaf bank is no longer needed; don't retain two full models.
        tree.values = []
    rounding = [r / step for r in _roundoff(source)]
    lows, highs = [], []
    for j in range(c):
        low = (bias_residual[j] + Fraction(minimum_sum[j], denominator) - rounding[j]) * fine
        high = (bias_residual[j] + Fraction(maximum_sum[j], denominator) + rounding[j]) * fine
        lows.append(low.numerator // low.denominator)
        highs.append(-((-high.numerator) // high.denominator))
    pair_upper = None
    if pairwise:
        pair_upper = [[0] * c for _ in range(c)]
        for winner in range(c):
            for other in range(c):
                if winner == other:
                    continue
                bound = (bias_residual[other] - bias_residual[winner] +
                         Fraction(pair_sum[winner][other], denominator) +
                         rounding[winner] + rounding[other]) * fine
                pair_upper[winner][other] = -((-bound.numerator) // bound.denominator)
    envelope = (len(source.trees) + 1) * limit * fine
    _require(envelope + max(map(abs, lows + highs)) < 2**61,
             'certificate accumulator envelope exceeds int64 safety budget')
    return {'format': reference.FORMAT, 'source_sha256': source.digest,
            'domain': {'features': source.features, 'maximum': maximum},
            'classes': c, 'binary_source': source.binary,
            'quantization': {'bits': bits, 'step_exponent': exponent,
                             'bound_fraction_bits': fine_bits},
            'bias': qb, 'trees': compiled_trees,
            'error_low': lows, 'error_high': highs, 'pair_upper': pair_upper,
            'contract': {'source_arithmetic': 'binary64 summation then positive scale and bias',
                         'rounding': 'nearest-even', 'gradual_underflow': True,
                         'no_overflow_envelope': '2**900',
                         'claim': 'source-model class-index agreement, not ground truth'}}
