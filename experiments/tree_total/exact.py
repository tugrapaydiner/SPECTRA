"""Exact dyadic reconstruction of the unchanged SPCERT02 certificate policy.

The large leaf arrays use integers on one binary grid, not per-leaf Fraction
objects. Scale, bias and source-roundoff expressions retain exact rational
arithmetic. This is a faster implementation of the existing proof, not a weaker
hash-only trust policy. The old oracle remains unchanged and is a test control.
"""
from __future__ import annotations
from fractions import Fraction
import hashlib
import math
import struct
from pathlib import Path
from ..certified_trees import packed
from ..certified_trees.reference import certificate_oracle as ref


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def scalar(value) -> float:
    need(type(value) in (int, float), 'finite scalar required')
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError('unrepresentable source scalar') from error
    need(math.isfinite(result), 'finite scalar required')
    return result


def scaled_integer(value: float, denominator_bits: int) -> int:
    numerator, denominator = value.as_integer_ratio()
    return numerator << (denominator_bits - (denominator.bit_length() - 1))


def nearest_dyadic(numerator: int, denominator_bits: int) -> int:
    """Nearest-even for signed numerator / 2**denominator_bits, exactly."""
    if denominator_bits <= 0:
        return numerator << (-denominator_bits)
    lower, remainder = divmod(numerator, 1 << denominator_bits)
    half = 1 << (denominator_bits - 1)
    return lower + int(remainder > half or (remainder == half and lower & 1))


class Analysis:
    """Bounded validated numeric source with exact integral leaf contrasts."""
    def __init__(self, raw: bytes, maximum: int, features: int | None = None):
        need(type(maximum) is int and 1 <= maximum <= 255, 'maximum must be 1..255')
        doc = ref.loads(raw)
        need(type(doc) is dict, 'source object required')
        info = doc.get('features_info', {})
        need(type(info) is dict, 'invalid feature metadata')
        for key in ('categorical_features', 'text_features', 'embedding_features', 'ctrs'):
            need(not info.get(key), 'only numeric source features are supported')
        fs = info.get('float_features')
        need(type(fs) is list and 1 <= len(fs) <= 256, 'numeric feature inventory required')
        mapping = {}
        for feature in fs:
            need(type(feature) is dict, 'invalid feature entry')
            index, flat = feature.get('feature_index'), feature.get('flat_feature_index')
            need(type(index) is int and type(flat) is int and 0 <= index < 256 and
                 0 <= flat < 256 and index not in mapping, 'invalid feature indices')
            mapping[index] = flat
        need(len(set(mapping.values())) == len(mapping), 'duplicate flat feature index')
        inferred = max(mapping.values()) + 1
        self.d = inferred if features is None else features
        need(type(self.d) is int and inferred <= self.d <= 256, 'invalid declared feature count')
        sb = doc.get('scale_and_bias')
        need(type(sb) is list and len(sb) == 2 and type(sb[1]) is list and
             1 <= len(sb[1]) <= 64, 'explicit scale/bias metadata required')
        self.scale = Fraction.from_float(scalar(sb[0]))
        need(self.scale > 0, 'only positive common output scales are supported')
        original_bias = tuple(Fraction.from_float(scalar(v)) for v in sb[1])
        self.binary = len(original_bias) == 1
        self.bias = (Fraction(0), original_bias[0]) if self.binary else original_bias
        self.c = len(self.bias)
        source_trees = doc.get('oblivious_trees')
        need(type(source_trees) is list and 1 <= len(source_trees) <= 4096,
             'bounded oblivious-tree inventory required')
        need(not doc.get('trees'), 'asymmetric trees not supported')
        self.D, self.digest = maximum, hashlib.sha256(raw).hexdigest()
        self.trees = []
        scalars = 0
        self.denominator_bits = 0
        for tree in source_trees:
            need(type(tree) is dict, 'invalid tree entry')
            splits = tree.get('splits') or []
            need(type(splits) is list and len(splits) <= 12, 'tree depth limit')
            fidx, thresholds = [], []
            for split in splits:
                need(type(split) is dict and split.get('split_type') == 'FloatFeature',
                     'only FloatFeature splits are supported')
                index = split.get('float_feature_index')
                need(type(index) is int and index in mapping, 'unknown split feature')
                border = scalar(split.get('border'))
                try:
                    border32 = struct.unpack('<f', struct.pack('<f', border))[0]
                except (OverflowError, struct.error) as error:
                    raise ValueError('unrepresentable split border') from error
                need(border == border32, 'border must identify an exact binary32 threshold')
                fidx.append(mapping[index])
                thresholds.append(min(maximum, max(-1, math.floor(border))))
            values = tree.get('leaf_values')
            original_classes = 1 if self.binary else self.c
            rows = 1 << len(splits)
            need(type(values) is list and len(values) == rows * original_classes,
                 'leaf-value inventory')
            scalars += rows * self.c
            need(scalars <= ref.MAX_SCALARS, 'reference-oracle leaf-scalar budget exceeded')
            for i, v in enumerate(values):
                value = scalar(v)
                values[i] = value
                k = value.as_integer_ratio()[1].bit_length() - 1
                self.denominator_bits = max(self.denominator_bits, k)
            self.trees.append({'features': fidx, 'thresholds': thresholds, 'values': values})
        # Convert the existing lists in place; avoid duplicate Fraction matrices.
        self.mass = [0] * self.c
        amplitude = 0
        for tree in self.trees:
            values = tree.pop('values')
            original_classes = 1 if self.binary else self.c
            leaves = []
            masses = [0] * self.c
            for start in range(0, len(values), original_classes):
                leaf = [scaled_integer(v, self.denominator_bits)
                        for v in values[start:start + original_classes]]
                if self.binary:
                    leaf = [0, leaf[0]]
                for j, value in enumerate(leaf):
                    masses[j] = max(masses[j], abs(value))
                zero = leaf[0]
                contrast = [value - zero for value in leaf]
                amplitude = max(amplitude, max(map(abs, contrast)))
                leaves.append(contrast)
            tree['contrasts'] = leaves
            for j in range(self.c):
                self.mass[j] += masses[j]
        self.normalized_bias = tuple((v - self.bias[0]) / self.scale for v in self.bias)
        denominator = 1 << self.denominator_bits
        self.amplitude = max(Fraction(amplitude, denominator),
                             max(map(abs, self.normalized_bias)))
        count = len(self.trees) + 4
        divisor = 1 - count * ref.U
        need(divisor > 0, 'arithmetic depth exceeds roundoff argument')
        gamma = count * ref.U / divisor
        self.rounding = []
        for j in range(self.c):
            mass = Fraction(self.mass[j], denominator)
            total = self.scale * mass + abs(self.bias[j])
            need(mass < ref.power2(900) and total < ref.power2(900),
                 'source exceeds conservative no-overflow envelope')
            error = gamma * total + count * ref.ETA * max(Fraction(1), self.scale) / divisor
            self.rounding.append(error / self.scale)

    def quantize(self, exponent: int, *, pairwise: bool = False):
        """Return integers and exact outward bounds at any safe dyadic step."""
        step = ref.power2(exponent)
        qb = [ref.nearest_even(v / step) for v in self.normalized_bias]
        bias_residual = [v / step - q for v, q in zip(self.normalized_bias, qb)]
        shift = self.denominator_bits + exponent
        denominator = 1 << max(0, shift)
        minima, maxima = [0] * self.c, [0] * self.c
        compiled, residual_trees = [], []
        if pairwise:
            work = sum(len(t['contrasts']) for t in self.trees) * self.c ** 2
            need(work <= ref.MAX_PAIR_WORK, 'exact pairwise reference-work budget exceeded')
        for tree in self.trees:
            qleaves, residuals = [], []
            low, high = [None] * self.c, [None] * self.c
            for contrast in tree['contrasts']:
                integers, residual = [], []
                for j, value in enumerate(contrast):
                    q = nearest_dyadic(value, shift)
                    err = value - q * denominator if shift >= 0 else 0
                    integers.append(q)
                    residual.append(err)
                    low[j] = err if low[j] is None else min(low[j], err)
                    high[j] = err if high[j] is None else max(high[j], err)
                qleaves.append(integers)
                if pairwise:
                    residuals.append(residual)
            for j in range(self.c):
                minima[j] += low[j]
                maxima[j] += high[j]
            compiled.append({'features': tree['features'], 'thresholds': tree['thresholds'],
                             'leaves': qleaves})
            if pairwise:
                residual_trees.append(residuals)
        fine = 1 << 20
        roundoff = [r / step for r in self.rounding]
        lo = [ref.floor_fraction((bias_residual[j] + Fraction(minima[j], denominator) - roundoff[j]) * fine)
              for j in range(self.c)]
        hi = [ref.ceil_fraction((bias_residual[j] + Fraction(maxima[j], denominator) + roundoff[j]) * fine)
              for j in range(self.c)]
        pair = None
        if pairwise:
            pair = [[0] * self.c for _ in range(self.c)]
            for a in range(self.c):
                for b in range(self.c):
                    if a == b:
                        continue
                    err = sum(max(row[b] - row[a] for row in tree) for tree in residual_trees)
                    bound = bias_residual[b] - bias_residual[a] + Fraction(err, denominator)
                    bound += roundoff[a] + roundoff[b]
                    pair[a][b] = ref.ceil_fraction(bound * fine)
        return qb, compiled, lo, hi, pair

    def compile(self, bits: int = 16, pairwise: bool = False) -> dict:
        need(type(bits) is int and bits in (8, 16), 'bits must be 8 or 16')
        need(type(pairwise) is bool, 'pairwise flag must be boolean')
        limit = (1 << (bits - 1)) - 1
        exponent = ref.ceil_log2(self.amplitude / limit) if self.amplitude else 0
        need(-900 <= exponent <= 900, 'quantization exponent outside reference scope')
        qb, trees, lo, hi, pairs = self.quantize(exponent, pairwise=pairwise)
        for tree in trees:
            need(all(-limit <= v <= limit for leaf in tree['leaves'] for v in leaf),
                 'quantizer range failure')
        envelope = (len(trees) + 1) * limit * (1 << 20)
        need(envelope + max(map(abs, lo + hi)) < 2**61,
             'certificate accumulator envelope exceeds int64 safety budget')
        return {'format': ref.FORMAT, 'source_sha256': self.digest,
                'domain': {'features': self.d, 'maximum': self.D},
                'classes': self.c, 'binary_source': self.binary,
                'quantization': {'bits': bits, 'step_exponent': exponent, 'bound_fraction_bits': 20},
                'bias': qb, 'trees': trees, 'error_low': lo, 'error_high': hi,
                'pair_upper': pairs,
                'contract': {'source_arithmetic': 'binary64 summation then positive scale and bias',
                             'rounding': 'nearest-even', 'gradual_underflow': True,
                             'no_overflow_envelope': '2**900',
                             'claim': 'source-model class-index agreement, not ground truth'}}


def compile_bytes(source: bytes, maximum: int, *, features=None, bits=16, pairwise=False):
    result = Analysis(source, maximum, features).compile(bits, pairwise)
    return packed.pack(result), result


def verify(source: bytes, raw: bytes) -> dict:
    meta = packed.metadata(raw)
    need(hashlib.sha256(source).hexdigest() == meta['source_sha256'], 'original source identity mismatch')
    expected, _ = compile_bytes(source, meta['maximum'], features=meta['features'],
                                bits=meta['bits'], pairwise=bool(meta['flags'] & 2))
    need(raw == expected, 'packed tree/bounds differ from exact dyadic reconstruction')
    return {'status': 'PASS', **meta, 'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'full exact source reconstruction, not empirical calibration or authentication'}
