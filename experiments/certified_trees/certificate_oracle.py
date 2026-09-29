"""Independent exact-rational oracle for compact CatBoost numeric tree decisions.

Research/reference implementation, NOT a fast native runtime. No training or
calibration. A certificate concerns the supplied source MODEL'S class index,
not the true class. Only numeric oblivious trees and bounded uint8 inputs.

The source comparison assumes IEEE binary64 round-to-nearest operations, gradual
underflow, no overflow, at most T-1 additions to sum T leaf values per class,
then one common positive scale multiplication and one bias addition. This is a
stated arithmetic contract, NOT automatic acceptance of an arbitrary library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
from typing import Any, Sequence

FORMAT = 'spectra.tree.certificate-oracle.v1'
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_SCALARS = 500_000
MAX_PAIR_WORK = 2_000_000
U = F(1, 2**53)
ETA = F(1, 2**1074)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        _check(key not in out, 'duplicate JSON key')
        out[key] = value
    return out


def _finite(value: str) -> float:
    number = float(value)
    _check(math.isfinite(number), 'nonfinite JSON number')
    return number


def loads(raw: bytes) -> Any:
    _check(type(raw) is bytes and len(raw) <= MAX_SOURCE_BYTES, 'JSON byte limit')
    def reject(value: str) -> Any:
        raise ValueError('nonfinite JSON token: ' + value)
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=_unique,
                          parse_float=_finite, parse_constant=reject)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('invalid JSON encoding or nesting') from error


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def rational(value: Any) -> F:
    _check(type(value) in (float, int) and math.isfinite(value), 'finite scalar required')
    # Interpret the source as stored binary64 values, not exact decimal literals.
    return F.from_float(float(value))


def power2(exponent: int) -> F:
    return F(2**exponent) if exponent >= 0 else F(1, 2**(-exponent))


def ceil_fraction(value: F) -> int:
    return -((-value.numerator) // value.denominator)


def floor_fraction(value: F) -> int:
    return value.numerator // value.denominator


def nearest_even(value: F) -> int:
    lower, remainder = divmod(value.numerator, value.denominator)
    twice = 2 * remainder
    return lower + int(twice > value.denominator or
                       (twice == value.denominator and lower % 2 != 0))


def ceil_log2(value: F) -> int:
    _check(value > 0, 'positive logarithm argument required')
    estimate = value.numerator.bit_length() - value.denominator.bit_length()
    return estimate + int(power2(estimate) < value)


@dataclass(frozen=True)
class SourceTree:
    features: tuple[int, ...]
    thresholds: tuple[int, ...]
    leaves: tuple[tuple[F, ...], ...]


@dataclass(frozen=True)
class Source:
    digest: str
    features: int
    maximum: int
    classes: int
    binary: bool
    scale: F
    bias: tuple[F, ...]
    trees: tuple[SourceTree, ...]


def parse_source(raw: bytes, maximum: int, features: int | None = None) -> Source:
    """Read exported CatBoost JSON; refuse unsupported feature/tree semantics."""
    _check(type(maximum) is int and 1 <= maximum <= 255, 'maximum must be 1..255')
    doc = loads(raw)
    _check(type(doc) is dict, 'source object required')
    info = doc.get('features_info', {})
    _check(type(info) is dict, 'invalid feature metadata')
    for key in ('categorical_features', 'text_features', 'embedding_features', 'ctrs'):
        _check(not info.get(key), 'only numeric source features are supported')
    fs = info.get('float_features')
    _check(type(fs) is list and 1 <= len(fs) <= 256, 'numeric feature inventory required')
    mapping: dict[int, int] = {}
    for feature in fs:
        _check(type(feature) is dict, 'invalid feature entry')
        index, flat = feature.get('feature_index'), feature.get('flat_feature_index')
        _check(type(index) is int and type(flat) is int and 0 <= index < 256 and
               0 <= flat < 256 and index not in mapping, 'invalid feature indices')
        mapping[index] = flat
    _check(len(set(mapping.values())) == len(mapping), 'duplicate flat feature index')
    inferred = max(mapping.values()) + 1
    width = inferred if features is None else features
    _check(type(width) is int and inferred <= width <= 256, 'invalid declared feature count')
    scale_bias = doc.get('scale_and_bias')
    _check(type(scale_bias) is list and len(scale_bias) == 2 and
           type(scale_bias[1]) is list and 1 <= len(scale_bias[1]) <= 64,
           'explicit scale/bias metadata required')
    scale = rational(scale_bias[0]); original_bias = tuple(map(rational, scale_bias[1]))
    _check(scale > 0, 'only positive common output scales are supported')
    binary = len(original_bias) == 1
    bias = (F(0), original_bias[0]) if binary else original_bias
    classes = len(bias)
    raw_trees = doc.get('oblivious_trees')
    _check(type(raw_trees) is list and 1 <= len(raw_trees) <= 4096,
           'bounded oblivious-tree inventory required')
    _check(not doc.get('trees'), 'asymmetric trees not supported')
    trees: list[SourceTree] = []; scalars = 0
    for tree in raw_trees:
        _check(type(tree) is dict, 'invalid tree entry')
        splits = tree.get('splits') or []
        _check(type(splits) is list and len(splits) <= 12, 'tree depth limit')
        fidx: list[int] = []; thresholds: list[int] = []
        for split in splits:
            _check(type(split) is dict and split.get('split_type') == 'FloatFeature',
                   'only FloatFeature splits are supported')
            idx = split.get('float_feature_index')
            _check(type(idx) is int and idx in mapping, 'unknown split feature')
            border = float(rational(split.get('border')))
            try:
                round32 = struct.unpack('<f', struct.pack('<f', border))[0]
            except (OverflowError, struct.error) as error:
                raise ValueError('unrepresentable split border') from error
            _check(round32 == border, 'border must identify an exact binary32 threshold')
            # q is a bounded exact integer. Clamping the predicate cutoff to these
            # two outer sentinels preserves constant true/false splits.
            cutoff = min(maximum, max(-1, math.floor(border)))
            fidx.append(mapping[idx]); thresholds.append(cutoff)
        values = tree.get('leaf_values')
        original_classes = 1 if binary else classes
        expected = (1 << len(splits)) * original_classes
        _check(type(values) is list and len(values) == expected, 'leaf-value inventory')
        scalars += (1 << len(splits)) * classes
        _check(scalars <= MAX_SCALARS, 'reference-oracle leaf-scalar budget exceeded')
        leaves = []
        for i in range(1 << len(splits)):
            values_at_leaf = tuple(rational(x) for x in
                                  values[i * original_classes:(i + 1) * original_classes])
            leaves.append((F(0), values_at_leaf[0]) if binary else values_at_leaf)
        trees.append(SourceTree(tuple(fidx), tuple(thresholds), tuple(leaves)))
    return Source(sha(raw), width, maximum, classes, binary, scale, bias, tuple(trees))


def validate_query(source: Source, query: Sequence[int]) -> tuple[int, ...]:
    _check(isinstance(query, (list, tuple, bytes, bytearray)), 'integer row sequence required')
    row = tuple(query)
    _check(len(row) == source.features and all(type(x) is int and
           0 <= x <= source.maximum for x in row), 'invalid query shape/domain')
    return row


def leaf_index(features: Sequence[int], thresholds: Sequence[int], query: Sequence[int]) -> int:
    result = 0
    for bit, (feature, threshold) in enumerate(zip(features, thresholds)):
        if query[feature] > threshold:
            result |= 1 << bit
    return result


def source_scores(source: Source, query: Sequence[int], exact: bool = False) -> list[Any]:
    row = validate_query(source, query)
    totals: list[Any] = [F(0) if exact else 0.0 for _ in range(source.classes)]
    for tree in source.trees:
        leaf = tree.leaves[leaf_index(tree.features, tree.thresholds, row)]
        for c in range(source.classes):
            totals[c] += leaf[c] if exact else float(leaf[c])
    if exact:
        return [source.scale * v + b for v, b in zip(totals, source.bias)]
    return [float(source.scale) * v + float(b) for v, b in zip(totals, source.bias)]


def _roundoff_bounds(source: Source) -> tuple[F, ...]:
    """Conservative absolute source-score errors, in source.scale-normalized units.

    The additive ETA term covers gradual underflow. Bounds use exact rationals,
    not inward-rounded floating arithmetic. Excessively large sources are refused.
    """
    count = len(source.trees) + 4
    _check(count * U < 1, 'arithmetic depth exceeds roundoff argument')
    denominator = 1 - count * U
    gamma = count * U / denominator
    errors = []
    for c in range(source.classes):
        mass = sum((max(abs(leaf[c]) for leaf in tree.leaves)
                    for tree in source.trees), F(0))
        total = source.scale * mass + abs(source.bias[c])
        _check(mass < power2(900) and total < power2(900),
               'source exceeds conservative no-overflow envelope')
        error = gamma * total + count * ETA * max(F(1), source.scale) / denominator
        errors.append(error / source.scale)
    return tuple(errors)


def compile_source(raw: bytes, maximum: int, *, features: int | None = None,
                   bits: int = 16, pairwise: bool = False) -> dict[str, Any]:
    """Reference compiler with exact-rational, outward-rounded certificate bounds.

    `pairwise=True` uses correlated class residual differences to tighten the
    certificate. Its additional O(classes**2) payload and offline work are counted.
    No examples or labels are used to choose or bound quantization.
    """
    _check(type(bits) is int and bits in (8, 16), 'bits must be 8 or 16')
    _check(type(pairwise) is bool, 'pairwise flag must be boolean')
    source = parse_source(raw, maximum, features)
    contrasts = [tuple(v - leaf[0] for v in leaf)
                 for tree in source.trees for leaf in tree.leaves]
    bias = tuple((v - source.bias[0]) / source.scale for v in source.bias)
    amplitude = max([abs(v) for leaf in contrasts for v in leaf] + [abs(v) for v in bias])
    limit = (1 << (bits - 1)) - 1
    exponent = ceil_log2(amplitude / limit) if amplitude else 0
    _check(-900 <= exponent <= 900, 'quantization exponent outside reference scope')
    step = power2(exponent)
    fine_bits = 20; fine = 1 << fine_bits
    qb = [nearest_even(v / step) for v in bias]
    bias_residual = [v / step - q for v, q in zip(bias, qb)]
    error_low = list(bias_residual); error_high = list(bias_residual)
    compiled_trees = []; residual_trees = []
    for tree in source.trees:
        qleaves = []; residuals = []
        for leaf in tree.leaves:
            normalized = [(v - leaf[0]) / step for v in leaf]
            integers = [nearest_even(v) for v in normalized]
            _check(all(-limit <= q <= limit for q in integers), 'quantizer range failure')
            qleaves.append(integers)
            residuals.append([v - q for v, q in zip(normalized, integers)])
        for c in range(source.classes):
            error_low[c] += min(r[c] for r in residuals)
            error_high[c] += max(r[c] for r in residuals)
        compiled_trees.append({'features': list(tree.features), 'thresholds': list(tree.thresholds),
                               'leaves': qleaves})
        residual_trees.append(residuals)
    rounding = [v / step for v in _roundoff_bounds(source)]
    lows = [floor_fraction((v - r) * fine) for v, r in zip(error_low, rounding)]
    highs = [ceil_fraction((v + r) * fine) for v, r in zip(error_high, rounding)]
    pair_upper = None
    if pairwise:
        work = sum(len(t.leaves) for t in source.trees) * source.classes**2
        _check(work <= MAX_PAIR_WORK, 'exact pairwise reference-work budget exceeded')
        pair_upper = [[0] * source.classes for _ in range(source.classes)]
        for winner in range(source.classes):
            for other in range(source.classes):
                if winner == other:
                    continue
                bound = bias_residual[other] - bias_residual[winner]
                for residuals in residual_trees:
                    bound += max(r[other] - r[winner] for r in residuals)
                bound += rounding[winner] + rounding[other]
                pair_upper[winner][other] = ceil_fraction(bound * fine)
    envelope = (len(source.trees) + 1) * limit * fine
    _check(envelope + max(map(abs, lows + highs)) < 2**61,
           'certificate accumulator envelope exceeds int64 safety budget')
    return {'format': FORMAT, 'source_sha256': source.digest,
            'domain': {'features': source.features, 'maximum': maximum},
            'classes': source.classes, 'binary_source': source.binary,
            'quantization': {'bits': bits, 'step_exponent': exponent,
                             'bound_fraction_bits': fine_bits},
            'bias': qb, 'trees': compiled_trees,
            'error_low': lows, 'error_high': highs, 'pair_upper': pair_upper,
            'contract': {'source_arithmetic': 'binary64 summation then positive scale and bias',
                         'rounding': 'nearest-even', 'gradual_underflow': True,
                         'no_overflow_envelope': '2**900',
                         'claim': 'source-model class-index agreement, not ground truth'}}


def verify_compiled(raw: bytes, compact: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct every bound and coefficient from source; no empirical trust.

    This is an exact reference check against the compiler's mathematical policy,
    not a second implementation of every compiler operation. The tests separately
    check full-domain source predictions and bounds. Do not call these independent
    algorithms when describing this function alone.
    """
    _check(type(compact) is dict and compact.get('format') == FORMAT, 'compact format')
    _check(compact.get('source_sha256') == sha(raw), 'source hash mismatch')
    try:
        d, q = compact['domain'], compact['quantization']
        expected = compile_source(raw, d['maximum'], features=d['features'], bits=q['bits'],
                                  pairwise=compact['pair_upper'] is not None)
    except (KeyError, TypeError) as error:
        raise ValueError('invalid compact metadata') from error
    _check(canonical(compact) == canonical(expected), 'compact coefficients/bounds/contract differ')
    return {'status': 'PASS', 'source_sha256': sha(raw),
            'compact_sha256': sha(canonical(compact)),
            'trees': len(expected['trees']), 'classes': expected['classes'],
            'scope': 'source-bound deterministic reconstruction; not empirical accuracy'}


class CertifiedOracle:
    """Reference predictor accepting only a source-verified compact object.

    Deep-copy after verification prevents accidental caller mutation. No pickle,
    downloaded executable, native extension, model training or numerical framework.
    """
    def __init__(self, raw_source: bytes, compact: dict[str, Any]):
        self.receipt = verify_compiled(raw_source, compact)
        self.source = parse_source(raw_source, compact['domain']['maximum'],
                                   compact['domain']['features'])
        self.model = loads(canonical(compact))
        c = self.source.classes
        remaining_low = [0] * c; remaining_high = [0] * c
        self.suffix = [(tuple(remaining_low), tuple(remaining_high))]
        for tree in reversed(self.model['trees']):
            for j in range(c):
                remaining_low[j] += min(leaf[j] for leaf in tree['leaves'])
                remaining_high[j] += max(leaf[j] for leaf in tree['leaves'])
            self.suffix.append((tuple(remaining_low), tuple(remaining_high)))
        self.suffix.reverse()

    def predict(self, query: Sequence[int], *, checkpoint: int = 16,
                use_pairwise: bool = True) -> dict[str, Any]:
        _check(type(checkpoint) is int and checkpoint >= 0, 'nonnegative checkpoint required')
        _check(type(use_pairwise) is bool, 'boolean pairwise flag required')
        row = validate_query(self.source, query)
        model = self.model; fine = 1 << model['quantization']['bound_fraction_bits']
        totals = list(model['bias']); count = len(model['trees'])
        def settle(processed: int) -> int | None:
            lo_rem, hi_rem = self.suffix[processed]
            lows = [(t + r) * fine + e for t, r, e in
                    zip(totals, lo_rem, model['error_low'])]
            highs = [(t + r) * fine + e for t, r, e in
                     zip(totals, hi_rem, model['error_high'])]
            for candidate in sorted(range(self.source.classes),
                                    key=lambda i: (-lows[i], i)):
                if all(candidate == j or lows[candidate] > highs[j] or
                       (candidate < j and lows[candidate] == highs[j])
                       for j in range(self.source.classes)):
                    return candidate
            if processed == count and use_pairwise and model['pair_upper'] is not None:
                for candidate in sorted(range(self.source.classes), key=lambda i: (-totals[i], i)):
                    if all(candidate == j or
                           (totals[candidate] - totals[j]) * fine > model['pair_upper'][candidate][j] or
                           (candidate < j and (totals[candidate] - totals[j]) * fine ==
                            model['pair_upper'][candidate][j])
                           for j in range(self.source.classes)):
                        return candidate
            return None
        winner = settle(0) if checkpoint else None; processed = 0
        if winner is None:
            for processed, tree in enumerate(model['trees'], 1):
                index = leaf_index(tree['features'], tree['thresholds'], row)
                for c, value in enumerate(tree['leaves'][index]):
                    totals[c] += value
                if processed == count or (checkpoint and processed % checkpoint == 0):
                    winner = settle(processed)
                    if winner is not None:
                        break
        # Partial sums are not a final approximate prediction after early stopping.
        approximate = (max(range(self.source.classes), key=lambda c: totals[c])
                       if processed == count else None)
        return {'status': 'CERTIFIED' if winner is not None else 'UNRESOLVED',
                'class_index': winner, 'approximate_class_index': approximate,
                'trees_evaluated': processed, 'total_trees': count,
                'source_sha256': self.receipt['source_sha256'],
                'compact_sha256': self.receipt['compact_sha256'],
                'query_sha256': sha(bytes(row)),
                'scope': 'conditional source-model decision agreement, not true-label correctness'}


def _read(path: Path) -> bytes:
    with path.open('rb') as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    _check(len(raw) <= MAX_SOURCE_BYTES, 'input byte limit')
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('compile'); c.add_argument('--source', type=Path, required=True)
    c.add_argument('--maximum', type=int, required=True); c.add_argument('--features', type=int)
    c.add_argument('--bits', type=int, choices=[8, 16], default=16)
    c.add_argument('--pairwise', action='store_true'); c.add_argument('--out', type=Path, required=True)
    v = sub.add_parser('verify'); v.add_argument('--source', type=Path, required=True)
    v.add_argument('--compact', type=Path, required=True)
    p = sub.add_parser('predict'); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--compact', type=Path, required=True); p.add_argument('--row', required=True)
    p.add_argument('--checkpoint', type=int, default=16)
    args = parser.parse_args()
    raw = _read(args.source)
    if args.command == 'compile':
        obj = compile_source(raw, args.maximum, features=args.features, bits=args.bits,
                             pairwise=args.pairwise)
        receipt = verify_compiled(raw, obj)
        with args.out.open('xb') as stream:
            stream.write(canonical(obj))
        print(json.dumps(receipt, indent=2))
    elif args.command == 'verify':
        print(json.dumps(verify_compiled(raw, loads(_read(args.compact))), indent=2))
    else:
        model = CertifiedOracle(raw, loads(_read(args.compact)))
        print(json.dumps(model.predict(loads(args.row.encode()), checkpoint=args.checkpoint), indent=2))


if __name__ == '__main__':
    main()
