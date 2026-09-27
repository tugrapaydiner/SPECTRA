"""Model/input-bound SVM decision receipts and a separate ordered replay checker.

Verification imports no native runtime, NumPy, or scikit-learn. It decodes the
model independently, recomputes every disclosed pair, and then checks the vote
bounds. This verifies the local ordered-binary64 computation, not exact-real
exponentials, model authenticity, or the correctness of the real-world label.
"""
from __future__ import annotations

from array import array
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import zlib

FORMAT = 'spectra.svm.decision-receipt.v1'
_MAX_MODEL = 64 * 1024 * 1024
_MAX_RECEIPT = 1024 * 1024
_FIELDS = {'format', 'model_sha256', 'input_dtype', 'input_hex', 'input_sha256',
           'class_index', 'label', 'pair_outcomes'}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('nonfinite JSON token')


def _json(raw):
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=_unique,
                          parse_constant=_constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('invalid UTF-8 or excessive JSON nesting') from error


def load_receipt(path: str | Path) -> dict:
    with Path(path).open('rb') as stream:
        raw = stream.read(_MAX_RECEIPT + 1)
    if not 1 <= len(raw) <= _MAX_RECEIPT:
        raise ValueError('receipt exceeds byte limit or is empty')
    receipt = _json(raw)
    if type(receipt) is not dict:
        raise ValueError('expected a receipt object')
    return receipt


def _input(values, features, dtype):
    if dtype not in ('float32', 'float64'):
        raise ValueError('explicit float32 or float64 precision required')
    row = list(itertools.islice(iter(values), features + 1))
    if len(row) != features:
        raise ValueError('expected input feature count differs')
    try:
        if any(isinstance(v, (bool, str, bytes)) or not math.isfinite(v) for v in row):
            raise ValueError('expected input must be finite numeric values')
        rounded = array('f' if dtype == 'float32' else 'd', row)
    except (TypeError, OverflowError) as error:
        raise ValueError('expected input is outside selected precision') from error
    if any(not math.isfinite(v) for v in rounded):
        raise ValueError('expected input is outside selected precision')
    return [float(v) for v in rounded]


def _input_hash(values, dtype):
    payload = dtype.encode('ascii') + b'\0' + struct.pack('<I', len(values))
    payload += struct.pack('<' + 'd' * len(values), *values)
    return hashlib.sha256(payload).hexdigest()


def create_receipt(worker, values, *, schedule='beretta_cert', hint=-1) -> dict:
    """Run one stock shared worker and capture its exact input and partial vote.

    Values are transformed model features, not raw preprocessing-schema values.
    Inputs can be sensitive: persisting this record persists those feature values.
    Normal prediction APIs and their cost are unchanged.
    """
    from .svm_shared import SharedSession
    if type(worker) is not SharedSession:
        raise ValueError('expected a stock SharedSession')
    row = _input(values, worker.features, worker.input_dtype)
    prediction = worker.predict_with_certificate(row, schedule=schedule, hint=hint)
    return {'format': FORMAT, 'model_sha256': worker.sha256,
            'input_dtype': worker.input_dtype, 'input_hex': [v.hex() for v in row],
            'input_sha256': _input_hash(row, worker.input_dtype),
            'class_index': prediction.class_index, 'label': prediction.label,
            'pair_outcomes': list(prediction.pair_outcomes)}


class _Model:
    """Independent wire decoder. It does not call the deployment model parser."""
    def __init__(self, raw):
        if not 24 <= len(raw) <= _MAX_MODEL:
            raise ValueError('invalid model byte count')
        magic = raw[:8]
        if magic == b'SPCSVM01':
            _, c, n, size, crc = struct.unpack_from('<8sIIII', raw)
            h, d, metadata_size = 24, 16, 0
            labels = list(range(c)) if c <= 128 else []
        elif magic == b'SPCSVM02' and len(raw) >= 32:
            _, c, n, d, metadata_size, size, crc = struct.unpack_from('<8sIIIIII', raw)
            h = 32
            if metadata_size > 65536 or h + metadata_size + size != len(raw):
                raise ValueError('invalid model inventory')
            metadata = _json(raw[h:h + metadata_size])
            if type(metadata) is not dict or set(metadata) != {'labels'}:
                raise ValueError('invalid label metadata')
            labels = metadata['labels']
        else:
            raise ValueError('unsupported model format')
        if not 2 <= c <= 128 or not 1 <= n <= 100000 or not 1 <= d <= 4096 or n * d > 8_000_000:
            raise ValueError('unsupported model geometry')
        expected = 8 * (1 + n * d + (c - 1) * n + c * (c - 1) // 2) + 4 * c
        if size != expected or h + metadata_size + size != len(raw):
            raise ValueError('invalid model inventory')
        if zlib.crc32(raw[h:]) != crc:
            raise ValueError('model CRC mismatch')
        if type(labels) is not list or len(labels) != c:
            raise ValueError('invalid labels')
        try:
            ints = all(type(v) is int and -(2**63) <= v < 2**63 for v in labels)
            strings = all(type(v) is str and len(v.encode('utf-8')) <= 256 for v in labels)
            if not (ints or strings) or len(set(labels)) != c:
                raise ValueError('labels must be unique bounded values of one type')
        except (UnicodeError, TypeError) as error:
            raise ValueError('invalid labels') from error
        offset = h + metadata_size
        gamma = struct.unpack_from('<d', raw, offset)[0]
        counts = struct.unpack_from('<' + 'I' * c, raw, offset + 8)
        if not math.isfinite(gamma) or gamma <= 0 or any(k == 0 for k in counts) or sum(counts) != n:
            raise ValueError('invalid gamma or class support counts')
        numbers = array('d')
        numbers.frombytes(raw[offset + 8 + 4 * c:])
        if sys.byteorder != 'little':
            numbers.byteswap()
        if any(not math.isfinite(v) for v in numbers):
            raise ValueError('nonfinite model parameter')
        bank = memoryview(numbers)
        self.vectors = bank[:n * d]
        self.coefficients = bank[n * d:n * d + (c - 1) * n]
        self.biases = bank[n * d + (c - 1) * n:]
        self.c, self.n, self.d, self.gamma = c, n, d, gamma
        self.labels = labels
        self.starts = [0]
        for count in counts:
            self.starts.append(self.starts[-1] + count)
        # Match admission for every pair, even those absent from the receipt.
        # This scans stored coefficients once; it evaluates no input kernels.
        pair = 0
        limit = sys.float_info.max / 4
        for i in range(c):
            for j in range(i + 1, c):
                magnitude = abs(self.biases[pair])
                pair += 1
                if magnitude >= limit:
                    raise ValueError('pair magnitude is outside runtime bounds')
                for cls, coefficient_row in ((i, j - 1), (j, i)):
                    for support in range(self.starts[cls], self.starts[cls + 1]):
                        value = abs(self.coefficients[coefficient_row * n + support])
                        if value >= limit - magnitude:
                            raise ValueError('pair magnitude is outside runtime bounds')
                        magnitude += value


def verify_receipt(model_path: str | Path, receipt: dict, *, expected_input,
                   input_dtype: str, max_kernel_work: int = 8_000_000) -> dict:
    """Replay disclosed pair signs, then check every possible undisclosed vote.

    The caller supplies the intended model and input independently of the receipt.
    A result from another input/model is not accepted merely because its vote bounds
    are valid. Runtime cost is charged by feature operations for evaluated kernels;
    the limit is not a wall-clock deadline or process-memory sandbox. Disagreement
    across libm/compiler environments rejects rather than applying a sign tolerance.
    """
    if type(receipt) is not dict or set(receipt) != _FIELDS or receipt['format'] != FORMAT:
        raise ValueError('invalid receipt schema')
    if type(max_kernel_work) is not int or max_kernel_work < 0:
        raise ValueError('max_kernel_work must be a nonnegative integer')
    if type(input_dtype) is not str or receipt['input_dtype'] != input_dtype:
        raise ValueError('receipt input precision differs')
    with Path(model_path).open('rb') as stream:
        raw = stream.read(_MAX_MODEL + 1)
    if len(raw) > _MAX_MODEL:
        raise ValueError('model exceeds byte limit')
    model_hash = hashlib.sha256(raw).hexdigest()
    if receipt['model_sha256'] != model_hash:
        raise ValueError('receipt belongs to a different model')
    model = _Model(raw)
    row = _input(expected_input, model.d, input_dtype)
    encoded = receipt['input_hex']
    if type(encoded) is not list or encoded != [v.hex() for v in row]:
        raise ValueError('receipt belongs to a different input')
    input_hash = _input_hash(row, input_dtype)
    if receipt['input_sha256'] != input_hash:
        raise ValueError('input digest differs')
    winner = receipt['class_index']
    if type(winner) is not int or not 0 <= winner < model.c:
        raise ValueError('invalid class index')
    if type(receipt['label']) is not type(model.labels[winner]) or receipt['label'] != model.labels[winner]:
        raise ValueError('label does not match model class ordering')
    pairs = receipt['pair_outcomes']
    if type(pairs) is not list or len(pairs) != model.c * (model.c - 1) // 2:
        raise ValueError('invalid pair inventory')
    wins, remaining = [0] * model.c, [model.c - 1] * model.c
    kernels = {}
    kernel_work = terms = evaluated = index = 0
    for i in range(model.c):
        for j in range(i + 1, model.c):
            outcome = pairs[index]
            if type(outcome) is not int or outcome not in (-1, i, j):
                raise ValueError('invalid pair outcome')
            bias = model.biases[index]
            index += 1
            if outcome == -1:
                continue
            score = 0.
            magnitude = abs(bias)
            for cls, coefficient_row in ((i, j - 1), (j, i)):
                for support in range(model.starts[cls], model.starts[cls + 1]):
                    coefficient = model.coefficients[coefficient_row * model.n + support]
                    if coefficient == 0.:
                        continue
                    if abs(coefficient) >= sys.float_info.max / 4 - magnitude:
                        raise ValueError('pair magnitude is outside runtime bounds')
                    magnitude += abs(coefficient)
                    if support not in kernels:
                        if kernel_work + model.d > max_kernel_work:
                            raise ValueError('kernel replay work limit exceeded')
                        distance = 0.
                        for feature, value in enumerate(row):
                            difference = value - model.vectors[support * model.d + feature]
                            distance += difference * difference
                        kernels[support] = math.exp(-model.gamma * distance)
                        kernel_work += model.d
                    score += coefficient * kernels[support]
                    terms += 1
            if magnitude >= sys.float_info.max / 4:
                raise ValueError('pair magnitude is outside runtime bounds')
            score += bias
            if not math.isfinite(score):
                raise ValueError('nonfinite replayed margin')
            actual = (1 if score >= 0. else 0) if model.c == 2 else (i if score > 0. else j)
            if actual != outcome:
                raise ValueError('disclosed pair outcome disagrees with model/input replay')
            wins[actual] += 1
            remaining[i] -= 1
            remaining[j] -= 1
            evaluated += 1
    # First-index tie handling is part of the model's decision contract.
    for opponent in range(model.c):
        if opponent == winner:
            continue
        upper = wins[opponent] + remaining[opponent]
        if wins[winner] < upper or (wins[winner] == upper and winner > opponent):
            raise ValueError('disclosed comparisons do not settle the claimed winner')
    return {'verified': True, 'model_sha256': model_hash, 'input_sha256': input_hash,
            'class_index': winner, 'label': model.labels[winner],
            'pairs_replayed': evaluated, 'kernels_replayed': len(kernels),
            'coefficient_terms_replayed': terms, 'kernel_feature_work': kernel_work,
            'numerical_contract': 'ordered binary64/libm replay, not exact-real arithmetic'}
