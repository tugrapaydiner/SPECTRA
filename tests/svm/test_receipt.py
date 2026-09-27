"""Receipt fidelity, tampering, precision, strict JSON and isolated replay."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import zlib

import pytest
from spectra.svm import verify_certificate
from spectra.svm_shared import PreparedModel, SHARED_SCHEDULES
from spectra.svm_receipt import create_receipt, load_receipt, verify_receipt, _Model


def fixture_model(path, features=7, classes=3, zero=False):
    rng = random.Random(71 + features + classes)
    count = 3 * classes
    labels = [f'class-{i}é' for i in range(classes)]
    meta = json.dumps({'labels': labels}, ensure_ascii=False).encode()
    values = [rng.choice([-.5, 0., .25, .75]) for _ in range(count * features)]
    values += [0. if zero else rng.uniform(-1., 1.) for _ in range((classes - 1) * count)]
    values += [0. if zero else rng.uniform(-.1, .1) for _ in range(classes * (classes - 1) // 2)]
    data = struct.pack('<d' + 'I' * classes, .25 / features, *([3] * classes))
    data += struct.pack('<' + 'd' * len(values), *values)
    body = meta + data
    path.write_bytes(struct.pack('<8sIIIIII', b'SPCSVM02', classes, count, features,
                                 len(meta), len(data), zlib.crc32(body)) + body)
    return path


@pytest.mark.parametrize('features,classes', [(1, 2), (7, 3), (16, 10), (33, 4)])
@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('tables', [False, True])
def test_replay_all_schedules_and_precisions(tmp_path, library, features, classes, precision, tables):
    model = fixture_model(tmp_path / 'model.srt', features, classes)
    row = [math.sin(i + 1) for i in range(features)]
    with PreparedModel(model, library, input_dtype=precision, tables=tables) as m, m.session() as w:
        for schedule in SHARED_SCHEDULES:
            record = create_receipt(w, row, schedule=schedule, hint=0)
            result = verify_receipt(model, record, expected_input=row, input_dtype=precision)
            assert result['verified'] and result['label'] == w.predict(row, schedule='exhaustive')
            assert result['pairs_replayed'] == sum(v != -1 for v in record['pair_outcomes'])
            assert result['kernel_feature_work'] == result['kernels_replayed'] * features
            assert 'spectra.svm.decision-receipt.v1' == record['format']


@pytest.fixture
def receipt_case(tmp_path, library):
    path = fixture_model(tmp_path / 'model.srt', zero=True)
    row = [-0., 1., 2., 3., 4., 5., 6.]
    with PreparedModel(path, library, input_dtype='float64') as m, m.session() as w:
        return path, row, create_receipt(w, row, schedule='exhaustive')


@pytest.mark.parametrize('change', ['model', 'input', 'digest', 'precision', 'class', 'label',
                                    'bool_label', 'pair', 'bool_pair', 'missing', 'extra',
                                    'size', 'unknown', 'float_index', 'signed_zero'])
def test_receipt_tampering_rejected(receipt_case, change):
    model, row, original = receipt_case
    record = deepcopy(original)
    if change == 'model': record['model_sha256'] = '0' * 64
    elif change == 'input': record['input_hex'][1] = (9.).hex()
    elif change == 'digest': record['input_sha256'] = '0' * 64
    elif change == 'precision': record['input_dtype'] = 'float32'
    elif change == 'class': record['class_index'] = -1
    elif change == 'label': record['label'] = 'wrong'
    elif change == 'bool_label': record['label'] = True
    elif change == 'pair': record['pair_outcomes'][0] = 2
    elif change == 'bool_pair': record['pair_outcomes'][0] = True
    elif change == 'missing': record.pop('input_hex')
    elif change == 'extra': record['hidden'] = 1
    elif change == 'size': record['pair_outcomes'].pop()
    elif change == 'unknown': record['pair_outcomes'] = [-1] * 3
    elif change == 'float_index': record['class_index'] = 2.
    elif change == 'signed_zero': record['input_hex'][0] = (0.).hex()
    with pytest.raises(ValueError):
        verify_receipt(model, record, expected_input=row, input_dtype='float64')


def test_valid_vote_certificate_with_false_pair_signs_is_rejected(receipt_case):
    model, row, record = receipt_case
    record['class_index'] = 0
    record['label'] = 'class-0é'
    record['pair_outcomes'] = [0, 0, 1]
    assert verify_certificate(3, 0, record['pair_outcomes'])
    with pytest.raises(ValueError, match='pair outcome disagrees'):
        verify_receipt(model, record, expected_input=row, input_dtype='float64')


def test_independently_supplied_input_is_required(receipt_case):
    model, row, record = receipt_case
    with pytest.raises(TypeError):
        verify_receipt(model, record, input_dtype='float64')
    with pytest.raises(ValueError, match='different input'):
        verify_receipt(model, record, expected_input=[8.] * 7, input_dtype='float64')


def test_work_budget_is_enforced(tmp_path, library):
    model = fixture_model(tmp_path / 'model.srt')
    row = [0.] * 7
    with PreparedModel(model, library, input_dtype='float64') as m, m.session() as w:
        record = create_receipt(w, row)
    with pytest.raises(ValueError, match='work limit'):
        verify_receipt(model, record, expected_input=row, input_dtype='float64', max_kernel_work=0)
    for limit in (True, -1, 1.5):
        with pytest.raises(ValueError):
            verify_receipt(model, record, expected_input=row, input_dtype='float64', max_kernel_work=limit)


@pytest.mark.parametrize('raw', [b'{}', b'{"format":1,"format":2}', b'{"x":NaN}',
                                 b'\xff', b'[]', b'[' * 10000, b' ' * (1048576 + 1)])
def test_strict_receipt_file_input(tmp_path, raw):
    file = tmp_path / 'receipt.json';file.write_bytes(raw)
    if raw == b'{}':
        assert load_receipt(file) == {}
    else:
        with pytest.raises(ValueError): load_receipt(file)


@pytest.mark.parametrize('damage', ['short', 'magic', 'crc', 'count', 'labels', 'nonfinite', 'bound'])
def test_independent_model_decoder_rejects_malformed(tmp_path, damage):
    path = fixture_model(tmp_path / 'model.srt')
    raw = bytearray(path.read_bytes())
    label_bytes = struct.unpack_from('<I', raw, 20)[0]
    offset = 32 + label_bytes
    if damage == 'short': raw.pop()
    elif damage == 'magic': raw[0] = 0
    elif damage == 'crc': raw[-1] ^= 1
    elif damage == 'count': struct.pack_into('<I', raw, offset + 8, 0)
    elif damage == 'labels': raw[32] = ord('[')
    elif damage == 'nonfinite': struct.pack_into('<d', raw, offset, float('inf'))
    elif damage == 'bound': struct.pack_into('<d', raw, len(raw) - 8, sys.float_info.max)
    if damage not in ('short', 'crc'): struct.pack_into('<I', raw, 28, zlib.crc32(raw[32:]))
    with pytest.raises(ValueError): _Model(bytes(raw))


def test_replay_without_site_packages_or_native_imports(receipt_case, tmp_path):
    model, row, record = receipt_case
    file = tmp_path / 'receipt.json';file.write_text(json.dumps(record))
    root = Path(__file__).resolve().parents[2]
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from spectra.svm_receipt import load_receipt, verify_receipt
record=load_receipt(sys.argv[3])
result=verify_receipt(sys.argv[2],record,expected_input=[-0.,1.,2.,3.,4.,5.,6.],input_dtype='float64')
assert result['verified']
assert not ({'ctypes','numpy','sklearn','torch','spectra.svm','spectra.svm_shared'} & sys.modules.keys())
print('independent receipt replay PASS')
"""
    done = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(root), str(model), str(file)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
