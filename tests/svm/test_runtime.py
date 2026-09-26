"""Framework-free package tests; fixture outcomes are not accuracy benchmarks."""
import concurrent.futures
import ctypes as C
import itertools
import math
import struct
import subprocess
import sys
import zlib

import pytest
from spectra.svm import Session, SCHEDULES, verify_certificate


def write_model(path, classes=3, per_class=32):
    n = classes * per_class
    bank = [float((i + d * 3) % 8) / 8 for i in range(n) for d in range(16)]
    coef = [((-1.0 if (i + r) % 3 else 1.0) / (1 + i % 7))
            if i % 4 else 0.0 for r in range(classes - 1) for i in range(n)]
    bias = [0.01 * (i - 1) for i in range(classes * (classes - 1) // 2)]
    raw = struct.pack('<d', 0.3) + struct.pack('<' + 'I' * classes, *([per_class] * classes))
    values = bank + coef + bias
    raw += struct.pack('<' + 'd' * len(values), *values)
    path.write_bytes(struct.pack('<8sIIII', b'SPCSVM01', classes, n, len(raw), zlib.crc32(raw)) + raw)
    return path


def oracle(path, values):
    """Independently interpret all original dense coefficient rows in Python."""
    raw = path.read_bytes(); _, c, n, _, _ = struct.unpack('<8sIIII', raw[:24])
    gamma = struct.unpack_from('<d', raw, 24)[0]
    counts = struct.unpack_from('<' + 'I' * c, raw, 32)
    pos = 32 + 4 * c
    bank = struct.unpack_from('<' + 'd' * (16 * n), raw, pos);pos += 128 * n
    coeff = struct.unpack_from('<' + 'd' * ((c - 1) * n), raw, pos);pos += 8 * (c - 1) * n
    bias = struct.unpack_from('<' + 'd' * (c * (c - 1) // 2), raw, pos)
    x = struct.unpack('<16f', struct.pack('<16f', *values))
    kernels = []
    for i in range(n):
        total = 0.0
        for d in range(16):
            a = x[d] - bank[16 * i + d];total += a * a
        kernels.append(math.exp(-gamma * total))
    starts = [sum(counts[:i]) for i in range(c)]
    wins = [0] * c;p = 0
    for i in range(c):
        for j in range(i + 1, c):
            value = 0.0
            for cls, row in ((i, j - 1), (j, i)):
                for k in range(starts[cls], starts[cls] + counts[cls]):
                    value += coeff[row * n + k] * kernels[k]
            value += bias[p];p += 1
            winner = (1 if value >= 0 else 0) if c == 2 else (i if value > 0 else j)
            wins[winner] += 1
    return max(range(c), key=lambda i: (wins[i], -i))


@pytest.mark.parametrize('classes', [2, 3, 10, 26])
@pytest.mark.parametrize('tables', [False, True])
def test_modes_hints_and_independent_certificate(tmp_path, library, classes, tables):
    model = write_model(tmp_path / 'm.srt', classes)
    with Session(model, library, tables=tables) as session:
        for x in [[0.0] * 16, [0.13 * i for i in range(16)], [-1.0] * 16]:
            expected = oracle(model, x)
            for schedule in SCHEDULES:
                for hint in (-1, 0, classes - 1):
                    result = session.predict_with_certificate(x, schedule=schedule, hint=hint)
                    assert result.class_index == expected
                    assert verify_certificate(classes, expected, result.pair_outcomes)
                    assert result.evaluated_pairs == sum(w >= 0 for w in result.pair_outcomes)
        original = session.info['classes'];session.info['classes'] = 0
        assert session.info['classes'] == original


@pytest.mark.parametrize('row', [[], [0.] * 15, [0.] * 17, [float('nan')] * 16,
                                [float('inf')] * 16, [1e300] * 16, ['0'] * 16, [True] * 16])
def test_bad_inputs_and_closed_session(tmp_path, library, row):
    session = Session(write_model(tmp_path / 'm.srt'), library)
    with pytest.raises(ValueError):session.predict(row)
    session.close();session.close()
    with pytest.raises(ValueError):session.predict([0.] * 16)


@pytest.mark.parametrize('field', ['magic', 'length', 'crc', 'class', 'nan'])
def test_malformed_model(tmp_path, library, field):
    p = write_model(tmp_path / 'm.srt');raw = bytearray(p.read_bytes())
    if field == 'magic':raw[0] = 0
    elif field == 'length':raw.pop()
    elif field == 'crc':raw[-1] ^= 1
    elif field == 'class':struct.pack_into('<I', raw, 8, 129)
    else:
        struct.pack_into('<d', raw, 24, float('nan'))
        struct.pack_into('<I', raw, 20, zlib.crc32(raw[24:]))
    p.write_bytes(raw)
    with pytest.raises(ValueError):Session(p, library)


def test_shared_session_serializes_certificate_and_close(tmp_path, library):
    path = write_model(tmp_path / 'm.srt', 10)
    rows = [[(i * 7 + j * 3) % 19 / 19 for j in range(16)] for i in range(24)]
    expected = [oracle(path, x) for x in rows]
    with Session(path, library, tables=True) as session:
        def request(i):
            result = session.predict_with_certificate(rows[i % 24])
            assert verify_certificate(10, result.class_index, result.pair_outcomes)
            return result.class_index
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(request, range(480))) == expected * 20


def test_first_index_certificate_predicate_against_all_completions():
    n = 4;edges = list(itertools.combinations(range(n), 2))
    for known in itertools.product(*[(-1, i, j) for i, j in edges]):
        answers = set()
        for completion in itertools.product(*[(i, j) if v == -1 else (v,) for v, (i, j) in zip(known, edges)]):
            answers.add(max(range(n), key=lambda i: (completion.count(i), -i)))
        for winner in range(n):
            assert verify_certificate(n, winner, known) == (answers == {winner})
    assert not verify_certificate(4, True, [0] * 6)
    assert not verify_certificate(4, 0, [True] * 6)
    assert not verify_certificate(4, 0, [0] * 5)


def test_framework_free_import():
    subprocess.run([sys.executable, '-S', '-c',
                    'import spectra.svm, spectra.svm_export, sys; '
                    'assert not ({"torch", "numpy", "sklearn"} & sys.modules.keys())'], check=True)
