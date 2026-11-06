"""Resource flags must not turn small valid files into huge read allocations."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import pytest

from spectra._json import load_file
from spectra.cli import main


@pytest.mark.parametrize('cap', [sys.maxsize, sys.maxsize + 1, 10 ** 100])
def test_unrepresentable_cap_is_a_settings_error(tmp_path, cap):
    path = tmp_path / 'small.json'
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='max-json-bytes'):
        load_file(path, max_bytes=cap)


def test_largest_supported_cap_does_not_preallocate_it(tmp_path):
    path = tmp_path / 'small.json'
    path.write_text('{"value": 7}', encoding='utf-8')
    assert load_file(path, max_bytes=sys.maxsize - 1) == {'value': 7}


@pytest.mark.parametrize('cap', [sys.maxsize, sys.maxsize + 1, 10 ** 100])
def test_cli_reports_extreme_cap_without_traceback(tmp_path, cap, capsys):
    formula = tmp_path / 'small.cnf'
    formula.write_text('p cnf 1 1\n1 0\n', encoding='utf-8')
    witness = tmp_path / 'witness.json'
    witness.write_text('{"witness":[true]}', encoding='utf-8')
    assert main(['cnf', 'check', str(formula), str(witness),
                 '--max-json-bytes', str(cap)]) == 2
    out = capsys.readouterr()
    assert not out.out
    assert 'max-json-bytes' in out.err and 'Traceback' not in out.err


@pytest.mark.parametrize('padding', [65520, 65536, 131072])
def test_multiple_chunks_keep_exact_byte_limit_and_utf8(tmp_path, padding):
    value = {'padding': 'x' * padding, 'text': '\u03b1\U0001f30d'}
    raw = json.dumps(value, ensure_ascii=False).encode('utf-8')
    path = tmp_path / 'large.json'
    path.write_bytes(raw)
    assert load_file(path, max_bytes=len(raw)) == value
    with pytest.raises(ValueError, match='exceeds'):
        load_file(path, max_bytes=len(raw) - 1)


def test_reads_are_bounded_even_when_the_declared_cap_is_large(monkeypatch):
    requests = []
    raw = json.dumps({'value': 'x' * 140000}).encode('utf-8')

    class Observed(io.BytesIO):
        def read(self, size=-1):
            requests.append(size)
            assert 0 < size <= 65536
            return super().read(size)

    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: Observed(raw))
    assert load_file(Path('unused'), max_bytes=sys.maxsize - 1) == json.loads(raw)
    assert len(requests) >= 3


def test_excess_input_stops_at_limit_plus_one(monkeypatch):
    consumed = []

    class Observed(io.BytesIO):
        def read(self, size=-1):
            result = super().read(size)
            consumed.append(len(result))
            return result

    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: Observed(b'x' * 200000))
    with pytest.raises(ValueError, match='exceeds'):
        load_file(Path('unused'), max_bytes=100000)
    assert sum(consumed) == 100001


def test_duplicate_keys_are_still_rejected_across_chunks(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_bytes(b'{"key":1,"padding":"' + b'x' * 70000 + b'","key":2}')
    with pytest.raises(ValueError, match='duplicate'):
        load_file(path)
