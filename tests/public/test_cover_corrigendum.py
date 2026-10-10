"""Regression coverage for the post-run transport correction, not new task evidence."""
import json
from pathlib import Path
import pytest
from experiments.cover_search import audit_corrected,study


def test_nonempty_edges_preserve_canonical_json_but_not_native_tuple_type():
    original=[{'edges':[(0,1),(1,2)],'n':3}]
    stored=json.loads(json.dumps(original))
    assert original!=stored
    assert audit_corrected.json_native(original)==stored
    assert study.digest(original)==study.digest(stored)


def test_correction_restores_original_function_after_failure(monkeypatch):
    original=lambda:[{'edges':[(0,1)]}]
    monkeypatch.setattr(study,'make_cases',original)
    with pytest.raises(RuntimeError):
        with audit_corrected.expected_json_inputs():
            assert study.make_cases()==[{'edges':[[0,1]]}]
            raise RuntimeError('deliberate audit failure')
    assert study.make_cases is original


def test_correction_rejects_different_edges_before_reading_receipts(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'make_cases',lambda:[{'edges':[(0,1)],'n':3}])
    (tmp_path/'cases.json').write_text(json.dumps([{'edges':[[0,2]],'n':3}]))
    with pytest.raises(ValueError,match='canonical input inventory differs'):
        audit_corrected.validate(tmp_path)


def test_correction_rejects_nonfinite_json():
    with pytest.raises(ValueError):audit_corrected.json_native({'x':float('nan')})
