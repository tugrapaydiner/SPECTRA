"""Recorded scores must substantiate their declared dimensions and class indices."""
import json
import math
import struct

import pytest

from .audit import audit
from .test_audit import digest, evidence, write_json


def replace_stress(root, kind, suffix, raw):
    folder = root / 'results/replay/letter'
    (folder / (kind + '-' + suffix)).write_bytes(raw)
    report = json.loads((folder / 'result.json').read_text())
    field = 'source_scores_sha256' if suffix == 'scores.f64' else 'indices_sha256'
    report['stress'][kind][field] = digest(raw)
    write_json(folder / 'result.json', report)


@pytest.mark.parametrize('kind', ['retained', 'uniform'])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_recorded_scores_rejected(evidence, kind, value):
    folder = evidence / 'results/replay/letter'
    path = folder / ('reference_scores.f64' if kind == 'retained' else kind + '-scores.f64')
    raw = bytearray(path.read_bytes())
    # Keep the old Python max() answer unchanged to isolate the missing finite check.
    struct.pack_into('<d', raw, 8 if value == -math.inf else 0, value)
    if kind == 'retained':
        path.write_bytes(raw)
    else:
        replace_stress(evidence, kind, 'scores.f64', raw)
    with pytest.raises(ValueError):
        audit(evidence)


@pytest.mark.parametrize('suffix,width', [('scores.f64', 8), ('indices.i32', 4)])
@pytest.mark.parametrize('extra', [False, True])
def test_stress_payload_inventory_must_match_rows(evidence, suffix, width, extra):
    path = evidence / 'results/replay/letter' / ('uniform-' + suffix)
    raw = path.read_bytes()
    raw = raw + b'\0' * width if extra else raw[:-width]
    replace_stress(evidence, 'uniform', suffix, raw)
    with pytest.raises(ValueError):
        audit(evidence)


@pytest.mark.parametrize('kind', ['uniform', 'boundary'])
def test_rehashed_stress_indices_must_follow_recorded_scores(evidence, kind):
    path = evidence / 'results/replay/letter' / (kind + '-indices.i32')
    raw = bytearray(path.read_bytes())
    struct.pack_into('<i', raw, 0, 1)
    replace_stress(evidence, kind, 'indices.i32', raw)
    with pytest.raises(ValueError, match='score'):
        audit(evidence)


@pytest.mark.parametrize('kind', ['retained', 'uniform'])
@pytest.mark.parametrize('bad_count', [-1, True])
def test_coverage_counts_must_be_nonnegative_integers(evidence, kind, bad_count):
    path = evidence / 'results/replay/letter/result.json'
    report = json.loads(path.read_text())
    if kind == 'retained':
        count, work = 2, report['models']['interned']['work']['total']
    else:
        count, work = 2048, report['stress'][kind]['work']
    work.update(coarse_certified=bad_count, exact_completed=count - bad_count)
    write_json(path, report)
    with pytest.raises(ValueError):
        audit(evidence)


def test_stress_coverage_must_account_for_every_row(evidence):
    path = evidence / 'results/replay/letter/result.json'
    report = json.loads(path.read_text())
    report['stress']['uniform']['work']['exact_completed'] = 1
    write_json(path, report)
    with pytest.raises(ValueError):
        audit(evidence)


def test_stress_input_must_be_inside_declared_domain(evidence):
    path = evidence / 'results/replay/letter/uniform.u8'
    raw = bytearray(path.read_bytes())
    raw[0] = 2
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        audit(evidence)


@pytest.mark.parametrize('field,value', [('stress_rows_per_kind', 1024), ('seed_rule', 'changed')])
def test_fixed_stress_parameters_cannot_change(evidence, field, value):
    path = evidence / 'results/replay/LOCK.json'
    lock = json.loads(path.read_text())
    lock[field] = value
    write_json(path, lock)
    with pytest.raises(ValueError):
        audit(evidence)


def test_recorded_score_ties_keep_first_class(evidence):
    path = evidence / 'results/replay/letter/uniform-scores.f64'
    raw = bytearray(path.read_bytes())
    struct.pack_into('<dd', raw, 0, -0., 0.)
    replace_stress(evidence, 'uniform', 'scores.f64', raw)
    assert audit(evidence)['status'] == 'PASS'
