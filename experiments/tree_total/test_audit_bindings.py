"""Recorded fixed-corpus identities must agree across audit receipts."""
import json

import pytest

from .audit import audit
from .test_audit import digest, evidence, write_json


@pytest.mark.parametrize('task', ['letter', 'pendigits', 'satellite', 'optdigits'])
@pytest.mark.parametrize('field', ['source', 'inputs', 'expected'])
def test_replay_identity_mismatch_rejected(evidence, task, field):
    path = evidence / 'results/replay/LOCK.json'
    lock = json.loads(path.read_text())
    lock['models'][task][field] = '0' * 64
    write_json(path, lock)
    with pytest.raises(ValueError):
        audit(evidence)


@pytest.mark.parametrize('fault', ['missing-models', 'empty-models', 'missing-task', 'missing-field'])
def test_incomplete_replay_identities_rejected(evidence, fault):
    path = evidence / 'results/replay/LOCK.json'
    lock = json.loads(path.read_text())
    if fault == 'missing-models':
        del lock['models']
    elif fault == 'empty-models':
        lock['models'] = {}
    elif fault == 'missing-task':
        del lock['models']['letter']
    else:
        del lock['models']['letter']['inputs']
    write_json(path, lock)
    with pytest.raises(ValueError):
        audit(evidence)


@pytest.mark.parametrize('index', range(8))
@pytest.mark.parametrize('fault', ['wrong-source', 'missing-source'])
def test_each_proof_source_binding_rejected(evidence, index, fault):
    path = evidence / 'results/independent.json'
    proof = json.loads(path.read_text())
    if fault == 'wrong-source':
        proof['files'][index]['source_sha256'] = '0' * 64
    else:
        del proof['files'][index]['source_sha256']
    write_json(path, proof)
    with pytest.raises(ValueError):
        audit(evidence)


def test_timed_model_cannot_differ_from_reconstructed_model(evidence):
    raw = b'different inert timed model'
    (evidence / 'different.sctt').write_bytes(raw)
    key = '/synthetic/results/replay/letter/flat.sctt'
    path = evidence / 'results/benchmark/LOCK.json'
    lock = json.loads(path.read_text())
    lock['files'][key] = digest(raw)
    write_json(path, lock)
    path = evidence / 'PATHS.json'
    mapping = json.loads(path.read_text())
    mapping[key] = 'different.sctt'
    write_json(path, mapping)
    with pytest.raises(ValueError):
        audit(evidence)


def test_identical_models_in_separate_relocated_files_pass(evidence):
    key = '/synthetic/results/replay/letter/flat.sctt'
    path = evidence / 'PATHS.json'
    mapping = json.loads(path.read_text())
    (evidence / 'separate.sctt').write_bytes((evidence / mapping[key]).read_bytes())
    mapping[key] = 'separate.sctt'
    write_json(path, mapping)
    assert audit(evidence)['status'] == 'PASS'


def test_distinct_replay_library_does_not_imply_changed_models(evidence):
    path = evidence / 'results/replay/LOCK.json'
    lock = json.loads(path.read_text())
    lock['library_sha256'] = digest(b'different valid native build')
    write_json(path, lock)
    # The recorded-data auditor does not establish this library's provenance.
    # It must not substitute equality with the timing build for that check.
    assert audit(evidence)['status'] == 'PASS'
