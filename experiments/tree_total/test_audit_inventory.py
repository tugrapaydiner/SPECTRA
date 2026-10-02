"""Omission regressions using inert, complete recorded-evidence inventories."""
import json

import pytest

from .audit import audit
from .test_audit import evidence, write_json


@pytest.mark.parametrize('file,field', [
    ('results/benchmark/LOCK.json', 'files'),
    ('results/benchmark/LOCK.json', 'source'),
    ('results/replay/LOCK.json', 'source_files'),
])
@pytest.mark.parametrize('mode', ['empty', 'omit-one', 'omit-and-corrupt'])
def test_inventory_omissions_rejected(evidence, file, field, mode):
    path = evidence / file
    lock = json.loads(path.read_text())
    inventory = lock[field]
    key = next(iter(inventory))
    if mode == 'empty':
        inventory.clear()
    else:
        del inventory[key]
    if mode == 'omit-and-corrupt':
        if field == 'source_files':
            target = evidence / 'source/experiments/tree_total' / key
        else:
            mapping = json.loads((evidence / ('PATHS.json' if field == 'files' else 'SOURCES.json')).read_text())
            target = evidence / mapping[key]
        target.write_bytes(b'changed bytes hidden by omitted binding')
    write_json(path, lock)
    with pytest.raises(ValueError):
        audit(evidence)


def test_complete_bindings_pass(evidence):
    assert audit(evidence)['status'] == 'PASS'


ARTIFACTS = [f'baseline_sdk/models/{task}/{name}'
             for task in ('letter', 'pendigits', 'satellite', 'optdigits')
             for name in ('model.json', 'model-16.sct', 'model.cbm', 'input.u8', 'indices.i32')]
ARTIFACTS += [f'results/replay/{task}/{name}.sctt'
              for task in ('letter', 'pendigits', 'satellite', 'optdigits')
              for name in ('flat', 'interned')]
ARTIFACTS += [f'residual_sdk/models/{task}/model.scr'
              for task in ('letter', 'pendigits', 'satellite', 'optdigits')]
ARTIFACTS += [f'baseline_evidence/replay/{task}/cpp/export.so'
              for task in ('letter', 'pendigits', 'satellite', 'optdigits')]
ARTIFACTS += ['native/total.so', 'residual/residual.so', 'baseline_evidence/native-register/trees.so',
              'baseline_evidence/official/libcatboostmodel-linux-x86_64-1.2.8.so',
              'baseline_evidence/official/libcatboostmodel-linux-x86_64-1.2.10.so']
SOURCES = ['benchmark.py', 'build.py', 'compiler.py', 'exact.py', 'replay.py', 'runtime.cpp', 'session.py']


@pytest.mark.parametrize('name', ARTIFACTS)
def test_each_required_artifact_rejected_even_when_mapping_also_removed(evidence, name):
    path = evidence / 'results/benchmark/LOCK.json'
    lock = json.loads(path.read_text())
    del lock['files']['/synthetic/' + name]
    write_json(path, lock)
    path = evidence / 'PATHS.json'
    mapping = json.loads(path.read_text())
    del mapping['/synthetic/' + name]
    write_json(path, mapping)
    with pytest.raises(ValueError):
        audit(evidence)


@pytest.mark.parametrize('name', ['experiments/tree_total/' + name for name in SOURCES] +
                         ['experiments/tree_residual/runtime.cpp', 'experiments/certified_trees/runtime.cpp'])
def test_each_required_timed_source_rejected(evidence, name):
    path = evidence / 'results/benchmark/LOCK.json'
    lock = json.loads(path.read_text())
    del lock['source'][name]
    write_json(path, lock)
    with pytest.raises(ValueError):
        audit(evidence)


@pytest.mark.parametrize('name', SOURCES)
def test_each_required_replay_source_rejected(evidence, name):
    path = evidence / 'results/replay/LOCK.json'
    lock = json.loads(path.read_text())
    del lock['source_files'][name]
    write_json(path, lock)
    with pytest.raises(ValueError):
        audit(evidence)


def test_ambiguous_artifact_role_rejected(evidence):
    path = evidence / 'results/benchmark/LOCK.json'
    lock = json.loads(path.read_text())
    original = '/synthetic/native/total.so'
    alias = '/ambiguous/total.so'
    lock['files'][alias] = lock['files'][original]
    write_json(path, lock)
    path = evidence / 'PATHS.json'
    mapping = json.loads(path.read_text())
    mapping[alias] = mapping[original]
    write_json(path, mapping)
    with pytest.raises(ValueError):
        audit(evidence)


def test_additional_bound_artifact_and_source_allowed(evidence):
    from .test_audit import digest
    raw = b'additional inert record'
    (evidence / 'extra.txt').write_bytes(raw)
    path = evidence / 'results/benchmark/LOCK.json'
    lock = json.loads(path.read_text())
    lock['files']['/synthetic/extra.txt'] = digest(raw)
    lock['source']['experiments/tree_total/extra.txt'] = digest(raw)
    write_json(path, lock)
    for file, key in [('PATHS.json', '/synthetic/extra.txt'),
                      ('SOURCES.json', 'experiments/tree_total/extra.txt')]:
        path = evidence / file
        mapping = json.loads(path.read_text())
        mapping[key] = 'extra.txt'
        write_json(path, mapping)
    assert audit(evidence)['status'] == 'PASS'

