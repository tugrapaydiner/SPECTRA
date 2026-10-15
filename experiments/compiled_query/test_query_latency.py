"""Failure and tampering tests for the development comparison's evidence."""
import json
from pathlib import Path

import pytest

from . import query_latency as bench
from .cases import digest
from .exact import build_exact_runtime


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    return build_exact_runtime(tmp_path_factory.mktemp('latency-test-build'))


@pytest.fixture
def run(tmp_path, monkeypatch, library):
    # Tiny, explicitly synthetic test schedule. Never a benchmark observation.
    monkeypatch.setattr(bench, 'COUNTS', (1, 64))
    monkeypatch.setattr(bench, 'REPEATS', 1)
    monkeypatch.setattr(bench, 'FAMILIES', ('random_tree',))
    monkeypatch.setattr(bench, 'REPLICAS', (0,))
    root = tmp_path / 'fixture'
    root.mkdir()
    case = bench.make_case('random_tree', 0, n=8, count=64)
    case_hash = bench.write_once(root/'cases'/(case.name+'.json'), case.payload())
    build = dict(library=str(library), library_sha256=digest(library.read_bytes()),
                 receipt_sha256=digest(library.with_name('build.json').read_bytes()))
    manifest = dict(schema=bench.SCHEMA, classification='test fixture only',
        sources=bench.sources(), builds={'baseline': build, 'candidate': build},
        cases=[dict(name=case.name, family='random_tree', replica=0, sha256=case_hash)],
        arms=bench.ARMS, counts=bench.COUNTS, repeats=bench.REPEATS, warmups=0,
        address_space_bytes=2 << 30, native_bytes=512 << 20,
        candidate_commit='test fixture', baseline_commit='test fixture')
    manifest_hash = bench.put(root/'manifest.json', manifest)
    bench.write_once(root/'manifest.sha256', (manifest_hash+'\n').encode())
    # Worker resource/affinity changes must not affect the pytest process.
    import resource
    monkeypatch.setattr(resource, 'setrlimit', lambda *args: None)
    monkeypatch.setattr(bench.os, 'sched_setaffinity', lambda *args: None)
    folder = root/'runs'/case.name
    folder.mkdir(parents=True)
    bench.worker(root, 0)
    bench.put(folder/'controller.json', dict(status='COMPLETE', returncode=0,
        worker_sha256=digest((folder/'worker.json').read_bytes())))
    return root, folder


def test_records_cannot_be_silently_replaced(tmp_path):
    path = tmp_path/'record.json'
    expected = bench.put(path, {'status': 'ERROR'})
    with pytest.raises(FileExistsError):
        bench.put(path, {'status': 'COMPLETE'})
    assert json.loads(bench.read_bound(path, expected))['status'] == 'ERROR'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        bench.read_bound(path, expected)


def test_audit_replays_a_complete_small_fixture(run):
    root, _ = run
    result = bench.summarize(root)
    assert result['functional_complete']
    assert result['counts']['COMPLETE'] == 12
    assert result['independently_replayed_decisions'] >= 65
    assert set(result['aggregates']) == {'1', '64'}
    assert not result['development_target_pass']  # No long-bank evidence in this fixture.


@pytest.mark.parametrize('damage', ('missing_record', 'changed_time', 'changed_output', 'changed_worker'))
def test_audit_rejects_missing_or_changed_evidence(run, damage):
    root, folder = run
    record = sorted((folder/'records').iterdir())[0]
    if damage == 'missing_record':
        record.unlink()
    elif damage == 'changed_time':
        row = json.loads(record.read_bytes())
        row['complete_ns'] = 1
        record.write_bytes(bench.canonical(row))
    elif damage == 'changed_output':
        output = sorted((folder/'outputs').iterdir())[0]
        output.write_bytes(output.read_bytes() + b'\0')
    else:
        path = folder/'worker.json'
        path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        bench.summarize(root)


def test_crashed_worker_cannot_produce_a_complete_panel_speedup(run):
    root, folder = run
    path = folder/'controller.json'
    path.write_bytes(bench.canonical(dict(status='CRASH', returncode=1)))
    result = bench.summarize(root)
    assert not result['functional_complete']
    assert not result['development_target_pass']
    assert result['aggregates'] == {}
    assert result['failures'][0]['reason'] == 'CRASH'


@pytest.mark.parametrize('damage', ('drop', 'duplicate', 'relabel'))
def test_manifest_cannot_reduce_or_relabel_the_fixed_panel(run, damage):
    root, _ = run
    path = root/'manifest.json'
    manifest = json.loads(path.read_bytes())
    if damage == 'drop':
        manifest['cases'].clear()
    elif damage == 'duplicate':
        manifest['cases'].append(manifest['cases'][0].copy())
    else:
        manifest['cases'][0]['family'] = 'another-family'
    # Even recomputing the hash cannot turn a partial schedule into a full panel.
    data = bench.canonical(manifest)
    path.write_bytes(data)
    (root/'manifest.sha256').write_text(digest(data)+'\n')
    with pytest.raises(ValueError, match='every fixed graph'):
        bench.summarize(root)
