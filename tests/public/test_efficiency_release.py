"""Synthetic release-integrity tests; these are not performance observations."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import pytest

SPEC = importlib.util.spec_from_file_location('efficiency_release', Path(__file__).resolve().parents[2]/'scripts/package_efficiency_release.py')
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_source_archive_duplicate_or_symlink_rejected():
    for kind in ('duplicate', 'symlink'):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as archive:
            info = tarfile.TarInfo('a.py'); info.size = 1
            archive.addfile(info, io.BytesIO(b'x'))
            info = tarfile.TarInfo('a.py' if kind == 'duplicate' else 'b.py')
            if kind == 'symlink':
                info.type = tarfile.SYMTYPE; info.linkname = 'a.py'
                archive.addfile(info)
            else:
                info.size = 1; archive.addfile(info, io.BytesIO(b'x'))
        with pytest.raises(ValueError):
            release.archive_members(buffer.getvalue())


@pytest.mark.parametrize('tag', ['failure', 'error', 'skipped'])
def test_junit_failed_or_skipped_contract_rejected(tmp_path, tag):
    path = tmp_path/'junit.xml'
    path.write_text(f'<testsuite><testcase><{tag}/></testcase></testsuite>')
    with pytest.raises(ValueError):
        release.check_tests(path)


def test_junit_empty_rejected_and_valid_counted(tmp_path):
    path = tmp_path/'junit.xml'; path.write_text('<testsuite/>')
    with pytest.raises(ValueError):
        release.check_tests(path)
    path.write_text('<testsuites><testsuite><testcase/><testcase/></testsuite></testsuites>')
    assert release.check_tests(path) == 2


def test_release_never_replaces_existing_output(tmp_path):
    out = tmp_path/'out'; out.mkdir(); (out/'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):
        release.package(tmp_path/'missing', out)
    assert (out/'sentinel').read_text() == 'keep'


def test_release_requires_complete_three_job_matrix(tmp_path):
    inputs = tmp_path/'inputs'; inputs.mkdir()
    with pytest.raises(ValueError, match='exactly'):
        release.package(inputs, tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.fixture
def output_fixture(tmp_path):
    root = tmp_path/'output'; root.mkdir()
    def write(name, obj):
        (root/name).write_text(json.dumps(obj))
    write('config.json', dict(dims=[16,64], depths=[1,4,16], batches=[1,4], seed=91426, rounds=5))
    sources = ['scripts/bench_output_only.py', 'spectra/inference.py', 'deploy/m10_runtime.py',
               'deploy/m10_artifact.py', 'deploy/m10_native.py', 'deploy/m10_dense_extension.cpp']
    write('source_sha256.json', {p: release.sha(release.ROOT/p) for p in sources})
    write('environment.json', {'fixture_only': True})
    rows, cells = [], {}
    for dim in (16,64):
        for depth in (1,4,16):
            (root/f'artifact-{dim}-{depth}.pt').write_bytes(b'synthetic fixture, not loadable model')
            for batch in (1,4):
                case = f'd{dim}:depth{depth}:batch{batch}'
                storage = {'trace': batch*(depth*(128*dim+324)+128), 'final': batch*452}
                cells[case] = dict(mean_ms={'trace': .0001, 'final': .00005}, mean_ratio=.5,
                                   retained_tensor_bytes=storage, retained_ratio=storage['final']/storage['trace'])
                for r in range(5):
                    for mode in (['trace','final'] if (r+batch+depth)%2 else ['final','trace']):
                        rows.append(dict(case=case, mode=mode, round=r, wall_ns=100 if mode=='trace' else 50,
                                         logits_sha256='a'*64, answer_sha256='b'*64, halt_sha256='c'*64,
                                         retained_tensor_bytes=storage[mode],
                                         native_calls=13*depth if mode=='trace' else 12*depth+1,
                                         halt_calls=depth if mode=='trace' else 1))
    (root/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write('summary.json', dict(schema='spectra.output_only_api.v1', measured_calls=120,
                              independent_artifacts=6, cells=cells, exact_final_outputs=True,
                              quality_benchmark=False, untrained_artifacts=True))
    write('SHA256.json', {p.name: release.sha(p) for p in root.iterdir()})
    return root


def rehash(root):
    (root/'SHA256.json').write_text(json.dumps({p.name: release.sha(p) for p in root.iterdir() if p.name!='SHA256.json'}))


def test_output_synthetic_inventory_verified(output_fixture):
    assert release.verify_output(output_fixture)['measured_calls'] == 120


@pytest.mark.parametrize('change', ['missing', 'hash', 'output', 'work', 'time', 'summary', 'source'])
def test_output_corruption_rejected(output_fixture, change):
    root = output_fixture
    rows = [json.loads(r) for r in (root/'rows.jsonl').read_text().splitlines()]
    if change == 'missing': rows.pop()
    elif change == 'output': rows[1]['logits_sha256'] = 'd'*64
    elif change == 'work': rows[0]['native_calls'] += 1
    elif change == 'time': rows[0]['wall_ns'] = False
    if change in ('missing', 'output', 'work', 'time'):
        (root/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif change == 'summary':
        report = release.load(root/'summary.json'); report['quality_benchmark'] = True
        (root/'summary.json').write_text(json.dumps(report))
    elif change == 'source':
        (root/'source_sha256.json').write_text('{}')
    elif change == 'hash':
        (root/'environment.json').write_text('{"changed":true}')
    if change != 'hash': rehash(root)
    with pytest.raises(ValueError):
        release.verify_output(root)
