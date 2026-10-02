"""A kit must cover its executable/model inventory before replay can succeed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from . import kit


def inventory():
    paths = {'LICENSE', 'ATTRIBUTION.md', 'README.md', 'selftest.py', 'run.py'}
    paths.update('spectra/' + name for name in ('__init__.py', 'svm_lifetime.py', 'svm_stream.py'))
    paths.update('experiments/certified_trees/' + name for name in
                 ('runtime.cpp', 'packed.py', 'reference/certificate_oracle.py'))
    paths.update('experiments/tree_total/' + name for name in
                 ('compiler.py', 'exact.py', 'runtime.cpp', 'session.py', 'build.py',
                  'deployment.py', 'kit.py', 'CONTRACT.md', 'README.md'))
    paths.update(f'models/{task}/{name}' for task in kit.TASKS for name in
                 ('model.json', 'model.sctt', 'input.u8', 'input.jsonl', 'indices.i32'))
    paths.update(f'native/{target}/{name}' for target in ('portable', 'avx2')
                 for name in ('total.so', 'build.json'))
    return paths


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'kit'
    root.mkdir()
    files = {}
    # Inert bytes only: these inventory tests never load a model or executable.
    for name in sorted(inventory()):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (b'\x00\x01' if name.endswith('/input.u8') else
               struct.pack('<ii', 0, 1) if name.endswith('/indices.i32') else b'x')
        path.write_bytes(raw)
        files[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    manifest = {'format': 'spectra.total-tree.kit.v1', 'files': files,
                'models': {task: {'rows': 2, 'features': 1, 'maximum': 1,
                                  'classes': [0, 1]} for task in kit.TASKS}}
    write_manifest(root, manifest)
    return root, manifest


def write_manifest(root, manifest):
    (root / 'MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')


def test_complete_inventory_passes(bundle):
    root, manifest = bundle
    assert kit.verify(root) == manifest


@pytest.mark.parametrize('name', ['native/portable/total.so', 'native/avx2/build.json',
                                 'models/letter/model.sctt', 'models/pendigits/model.json',
                                 'models/satellite/indices.i32', 'models/optdigits/input.u8',
                                 'run.py', 'experiments/tree_total/session.py'])
def test_required_asset_cannot_be_dropped_from_inventory(bundle, name):
    root, manifest = bundle
    del manifest['files'][name]
    (root / name).unlink()
    write_manifest(root, manifest)
    with pytest.raises(ValueError):
        kit.verify(root)


@pytest.mark.parametrize('field', ['models', 'files'])
def test_empty_inventory_rejected(bundle, field):
    root, manifest = bundle
    manifest[field] = {}
    write_manifest(root, manifest)
    with pytest.raises(ValueError):
        kit.verify(root)


def test_unlisted_member_rejected(bundle):
    root, _ = bundle
    (root / 'unlisted.py').write_text('raise RuntimeError("unlisted source")')
    with pytest.raises(ValueError):
        kit.verify(root)


@pytest.mark.parametrize('field, value', [('rows', True), ('rows', 0), ('rows', 65537),
                                       ('features', 0), ('maximum', 256),
                                       ('classes', [0, 0]), ('classes', [False, True])])
def test_model_metadata_checked_before_replay(bundle, field, value):
    root, manifest = bundle
    manifest['models']['letter'][field] = value
    write_manifest(root, manifest)
    with pytest.raises(ValueError):
        kit.verify(root)


@pytest.mark.parametrize('field, value', [('bytes', True), ('bytes', -1),
                                       ('sha256', 'X' * 64), ('sha256', '0' * 63)])
def test_member_metadata_rejected(bundle, field, value):
    root, manifest = bundle
    manifest['files']['README.md'][field] = value
    write_manifest(root, manifest)
    with pytest.raises(ValueError):
        kit.verify(root)


def test_duplicate_manifest_fields_rejected(bundle):
    root, _ = bundle
    path = root / 'MANIFEST.json'
    raw = path.read_text()
    path.write_text('{"format":"wrong",' + raw[1:])
    with pytest.raises(ValueError):
        kit.verify(root)


def test_manifest_is_bounded_before_decoding(bundle):
    root, _ = bundle
    path = root / 'MANIFEST.json'
    path.write_bytes(path.read_bytes() + b' ' * (1024 * 1024))
    with pytest.raises(ValueError):
        kit.verify(root)


def test_manifest_symlink_rejected(bundle, tmp_path):
    root, _ = bundle
    path = root / 'MANIFEST.json'
    outside = tmp_path / 'external.json'
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError):
        kit.verify(root)


def test_empty_kit_cannot_publish_success(tmp_path):
    root = tmp_path / 'empty'
    root.mkdir()
    write_manifest(root, {'format': 'spectra.total-tree.kit.v1', 'models': {}, 'files': {}})
    destination = tmp_path / 'report.json'
    source = Path(__file__).resolve().parents[2]
    code = ('import sys; from pathlib import Path; sys.path.insert(0,sys.argv.pop(1)); '
            'from experiments.tree_total.kit import selftest_main; '
            'root=Path(sys.argv.pop(1)); selftest_main(root)')
    run = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(source),
                          str(root), '--out', str(destination)], capture_output=True, text=True)
    assert run.returncode != 0, run.stdout
    assert not destination.exists()


@pytest.mark.parametrize('raw', [b'[]', b'null', b'{"format":NaN}',
                               b'{"extra":1e9999}', b'\xff', b'[' * 2000])
def test_malformed_manifest_rejected(bundle, raw):
    root, _ = bundle
    (root / 'MANIFEST.json').write_bytes(raw)
    with pytest.raises(ValueError):
        kit.verify(root)


def test_member_corruption_rejected(bundle):
    root, _ = bundle
    (root / 'native/portable/total.so').write_bytes(b'y')
    with pytest.raises(ValueError, match='member differs'):
        kit.verify(root)


def test_symlink_directory_rejected(bundle, tmp_path):
    root, _ = bundle
    member = root / 'models/letter'
    outside = tmp_path / 'external-model'
    member.rename(outside)
    member.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        kit.verify(root)


def test_selftest_refuses_output_inside_immutable_bundle(bundle, monkeypatch):
    root, _ = bundle
    destination = root / 'result.json'
    monkeypatch.setattr(sys, 'argv', ['selftest.py', '--out', str(destination)])
    with pytest.raises(SystemExit):
        kit.selftest_main(root)
    assert not destination.exists()


def test_assembled_synthetic_kit_runs_without_frameworks(tmp_path):
    # These four tiny fixtures test packaging and replay, not retained-model quality.
    from ..certified_trees.test_native import source
    from .build import build
    from .compiler import compile_bytes

    checkout = Path(__file__).resolve().parents[2]
    parent, models, native, root = (tmp_path / name for name in ('parent', 'models', 'native', 'kit'))
    parent.mkdir()
    (parent / 'ATTRIBUTION.md').write_text('Synthetic packaging test; no retained dataset.\n')
    raw = source([[-1., 1.]])
    compiled = compile_bytes(raw, 1)
    metadata = {}
    for task in kit.TASKS:
        folder = parent / 'models' / task
        folder.mkdir(parents=True)
        (folder / 'model.json').write_bytes(raw)
        (folder / 'input.u8').write_bytes(bytes([0, 1, 0, 1]))
        (folder / 'input.jsonl').write_text('[0]\n[1]\n[0]\n[1]\n')
        (folder / 'indices.i32').write_bytes(struct.pack('<iiii', 0, 1, 0, 1))
        compiled_folder = models / task
        compiled_folder.mkdir(parents=True)
        (compiled_folder / 'interned.sctt').write_bytes(compiled)
        metadata[task] = {'rows': 4, 'features': 1, 'maximum': 1, 'classes': [0, 1]}
    (parent / 'SDK_MANIFEST.json').write_text(json.dumps({'models': metadata}))
    for target in ('portable', 'avx2'):
        build(native / ('native-' + target), target=target)
    manifest = kit.assemble(checkout, models, parent, native, root)
    assert set(manifest['files']) == inventory()
    targets = ['portable']
    if 'avx2' in Path('/proc/cpuinfo').read_text().lower():
        targets.append('avx2')
    for target in targets:
        report = tmp_path / (target + '-report.json')
        subprocess.run([sys.executable, '-I', '-S', str(root / 'selftest.py'),
                        '--target', target, '--out', str(report)], cwd=tmp_path,
                       check=True, capture_output=True, text=True)
        result = json.loads(report.read_text())
        assert result['status'] == 'PASS'
        assert result['rows'] == 16 and result['repeated_predictions'] == 48
        assert len(result['models']) == 4 and not result['external_engine_loaded']
    output = tmp_path / 'predictions.jsonl'
    subprocess.run([sys.executable, '-I', '-S', str(root / 'run.py'),
                    '--model', 'letter', '--input', str(root / 'models/letter/input.jsonl'),
                    '--output', str(output)], cwd=tmp_path, check=True,
                   capture_output=True, text=True)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row['class_index'] for row in records if row['type'] == 'prediction'] == [0, 1, 0, 1]
    assert records[-1]['type'] == 'complete' and records[-1]['rows'] == 4
    assert kit.verify(root) == manifest
