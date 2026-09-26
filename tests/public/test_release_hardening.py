"""Public-input and packaging regression tests; no research dependencies."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import warnings
import zipfile

import pytest

from spectra import __version__
from spectra._json import load_file
from spectra.cli import main
from spectra.evidence import capture

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


wheel_check = module('check_current_installation')
readiness = module('check_release_readiness')


@pytest.mark.parametrize('raw', [
    b'{"witness":[false],"witness":[true]}',
    b'{"metadata":{"key":1,"key":2},"witness":[true]}',
    b'{"witness":[true],"value":NaN}',
    b'{"witness":[true],"value":Infinity}',
    b'{"witness":[true],"value":-Infinity}',
    b'{"witness":[true],"value":1e999}',
    b'{"witness":[true],"value":-1e999}',
    b'{"witness":[true],"value":"\xff"}',
    b'[' * 10000 + b']' * 10000,
    b'{"witness":[true]',
])
def test_bad_witness_json_is_input_error(tmp_path, capsys, raw):
    problem = tmp_path / 'problem.cnf'
    problem.write_text('p cnf 1 1\n1 0\n')
    record = tmp_path / 'answer.json'
    record.write_bytes(raw)
    assert main(['cnf', 'check', str(problem), str(record)]) == 2
    output = capsys.readouterr()
    assert output.out == ''
    assert output.err.startswith('spectra:')


@pytest.mark.parametrize('limit', [0, -1, True, 1.5, '10'])
def test_reader_requires_positive_integer_limit(tmp_path, limit):
    source = tmp_path / 'data.json'; source.write_bytes(b'{}')
    with pytest.raises(ValueError, match='positive integer'):
        load_file(source, max_bytes=limit)


def test_reader_byte_boundary_and_finite_numbers(tmp_path):
    source = tmp_path / 'data.json'
    raw = '{"label":"é","value":1.25,"flag":true}'.encode('utf-8')
    source.write_bytes(raw)
    assert load_file(source, max_bytes=len(raw)) == {'label': 'é', 'value': 1.25, 'flag': True}
    with pytest.raises(ValueError, match='exceeds'):
        load_file(source, max_bytes=len(raw) - 1)


@pytest.mark.parametrize('command', ['witness', 'evidence'])
def test_cli_byte_limit_and_normal_records(tmp_path, capsys, command):
    data = tmp_path / 'data'; data.write_bytes(b'kept')
    source = tmp_path / 'input.cnf'; source.write_text('p cnf 1 1\n1 0\n')
    record = tmp_path / 'record.json'
    value = {'witness': [True]} if command == 'witness' else capture(tmp_path, ['data'])
    raw = json.dumps(value).encode(); record.write_bytes(raw)
    argv = ['cnf', 'check', str(source), str(record)] if command == 'witness' else ['evidence', str(tmp_path), str(record)]
    for limit in (0, -1, len(raw) - 1):
        assert main(argv + ['--max-json-bytes', str(limit)]) == 2
        assert capsys.readouterr().out == ''
    assert main(argv + ['--max-json-bytes', str(len(raw))]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed.get('valid', parsed.get('verified')) is True


@pytest.mark.parametrize('where', ['root', 'artifact'])
def test_duplicate_evidence_fields_rejected(tmp_path, capsys, where):
    (tmp_path / 'data').write_bytes(b'kept')
    manifest = capture(tmp_path, ['data'])
    raw = json.dumps(manifest)
    if where == 'root':
        raw = '{"format":"wrong",' + raw[1:]
    else:
        raw = raw.replace('"bytes": 4', '"bytes": 999,"bytes": 4')
    path = tmp_path / 'manifest.json'; path.write_text(raw)
    assert main(['evidence', str(tmp_path), str(path)]) == 2
    assert 'duplicate JSON key' in capsys.readouterr().err


def test_strict_reader_stays_dependency_free():
    subprocess.run([sys.executable, '-S', '-c',
                    'import spectra.cli, spectra._json, sys; '
                    'assert not ({"torch", "numpy"} & sys.modules.keys())'], check=True, cwd=ROOT)


def make_wheel(directory, *, metadata=None, missing=None, extra=None):
    directory.mkdir(exist_ok=True)
    wheel = directory / f'spectra-{__version__}-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr(f'spectra-{__version__}.dist-info/METADATA', metadata or
                         f'Metadata-Version: 2.4\nName: spectra\nVersion: {__version__}\n')
        for name in sorted(wheel_check.REQUIRED_SOURCES):
            if name != missing:
                archive.writestr(name, b'// synthetic test fixture, not executable')
        if extra:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                archive.writestr(*extra)
    return wheel


def test_current_wheel_selected(tmp_path):
    path = make_wheel(tmp_path)
    assert wheel_check.select_wheel(tmp_path) == path


@pytest.mark.parametrize('metadata', [
    f'Name: other\nVersion: {__version__}\n',
    'Name: spectra\nVersion: 9.9.9\n',
    f'Name: spectra\nName: spectra\nVersion: {__version__}\n',
    f'Name: spectra\nVersion: {__version__}\nVersion: {__version__}\n',
    'Name: spectra\n',
])
def test_wheel_metadata_must_match_source(tmp_path, metadata):
    make_wheel(tmp_path, metadata=metadata)
    with pytest.raises(ValueError, match='metadata'):
        wheel_check.select_wheel(tmp_path)


@pytest.mark.parametrize('case', ['empty', 'multiple', 'stale', 'not_directory', 'missing_directory', 'bad_zip'])
def test_bad_build_directories_rejected(tmp_path, case):
    directory = tmp_path
    if case not in ('empty', 'missing_directory'):
        path = make_wheel(directory)
        if case == 'multiple':
            (directory / 'spectra-9.9.9-py3-none-any.whl').write_bytes(path.read_bytes())
        elif case == 'stale':
            path.rename(directory / 'spectra-9.9.9-py3-none-any.whl')
        elif case == 'not_directory':
            directory = path
        elif case == 'bad_zip':
            path.write_bytes(b'not a zip')
    elif case == 'missing_directory':
        directory = directory / 'missing'
    with pytest.raises((ValueError, OSError, zipfile.BadZipFile)):
        wheel_check.select_wheel(directory)


@pytest.mark.parametrize('case', ['source', 'duplicate', 'extra_distribution'])
def test_wheel_inventory_checked(tmp_path, case):
    kwargs = {'missing': next(iter(wheel_check.REQUIRED_SOURCES))} if case == 'source' else {
        'extra': ('spectra-duplicate.dist-info/METADATA' if case == 'extra_distribution' else
                  next(iter(wheel_check.REQUIRED_SOURCES)), b'duplicate')}
    make_wheel(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        wheel_check.select_wheel(tmp_path)


@pytest.fixture
def guide_tree(tmp_path):
    (tmp_path / 'spectra').mkdir()
    (tmp_path / 'spectra/__init__.py').write_text('__version__ = "1.2.3"\n')
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "spectra"\nversion = "1.2.3"\n')
    for guide in readiness.GUIDES:
        path = tmp_path / guide; path.parent.mkdir(exist_ok=True)
        path.write_text('# Example heading\n\n## Second heading\n')
    workflows = tmp_path / '.github/workflows'; workflows.mkdir(parents=True)
    for name in ('public-package.yml', 'indexed-efficiency.yml'):
        (workflows / name).write_text('python scripts/check_current_installation.py\n')
    return tmp_path


def test_metadata_and_links_validated(guide_tree):
    (guide_tree / 'README.md').write_text('# Example\n[api](docs/API.md#second-heading)\n'
                                       '[web](https://example.invalid/)\n')
    report = readiness.check(guide_tree)
    assert report['status'] == 'PASS' and report['version'] == '1.2.3'
    assert report['local_links_checked'] == 1 and report['external_links_not_fetched'] == 1


@pytest.mark.parametrize('case', ['missing', 'anchor', 'escape', 'version', 'duplicate_version',
                                  'missing_version', 'hardcoded_wheel', 'missing_check'])
def test_bad_release_metadata_fails(guide_tree, case):
    if case in ('missing', 'anchor', 'escape'):
        targets = {'missing': 'missing.md', 'anchor': 'docs/API.md#not-a-heading', 'escape': '../'}
        (guide_tree / 'README.md').write_text(f'[bad]({targets[case]})')
    elif case == 'version':
        (guide_tree / 'spectra/__init__.py').write_text('__version__ = "9.9.9"\n')
    elif case == 'duplicate_version':
        with (guide_tree / 'pyproject.toml').open('a') as stream:
            stream.write('version = "1.2.3"\n')
    elif case == 'missing_version':
        (guide_tree / 'spectra/__init__.py').write_text('')
    else:
        path = guide_tree / '.github/workflows/public-package.yml'
        path.write_text('check_current_installation.py spectra-1.2.3-py3-none-any.whl' if case == 'hardcoded_wheel' else '')
    with pytest.raises(ValueError):
        readiness.check(guide_tree)


def test_duplicate_heading_suffixes_and_fences(guide_tree):
    (guide_tree / 'docs/API.md').write_text('# Heading\n# Heading\n```python\n# not a heading\n```\n')
    (guide_tree / 'README.md').write_text('[api](docs/API.md#heading-1)\n````\n')
    assert readiness.check(guide_tree)['local_links_checked'] == 1
    assert 'not-a-heading' not in readiness.anchors(guide_tree / 'docs/API.md')


def test_real_checkout_release_metadata():
    assert readiness.check(ROOT)['version'] == __version__


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'dir\\escape'])
def test_unsafe_wheel_paths_rejected(tmp_path, name):
    make_wheel(tmp_path, extra=(name, b'x'))
    with pytest.raises(ValueError, match='unsafe'):
        wheel_check.select_wheel(tmp_path)


def test_symlinked_wheel_member_rejected(tmp_path):
    path = make_wheel(tmp_path)
    with zipfile.ZipFile(path, 'a') as archive:
        info = zipfile.ZipInfo('link'); info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, 'somewhere')
    with pytest.raises(ValueError, match='linked'):
        wheel_check.select_wheel(tmp_path)


@pytest.mark.parametrize('change', [None, 'changed', 'missing', 'unexpected', 'duplicate'])
def test_every_shipped_source_is_bound(tmp_path, change):
    source = tmp_path / 'source'; source.mkdir()
    members = {}
    for name in wheel_check.SOURCE_PACKAGES:
        (source / name).mkdir()
        raw = f'"""{name} fixture"""\n'.encode()
        (source / name / '__init__.py').write_bytes(raw)
        members[f'{name}/__init__.py'] = raw
    members['spectra/cli.py'] = b'# CLI that older three-file checks missed\n'
    (source / 'spectra/cli.py').write_bytes(members['spectra/cli.py'])
    if change == 'changed':
        members['spectra/cli.py'] = b'# mismatched CLI\n'
    elif change == 'missing':
        members.pop('spectra/cli.py')
    elif change == 'unexpected':
        members['unreviewed.py'] = b'# extra code\n'
    wheel = tmp_path / 'fixture.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr('spectra-1.dist-info/METADATA', b'test fixture')
        if change == 'duplicate':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                archive.writestr('spectra/cli.py', b'duplicate')
    if change:
        with pytest.raises(ValueError):
            wheel_check.verify_sources(wheel, source)
    else:
        assert wheel_check.verify_sources(wheel, source) == {
            'matched_source_files': 8, 'source_identity': 'PASS'}
