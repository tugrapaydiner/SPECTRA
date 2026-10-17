"""Repository-size and workflow hygiene contracts use disposable fixtures."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / 'scripts/check_active_tree.py'
    spec = importlib.util.spec_from_file_location('check_active_tree', path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


active = _module()


def fixture(tmp_path: Path) -> Path:
    for name in active.ACTIVE_WORKFLOWS:
        path = tmp_path / '.github/workflows' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('name: fixture\non:\n  workflow_dispatch:\n')
    (tmp_path / 'README.md').write_text('archive/research-history-20261016\n')
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results/README.md').write_text('compact results\n')
    (tmp_path / 'results/sat_workload').mkdir()
    (tmp_path / 'results/sat_workload/pilot.zip').write_bytes(b'small fixture')
    return tmp_path


def test_real_active_tree_passes():
    report = active.check(ROOT)
    assert report['status'] == 'PASS'
    assert report['active_workflows'] == sorted(active.ACTIVE_WORKFLOWS)
    assert report['tracked_bytes'] <= report['limits']['tracked_bytes']


@pytest.mark.parametrize('mutation', ['archive', 'workflow', 'branch', 'png', 'delivery', 'large'])
def test_active_tree_rejects_bloat_and_dead_automation(tmp_path, monkeypatch, mutation):
    root = fixture(tmp_path)
    monkeypatch.setattr(active, '_tracked', lambda _root: sorted(p for p in root.rglob('*') if p.is_file()))
    if mutation == 'archive':
        (root / 'results/old.tar.gz').write_bytes(b'x')
    elif mutation == 'workflow':
        (root / '.github/workflows/old.yml').write_text('name: old\n')
    elif mutation == 'branch':
        path = root / '.github/workflows/public-package.yml'
        path.write_text('research/100-blitz-20261016\n')
    elif mutation == 'png':
        (root / 'assets').mkdir()
        (root / 'assets/01_old.png').write_bytes(b'x')
    elif mutation == 'delivery':
        (root / 'delivery').mkdir()
        (root / 'delivery/payload').write_bytes(b'x')
    else:
        (root / 'oversized.bin').write_bytes(b'x' * (active.MAX_SINGLE_FILE_BYTES + 1))
    with pytest.raises(ValueError):
        active.check(root)
