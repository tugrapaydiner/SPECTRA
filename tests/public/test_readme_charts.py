"""Documentation figures must be source-bound, reproducible and complete."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from scripts import render_readme_charts as charts


def test_committed_figures_are_current():
    assert charts.main(['--check']) == 0


def test_generator_is_deterministic():
    assert charts.outputs() == charts.outputs()


def test_every_published_cell_and_missing_comparator_are_preserved():
    data = charts.load_data()
    assert len(data['indexed']['cells']) == 6
    assert sum(r['long_ratio'] > 1 for r in data['indexed']['cells']) == 2
    rows = data['native']['cells']
    assert len(rows) == 7
    assert [r['model'] for r in rows if r['generated_us'] is None] == ['HAR']
    assert sum(r['generated_us'] is not None and r['generated_us'] < r['spectra_default_us'] for r in rows) == 5
    assert data['native']['har_different_linear_model_us'] < rows[-1]['spectra_default_us']
    assert json.loads(charts.outputs()['chart-data.json']) == data


def test_svg_has_accessible_metadata_and_no_external_content():
    for name, content in charts.outputs().items():
        if not name.endswith('.svg'):
            continue
        root = ET.fromstring(content)
        ns = '{http://www.w3.org/2000/svg}'
        assert root.attrib['role'] == 'img'
        assert root.find(ns+'title').text and root.find(ns+'desc').text
        assert root.attrib['viewBox'].startswith('0 0 1000 ')
        assert not any(node.tag in {ns+'script', ns+'image', ns+'foreignObject'} for node in root.iter())
        assert not any(key.endswith('href') for node in root.iter() for key in node.attrib)


@pytest.mark.parametrize('mode', ['missing', 'modified'])
def test_check_refuses_stale_assets_without_rewriting(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(charts, 'OUT', tmp_path)
    for name, value in charts.outputs().items():
        (tmp_path/name).write_text(value, encoding='utf-8')
    target = tmp_path/'indexed-search.svg'
    if mode == 'missing':
        target.unlink()
    else:
        target.write_text('not the generated figure', encoding='utf-8')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(SystemExit) as failure:
        charts.main(['--check'])
    assert failure.value.code == 1
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_changed_historical_source_requires_explicit_review(tmp_path, monkeypatch):
    monkeypatch.setattr(charts, 'ROOT', tmp_path)
    for name in charts.SOURCES:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('altered summary', encoding='utf-8')
    with pytest.raises(ValueError, match='historical chart source changed'):
        charts.load_data()
