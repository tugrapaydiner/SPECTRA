"""Fail-closed evidence, pairing and source identity tests."""
import copy
import gzip
import json

import pytest

from eval import cnf_cache_benchmark as bench


@pytest.fixture
def tiny(monkeypatch):
    for key, value in {'sizes': [4], 'per_cell': 2, 'moves': 4,
                       'bootstrap_repeats': 40}.items():
        monkeypatch.setitem(bench.CONFIG, key, value)


@pytest.fixture
def evidence(tiny, tmp_path):
    out = tmp_path / 'evidence'
    bench.run(out)
    cases = json.loads((out / 'cases.json').read_text())
    with gzip.open(out / 'rows.jsonl.gz', 'rt') as f:
        rows = [json.loads(line) for line in f]
    return out, cases, rows


def test_full_round_trip_and_replay(evidence):
    out, _, rows = evidence
    receipt = bench.verify(out, replay=True)
    assert receipt['integrity'] == 'PASS'
    assert receipt['original_formula_answer_checks'] == len(rows)
    assert receipt['replayed_paths'] == len(rows) // bench.CONFIG['rounds']


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'order', 'extra_key',
    'false_witness', 'false_status', 'false_residual', 'negative_time', 'bool_time',
    'nan_time', 'bad_hash', 'path_mismatch', 'work_change', 'missing_work', 'bool_work',
    'changed_input', 'changed_seed', 'too_many_moves', 'wrong_query_cost'])
def test_semantic_tampering_is_rejected(evidence, mutation):
    _, cases, rows = evidence
    cases, rows = copy.deepcopy(cases), copy.deepcopy(rows)
    if mutation == 'missing': rows.pop()
    elif mutation == 'duplicate': rows[-1] = rows[0]
    elif mutation == 'order': rows[0], rows[1] = rows[1], rows[0]
    elif mutation == 'extra_key': rows[0]['unrecognized'] = True
    elif mutation == 'false_witness': rows[0]['witness'][0] = 1
    elif mutation == 'false_status': rows[0]['status'] = 'UNSAT'
    elif mutation == 'false_residual': rows[0]['unsatisfied'] = [-1]
    elif mutation == 'negative_time': rows[0]['elapsed_ns'] = -1
    elif mutation == 'bool_time': rows[0]['elapsed_ns'] = True
    elif mutation == 'nan_time': rows[0]['elapsed_ns'] = float('nan')
    elif mutation == 'bad_hash': rows[0]['path_sha256'] = 'g' * 64
    elif mutation == 'path_mismatch': rows[0]['path_sha256'] = '0' * 64
    elif mutation == 'work_change': rows[0]['work']['literal_updates'] += 1
    elif mutation == 'missing_work': rows[0]['work'].pop('literal_updates')
    elif mutation == 'bool_work': rows[0]['work']['literal_updates'] = True
    elif mutation == 'changed_input': cases[0]['formula']['clauses'].pop()
    elif mutation == 'changed_seed': rows[0]['search_seed'] += 1
    elif mutation == 'too_many_moves': rows[0]['flips'] = bench.CONFIG['moves'] + 1
    elif mutation == 'wrong_query_cost':
        row = next(r for r in rows if r['engine'] == 'cached')
        row['work']['feature_literal_visits'] = 1
    with pytest.raises((ValueError, TypeError)):
        bench.analysis(cases, rows)


@pytest.mark.parametrize('mutation', ['hash', 'inventory', 'source', 'summary', 'config'])
def test_archive_and_recomputed_statistics(evidence, mutation):
    out, _, _ = evidence
    path = out / {'hash': 'summary.json', 'source': 'sources.json',
                  'summary': 'summary.json', 'config': 'config.json',
                  'inventory': 'unexpected.json'}[mutation]
    if mutation == 'inventory':
        path.write_text('{}')
    else:
        value = json.loads(path.read_text())
        if mutation in ('hash', 'summary'): value['systems_gate'] = 'FORGED'
        elif mutation == 'config': value['moves'] += 1
        else: value['data/cnf.py'] = '0' * 64
        bench.write(path, value)
        if mutation != 'hash':
            hashes = json.loads((out / 'SHA256.json').read_text())
            hashes[path.name] = bench.sha(path)
            bench.write(out / 'SHA256.json', hashes)
    with pytest.raises(ValueError):
        bench.verify(out)


def test_reject_overwrite(evidence):
    out, _, _ = evidence
    with pytest.raises(FileExistsError):
        bench.run(out)


def test_bootstrap_is_formula_clustered_and_reports_negative_gate(evidence):
    _, cases, rows = evidence
    for row in rows:
        row['elapsed_ns'] = 100 if row['engine'] == 'reference' else 200
    report = bench.analysis(cases, rows)
    assert report['overall']['mean_ratio'] == 2
    assert report['overall']['mean_ratio_ci95'] == [2, 2]
    assert report['overall']['formulas'] == len(cases)
    assert report['systems_gate'] == 'FAIL'
    assert not any(report['conditions'].values())
