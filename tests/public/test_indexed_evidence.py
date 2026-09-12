"""Analysis corruption contracts use synthetic records, not benchmark evidence."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import pytest
from data.cnf import CNF

SPEC = importlib.util.spec_from_file_location('indexed_benchmark', Path(__file__).resolve().parents[2]/'scripts/bench_indexed_search.py')
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


@pytest.fixture
def records(monkeypatch):
    # A tiny observation inventory with a dimension-compatible, trivially valid
    # synthetic formula. This fixture is never a measured performance result.
    config = dict(bench.CONFIG, sizes=[4096], families=['uniform'], per_cell=1,
                  search_seeds=[19101], caps=[4096], rounds=1, bootstrap_repeats=20)
    monkeypatch.setattr(bench, 'CONFIG', config)
    p = CNF(4096, ((1, 2, 3),)*(4096*21//5))
    case = dict(id='4096:uniform:0', nvars=4096, family='uniform', seed=91326000,
                sha256=p.sha256(), formula=p.record())
    rows = []
    for c, seed, cap, r, e in bench.schedule([case]):
        result = dict(schema='spectra.cnf.solve.v1', status='SAT_VERIFIED', witness=[True]*4096,
                      unsatisfied=[], flips=0, queries=0, path_sha256=hashlib.sha256(b'').hexdigest(),
                      elapsed_ns=50 if e == 'reference' else 25, seed=seed, max_flips=cap,
                      algorithm='focused_classical_local_search', learned=False)
        rows.append(dict(case=c, seed=seed, cap=cap, round=r, engine=e,
                         wall_ns=100 if e == 'reference' else 50, result=result))
    memory = [dict(case=case['id'], engine=e, current_python_bytes=10, peak_python_bytes=20)
              for e in bench.ENGINES]
    preparation = [dict(case=case['id'], wall_ns=10, current_python_bytes=10, peak_python_bytes=20)]
    return [case], rows, memory, preparation


def test_synthetic_analysis_inventory_accepted(records):
    result = bench.analyze(*records)
    assert result['observations'] == 4
    assert result['paired_paths'] == 1
    assert result['primary']['mean_ratio'] == .5
    assert result['confirmation'] is False


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'reorder', 'seed', 'cap', 'round_type',
                                    'zero_time', 'nan_time', 'inner_larger', 'wrong_answer', 'status',
                                    'bool_witness', 'path', 'flips', 'queries', 'memory', 'prep',
                                    'learned', 'extra', 'formula_hash', 'residual_type'])
def test_corrupt_records_are_rejected(records, change):
    cases, rows, memory, prep = copy.deepcopy(records)
    row = rows[0]
    if change == 'missing': rows.pop()
    elif change == 'duplicate': rows[1] = copy.deepcopy(row)
    elif change == 'reorder': rows.reverse()
    elif change == 'seed': row['result']['seed'] += 1
    elif change == 'cap': row['result']['max_flips'] += 1
    elif change == 'round_type': row['round'] = False
    elif change == 'zero_time': row['wall_ns'] = 0
    elif change == 'nan_time': row['wall_ns'] = float('nan')
    elif change == 'inner_larger': row['result']['elapsed_ns'] = row['wall_ns']+1
    elif change == 'wrong_answer': row['result']['witness'] = [False]*4096
    elif change == 'status': row['result']['status'] = 'UNKNOWN'
    elif change == 'bool_witness': row['result']['witness'][0] = 1
    elif change == 'path': row['result']['path_sha256'] = '1'*64
    elif change == 'flips': row['result']['flips'] = -1
    elif change == 'queries': row['result']['queries'] = 1
    elif change == 'memory': memory.pop()
    elif change == 'prep': prep[0]['peak_python_bytes'] = 1
    elif change == 'learned': row['result']['learned'] = True
    elif change == 'extra': row['unrecorded'] = True
    elif change == 'formula_hash': cases[0]['sha256'] = '0'*64
    elif change == 'residual_type': row['result']['unsatisfied'] = [False]
    with pytest.raises((ValueError, TypeError)):
        bench.analyze(cases, rows, memory, prep)


def test_schedule_balanced_reproducible(records):
    cases, rows, _, _ = records
    assert bench.schedule(cases) == bench.schedule(cases)
    assert {r['engine'] for r in rows} == set(bench.ENGINES)


def test_verifier_rejects_unexpected_file_inventory(tmp_path):
    bench.write(tmp_path/'SHA256.json', {})
    with pytest.raises(ValueError, match='inventory'):
        bench.verify(tmp_path)


def test_verifier_rejects_source_change_before_data_replay(tmp_path):
    import json
    files = {'config.json', 'environment.json', 'source_sha256.json', 'executable_source.tar.gz',
             'cases.json.gz', 'timings.jsonl.gz', 'memory.json', 'preparation.json', 'summary.json'}
    for name in files:
        (tmp_path/name).write_text('{}')
    (tmp_path/'config.json').write_text(json.dumps(bench.CONFIG))
    hashes = {name: hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in files}
    bench.write(tmp_path/'SHA256.json', hashes)
    with pytest.raises(ValueError, match='source mismatch'):
        bench.verify(tmp_path)
