"""Frozen optional-cache systems comparison with full retained observations.

Run/verify with ``python -m eval.cnf_cache_benchmark``. This is not a SAT
competition benchmark, training pipeline, or learned-capability experiment.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time

import numpy as np

from data.cnf import CNF, SplitMix64, random_3sat
from eval.cnf_cached import CachedCNFRepairState
from eval.cnf_repair import CNFRepairState

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {'schema': 'spectra.cnf_cache_benchmark.v1', 'families': ['uniform', 'planted'],
          'sizes': [64, 128, 256, 512], 'per_cell': 8, 'case_seed': 620260911,
          'search_seeds': [17001, 27002], 'rounds': 3, 'moves': 1024,
          'bootstrap_repeats': 4000, 'bootstrap_seed': 90260911,
          'maximum_mean_upper': .80, 'maximum_cell_upper': 1.10,
          'maximum_p95_ratio': 1.00}
ENGINES = {'reference': CNFRepairState, 'cached': CachedCNFRepairState}
SOURCES = ['data/cnf.py', 'eval/cnf_repair.py', 'eval/cnf_cached.py',
           'eval/cnf_cache_benchmark.py', 'docs/CACHED_RESIDUAL_PROTOCOL.md']
SEMANTICS = ['witness', 'status', 'flips', 'path_sha256', 'unsatisfied', 'queries']
ROW_KEYS = set(SEMANTICS + ['case_id', 'engine', 'search_seed', 'round',
                           'elapsed_ns', 'work'])


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_cases():
    result = []
    for family in CONFIG['families']:
        for n in CONFIG['sizes']:
            for index in range(CONFIG['per_cell']):
                seed = CONFIG['case_seed'] + len(result)
                problem, _ = random_3sat(n, n * 21 // 5, seed, planted=family == 'planted')
                result.append({'id': f'{family}:{n}:{index}', 'family': family,
                               'nvars': n, 'seed': seed, 'formula': problem.record(),
                               'sha256': problem.sha256()})
    return result


def schedule(cases):
    for r in range(CONFIG['rounds']):
        for i, case in enumerate(cases):
            for j, seed in enumerate(CONFIG['search_seeds']):
                order = ['reference', 'cached'] if (r + i + j) % 2 == 0 else ['cached', 'reference']
                for engine in order:
                    yield case, seed, r, engine


def original_check(problem, witness):
    problem.validate_witness(witness)
    literals = {v + 1 if value else -v - 1 for v, value in enumerate(witness)}
    return all(bool(literals.intersection(clause)) for clause in problem.clauses)


def execute(problem, seed, engine, weights):
    """Only state representation differs; selection, logging and timing are shared."""
    start = time.perf_counter_ns()
    rng = SplitMix64(seed)
    witness = tuple(bool(rng.below(2)) for _ in range(problem.nvars))
    state = ENGINES[engine](problem, witness)
    path = []
    queries = 0
    while len(path) < CONFIG['moves'] and not state.valid:
        violated = state.unsatisfied
        clause = problem.clauses[violated[rng.below(len(violated))]]
        candidates = sorted({abs(lit) - 1 for lit in clause})
        if not candidates:
            break
        scores = [weights[state.make_break(v)[1]] for v in candidates]
        queries += len(candidates)
        threshold = (rng.below(2**53) / 2**53) * sum(scores)
        picked = candidates[-1]
        for variable, weight in zip(candidates, scores):
            threshold -= weight
            if threshold < 0:
                picked = variable
                break
        state.flip(picked)
        path.append(picked)
    witness = state.witness
    unsatisfied = list(state.unsatisfied)
    checked = original_check(problem, witness)
    if checked != state.valid:
        raise AssertionError('incremental state disagrees with original-formula semantics')
    work = {k: getattr(state, k) for k in ['literal_updates', 'feature_literal_visits']}
    if engine == 'cached':
        work.update({k: getattr(state, k) for k in ['cache_literal_visits',
                         'initialization_cache_visits', 'score_queries']})
    del state
    elapsed = time.perf_counter_ns() - start
    return {'witness': list(witness), 'status': 'SAT_VERIFIED' if checked else 'UNKNOWN',
            'flips': len(path), 'path_sha256': hashlib.sha256(json.dumps(path).encode()).hexdigest(),
            'queries': queries, 'unsatisfied': unsatisfied, 'work': work, 'elapsed_ns': elapsed}


def validate(cases, rows):
    if cases != generate_cases():
        raise ValueError('input inventory differs from the frozen unfiltered generator')
    expected = [(c['id'], s, r, e) for c, s, r, e in schedule(cases)]
    actual = [(r.get('case_id'), r.get('search_seed'), r.get('round'), r.get('engine')) for r in rows]
    if actual != expected:
        raise ValueError('missing, duplicated, reordered or unexpected observations')
    by_case = {c['id']: c for c in cases}
    invariants = {}
    work_records = {}
    for row in rows:
        if type(row) is not dict or set(row) != ROW_KEYS:
            raise ValueError('invalid record keys')
        for k in ['elapsed_ns', 'flips', 'queries', 'search_seed', 'round']:
            if type(row[k]) is not int or row[k] < (1 if k == 'elapsed_ns' else 0):
                raise ValueError('invalid integer observation: ' + k)
        if row['flips'] > CONFIG['moves'] or not 0 <= row['queries'] <= 3 * CONFIG['moves']:
            raise ValueError('impossible work count')
        work = row['work']
        keys = {'literal_updates', 'feature_literal_visits'}
        if row['engine'] == 'cached':
            keys |= {'cache_literal_visits', 'initialization_cache_visits', 'score_queries'}
        if (type(work) is not dict or set(work) != keys or
                any(type(v) is not int or v < 0 for v in work.values())):
            raise ValueError('invalid work counters')
        if row['engine'] == 'cached' and (work['feature_literal_visits'] != 0 or work['score_queries'] != row['queries']):
            raise ValueError('cached single-variable query work differs')
        problem = CNF.from_record(by_case[row['case_id']]['formula'])
        if type(row['witness']) is not list:
            raise ValueError('witness must be a complete Boolean list')
        witness = tuple(row['witness'])
        correct = original_check(problem, witness)
        if row['status'] != ('SAT_VERIFIED' if correct else 'UNKNOWN'):
            raise ValueError('false solver outcome')
        if (type(row['unsatisfied']) is not list or any(type(x) is not int for x in row['unsatisfied'])
                or row['unsatisfied'] != list(problem.violated(witness))):
            raise ValueError('incorrect final residual')
        digest = row['path_sha256']
        if type(digest) is not str or len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
            raise ValueError('invalid trajectory digest')
        key = (row['case_id'], row['search_seed'])
        semantics = {k: row[k] for k in SEMANTICS}
        if key in invariants and semantics != invariants[key]:
            raise ValueError('trajectory/outcome mismatch between engines or rounds')
        invariants[key] = semantics
        work_key = (*key, row['engine'])
        if work_key in work_records and work != work_records[work_key]:
            raise ValueError('nondeterministic work across timing rounds')
        work_records[work_key] = work
    return invariants


def analysis(cases, rows):
    invariants = validate(cases, rows)
    # Keep two search seeds nested in each formula, not as independent examples.
    costs = np.zeros((len(cases), len(CONFIG['search_seeds']), 2), dtype=np.float64)
    index = {c['id']: i for i, c in enumerate(cases)}
    for row in rows:
        i = index[row['case_id']]
        j = CONFIG['search_seeds'].index(row['search_seed'])
        k = ['reference', 'cached'].index(row['engine'])
        costs[i, j, k] += row['elapsed_ns'] / CONFIG['rounds']
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])

    def summarize(selected):
        values = costs[selected]
        means = values.mean(axis=1)
        # Stratified sampling within family/size for overall; one cell for strata.
        cells = {}
        for local, global_index in enumerate(selected):
            c = cases[global_index]
            cells.setdefault((c['family'], c['nvars']), []).append(local)
        draws = np.concatenate([rng.choice(ix, (CONFIG['bootstrap_repeats'], len(ix)))
                                for ix in cells.values()], axis=1)
        sampled = means[draws].mean(axis=1)
        ratios = sampled[:, 1] / sampled[:, 0]
        interval = np.percentile(ratios, [2.5, 97.5]).tolist()
        ids = {cases[i]['id'] for i in selected}
        successes = sum(r['status'] == 'SAT_VERIFIED' for key, r in invariants.items() if key[0] in ids)
        return {'formulas': len(selected), 'formula_search_pairs': len(selected) * 2,
                'solved_pairs_each_engine': successes,
                'reference_mean_ms': float(values[:, :, 0].mean() / 1e6),
                'cached_mean_ms': float(values[:, :, 1].mean() / 1e6),
                'mean_ratio': float(means[:, 1].mean() / means[:, 0].mean()),
                'mean_ratio_ci95': interval,
                'p95_ratio': float(np.percentile(values[:, :, 1], 95) / np.percentile(values[:, :, 0], 95))}
    overall = summarize(list(range(len(cases))))
    cells = {f'{family}:{n}': summarize([i for i, c in enumerate(cases)
             if c['family'] == family and c['nvars'] == n])
             for family in CONFIG['families'] for n in CONFIG['sizes']}
    conditions = {'overall_mean_upper': overall['mean_ratio_ci95'][1] <= CONFIG['maximum_mean_upper'],
                  'all_cell_upper': all(c['mean_ratio_ci95'][1] <= CONFIG['maximum_cell_upper'] for c in cells.values()),
                  'overall_p95': overall['p95_ratio'] <= CONFIG['maximum_p95_ratio']}
    return {'schema': CONFIG['schema'], 'correctness': 'PASS', 'observations': len(rows),
            'paired_paths': len(invariants), 'overall': overall, 'cells': cells,
            'conditions': conditions, 'systems_gate': 'PASS' if all(conditions.values()) else 'FAIL',
            'learned_capability': False, 'novel_algorithm_claim': False, 'energy_measured': False,
            'scope': 'same Python stochastic policy; capped full executions; paired formula-cluster descriptive intervals; one host'}


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'config.json', CONFIG)
    write(out / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
                                   'numpy': np.__version__, 'clock': vars(time.get_clock_info('perf_counter'))})
    write(out / 'sources.json', {p: sha(ROOT / p) for p in SOURCES})
    cases = generate_cases()
    write(out / 'cases.json', cases)
    warm, _ = random_3sat(32, 134, 720260911)
    for engine in ENGINES:
        execute(warm, 730260911, engine, [(1 + b)**-2.3 for b in range(135)])
    problems = {c['id']: CNF.from_record(c['formula']) for c in cases}
    tables = {c['id']: [(1 + b)**-2.3 for b in range(len(c['formula']['clauses']) + 1)] for c in cases}
    rows = []
    with gzip.open(out / 'rows.jsonl.gz', 'wt') as stream:
        for case, seed, r, engine in schedule(cases):
            row = execute(problems[case['id']], case['seed'] ^ seed, engine, tables[case['id']])
            row.update({'case_id': case['id'], 'search_seed': seed, 'round': r, 'engine': engine})
            rows.append(row)
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            if len(rows) % 128 == 0:
                print('observations', len(rows), flush=True)
    report = analysis(cases, rows)
    write(out / 'summary.json', report)
    write(out / 'SHA256.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != 'SHA256.json'})
    return report


def verify(out, replay=False):
    out = Path(out)
    inventory = json.loads((out / 'SHA256.json').read_text())
    if set(inventory) != {p.name for p in out.iterdir() if p.name != 'SHA256.json'}:
        raise ValueError('evidence inventory differs')
    if any(sha(out / name) != digest for name, digest in inventory.items()):
        raise ValueError('evidence hash differs')
    if json.loads((out / 'config.json').read_text()) != CONFIG:
        raise ValueError('protocol differs')
    if json.loads((out / 'sources.json').read_text()) != {p: sha(ROOT / p) for p in SOURCES}:
        raise ValueError('execution source differs')
    cases = json.loads((out / 'cases.json').read_text())
    with gzip.open(out / 'rows.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    report = analysis(cases, rows)
    if report != json.loads((out / 'summary.json').read_text()):
        raise ValueError('recomputed statistics differ')
    replays = 0
    if replay:
        firsts = {(r['case_id'], r['search_seed'], r['engine']): r for r in rows if r['round'] == 0}
        for case in cases:
            problem = CNF.from_record(case['formula'])
            weights = [(1 + b)**-2.3 for b in range(len(problem.clauses) + 1)]
            for seed in CONFIG['search_seeds']:
                for engine in ENGINES:
                    actual = execute(problem, case['seed'] ^ seed, engine, weights)
                    expected = firsts[(case['id'], seed, engine)]
                    if any(actual[k] != expected[k] for k in SEMANTICS + ['work']):
                        raise ValueError('full deterministic path/work replay differs')
                    replays += 1
    return {'integrity': 'PASS', 'original_formula_answer_checks': len(rows),
            'replayed_paths': replays, 'systems_gate': report['systems_gate']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'verify'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    report = run(args.out) if args.command == 'run' else verify(args.out, args.replay)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
