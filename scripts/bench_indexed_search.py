"""Frozen four-arm exact-path systems comparison; see INDEXED_SEARCH_PROTOCOL.md.

Run and verify write new paths only. Full call wall time (including returned
result construction and per-call release) is the primary latency observation.
The historical internal timer is also retained without redefining its scope.
"""
from __future__ import annotations
import argparse
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tarfile
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.cnf import CNF, SplitMix64, random_3sat
from spectra.cnf.search import solve
from spectra.cnf.indexed import PreparedCNF, solve_indexed, _solve_ranked

CONFIG = dict(sizes=[512, 4096, 16384], families=['uniform', 'planted'], per_cell=4,
              case_seed=91326000, search_seeds=[19101, 29102], caps=[128, 4096],
              rounds=3, order_seed=91326, bootstrap_seed=91327, bootstrap_repeats=4000,
              primary_maximum_ratio=0.70, primary_maximum_upper95=1.0)
ENGINES = ('reference', 'ranked', 'indexed_cold', 'indexed_warm')
SOURCES = ('data/cnf.py', 'eval/cnf_cached.py', 'spectra/cnf/state.py',
           'spectra/cnf/search.py', 'spectra/cnf/ranked.py', 'spectra/cnf/indexed.py',
           'scripts/bench_indexed_search.py')
RESULT_FIELDS = {'schema', 'status', 'witness', 'unsatisfied', 'flips', 'queries',
                 'path_sha256', 'elapsed_ns', 'seed', 'max_flips', 'algorithm', 'learned'}
ROW_FIELDS = {'case', 'seed', 'cap', 'round', 'engine', 'wall_ns', 'result'}


def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def load(path):
    return json.loads(Path(path).read_text())


def generate():
    cases = []
    for n in CONFIG['sizes']:
        for family in CONFIG['families']:
            for i in range(CONFIG['per_cell']):
                seed = CONFIG['case_seed'] + len(cases)
                p, _ = random_3sat(n, n*21//5, seed, planted=family == 'planted')
                cases.append(dict(id=f'{n}:{family}:{i}', nvars=n, family=family,
                                  seed=seed, sha256=p.sha256(), formula=p.record()))
    return cases


def shuffle(items, rng):
    for i in range(len(items)-1, 0, -1):
        j = rng.below(i+1)
        items[i], items[j] = items[j], items[i]


def schedule(cases):
    rng = SplitMix64(CONFIG['order_seed'])
    groups = [(c['id'], s, cap, r) for r in range(CONFIG['rounds']) for c in cases
              for s in CONFIG['search_seeds'] for cap in CONFIG['caps']]
    shuffle(groups, rng)
    rows = []
    for case, seed, cap, round_id in groups:
        engines = list(ENGINES)
        shuffle(engines, rng)
        rows.extend((case, seed, cap, round_id, e) for e in engines)
    return rows


def execute(problem, prepared, engine, seed, cap):
    if engine == 'reference':
        return solve(problem, seed=seed, max_flips=cap)
    if engine == 'ranked':
        return _solve_ranked(problem, seed=seed, max_flips=cap)
    if engine == 'indexed_cold':
        return solve_indexed(problem, seed=seed, max_flips=cap)
    if engine == 'indexed_warm':
        return prepared.solve(seed=seed, max_flips=cap)
    raise ValueError('unknown engine')


def semantic(result):
    return {k: v for k, v in result.items() if k != 'elapsed_ns'}


def positive_int(value):
    return type(value) is int and value > 0


def quantile(values, p):
    values = sorted(values)
    x = p*(len(values)-1)
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi]-values[lo])*(x-lo)


def ratios_by_formula(case_ids, table, cap, engine, bootstrap_seed):
    # The declared inventory is balanced: every formula has the same number of
    # seeds/rounds under both arms. Keep nanosecond totals as exact integers until
    # the final division. Float sum() changed across supported Python versions;
    # averaging float means first made otherwise identical reports differ by ULPs.
    counts = {len(table[(c, cap, arm)]) for c in case_ids for arm in ('reference', engine)}
    if len(counts) != 1 or not counts or min(counts) == 0:
        raise ValueError('formula aggregation requires a nonempty balanced inventory')
    observations = counts.pop()
    pairs = []
    for case in case_ids:
        arms = [table[(case, cap, arm)] for arm in ('reference', engine)]
        if any(not positive_int(value) for arm in arms for value in arm):
            raise ValueError('aggregation requires positive integer nanoseconds')
        pairs.append(tuple(sum(arm) for arm in arms))
    before, after = sum(a for a, _ in pairs), sum(b for _, b in pairs)
    denominator = len(pairs) * observations * 1_000_000
    rng = SplitMix64(bootstrap_seed)
    draws = []
    for _ in range(CONFIG['bootstrap_repeats']):
        selected = [pairs[rng.below(len(pairs))] for _ in pairs]
        draws.append(sum(b for _, b in selected)/sum(a for a, _ in selected))
    return dict(reference_mean_ms=before/denominator, candidate_mean_ms=after/denominator,
                mean_ratio=after/before, ratio_ci95=[quantile(draws, .025), quantile(draws, .975)],
                independent_formulas=len(pairs))


def validate_cases(cases):
    expected = [(n, f, i) for n in CONFIG['sizes'] for f in CONFIG['families']
                for i in range(CONFIG['per_cell'])]
    if len(cases) != len(expected):
        raise ValueError('case inventory mismatch')
    problems = {}
    for j, (case, (n, family, i)) in enumerate(zip(cases, expected)):
        if (set(case) != {'id', 'nvars', 'family', 'seed', 'sha256', 'formula'} or
                case['id'] != f'{n}:{family}:{i}' or case['nvars'] != n or
                case['family'] != family or case['seed'] != CONFIG['case_seed']+j):
            raise ValueError('case metadata mismatch')
        problem = CNF.from_record(case['formula'])
        if (problem.nvars != n or len(problem.clauses) != n*21//5 or
                problem.sha256() != case['sha256']):
            raise ValueError('case formula/hash mismatch')
        problems[case['id']] = problem
    return problems


def analyze(cases, rows, memory, preparation):
    problems = validate_cases(cases)
    expected = schedule(cases)
    if len(rows) != len(expected):
        raise ValueError('timing inventory mismatch')
    deterministic = {}
    table = {}
    solves = {e: 0 for e in ENGINES}
    for row, key in zip(rows, expected):
        if set(row) != ROW_FIELDS or tuple(row[k] for k in ['case', 'seed', 'cap', 'round', 'engine']) != key:
            raise ValueError('timing inventory/order mismatch')
        for k in ('seed', 'cap', 'round'):
            if type(row[k]) is not int:
                raise ValueError('invalid schedule integer')
        record = row['result']
        if type(record) is not dict or set(record) != RESULT_FIELDS:
            raise ValueError('result schema mismatch')
        if (not positive_int(row['wall_ns']) or not positive_int(record['elapsed_ns']) or
                row['wall_ns'] < record['elapsed_ns']):
            raise ValueError('invalid timing')
        if record['schema'] != 'spectra.cnf.solve.v1' or record['algorithm'] != 'focused_classical_local_search' or record['learned'] is not False:
            raise ValueError('unexpected algorithm')
        if (type(record['seed']) is not int or type(record['max_flips']) is not int or
                record['seed'] != row['seed'] or record['max_flips'] != row['cap']):
            raise ValueError('wrong execution settings')
        if (type(record['flips']) is not int or not 0 <= record['flips'] <= row['cap'] or
                type(record['queries']) is not int or record['queries'] != 3*record['flips']):
            raise ValueError('invalid work')
        digest = record['path_sha256']
        if type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid path digest')
        if type(record['witness']) is not list or type(record['unsatisfied']) is not list or any(type(v) is not int for v in record['unsatisfied']):
            raise ValueError('invalid output arrays')
        violated = list(problems[row['case']].violated(tuple(record['witness'])))
        if record['unsatisfied'] != violated or record['status'] != ('UNKNOWN' if violated else 'SAT_VERIFIED'):
            raise ValueError('original-formula witness mismatch')
        pair = (row['case'], row['seed'], row['cap'])
        observed = semantic(record)
        if pair in deterministic and deterministic[pair] != observed:
            raise ValueError('trajectory mismatch between backends/rounds')
        deterministic[pair] = observed
        solves[row['engine']] += not violated
        table.setdefault((row['case'], row['cap'], row['engine']), []).append(row['wall_ns'])
    expected_memory = [(c['id'], e) for c in cases for e in ENGINES]
    if len(memory) != len(expected_memory):
        raise ValueError('memory inventory mismatch')
    for row, key in zip(memory, expected_memory):
        if (set(row) != {'case', 'engine', 'current_python_bytes', 'peak_python_bytes'} or
                (row['case'], row['engine']) != key or
                not positive_int(row['current_python_bytes']) or
                not positive_int(row['peak_python_bytes']) or
                row['peak_python_bytes'] < row['current_python_bytes']):
            raise ValueError('invalid memory record')
    if len(preparation) != len(cases):
        raise ValueError('preparation inventory mismatch')
    for row, case in zip(preparation, cases):
        if (set(row) != {'case', 'wall_ns', 'current_python_bytes', 'peak_python_bytes'} or
                row['case'] != case['id'] or any(not positive_int(row[k]) for k in ['wall_ns', 'current_python_bytes', 'peak_python_bytes']) or
                row['peak_python_bytes'] < row['current_python_bytes']):
            raise ValueError('invalid preparation record')
    cells = {}
    for n in CONFIG['sizes']:
        for family in CONFIG['families']:
            ids = [c['id'] for c in cases if c['nvars'] == n and c['family'] == family]
            for cap in CONFIG['caps']:
                cell = {}
                for engine in ENGINES[1:]:
                    cell[engine] = ratios_by_formula(ids, table, cap, engine, CONFIG['bootstrap_seed'])
                cells[f'{n}:{family}:{cap}'] = cell
    large = [c['id'] for c in cases if c['nvars'] >= 4096]
    primary = ratios_by_formula(large, table, 4096, 'indexed_cold', CONFIG['bootstrap_seed'])
    gates = dict(exact_paths=True, primary_mean=primary['mean_ratio'] <= CONFIG['primary_maximum_ratio'],
                 primary_upper95=primary['ratio_ci95'][1] < CONFIG['primary_maximum_upper95'])
    return dict(schema='spectra.indexed_benchmark.v2', aggregation='balanced_integer_nanoseconds_v1', observations=len(rows), paired_paths=len(deterministic),
                independent_formulas=len(cases), timing='outer complete-call wall_ns', cells=cells,
                primary=primary, gates=gates, gate='PASS' if all(gates.values()) else 'FAIL',
                verified_sat_observations=solves, memory=memory, preparation=preparation,
                memory_scope='tracemalloc Python allocations; input outside, reusable warm index separately reported',
                confirmation=False, learned_capability=False, external_solver_superiority=False)


def environment():
    info = dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                gc_enabled=gc.isenabled(), affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)
    for name, path in [('cpu_max', '/sys/fs/cgroup/cpu.max'), ('memory_max', '/sys/fs/cgroup/memory.max')]:
        info[name] = Path(path).read_text().strip() if Path(path).is_file() else None
    if Path('/proc/cpuinfo').is_file():
        info['cpu_model'] = next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')), None)
    return info


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', CONFIG)
    write(out/'environment.json', environment())
    write(out/'source_sha256.json', {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES})
    with tarfile.open(out/'executable_source.tar.gz', 'w:gz') as archive:
        for path in SOURCES:
            archive.add(ROOT/path, arcname=path)
    cases = generate()
    with gzip.open(out/'cases.json.gz', 'wt') as f:
        json.dump(cases, f, separators=(',', ':'))
    problems = validate_cases(cases)
    prepared, preparation = {}, []
    for c in cases:
        start = time.perf_counter_ns()
        index = PreparedCNF(problems[c['id']])
        elapsed = time.perf_counter_ns()-start
        prepared[c['id']] = index
        preparation.append(dict(case=c['id'], wall_ns=elapsed))
    rows = []
    with (out/'timings.partial.jsonl').open('x') as stream:
        for i, (case, seed, cap, round_id, engine) in enumerate(schedule(cases)):
            start = time.perf_counter_ns()
            result = execute(problems[case], prepared[case], engine, seed, cap)
            elapsed = time.perf_counter_ns()-start
            row = dict(case=case, seed=seed, cap=cap, round=round_id, engine=engine,
                       wall_ns=elapsed, result=result.record())
            rows.append(row)
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
            stream.flush()
            del result
            if (i+1) % 48 == 0:
                print('measured', i+1, 'of', len(schedule(cases)), flush=True)
    memory = []
    for i, case in enumerate(cases):
        p = problems[case['id']]
        gc.collect(); tracemalloc.start()
        index = PreparedCNF(p)
        current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        preparation[i].update(current_python_bytes=current, peak_python_bytes=peak)
        del index
        for engine in ENGINES:
            gc.collect(); tracemalloc.start()
            result = execute(p, prepared[case['id']], engine, 19101, 128)
            current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            if tuple(result.unsatisfied) != p.violated(result.witness):
                raise AssertionError('memory pass witness mismatch')
            memory.append(dict(case=case['id'], engine=engine,
                               current_python_bytes=current, peak_python_bytes=peak))
            del result
        print('memory', case['id'], flush=True)
    write(out/'memory.json', memory)
    write(out/'preparation.json', preparation)
    summary = analyze(cases, rows, memory, preparation)
    write(out/'summary.json', summary)
    with (out/'timings.partial.jsonl').open('rb') as source, gzip.open(out/'timings.jsonl.gz', 'wb') as dest:
        import shutil
        shutil.copyfileobj(source, dest)
    (out/'timings.partial.jsonl').unlink()  # Exact logical bytes are now in the gzip stream.
    write(out/'SHA256.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())})
    return summary


def verify(out, replay=False):
    hashes = load(out/'SHA256.json')
    expected_files = {'config.json', 'environment.json', 'source_sha256.json', 'executable_source.tar.gz',
                      'cases.json.gz', 'timings.jsonl.gz', 'memory.json', 'preparation.json', 'summary.json'}
    if set(hashes) != expected_files or {p.name for p in out.iterdir()} != expected_files | {'SHA256.json'}:
        raise ValueError('artifact inventory mismatch')
    for name, digest in hashes.items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest() != digest:
            raise ValueError('artifact hash mismatch')
    if load(out/'config.json') != CONFIG:
        raise ValueError('configuration mismatch')
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}
    if load(out/'source_sha256.json') != sources:
        raise ValueError('executable source mismatch')
    with tarfile.open(out/'executable_source.tar.gz', 'r:gz') as archive:
        if archive.getnames() != list(SOURCES):
            raise ValueError('source archive inventory mismatch')
        for path in SOURCES:
            if archive.extractfile(path).read() != (ROOT/path).read_bytes():
                raise ValueError('source archive bytes mismatch')
    with gzip.open(out/'cases.json.gz', 'rt') as f:
        cases = json.load(f)
    if cases != generate():
        raise ValueError('frozen generated inputs mismatch')
    with gzip.open(out/'timings.jsonl.gz', 'rt') as f:
        rows = [json.loads(line) for line in f]
    summary = analyze(cases, rows, load(out/'memory.json'), load(out/'preparation.json'))
    if summary != load(out/'summary.json'):
        raise ValueError('analysis mismatch')
    replayed = 0
    if replay:
        problems = validate_cases(cases)
        prepared = {key: PreparedCNF(p) for key, p in problems.items()}
        seen = set()
        for row in rows:
            key = tuple(row[k] for k in ['case', 'seed', 'cap', 'engine'])
            if key in seen:
                continue
            seen.add(key)
            actual = execute(problems[row['case']], prepared[row['case']], row['engine'], row['seed'], row['cap']).record()
            if semantic(actual) != semantic(row['result']):
                raise ValueError('exact trajectory replay mismatch')
            replayed += 1
    return dict(integrity='PASS', answers=len(rows), exact_path_replays=replayed,
                systems_gate=summary['gate'], primary=summary['primary'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'verify'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.out) if args.command == 'run' else verify(args.out, args.replay), indent=2))
