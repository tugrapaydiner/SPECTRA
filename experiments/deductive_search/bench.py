"""Fresh, declared capability/cost comparison; preserve every outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.cnf import CNF, SplitMix64, random_3sat
from spectra.cnf import solve_deductive, solve_indexed

FAMILIES = ('binary', 'forced', 'planted3', 'uniform3', 'ring')
ARMS = ('indexed', 'deductive', 'glucose4')
SEEDS = (17, 73)


def shuffle(items, rng):
    for i in range(len(items)-1, 0, -1):
        j = rng.below(i+1)
        items[i], items[j] = items[j], items[i]


def generate(family, nvars, seed):
    if family in ('planted3', 'uniform3'):
        return random_3sat(nvars, nvars*42//10, seed, planted=family == 'planted3')[0]
    rng = SplitMix64(seed)
    if family == 'ring':
        witness = tuple(bool(rng.below(2)) for _ in range(nvars))
        order = list(range(1, nvars+1))
        shuffle(order, rng)
        signs = [v if witness[v-1] else -v for v in order]
        clauses = [clause for a, b in zip(signs, signs[1:]+signs[:1])
                   for clause in ((-a, b), (a, -b))]
    elif family == 'forced':
        base, witness = random_3sat(nvars, 3*nvars, seed, planted=True)
        order = list(range(1, nvars+1))
        shuffle(order, rng)
        signs = [v if witness[v-1] else -v for v in order]
        clauses = [(signs[0],)] + [(-a, b) for a, b in zip(signs, signs[1:])] + list(base.clauses)
    else:
        witness = tuple(bool(rng.below(2)) for _ in range(nvars))
        clauses, seen = [], set()
        while len(clauses) < 4*nvars:
            a, b = 1+rng.below(nvars), 1+rng.below(nvars)
            if a == b:
                continue
            clause = tuple(sorted((a if rng.below(2) else -a, b if rng.below(2) else -b), key=abs))
            if clause not in seen and any(witness[abs(lit)-1] == (lit > 0) for lit in clause):
                seen.add(clause)
                clauses.append(clause)
    shuffle(clauses, rng)
    problem = CNF(nvars, tuple(clauses))
    if not problem.satisfied(witness):
        raise ValueError('invalid generated witness')
    return problem


def cases(split):
    sizes, count, base = ((128,), 6, 202610021000) if split == 'development' else ((128, 512), 8, 202610022000)
    result = []
    for family in FAMILIES:
        for nvars in sizes:
            for i in range(count):
                seed = base+len(result)
                problem = generate(family, nvars, seed)
                result.append({'id': f'{family}:{nvars}:{i}', 'family': family, 'nvars': nvars,
                               'seed': seed, 'sha256': problem.sha256(), 'formula': problem.record()})
    if len({c['sha256'] for c in result}) != len(result):
        raise ValueError('duplicate input')
    return result


def independent_check(problem, witness):
    problem.validate_witness(witness)
    true = {i+1 if value else -i-1 for i, value in enumerate(witness)}
    return all(bool(set(clause) & true) for clause in problem.clauses)


def call(problem, arm, seed):
    if arm == 'glucose4':
        from pysat.solvers import Solver
        with Solver(name='glucose4', bootstrap_with=[list(c) for c in problem.clauses]) as solver:
            solver.conf_budget(2000)
            answer = solver.solve_limited()
            model = solver.get_model() if answer is True else None
            stats = solver.accum_stats()
        if model is not None:
            values = {abs(lit): lit > 0 for lit in model}
            witness = tuple(values.get(i+1, False) for i in range(problem.nvars))
        else:
            witness = None
        result = {'status': 'SAT_VERIFIED' if answer is True else 'UNSAT_REPORTED' if answer is False else 'UNKNOWN',
                  'witness': witness, 'stats': stats, 'conflict_budget_requested': 2000}
    else:
        solver = solve_indexed if arm == 'indexed' else solve_deductive
        result = solver(problem, seed=seed, max_flips=2048).record()
    if result['status'] == 'SAT_VERIFIED' and not independent_check(problem, tuple(result['witness'])):
        raise ValueError('false SAT witness')
    return result


def summary(rows):
    out = {}
    for family, nvars in sorted({(r['family'], r['nvars']) for r in rows}):
        cell = {}
        for arm in ARMS:
            group = [r for r in rows if (r['family'], r['nvars'], r['arm']) == (family, nvars, arm)]
            by_seed = {}
            for row in group:
                key = row['id'], row['search_seed']
                if key in by_seed and by_seed[key]['result']['status'] != row['result']['status']:
                    raise ValueError('nonrepeatable solve outcome')
                by_seed[key] = row
            times = sorted(r['complete_ns']/1e6 for r in group)
            cell[arm] = {'distinct_formula_seed_calls': len(by_seed),
                         'sat_verified': sum(r['result']['status'] == 'SAT_VERIFIED' for r in by_seed.values()),
                         'unknown': sum(r['result']['status'] == 'UNKNOWN' for r in by_seed.values()),
                         'unsat_reported': sum(r['result']['status'] == 'UNSAT_REPORTED' for r in by_seed.values()),
                         'mean_ms': statistics.mean(times),
                         'p95_ms': times[min(len(times)-1, (95*len(times)+99)//100-1)],
                         'solved_by_observed_deadline': {str(ms): sum(r['result']['status'] == 'SAT_VERIFIED' and
                            statistics.median(s['complete_ns'] for s in group if (s['id'], s['search_seed']) == key) <= ms*1e6
                            for key, r in by_seed.items()) for ms in (5, 20, 100)}}
        cell['candidate_over_indexed_mean'] = cell['deductive']['mean_ms']/cell['indexed']['mean_ms']
        out[f'{family}:{nvars}'] = cell
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=('development', 'evaluation'), default='development')
    p.add_argument('--dependencies', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--rss-case', type=Path)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--memory', action='store_true')
    a = p.parse_args()
    sys.path.insert(0, str(a.dependencies.resolve()))
    import pysat
    if pysat.__version__ != '1.9.dev15':
        raise ValueError('unexpected python-sat version')
    # Import the native comparator before measured calls, as declared for all imports.
    from pysat.solvers import Solver
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if a.rss_case:
        case = json.loads(a.rss_case.read_text())
        result = call(CNF.from_record(case['formula']), a.arm, SEEDS[0])
        print(json.dumps({'status': result['status'], 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
        return
    a.out.mkdir(parents=True, exist_ok=False)
    source_files = ['spectra/cnf/deductive.py', 'spectra/cnf/indexed.py', 'spectra/cnf/ranked.py',
                    'data/cnf.py', 'experiments/deductive_search/bench.py', 'experiments/deductive_search/PROTOCOL.md',
                    'experiments/deductive_search/AMENDMENT-01.md']
    lock = {'split': a.split, 'python': sys.version, 'platform': platform.platform(), 'python_sat': pysat.__version__,
            'cpu': next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
            'affinity': sorted(os.sched_getaffinity(0)), 'search_seeds': SEEDS, 'rounds': 3,
            'max_flips': 2048, 'conflict_budget_requested': 2000,
            'sources': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}}
    (a.out/'LOCK.json').write_text(json.dumps(lock, indent=2)+'\n')
    corpus = cases(a.split)
    (a.out/'cases.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in corpus))
    rows = []
    with (a.out/'rows.jsonl').open('x') as stream:
        for round_id in range(3):
            for index, case in enumerate(corpus):
                problem = CNF.from_record(case['formula'])
                for seed in SEEDS:
                    order = ARMS if (round_id+index+seed) % 2 else ARMS[::-1]
                    for arm in order:
                        start = time.perf_counter_ns()
                        result = call(problem, arm, seed)
                        elapsed = time.perf_counter_ns()-start
                        row = {k:v for k,v in case.items() if k != 'formula'}
                        row.update(arm=arm, round=round_id, search_seed=seed, complete_ns=elapsed, result=result)
                        rows.append(row)
                        stream.write(json.dumps(row)+'\n')
                        stream.flush()
            print('round', round_id, 'complete', flush=True)
    (a.out/'SUMMARY.json').write_text(json.dumps(summary(rows), indent=2)+'\n')
    if a.memory:
        with (a.out/'memory.jsonl').open('x') as stream:
            for case in corpus:
                problem = CNF.from_record(case['formula'])
                path = a.out/'memory-input.json'
                path.write_text(json.dumps(case))
                for arm in ARMS:
                    peak = None
                    if arm != 'glucose4':
                        tracemalloc.start()
                        call(problem, arm, SEEDS[0])
                        _, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                    result = subprocess.run([sys.executable, '-I', '-S', str(Path(__file__).resolve()),
                        '--dependencies', str(a.dependencies.resolve()), '--rss-case', str(path.resolve()), '--arm', arm],
                        check=True, capture_output=True, text=True, timeout=30)
                    stream.write(json.dumps({'id': case['id'], 'family': case['family'], 'nvars': case['nvars'],
                        'arm': arm, 'python_peak_bytes': peak, **json.loads(result.stdout)})+'\n')
                    stream.flush()
            path.unlink()
    print(json.dumps(summary(rows), indent=2))


if __name__ == '__main__':
    main()
