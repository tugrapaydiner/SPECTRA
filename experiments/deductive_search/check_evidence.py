"""Check retained bytes and witnesses, then summarize; never rerun timings.

Uses only the standard library and reads the archive without extracting it.
Historical development incompleteness is reported separately from evaluation.
"""
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import zipfile


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = ('indexed', 'deductive', 'glucose4')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = json.loads((HERE/'EVIDENCE.json').read_text())
    archive = HERE/manifest['archive']
    require(archive.stat().st_size == manifest['bytes'], 'archive size')
    require(hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['sha256'], 'archive digest')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        require(len(set(names)) == len(names), 'duplicate archive members')
        require(set(names) == set(manifest['members']), 'archive inventory')
        for name, identity in manifest['members'].items():
            data = z.read(name)
            require(len(data) == identity['bytes'], f'{name}: size')
            require(hashlib.sha256(data).hexdigest() == identity['sha256'], f'{name}: digest')
        frozen = json.loads(z.read('FROZEN.json'))
        for name, digest in frozen['sources'].items():
            require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, f'{name}: frozen source')
        reports = {}
        for split in ('development', 'development-02', 'development-03', 'evaluation'):
            lock = json.loads(z.read(f'{split}/LOCK.json'))
            for name, digest in lock['sources'].items():
                require(hashlib.sha256(z.read(f'{split}/sources/{name}')).hexdigest() == digest,
                        f'{split}/{name}: source snapshot')
            corpus = [json.loads(s) for s in z.read(f'{split}/cases.jsonl').splitlines()]
            cases = {case['id']: case for case in corpus}
            require(len(cases) == len(corpus), f'{split}: duplicate cases')
            require(len({c['sha256'] for c in corpus}) == len(corpus), f'{split}: duplicate formulas')
            for c in corpus:
                formula = c['formula']
                payload = json.dumps([formula['nvars'], formula['clauses']], separators=(',', ':')).encode('ascii')
                require(hashlib.sha256(b'spectra.cnf.v1\0'+payload).hexdigest() == c['sha256'], 'formula identity')
            rows = [json.loads(s) for s in z.read(f'{split}/rows.jsonl').splitlines()]
            keys = [(r['id'], r['search_seed'], r['arm'], r['round']) for r in rows]
            expected = set(itertools.product(cases, (17, 73), ARMS, range(3)))
            require(len(keys) == len(set(keys)), f'{split}: duplicate observations')
            require(set(keys) <= expected, f'{split}: unexpected observation')
            missing = sorted(expected-set(keys))
            require(not missing or (split == 'development' and len(missing) == 25), f'{split}: missing observations')
            for r in rows:
                c, result = cases[r['id']], r['result']
                require(all(r[k] == c[k] for k in ('family', 'nvars', 'seed', 'sha256')), 'row identity')
                require(type(r['complete_ns']) is int and r['complete_ns'] > 0, 'complete-call time')
                require(result['status'] in ('SAT_VERIFIED', 'UNKNOWN', 'UNSAT_REPORTED'), 'status')
                if result['status'] == 'SAT_VERIFIED':
                    w = result['witness']
                    require(len(w) == c['nvars'] and all(type(v) is bool for v in w), 'Boolean witness')
                    require(all(any(w[abs(lit)-1] == (lit > 0) for lit in clause)
                                for clause in c['formula']['clauses']), 'invalid original-clause SAT witness')
            reports[split] = {'formulas': len(corpus), 'observations': len(rows), 'missing': missing}
            if split != 'evaluation':
                continue
            require(len(corpus) == 80, 'evaluation formula count')
            order = list(itertools.product(('binary', 'forced', 'planted3', 'uniform3', 'ring'), (128, 512), range(8)))
            for i, (case, (family, nvars, index)) in enumerate(zip(corpus, order)):
                require((case['id'], case['seed']) == (f'{family}:{nvars}:{index}', 202610022000+i), 'evaluation order')
            indexed = {(r['id'], r['search_seed'], r['round']): r['result'] for r in rows if r['arm'] == 'indexed'}
            semantic = ('status', 'witness', 'unsatisfied', 'flips', 'queries', 'path_sha256', 'seed', 'max_flips')
            general_pairs = 0
            for r in rows:
                if r['arm'] == 'deductive' and r['family'] in ('planted3', 'uniform3'):
                    before = indexed[r['id'], r['search_seed'], r['round']]
                    require(all(before[k] == r['result'][k] for k in semantic), 'general search changed')
                    general_pairs += 1
            memory = [json.loads(s) for s in z.read('evaluation/memory.jsonl').splitlines()]
            rss = [json.loads(s) for s in z.read('evaluation/rss-corrected.jsonl').splitlines()]
            for records in (memory, rss):
                require(len(records) == 240 and {(r['id'], r['arm']) for r in records} == set(itertools.product(cases, ARMS)),
                        'memory inventory')
                outcomes = {(r['id'], r['arm']): r['result']['status'] for r in rows if r['search_seed'] == 17 and r['round'] == 0}
                require(all(r['status'] == outcomes[r['id'], r['arm']] for r in records), 'memory status changed')
            metrics = {}
            for arm in ARMS:
                group = [r for r in rows if r['arm'] == arm]
                distinct = {}
                for r in group:
                    key = r['id'], r['search_seed']
                    if key in distinct:
                        require(distinct[key]['result']['status'] == r['result']['status'], 'repeat outcome changed')
                    distinct[key] = r
                times = sorted(r['complete_ns']/1e6 for r in group)
                peaks = [r['peak_rss_kib']/1024 for r in rss if r['arm'] == arm]
                metrics[arm] = {
                    'sat_verified': sum(r['result']['status'] == 'SAT_VERIFIED' for r in distinct.values()),
                    'unknown': sum(r['result']['status'] == 'UNKNOWN' for r in distinct.values()),
                    'unsat_reported': sum(r['result']['status'] == 'UNSAT_REPORTED' for r in distinct.values()),
                    'unique_formulas_solved': len({r['id'] for r in distinct.values() if r['result']['status'] == 'SAT_VERIFIED'}),
                    'mean_ms': statistics.mean(times), 'p95_ms': times[(95*len(times)+99)//100-1],
                    'rss_mean_mib': statistics.mean(peaks), 'rss_range_mib': [min(peaks), max(peaks)],
                    'solved_by_observed_deadline': {str(ms): sum(r['result']['status'] == 'SAT_VERIFIED' and
                        statistics.median(s['complete_ns'] for s in group if (s['id'], s['search_seed']) == key) <= ms*1e6
                        for key, r in distinct.items()) for ms in (5, 20, 100)},
                }
            conflicts = [r['result']['stats']['conflicts'] for r in rows if r['arm'] == 'glucose4']
            reports[split].update(metrics=metrics, general_identical_pairs=general_pairs,
                                 glucose_conflicts_max=max(conflicts), glucose_budget_overshoot_observations=sum(c > 2000 for c in conflicts))
    print(json.dumps({'evaluation_status': 'PASS', 'development_status': 'INCOMPLETE_FIRST_RAW_RUN',
                      'frozen': frozen, 'splits': reports,
                      'scope': 'retained bytes and independent SAT checks; no UNSAT proof or historical timing reproduction'}, indent=2))


if __name__ == '__main__':
    main()
