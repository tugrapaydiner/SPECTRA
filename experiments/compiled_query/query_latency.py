"""Paired development comparison on the already exposed structural panel.

See QUERY_LATENCY_PLAN.md. No retries, no changed frozen protocol, no holdout claim.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback

from . import exact
from .cases import Case, canonical, digest
from .proof_replay import OriginalReplay
from .sessions import decode_certified_output
from .transfer_panel import COUNTS, FAMILIES, make_case, seed_for

ROOT = Path(__file__).resolve().parents[2]
BASELINE = '39e9ee0b89f1261106231ee94546758c510291ba'
SCHEMA = 'spectra.query_latency_development.v1'
ARMS = ('baseline-raw', 'baseline-route', 'candidate-raw', 'candidate-scc',
        'candidate-route', 'candidate-auto')
REPEATS = 9
REPLICAS = tuple(range(6))
COMPARISONS = {
    'auto_vs_raw': ('candidate-auto', 'candidate-raw'),
    'auto_vs_baseline_route': ('candidate-auto', 'baseline-route'),
    'route_vs_baseline_route': ('candidate-route', 'baseline-route'),
    'raw_vs_baseline_raw': ('candidate-raw', 'baseline-raw'),
    'route_vs_raw': ('candidate-route', 'candidate-raw'),
}


def write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(payload)


def put(path: Path, value) -> str:
    return write_once(path, canonical(value))


def sources() -> dict:
    folder = Path(__file__).parent
    paths = sorted(folder.glob('*.py')) + sorted(folder.glob('*.cpp'))
    paths.append(folder / 'QUERY_LATENCY_PLAN.md')
    return {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths}


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', *args], cwd=ROOT)


def read_bound(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f'evidence hash differs: {path.name}')
    return data


def prepare(root: Path) -> dict:
    # Untracked source files are also forbidden: a Git commit must reproduce this run.
    if git('status', '--porcelain', '--', 'experiments/compiled_query').strip():
        raise ValueError('commit the comparison sources before running')
    root.mkdir(parents=True, exist_ok=False)
    identity = sources()
    commit = git('rev-parse', 'HEAD').decode().strip()
    source_path = 'experiments/compiled_query/exact_native.cpp'
    baseline_source = git('show', BASELINE + ':' + source_path)
    write_once(root / 'baseline.cpp', baseline_source)
    write_once(root / 'candidate.cpp', exact.SOURCE.read_bytes())
    current = exact.build_exact_runtime(root / 'candidate')
    original = exact.SOURCE
    try:
        exact.SOURCE = (root / 'baseline.cpp').resolve()
        baseline = exact.build_exact_runtime(root / 'baseline')
    finally:
        exact.SOURCE = original
    builds = {}
    for name, library, source in (('baseline', baseline, baseline_source),
                                  ('candidate', current, exact.SOURCE.read_bytes())):
        receipt = json.loads(library.with_name('build.json').read_bytes())
        if receipt['source_sha256'] != digest(source) or receipt['library_sha256'] != digest(library.read_bytes()):
            raise ValueError('native build receipt differs')
        builds[name] = dict(library=str(library.relative_to(root.resolve())),
                            library_sha256=receipt['library_sha256'],
                            receipt_sha256=digest(library.with_name('build.json').read_bytes()))
    a = json.loads(baseline.with_name('build.json').read_bytes())
    b = json.loads(current.with_name('build.json').read_bytes())
    if a['command'][:-3] != b['command'][:-3] or a['compiler'] != b['compiler']:
        raise ValueError('compiler or flags differ between arms')
    cases = []
    for family in FAMILIES:
        for replica in REPLICAS:
            case = make_case(family, replica)
            case_hash = write_once(root / 'cases' / (case.name + '.json'), case.payload())
            cases.append(dict(name=case.name, family=family, replica=replica, sha256=case_hash))
    manifest = dict(schema=SCHEMA, classification='exposed synthetic development panel; not confirmation',
                    candidate_commit=commit, baseline_commit=BASELINE, sources=identity,
                    baseline_source_sha256=digest(baseline_source), builds=builds, cases=cases,
                    arms=ARMS, counts=COUNTS, repeats=REPEATS, warmups=2,
                    deadline_seconds=120, address_space_bytes=2 << 30, native_bytes=512 << 20,
                    python=sys.version, platform=platform.platform(), machine=platform.machine())
    manifest_hash = put(root / 'manifest.json', manifest)
    write_once(root / 'manifest.sha256', (manifest_hash + '\n').encode())
    return manifest


def manifest_at(root: Path, *, executing: bool = False) -> dict:
    manifest = json.loads(read_bound(root / 'manifest.json', (root / 'manifest.sha256').read_text().strip()))
    if (manifest['schema'] != SCHEMA or tuple(manifest['arms']) != ARMS
            or tuple(manifest['counts']) != COUNTS or manifest['repeats'] != REPEATS):
        raise ValueError('comparison schedule or schema differs')
    expected_cases = {(family, replica) for family in FAMILIES for replica in REPLICAS}
    actual_cases = {(item['family'], item['replica']) for item in manifest['cases']}
    if (actual_cases != expected_cases or len(manifest['cases']) != len(expected_cases)
            or len({item['name'] for item in manifest['cases']}) != len(expected_cases)):
        raise ValueError('comparison must retain every fixed graph realization exactly once')
    if executing and manifest['sources'] != sources():
        raise ValueError('executing sources differ from the recorded commit')
    return manifest


def record_name(count: int, repeat: int, arm: str) -> str:
    return f'{count}-{repeat:02d}-{arm}.json'


def worker(root: Path, index: int) -> None:
    import resource
    manifest = manifest_at(root, executing=True)
    resource.setrlimit(resource.RLIMIT_AS, (manifest['address_space_bytes'],) * 2)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    item = manifest['cases'][index]
    case = Case.decode(read_bound(root / 'cases' / (item['name'] + '.json'), item['sha256']))
    folder = root / 'runs' / case.name
    runtimes = {}
    for name, build in manifest['builds'].items():
        library = root / build['library']
        read_bound(library, build['library_sha256'])
        read_bound(library.with_name('build.json'), build['receipt_sha256'])
        runtimes[name] = exact.ExactRuntime(library)
    rng = random.Random(seed_for('query-latency/' + case.name))
    packets, outputs, receipts = {}, set(), {}
    for count in COUNTS:
        for repeat in range(REPEATS):
            order = list(range(count))
            if repeat % 3 == 1:
                order.reverse()
            elif repeat % 3 == 2:
                random.Random(seed_for(f'{case.name}/{count}/{repeat}/order')).shuffle(order)
            rows = tuple(case.queries[i] for i in order)
            tick = time.perf_counter_ns()
            packet = exact.pack_session_packet(case.problem, rows)
            packing_ns = time.perf_counter_ns() - tick
            packet_hash = digest(packet)
            if packet_hash not in packets:
                write_once(folder / 'packets' / (packet_hash + '.sbc1'), packet)
                packets[packet_hash] = dict(count=count, order=order)
            arms = list(ARMS)
            rng.shuffle(arms)
            for arm in arms:
                build, mode = arm.split('-')
                record = dict(arm=arm, count=count, repeat=repeat,
                              packet_sha256=packet_hash, packing_ns=packing_ns)
                try:
                    rt = runtimes[build]
                    if repeat == 0:
                        for _ in range(manifest['warmups']):
                            rt.solve_session_packet(packet, mode=mode, max_bytes=manifest['native_bytes'])
                    tick = time.perf_counter_ns()
                    if digest(packet) != packet_hash:
                        raise ValueError('input bytes changed')
                    answer = rt.solve_session_packet(packet, mode=mode, max_bytes=manifest['native_bytes'])
                    output_hash = digest(answer.output)
                    elapsed = time.perf_counter_ns() - tick
                    if output_hash not in outputs:
                        write_once(folder / 'outputs' / (output_hash + '.scq2'), answer.output)
                        outputs.add(output_hash)
                    record.update(status='COMPLETE', complete_ns=elapsed,
                                  output_sha256=output_hash, output_bytes=len(answer.output),
                                  sat=answer.sat_count, unsat=answer.unsat_count,
                                  proof_bytes=answer.proof_bytes, info=answer.info)
                    del answer
                except Exception as exc:
                    record.update(status='ERROR', error=repr(exc), traceback=traceback.format_exc())
                filename = record_name(count, repeat, arm)
                receipts[filename] = put(folder / 'records' / filename, record)
    packets_hash = put(folder / 'packets.json', packets)
    put(folder / 'worker.json', dict(case=case.name, records=receipts, packets_sha256=packets_hash,
        manifest_sha256=digest((root / 'manifest.json').read_bytes()), builds=manifest['builds'],
        python=sys.version, machine=platform.machine(), affinity=sorted(os.sched_getaffinity(0)),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))


def execute(root: Path) -> None:
    manifest = prepare(root)
    for index, item in enumerate(manifest['cases']):
        folder = root / 'runs' / item['name']
        folder.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-m', 'experiments.compiled_query.query_latency', '_worker',
                   '--out', str(root.resolve()), '--case', str(index)]
        tick = time.perf_counter_ns()
        status = 'COMPLETE'
        with (folder / 'stdout.log').open('xb') as stdout, (folder / 'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                start_new_session=True, env={**os.environ, 'PYTHONHASHSEED': '0',
                    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
            try:
                code = process.wait(timeout=manifest['deadline_seconds'])
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
                status = 'TIMEOUT'
            except BaseException:
                import signal
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        if code and status == 'COMPLETE':
            status = 'CRASH'
        put(folder / 'controller.json', dict(status=status, returncode=code,
            cold_wall_ns=time.perf_counter_ns()-tick,
            worker_sha256=digest((folder/'worker.json').read_bytes()) if (folder/'worker.json').exists() else None))
        print(item['name'], status, flush=True)


def gmean(values) -> float:
    return math.exp(statistics.mean(math.log(x) for x in values))


def summarize(root: Path) -> dict:
    manifest = manifest_at(root)
    cells, counts, selections = defaultdict(list), Counter(), Counter()
    case_results, failures = [], []
    checked_outputs = checked_decisions = 0
    manifest_hash = digest((root / 'manifest.json').read_bytes())
    for item in manifest['cases']:
        case = Case.decode(read_bound(root / 'cases' / (item['name'] + '.json'), item['sha256']))
        folder = root / 'runs' / case.name
        control = json.loads((folder/'controller.json').read_bytes()) if (folder/'controller.json').exists() else {'status': 'MISSING'}
        counts['workers_' + control['status']] += 1
        if control['status'] != 'COMPLETE':
            failures.append(dict(case=case.name, reason=control['status']))
            continue
        worker_receipt = json.loads(read_bound(folder/'worker.json', control['worker_sha256']))
        if worker_receipt['manifest_sha256'] != manifest_hash or worker_receipt['builds'] != manifest['builds']:
            raise ValueError('worker identity differs')
        expected = {record_name(c, r, a) for c in COUNTS for r in range(REPEATS) for a in ARMS}
        if set(worker_receipt['records']) != expected or {p.name for p in (folder/'records').iterdir()} != expected:
            raise ValueError('missing, extra or duplicate timing records')
        packets = json.loads(read_bound(folder/'packets.json', worker_receipt['packets_sha256']))
        replay = OriginalReplay(case.problem)
        seen, decisions, output_by_arm, auto_modes = set(), {}, {}, {}
        for filename, expected_hash in worker_receipt['records'].items():
            row = json.loads(read_bound(folder/'records'/filename, expected_hash))
            arm, count, repeat = row['arm'], row['count'], row['repeat']
            if filename != record_name(count, repeat, arm):
                raise ValueError('record schedule differs from its filename')
            counts[row['status']] += 1
            if row['status'] != 'COMPLETE':
                failures.append(dict(case=case.name, record=filename, reason=row['status']))
                continue
            if type(row['complete_ns']) is not int or row['complete_ns'] <= 0:
                raise ValueError('invalid measurement duration')
            packet_hash, output_hash = row['packet_sha256'], row['output_sha256']
            packet = read_bound(folder/'packets'/(packet_hash+'.sbc1'), packet_hash)
            output = read_bound(folder/'outputs'/(output_hash+'.scq2'), output_hash)
            order = packets[packet_hash]['order']
            if len(order) != count or sorted(order) != list(range(count)):
                raise ValueError('request ordering differs')
            query_rows = tuple(case.queries[i] for i in order)
            if packet != exact.pack_session_packet(case.problem, query_rows):
                raise ValueError('packet differs from original problem and requests')
            answers = decode_certified_output(output, case.problem.n, count)
            outcome = tuple(code for code, _, _ in answers)
            if (len(output) != row['output_bytes'] or sum(outcome) != row['sat']
                    or count-sum(outcome) != row['unsat']
                    or sum(len(p) for _, _, p in answers) != row['proof_bytes']):
                raise ValueError('answer accounting differs')
            if packet_hash in decisions and outcome != decisions[packet_hash]:
                raise ValueError('solver decisions disagree')
            decisions[packet_hash] = outcome
            if (packet_hash, output_hash) not in seen:
                for query, (code, labels, certificate) in zip(query_rows, answers):
                    if not replay.verify(query, 'SAT_VERIFIED' if code else 'UNSAT_VERIFIED',
                                         labels if code else b'', certificate):
                        raise ValueError('original-input witness or proof rejected')
                seen.add((packet_hash, output_hash))
                checked_outputs += 1
                checked_decisions += count
            cells[case.name, count, arm].append(row['complete_ns'])
            output_by_arm[count, repeat, arm] = output_hash
            if arm == 'candidate-auto':
                selected = row['info']['selected_mode']
                if selected not in ('raw', 'scc', 'route'):
                    raise ValueError('auto reported an unknown engine')
                auto_modes[count, repeat] = selected
                selections[selected] += 1
        for count in COUNTS:
            for repeat in range(REPEATS):
                for mode in ('raw', 'route'):
                    keys = [(count, repeat, prefix + '-' + mode) for prefix in ('baseline', 'candidate')]
                    if all(k in output_by_arm for k in keys) and output_by_arm[keys[0]] != output_by_arm[keys[1]]:
                        raise ValueError('same-mode output bytes changed')
                if (count, repeat) in auto_modes:
                    target = count, repeat, 'candidate-' + auto_modes[count, repeat]
                    if target in output_by_arm and output_by_arm[target] != output_by_arm[count, repeat, 'candidate-auto']:
                        raise ValueError('auto differs from its reported engine')
        medians = {str(count): {arm: statistics.median(cells[case.name, count, arm])
            for arm in ARMS if len(cells[case.name, count, arm]) == REPEATS} for count in COUNTS}
        case_results.append({**item, 'base_satisfiable': replay.feasible(()), 'medians_ns': medians,
                             'peak_process_rss_bytes': worker_receipt['peak_process_rss_bytes']})
    expected_calls = len(manifest['cases']) * len(COUNTS) * REPEATS * len(ARMS)
    complete = counts['COMPLETE'] == expected_calls and counts['workers_COMPLETE'] == len(manifest['cases']) and not failures
    aggregates = {}
    if complete:
        for count in COUNTS:
            comparisons = {}
            for name, (a, b) in COMPARISONS.items():
                ratios = {r['name']: r['medians_ns'][str(count)][a]/r['medians_ns'][str(count)][b] for r in case_results}
                strata = [[ratios[r['name']] for r in case_results if r['family'] == f] for f in FAMILIES]
                rng = random.Random(seed_for(f'query-latency/bootstrap/{count}/{name}'))
                boot = sorted(gmean([rng.choice(group) for group in strata for _ in group]) for _ in range(4000))
                comparisons[name] = dict(ratio=gmean(ratios.values()), ci95=[boot[99], boot[3899]],
                    families={f: gmean(group) for f, group in zip(FAMILIES, strata)},
                    regressions_over_25pct={k: v for k, v in ratios.items() if v > 1.25})
            aggregates[str(count)] = comparisons
    long = aggregates.get('1024', {}).get('auto_vs_raw', {})
    target_pass = bool(complete and long and long['ratio'] <= .90 and long['ci95'][1] < 1
        and max(long['families'].values()) <= 1.10 and aggregates['1']['auto_vs_raw']['ratio'] <= 1.10)
    return dict(schema=SCHEMA, classification=manifest['classification'], functional_complete=complete,
        development_target_pass=target_pass, expected_calls=expected_calls, counts=dict(counts),
        selected_engines=dict(selections), independently_replayed_outputs=checked_outputs,
        independently_replayed_decisions=checked_decisions, aggregates=aggregates,
        cases=case_results, failures=failures, candidate_commit=manifest['candidate_commit'],
        baseline_commit=manifest['baseline_commit'], manifest_sha256=manifest_hash)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'audit', '_worker'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', type=int)
    args = parser.parse_args()
    root = args.out.resolve()
    if args.command == 'run':
        execute(root)
    elif args.command == '_worker':
        worker(root, args.case)
    else:
        result = summarize(root)
        put(root/'audit.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
        if not result['functional_complete']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
