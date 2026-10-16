"""Prospectively specified synthetic structural panel; never a deployment claim.

Generate only after publishing TRANSFER_PROTOCOL_20261015.md and source identity.
Each graph runs in an isolated process. All failures and paired timing records
are retained. Repeated answers are stored by SHA256, not counted as new problems.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
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

from .binary_list import BinaryListProblem
from .cases import Case, canonical, digest
from .exact import ExactRuntime, pack_session_packet
from .proof_replay import OriginalReplay
from .sessions import decode_certified_output

FAMILIES = ('random_tree', 'percolated_grid', 'geometric', 'preferential',
            'rewired_ring', 'communities', 'percolated_cube', 'random_bipartite')
COUNTS = (1, 63, 64, 65, 1024)
ARMS = ('prior-route', 'raw', 'scc', 'route')
REPEATS = 9
ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'spectra.prospective_structural_panel.v1'
PRIOR_SOURCE_SHA256 = '2de36dff7b0313108408452b9409c8568daf25d74780df119c5b5fcf01bbe8ec'


def seed_for(label: str) -> int:
    return int.from_bytes(hashlib.sha256(('SPECTRA-20261015-v1/' + label).encode()).digest()[:8], 'big')


def put(path: Path, value: object) -> None:
    """Immutable small receipt; never overwrite a failed or successful result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value))


def topology(family: str, n: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    if family not in FAMILIES or n < 8 or n & (n-1):
        raise ValueError('known family and power-of-two n >= 8 required')
    edges: set[tuple[int, int]] = set()
    def add(u, v):
        if u != v:
            edges.add((min(u, v), max(u, v)))
    if family == 'random_tree':
        for v in range(1, n):
            add(v, rng.randrange(v))
    elif family == 'percolated_grid':
        width = 1 << ((n.bit_length()-1)//2)
        for v in range(n):
            if v % width and rng.random() < .88:
                add(v, v-1)
            if v >= width and rng.random() < .88:
                add(v, v-width)
            if v >= width and v % width and rng.random() < .08:
                add(v, v-width-1)
    elif family == 'geometric':
        points = [(rng.random(), rng.random()) for _ in range(n)]
        radius2 = 6 / (math.pi*n)
        for u in range(n):
            for v in range(u):
                if sum((a-b)**2 for a, b in zip(points[u], points[v])) <= radius2:
                    add(u, v)
    elif family == 'preferential':
        edges.update(((0, 1), (0, 2), (1, 2)))
        urn = [0, 0, 1, 1, 2, 2]
        for v in range(3, n):
            targets = set()
            while len(targets) < 2:
                targets.add(rng.choice(urn))
            for u in sorted(targets):
                add(u, v)
                urn.extend((u, v))
    elif family == 'rewired_ring':
        for v in range(n):
            for distance in (1, 2):
                u = rng.randrange(n) if rng.random() < .15 else (v+distance) % n
                add(v, u)
    elif family == 'communities':
        for u in range(n):
            for v in range(u):
                probability = .07 if u//(n//4) == v//(n//4) else .005
                if rng.random() < probability:
                    add(u, v)
    elif family == 'percolated_cube':
        for v in range(n):
            for bit in range(n.bit_length()-1):
                u = v ^ (1 << bit)
                if u > v and rng.random() < .88:
                    add(u, v)
    else:
        for u in range(n//2):
            for v in range(n//2, n):
                if rng.random() < 8/n:
                    add(u, v)
    permutation = list(range(n))
    rng.shuffle(permutation)
    return tuple(sorted((min(permutation[u], permutation[v]), max(permutation[u], permutation[v]))
                        for u, v in edges))


def make_case(family: str, replica: int, *, count: int = 1024, n: int | None = None) -> Case:
    """No solver calls, filtering, repair, rejection, or answer-dependent loops."""
    if type(replica) is not int or not 0 <= replica < 6 or type(count) is not int or not 1 <= count <= 1024:
        raise ValueError('replica/count outside the fixed panel')
    n = n if n is not None else (128 if replica < 3 else 512)
    rng = random.Random(seed_for(f'{family}/{replica}'))
    edges = topology(family, n, rng)
    regime = replica % 3
    k = 2 if regime == 0 else 8
    masks = []
    for v in range(n):
        if regime == 0:
            masks.append(3)
        else:
            colors = rng.sample(range(k), 1 if regime == 2 and v % 7 == 0 else 2)
            masks.append(sum(1 << c for c in colors))
    p = BinaryListProblem.create(n, k, edges, masks)
    queries = []
    for i in range(count):
        if (i+1) % 29 == 0:
            queries.append(())
            continue
        width = (1, 2, 4, 8)[i % 4]
        row = []
        for v in rng.sample(range(n), min(width, n)):
            colors = [c for c in range(k) if masks[v] >> c & 1]
            allowed = 1 << rng.choice(colors)
            if (i+1) % 11 == 0:
                allowed = masks[v]
            if (i+1) % 23 == 0:
                allowed |= 1 << 63  # irrelevant bits cannot add original colours
            row.append((v, allowed))
        if (i+1) % 17 == 0:
            row.append((row[0][0], masks[row[0][0]] ^ (row[0][1] & masks[row[0][0]])))
        if (i+1) % 31 == 0:
            row.append((row[0][0], 0))
        queries.append(tuple(row))
    name = f'{family}-r{replica}-n{n}-d{regime}'
    return Case(name, name, 'prospective synthetic structural panel v1; no selection by outcomes', p, tuple(queries))


def source_receipt() -> dict:
    files = sorted(Path(__file__).parent.glob('*.py')) + sorted(Path(__file__).parent.glob('*.cpp'))
    return {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files}


def freeze(destination: Path, protocol_commit: str) -> dict:
    if len(protocol_commit) != 40 or any(c not in '0123456789abcdef' for c in protocol_commit):
        raise ValueError('an exact public protocol commit is required')
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for family in FAMILIES:
        for replica in range(6):
            case = make_case(family, replica)
            payload = case.payload()
            (destination / (case.name+'.json')).write_bytes(payload)
            records.append(dict(name=case.name, family=family, replica=replica, n=case.problem.n,
                                case_sha256=digest(payload), edges=len(case.problem.edges),
                                domain_regime=replica%3))
    manifest = dict(schema=SCHEMA, classification='prospective synthetic structural panel, not external replication',
                    protocol_commit=protocol_commit, sources=source_receipt(), cases=records,
                    counts=COUNTS, arms=ARMS, repeats=REPEATS, warmups=2, deadline_seconds=120,
                    native_bytes=512 << 20, address_space_bytes=2 << 30,
                    thresholds=dict(route_raw_long=.90, upper95=1., family_max=1.10, short_max=1.10),
                    generation_complete_time_ns=time.time_ns())
    put(destination/'manifest.json', manifest)
    (destination/'manifest.sha256').write_text(digest(canonical(manifest))+'\n')
    return manifest


def load_manifest(root: Path, check_sources: bool = True) -> dict:
    raw = (root/'manifest.json').read_bytes()
    if digest(raw) != (root/'manifest.sha256').read_text().strip():
        raise ValueError('panel manifest modified')
    manifest = json.loads(raw)
    if manifest['schema'] != SCHEMA or (check_sources and manifest['sources'] != source_receipt()):
        raise ValueError('panel execution source/schema modified')
    for record in manifest['cases']:
        if digest((root/(record['name']+'.json')).read_bytes()) != record['case_sha256']:
            raise ValueError('panel input modified')
    return manifest


def worker(root: Path, result: Path, case_index: int, current: Path, prior: Path):
    import resource
    manifest = load_manifest(root)
    resource.setrlimit(resource.RLIMIT_AS, (manifest['address_space_bytes'],)*2)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    item = manifest['cases'][case_index]
    case = Case.decode((root/(item['name']+'.json')).read_bytes())
    runtime, historical = ExactRuntime(current), ExactRuntime(prior)
    rng = random.Random(seed_for(case.name+'/schedule'))
    records = []
    packet_inventory = {}
    outputs = result/'outputs'
    outputs.mkdir(parents=True)
    result.joinpath('packets').mkdir()
    for count in manifest['counts']:
        for repeat in range(manifest['repeats']):
            order = list(range(count))
            if repeat % 3 == 1:
                order.reverse()
            elif repeat % 3 == 2:
                random.Random(seed_for(f'{case.name}/{count}/{repeat}/order')).shuffle(order)
            rows = tuple(case.queries[i] for i in order)
            tick = time.perf_counter_ns()
            packet = pack_session_packet(case.problem, rows)
            packing_ns = time.perf_counter_ns()-tick
            packet_hash = digest(packet)
            if packet_hash not in packet_inventory:
                result.joinpath('packets', packet_hash+'.sbc1').write_bytes(packet)
                packet_inventory[packet_hash] = dict(count=count, order=order)
            arms = list(manifest['arms'])
            rng.shuffle(arms)
            for arm in arms:
                rt, mode = (historical, 'route') if arm == 'prior-route' else (runtime, arm)
                try:
                    if repeat == 0:
                        for _ in range(manifest['warmups']):
                            rt.solve_session_packet(packet, mode=mode, max_bytes=manifest['native_bytes'])
                    tick = time.perf_counter_ns()
                    assert digest(packet) == packet_hash
                    answer = rt.solve_session_packet(packet, mode=mode, max_bytes=manifest['native_bytes'])
                    output_hash = digest(answer.output)
                    elapsed = time.perf_counter_ns()-tick
                    target = outputs/(output_hash+'.scq2')
                    if not target.exists():
                        target.write_bytes(answer.output)
                    record = dict(status='COMPLETE', arm=arm, count=count, repeat=repeat,
                                  complete_ns=elapsed, packing_ns=packing_ns, packet_sha256=packet_hash,
                                  output_sha256=output_hash, output_bytes=len(answer.output),
                                  sat=answer.sat_count, unsat=answer.unsat_count, proof_bytes=answer.proof_bytes,
                                  info=answer.info)
                except Exception as exc:
                    record = dict(status='ERROR', arm=arm, count=count, repeat=repeat,
                                  packet_sha256=packet_hash, error=repr(exc), traceback=traceback.format_exc())
                if record['status'] == 'COMPLETE':
                    del answer
                records.append(record)
                # Keep every measurement as it finishes, even if the worker is killed later.
                with (result/'records.jsonl').open('ab') as stream:
                    stream.write(canonical(record))
                    stream.flush()
    put(result/'packets.json', packet_inventory)
    put(result/'worker.json', dict(case=case.name, records=len(records),
        records_sha256=digest((result/'records.jsonl').read_bytes()),
        packets_sha256=digest((result/'packets.json').read_bytes()), python=sys.version,
        platform=platform.platform(), machine=platform.machine(), affinity=sorted(os.sched_getaffinity(0)),
        current_library_sha256=digest(current.read_bytes()), prior_library_sha256=digest(prior.read_bytes()),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def run(root: Path, current: Path, prior: Path):
    manifest = load_manifest(root)
    if current is None or prior is None or not current.is_file() or not prior.is_file():
        raise ValueError('two explicit native libraries required')
    builds = [json.loads(p.with_name('build.json').read_bytes()) for p in (current, prior)]
    expected_sources = [digest(Path(__file__).with_name('exact_native.cpp').read_bytes()), PRIOR_SOURCE_SHA256]
    for library, build, source_hash in zip((current, prior), builds, expected_sources):
        if build['source_sha256'] != source_hash or build['library_sha256'] != digest(library.read_bytes()):
            raise ValueError('explicit native baseline/source receipt differs')
    if builds[0]['command'][:-3] != builds[1]['command'][:-3] or builds[0]['compiler'] != builds[1]['compiler']:
        raise ValueError('historical and current compiler/options differ')
    identity = dict(builds=builds, manifest_sha256=digest((root/'manifest.json').read_bytes()),
                    current_library_sha256=digest(current.read_bytes()), prior_library_sha256=digest(prior.read_bytes()),
                    python=sys.version, platform=platform.platform())
    put(root/'execution.json', identity)
    for index, case in enumerate(manifest['cases']):
        destination = root/'runs'/case['name']
        destination.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-m', 'experiments.compiled_query.transfer_panel', '_worker',
                   '--root', str(root.resolve()), '--current', str(current.resolve()),
                   '--prior', str(prior.resolve()), '--case', str(index)]
        tick = time.perf_counter_ns()
        status = 'COMPLETE'
        with (destination/'stdout.log').open('wb') as out, (destination/'stderr.log').open('wb') as err:
            p = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True,
                                 env={**os.environ, 'PYTHONHASHSEED':'0', 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'})
            try:
                code = p.wait(timeout=manifest['deadline_seconds'])
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(p.pid, signal.SIGKILL)
                code = p.wait()
                status = 'TIMEOUT'
            except BaseException:
                import signal
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                raise
        if code and status == 'COMPLETE':
            status = 'CRASH'
        put(destination/'controller.json', dict(status=status, returncode=code, cold_wall_ns=time.perf_counter_ns()-tick,
            worker_sha256=digest((destination/'worker.json').read_bytes()) if (destination/'worker.json').exists() else None))
        print(case['name'], status, flush=True)


def gmean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def summarize(root: Path) -> dict:
    manifest = load_manifest(root, check_sources=False)
    execution = json.loads((root/'execution.json').read_bytes())
    if execution['manifest_sha256'] != digest((root/'manifest.json').read_bytes()):
        raise ValueError('execution/manifest identity differs')
    counts = Counter()
    cells = defaultdict(list)
    checked = set()
    semantic = Counter()
    case_rows = []
    paired_decisions = {}
    for item in manifest['cases']:
        case = Case.decode((root/(item['name']+'.json')).read_bytes())
        observer = OriginalReplay(case.problem)
        folder = root/'runs'/case.name
        controller = json.loads((folder/'controller.json').read_bytes()) if (folder/'controller.json').exists() else {'status':'MISSING'}
        counts['workers_'+controller['status']] += 1
        records = [json.loads(line) for line in (folder/'records.jsonl').read_bytes().splitlines()] if (folder/'records.jsonl').exists() else []
        if controller['status'] == 'COMPLETE':
            raw_worker = (folder/'worker.json').read_bytes()
            worker_record = json.loads(raw_worker)
            if (digest(raw_worker) != controller['worker_sha256'] or
                    digest((folder/'records.jsonl').read_bytes()) != worker_record['records_sha256'] or
                    digest((folder/'packets.json').read_bytes()) != worker_record['packets_sha256']):
                raise ValueError('worker/timing receipt modified')
            for key in ('current_library_sha256', 'prior_library_sha256'):
                if worker_record[key] != execution[key]:
                    raise ValueError('worker used another native library')
        expected = {(a, c, r) for a in manifest['arms'] for c in manifest['counts'] for r in range(manifest['repeats'])}
        seen = set()
        packet_meta = json.loads((folder/'packets.json').read_bytes()) if (folder/'packets.json').exists() else {}
        for row in records:
            key = row['arm'], row['count'], row['repeat']
            if key not in expected or key in seen:
                raise ValueError('unexpected or duplicate measurement')
            seen.add(key)
            counts[row['status']] += 1
            if row['status'] != 'COMPLETE':
                continue
            if type(row['complete_ns']) is not int or row['complete_ns'] <= 0:
                raise ValueError('invalid duration')
            cells[(case.name, row['count'], row['arm'])].append(row['complete_ns'])
            payload = (folder/'outputs'/(row['output_sha256']+'.scq2')).read_bytes()
            packet = (folder/'packets'/(row['packet_sha256']+'.sbc1')).read_bytes()
            if digest(payload) != row['output_sha256'] or digest(packet) != row['packet_sha256']:
                raise ValueError('output or input hash mismatch')
            order = packet_meta[row['packet_sha256']]['order']
            if len(order) != row['count'] or sorted(order) != list(range(row['count'])):
                raise ValueError('request ordering mismatch')
            query_rows = tuple(case.queries[i] for i in order)
            if packet != pack_session_packet(case.problem, query_rows):
                raise ValueError('prepared packet does not encode original requests')
            verified_key = case.name, row['packet_sha256'], row['output_sha256']
            answers = decode_certified_output(payload, case.problem.n, row['count'])
            if len(payload) != row['output_bytes'] or sum(len(p) for _, _, p in answers) != row['proof_bytes']:
                raise ValueError('output/proof byte accounting mismatch')
            if sum(code for code, _, _ in answers) != row['sat'] or len(answers)-row['sat'] != row['unsat']:
                raise ValueError('outcome accounting mismatch')
            decisions = tuple(code for code, _, _ in answers)
            decision_key = case.name, row['packet_sha256']
            if decision_key in paired_decisions and paired_decisions[decision_key] != decisions:
                raise ValueError('paired native solvers disagree')
            paired_decisions[decision_key] = decisions
            if verified_key not in checked:
                for query, (code, labels, proof) in zip(query_rows, answers):
                    status = 'SAT_VERIFIED' if code else 'UNSAT_VERIFIED'
                    if not observer.verify(query, status, labels if code else b'', proof):
                        raise ValueError('original-input proof/witness verification failed')
                    semantic[status] += 1
                checked.add(verified_key)
        counts['MISSING'] += len(expected-seen)
        item_result = {**item, 'base_satisfiable':observer.feasible(()), 'ratios':{}}
        for count in manifest['counts']:
            medians = {a:statistics.median(cells[(case.name,count,a)]) for a in ARMS
                       if len(cells[(case.name,count,a)]) == manifest['repeats']}
            if len(medians) != len(ARMS):
                item_result['ratios'][str(count)] = None
            else:
                item_result['ratios'][str(count)] = dict(medians_ns=medians,
                    route_raw=medians['route']/medians['raw'], route_scc=medians['route']/medians['scc'],
                    route_prior=medians['route']/medians['prior-route'])
        case_rows.append(item_result)
    complete = counts['COMPLETE'] == len(manifest['cases'])*len(manifest['counts'])*len(manifest['arms'])*manifest['repeats'] and counts['workers_COMPLETE'] == len(manifest['cases'])
    aggregates = {}
    if complete:
        for count in COUNTS:
            key=str(count)
            ratios = [row['ratios'][key]['route_raw'] for row in case_rows]
            family_means = {family:gmean([row['ratios'][key]['route_raw'] for row in case_rows if row['family']==family]) for family in FAMILIES}
            rng = random.Random(seed_for('bootstrap/'+key))
            strata = [[row['ratios'][key]['route_raw'] for row in case_rows if row['family']==family] for family in FAMILIES]
            boot = sorted(gmean([rng.choice(values) for values in strata for _ in values]) for _ in range(4000))
            aggregates[key] = dict(route_raw=gmean(ratios), route_raw_ci95=[boot[99],boot[3899]],
                route_prior=gmean([row['ratios'][key]['route_prior'] for row in case_rows]),
                route_scc=gmean([row['ratios'][key]['route_scc'] for row in case_rows]), families=family_means,
                regressions_over_25pct=sum(x>1.25 for x in ratios))
    speed_pass = bool(complete and aggregates['1024']['route_raw']<=.90 and
                      aggregates['1024']['route_raw_ci95'][1]<1 and
                      max(aggregates['1024']['families'].values())<=1.10 and aggregates['1']['route_raw']<=1.10)
    return dict(schema=SCHEMA, classification=manifest['classification'], functional_complete=complete,
                precommitted_speed_gate_pass=speed_pass, counts=dict(counts), unique_semantically_checked_outputs=len(checked),
                unique_output_decisions_checked=dict(semantic), aggregates=aggregates, cases=case_rows,
                notes=['Bootstrap resamples graph realizations within fixed families, not queries or timed repetitions.',
                       'Full native verification runs on every call; Python replay deduplicates identical packet/output bytes.',
                       'Base-infeasible inputs are retained and marked per graph; aggregate speed does not imply every query is difficult.',
                       'Ready bytes input and output hashing included; file read, packing, process launch and Python replay excluded.',
                       'These are fresh synthetic instances, not an operator workload, real deployment or external replication.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze','run','audit','_worker'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol-commit')
    parser.add_argument('--current',type=Path)
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--case',type=int)
    args=parser.parse_args()
    if args.command=='freeze':
        freeze(args.root,args.protocol_commit)
    elif args.command=='run':
        run(args.root,args.current,args.prior)
    elif args.command=='audit':
        result=summarize(args.root)
        put(args.root/'audit.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    else:
        manifest=load_manifest(args.root)
        destination=args.root/'runs'/manifest['cases'][args.case]['name']
        worker(args.root,destination,args.case,args.current,args.prior)


if __name__=='__main__':
    main()
