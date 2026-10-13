"""Process-isolated, resumable development comparisons with retained failures.

Run `python -m experiments.compiled_query.study --help`. No old WAP freeze,
threshold, claim, or raw evidence is modified by this separately named study.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time
import traceback

from .cases import canonical, digest, export_wap, load_case, synthetic_cases, write_case
from .sessions import ARMS, preload, run_session

ROOT = Path(__file__).resolve().parents[2]
ORDERS = ('forward', 'reverse', 'shuffle')


@dataclass(frozen=True)
class Config:
    arms: tuple[str, ...] = ARMS
    prefixes: tuple[int, ...] = (1, 32, 128)
    repeats: int = 3
    seed: int = 20261013
    wall_seconds: float = 30.0
    address_space_bytes: int = 2 << 30
    native_bytes: int = 512 << 20
    bdd_nodes: int = 2_000_000

    def validate(self):
        if not self.arms or len(set(self.arms)) != len(self.arms) or any(a not in ARMS for a in self.arms):
            raise ValueError('invalid/duplicate arms')
        if (not self.prefixes or len(set(self.prefixes)) != len(self.prefixes)
                or any(type(n) is not int or n < 1 for n in self.prefixes)):
            raise ValueError('invalid/duplicate prefixes')
        if type(self.repeats) is not int or not 1 <= self.repeats <= 1000:
            raise ValueError('repeats outside [1, 1000]')
        if type(self.seed) is not int or not 0 <= self.seed < 1 << 64:
            raise ValueError('seed must be uint64')
        if (type(self.wall_seconds) not in (int, float) or not math.isfinite(self.wall_seconds)
                or self.wall_seconds <= 0):
            raise ValueError('invalid wall deadline')
        for value in (self.address_space_bytes, self.native_bytes, self.bdd_nodes):
            if type(value) is not int or value <= 0:
                raise ValueError('resource limits must be positive exact integers')
        if self.native_bytes < 1 << 20 or self.address_space_bytes < 64 << 20:
            raise ValueError('memory limits below supported minimum')


def write_once(path: Path, payload: bytes):
    """Publish complete bytes without replacing an existing record."""
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temp.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def append_event(root: Path, value: dict):
    with (root / 'events.jsonl').open('ab') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def locked(root: Path):
    import fcntl
    with (root / '.controller.lock').open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another controller holds this study') from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def sources() -> dict:
    paths = sorted((ROOT / 'experiments/compiled_query').glob('*.py'))
    paths += [ROOT / 'spectra/cnf/quotient_query.py', ROOT / 'spectra/_native/quotient_query.cpp']
    return {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths}



def execution_identity() -> dict:
    """Hash installed direct backend code without importing native libraries."""
    dependencies = {}
    for name in ('dd', 'python-sat'):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
            continue
        files = {}
        for item in sorted(distribution.files or (), key=str):
            if str(item).endswith(('.py', '.so', '.pyd')):
                path = distribution.locate_file(item)
                files[str(item)] = digest(path.read_bytes())
        dependencies[name] = dict(version=distribution.version, files=files)
    cpu = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    model = next((line.split(':', 1)[1].strip() for line in cpu.splitlines()
                  if line.startswith('model name')), 'unknown')
    return dict(python=sys.version, executable_sha256=digest(Path(sys.executable).read_bytes()),
                platform=platform.platform(), cpu=model, affinity=sorted(os.sched_getaffinity(0)),
                dependencies=dependencies)


def schedule(cases: list[dict], config: Config) -> list[dict]:
    config.validate()
    groups = [(i, count, rep) for i, case in enumerate(cases)
              for count in config.prefixes for rep in range(config.repeats)]
    if any(max(config.prefixes) > case['query_count'] for case in cases):
        raise ValueError('requested prefix exceeds a case query bank')
    rng = random.Random(config.seed)
    rng.shuffle(groups)
    jobs = []
    for case_index, count, rep in groups:
        arms = list(config.arms)
        rng.shuffle(arms)
        # Identical ordering/seed for every arm in a paired group.
        order_seed = rng.getrandbits(64)
        for arm in arms:
            jobs.append(dict(id=f'{len(jobs):06d}', case_index=case_index,
                             case_sha256=cases[case_index]['inputs'][str(count)]['sha256'], arm=arm,
                             count=count, repeat=rep, order=ORDERS[rep % len(ORDERS)],
                             order_seed=order_seed))
    return jobs


def create_study(case_paths: list[Path], library: Path, output: Path, config: Config) -> dict:
    config.validate()
    if not case_paths:
        raise ValueError('no case files')
    cases = []
    snapshots = []
    for index, path in enumerate(case_paths):
        case, _ = load_case(path)
        payload = case.payload()
        snapshots.append((f'cases/{index:04d}.json', payload))
        inputs = {}
        for count in config.prefixes:
            if count > len(case.queries):
                raise ValueError('requested prefix exceeds a case query bank')
            prefix_payload = replace(case, queries=case.queries[:count]).payload()
            filename = f'cases/{index:04d}-q{count}.json'
            inputs[str(count)] = dict(file=filename, sha256=digest(prefix_payload))
            snapshots.append((filename, prefix_payload))
        cases.append(dict(file=f'cases/{index:04d}.json', sha256=digest(payload),
                          name=case.name, cluster=case.cluster, provenance=case.provenance,
                          query_count=len(case.queries), n=case.problem.n, inputs=inputs))
    if len({c['name'] for c in cases}) != len(cases):
        raise ValueError('case names must be unique')
    if len({c['sha256'] for c in cases}) != len(cases):
        raise ValueError('duplicate cases would inflate replication counts')
    jobs = schedule(cases, config)
    library = Path(library)
    native = library.read_bytes()
    build_path = library.parent / 'build.json'
    build = json.loads(build_path.read_text())
    if (build['library_sha256'] != digest(native)
            or build['source_sha256'] != digest((ROOT / 'spectra/_native/quotient_query.cpp').read_bytes())):
        raise ValueError('native build receipt does not match source and library')
    manifest = dict(schema='spectra.compiled_query.study.v1', classification='development-only',
                    config=asdict(config), cases=cases, jobs=jobs, sources=sources(),
                    execution_identity=execution_identity(),
                    library='runtime/' + library.name, library_sha256=digest(native),
                    build=build, endpoint='validated JSON -> full checked SCQ1 output + SHA256 + disposal',
                    external_replication=False, rss_is_not_hard_capped=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'cases').mkdir()
    (output / 'runtime').mkdir()
    (output / 'sessions').mkdir()
    for filename, payload in snapshots:
        write_once(output / filename, payload)
    write_once(output / manifest['library'], native)
    write_once(output / 'manifest.json', canonical(manifest))
    write_once(output / 'manifest.sha256', (digest(canonical(manifest)) + '\n').encode())
    return manifest


def read_manifest(root: Path, *, check_sources: bool = True) -> dict:
    payload = (root / 'manifest.json').read_bytes()
    if digest(payload) != (root / 'manifest.sha256').read_text().strip():
        raise ValueError('manifest changed')
    manifest = json.loads(payload)
    if check_sources and manifest['sources'] != sources():
        raise ValueError('benchmark source changed; start a new study')
    if check_sources and manifest['execution_identity'] != execution_identity():
        raise ValueError('interpreter, host or backend code changed; start a new study')
    if digest((root / manifest['library']).read_bytes()) != manifest['library_sha256']:
        raise ValueError('native library changed')
    for case in manifest['cases']:
        if digest((root / case['file']).read_bytes()) != case['sha256']:
            raise ValueError('case snapshot changed')
        for prefix in case['inputs'].values():
            if digest((root / prefix['file']).read_bytes()) != prefix['sha256']:
                raise ValueError('prefix snapshot changed')
    config = Config(**{**manifest['config'], 'arms': tuple(manifest['config']['arms']),
                       'prefixes': tuple(manifest['config']['prefixes'])})
    if schedule(manifest['cases'], config) != manifest['jobs']:
        raise ValueError('schedule is not the committed paired schedule')
    return manifest


def environment() -> dict:
    import resource
    versions = {}
    for name in ('dd', 'python-sat', 'networkx', 'setuptools'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=versions,
                affinity=sorted(os.sched_getaffinity(0)),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def worker(root: Path, job_id: str):
    import resource
    manifest = read_manifest(root)
    job = next(j for j in manifest['jobs'] if j['id'] == job_id)
    config = manifest['config']
    limit = config['address_space_bytes']
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    cpus = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(cpus)})
    destination = root / 'sessions' / job_id
    try:
        preload(job['arm'])  # Process/import wall is retained by the controller.
        from spectra.cnf.quotient_query import QuotientRuntime
        runtime = QuotientRuntime(root / manifest['library'])
        record, output = run_session(root / manifest['cases'][job['case_index']]['inputs'][str(job['count'])]['file'],
                                     runtime, job['arm'], count=job['count'], order=job['order'],
                                     seed=job['order_seed'], max_bytes=config['native_bytes'],
                                     max_nodes=config['bdd_nodes'])
        write_once(destination / 'outputs.bin', output)
    except Exception as exc:
        from .bdd import BackendUnavailable, CompilationLimit
        status = ('RESOURCE_LIMIT' if isinstance(exc, (MemoryError, CompilationLimit)) else
                  'UNAVAILABLE' if isinstance(exc, (ImportError, BackendUnavailable)) else 'ERROR')
        record = dict(status=status, error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    record.update(job=job, environment=environment())
    write_once(destination / 'record.json', canonical(record))


def terminal_receipt(destination: Path, job: dict, returncode, cold_ns, status=None):
    record_path = destination / 'record.json'
    record = json.loads(record_path.read_bytes()) if record_path.exists() else None
    if record is not None and record.get('job') != job:
        raise ValueError('worker receipt belongs to another job')
    actual_status = status or (record['status'] if record and returncode == 0 else 'CRASH')
    result = dict(job=job, status=actual_status, returncode=returncode,
                  cold_wall_ns=cold_ns,
                  record_sha256=digest(record_path.read_bytes()) if record else None)
    write_once(destination / 'controller.json', canonical(result))
    return result


def run_study(root: Path, max_new: int | None = None) -> list[dict]:
    if platform.system() != 'Linux':
        raise NotImplementedError('hard process isolation currently requires Linux')
    if max_new is not None and (type(max_new) is not int or max_new < 0):
        raise ValueError('max_new must be a nonnegative exact integer')
    completed = []
    with locked(root):
        manifest = read_manifest(root)
        for job in manifest['jobs']:
            destination = root / 'sessions' / job['id']
            terminal = destination / 'controller.json'
            if terminal.exists():
                receipt = json.loads(terminal.read_bytes())
                if receipt['job'] != job:
                    raise ValueError('terminal job differs')
                if receipt['record_sha256'] is not None:
                    if digest((destination / 'record.json').read_bytes()) != receipt['record_sha256']:
                        raise ValueError('worker record changed')
                    record = json.loads((destination / 'record.json').read_bytes())
                    if record.get('status') in ('COMPLETE', 'INCOMPLETE'):
                        payload = (destination / 'outputs.bin').read_bytes()
                        if digest(payload) != record['output_sha256']:
                            raise ValueError('worker output changed')
                continue  # No retry-until-success selection.
            if max_new is not None and len(completed) >= max_new:
                break
            if destination.exists():
                receipt = terminal_receipt(destination, job, None, None, 'INTERRUPTED')
            else:
                destination.mkdir()
                append_event(root, dict(event='STARTED', job=job, time_ns=time.time_ns()))
                command = [sys.executable, '-m', 'experiments.compiled_query.study',
                           '_worker', '--output', str(root.resolve()), '--job', job['id']]
                tick = time.perf_counter_ns()
                status = None
                with (destination / 'stdout.log').open('wb') as stdout, (destination / 'stderr.log').open('wb') as stderr:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                               start_new_session=True,
                                               env={**os.environ, 'PYTHONHASHSEED': '0',
                                                    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
                    try:
                        returncode = process.wait(timeout=manifest['config']['wall_seconds'])
                    except subprocess.TimeoutExpired:
                        status = 'TIMEOUT'
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        returncode = process.wait()
                    except BaseException:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                        raise
                receipt = terminal_receipt(destination, job, returncode,
                                           time.perf_counter_ns()-tick, status)
            append_event(root, dict(event='TERMINAL', **receipt))
            completed.append(receipt)
            print(json.dumps(dict(job=job['id'], arm=job['arm'], status=receipt['status'])), flush=True)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    synth = sub.add_parser('synthetic')
    synth.add_argument('--output', type=Path, required=True)
    synth.add_argument('--n', type=int, default=96)
    synth.add_argument('--queries', type=int, default=128)
    wap = sub.add_parser('export-wap')
    wap.add_argument('--graphs', type=Path, required=True)
    wap.add_argument('--output', type=Path, required=True)
    new = sub.add_parser('new')
    new.add_argument('--cases', type=Path, required=True)
    new.add_argument('--library', type=Path, required=True)
    new.add_argument('--output', type=Path, required=True)
    new.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    new.add_argument('--prefixes', nargs='+', type=int, default=[1, 32, 128])
    new.add_argument('--repeats', type=int, default=3)
    new.add_argument('--deadline', type=float, default=30)
    new.add_argument('--bdd-nodes', type=int, default=2_000_000)
    run = sub.add_parser('run')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--max-new', type=int)
    work = sub.add_parser('_worker')
    work.add_argument('--output', type=Path, required=True)
    work.add_argument('--job', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'synthetic':
        args.output.mkdir(parents=True, exist_ok=False)
        for case in synthetic_cases(args.queries, args.n):
            write_case(args.output / (case.name + '.json'), case)
    elif args.command == 'export-wap':
        export_wap(args.graphs, args.output)
    elif args.command == 'new':
        paths = sorted(p for p in args.cases.glob('*.json') if not p.name.endswith('.generation.json'))
        create_study(paths, args.library, args.output,
                     Config(arms=tuple(args.arms), prefixes=tuple(args.prefixes), repeats=args.repeats,
                            wall_seconds=args.deadline, bdd_nodes=args.bdd_nodes))
    elif args.command == 'run':
        run_study(args.output, args.max_new)
    elif args.command == '_worker':
        worker(args.output, args.job)
    else:
        from .audit import audit_study
        result = audit_study(args.output)
        print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
