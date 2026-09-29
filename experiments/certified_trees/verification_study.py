"""Fresh-process, exact-output comparison of two complete source verifiers.

No fitting or warm prediction benchmark. Both reconstruct all certificate bytes;
the session phase additionally loads the same native and official fallback code.
All predetermined attempts are retained; timings do not establish generality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

TASKS = ('letter', 'pendigits', 'satellite', 'optdigits')
SEED = 2026092953
REPETITIONS = 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def own_memory():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmHWM', 'VmRSS'):
            values[key + '_KiB'] = int(value.split()[0])
    return values


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def worker(args):
    # -I prevents checkout/environment substitution; the explicitly named source
    # root is inserted once, as the experiment's recorded implementation.
    sys.path.insert(0, str(args.source.resolve()))
    from experiments.certified_trees.packed import verify
    from experiments.certified_trees.session import VerifiedCompact, TreeSession
    import struct
    model = args.evidence / 'models' / args.task
    fit = json.loads((model / 'FIT.json').read_text())
    packed = args.evidence / 'compiled' / f'{args.task}-{args.bits}.sct'
    start = time.perf_counter_ns()
    cpu = time.process_time_ns()
    session = None
    if args.phase == 'official':
        session = TreeSession(args.library, official_model=model / 'model.cbm',
            official_library=args.official, features=fit['features'],
            maximum=fit['maximum'], classes=len(fit['classes']))
        identity = {}  # Model/source hashes below are outside the official-load timer.
    else:
        source_bytes = (model / 'model.json').read_bytes()
        compact_bytes = packed.read_bytes()
        if args.phase == 'verify':
            identity = verify(source_bytes, compact_bytes, backend=args.backend)
        else:
            proof = VerifiedCompact(source_bytes, compact_bytes, backend=args.backend)
            identity = dict(proof.info)
            session = TreeSession(args.library, first=proof,
                official_model=model / 'model.cbm', official_library=args.official)
    elapsed = time.perf_counter_ns() - start
    cpu_ns = time.process_time_ns() - cpu
    memory = own_memory()  # Own kernel high-water statistic, not inherited rusage.
    if session is not None:
        try:
            row = bytearray((args.evidence / 'evaluation' / args.task / 'input.u8').read_bytes()[:fit['features']])
            expected = struct.unpack('<i', (args.evidence / 'evaluation' / args.task / 'indices.i32').read_bytes()[:4])[0]
            result = session.predict_buffer(row, fallback=True)
            if result != [expected]:
                raise ValueError('source output mismatch after preparation')
        finally:
            session.close()
    if {'numpy', 'scipy', 'sklearn', 'catboost', 'torch', 'pandas'} & sys.modules.keys():
        raise ValueError('unexpected numerical Python import')
    return {'status': 'PASS', 'task': args.task, 'phase': args.phase,
            'backend': args.backend, 'bits': args.bits,
            'elapsed_ns': elapsed, 'cpu_ns': cpu_ns, **memory,
            'source_sha256': sha(model / 'model.json'), 'packed_sha256': sha(packed),
            'oracle_sha256': identity.get('oracle_sha256'),
            'library_sha256': sha(args.library), 'official_sha256': sha(args.official),
            'python': sys.version, 'affinity': sorted(os.sched_getaffinity(0)),
            'scope': 'fresh-process parsing + exact source reconstruction; session also loads native/original fallback; one prediction checked after timing; no fitting'}


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = [(task, phase, bits, backend, repeat) for task in TASKS
            for phase in ('verify', 'session') for bits in (8, 16)
            for backend in ('reference', 'dyadic') for repeat in range(REPETITIONS)]
    jobs += [(task, 'official', 16, 'official', repeat) for task in TASKS
             for repeat in range(REPETITIONS)]
    random.Random(SEED).shuffle(jobs)
    core = min(os.sched_getaffinity(0))
    source_paths = ['experiments/certified_trees/' + name for name in
                    ('packed.py', 'dyadic.py', 'session.py', 'verification_study.py',
                     'reference/certificate_oracle.py', 'runtime.cpp')]
    protocol = {'seed': SEED, 'jobs': jobs, 'repetitions': REPETITIONS, 'cpu_core': core,
                'sources': {p: sha(args.source / p) for p in source_paths},
                'models': {task: {name: sha(args.evidence / 'models' / task / name)
                                 for name in ('model.json', 'model.cbm', 'FIT.json')}
                           for task in TASKS},
                'compiled': {f'{task}-{bits}.sct': sha(args.evidence / 'compiled' / f'{task}-{bits}.sct')
                             for task in TASKS for bits in (8, 16)},
                'library_sha256': sha(args.library), 'official_sha256': sha(args.official),
                'cpu': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                             if line.startswith('model name')), 'unknown'),
                'source_root': str(args.source.resolve()), 'model_root': str(args.evidence.resolve()),
                'scope': 'all 108 fixed attempts, one process at a time; timings include file reads, no proof cache; filesystem cache may be warm'}
    write_new(args.out / 'PROTOCOL.json', protocol)
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (args.out / 'observations.jsonl').open('x', encoding='utf-8') as log:
        for number, (task, phase, bits, backend, repeat) in enumerate(jobs):
            command = ['taskset', '-c', str(core), sys.executable, '-I', '-S', str(Path(__file__).resolve()),
                       'worker', '--source', str(args.source.resolve()), '--evidence', str(args.evidence.resolve()),
                       '--library', str(args.library.resolve()), '--official', str(args.official.resolve()),
                       '--task', task, '--phase', phase, '--bits', str(bits), '--backend', backend]
            start = time.perf_counter_ns()
            process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            wall = time.perf_counter_ns() - start
            write_new(args.out / f'process-{number:03d}.json',
                      {'command': command, 'returncode': process.returncode,
                       'stdout': process.stdout, 'stderr': process.stderr, 'wall_ns': wall})
            record = {'job': number, 'task': task, 'phase': phase, 'bits': bits,
                      'backend': backend, 'repeat': repeat, 'process_wall_ns': wall}
            if process.returncode:
                record.update(status='ERROR', returncode=process.returncode)
            else:
                record.update(json.loads(process.stdout))
            log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
            print(number + 1, '/', len(jobs), task, phase, bits, backend,
                  record['status'], round(record.get('elapsed_ns', 0) / 1e6, 2), 'ms', flush=True)
    return {'status': 'COMPLETE', 'attempts': len(jobs)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['run', 'worker'])
    for field in ('source', 'evidence', 'library', 'official', 'out'):
        parser.add_argument('--' + field, type=Path)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--phase', choices=['verify', 'session', 'official'])
    parser.add_argument('--bits', type=int, choices=[8, 16])
    parser.add_argument('--backend', choices=['reference', 'dyadic', 'official'])
    args = parser.parse_args()
    print(json.dumps((run if args.operation == 'run' else worker)(args), allow_nan=False))
