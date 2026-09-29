"""Independent standard-library audit of the complete verification timing study.

No compiler, native runtime, model fitting or reference math executes here.
This checks original bytes, process receipts, inventory and derived arithmetic;
it cannot authenticate clocks or a malicious replacement of the entire packet.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct

TASKS = ('letter', 'pendigits', 'satellite', 'optdigits')
HEADER = struct.Struct('<8s5Ii7I32s32s')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loads(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def fail(value):
        raise ValueError('nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=fail)


def read(path):
    return loads(Path(path).read_text(encoding='utf-8'))


def audit(study, source, evidence, library, official):
    study, source, evidence = Path(study), Path(source), Path(evidence)
    protocol = read(study / 'PROTOCOL.json')
    jobs = [(task, phase, bits, backend, repeat) for task in TASKS
            for phase in ('verify', 'session') for bits in (8, 16)
            for backend in ('reference', 'dyadic') for repeat in range(3)]
    jobs += [(task, 'official', 16, 'official', repeat) for task in TASKS for repeat in range(3)]
    random.Random(2026092953).shuffle(jobs)
    require(protocol['seed'] == 2026092953 and protocol['repetitions'] == 3 and
            protocol['jobs'] == [list(j) for j in jobs], 'study schedule changed')
    expected_sources = {'experiments/certified_trees/' + name for name in
                        ('packed.py', 'dyadic.py', 'session.py', 'verification_study.py',
                         'reference/certificate_oracle.py', 'runtime.cpp')}
    require(set(protocol['sources']) == expected_sources, 'source inventory incomplete')
    require(set(protocol['models']) == set(TASKS), 'model inventory incomplete')
    require(set(protocol['compiled']) == {f'{t}-{b}.sct' for t in TASKS for b in (8, 16)},
            'compact inventory incomplete')
    for name, digest in protocol['sources'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'unsafe source path')
        require(sha(source / path) == digest, 'measured source bytes changed: ' + name)
    for task in TASKS:
        require(set(protocol['models'][task]) == {'model.json', 'model.cbm', 'FIT.json'},
                'source model roles incomplete')
        for name, digest in protocol['models'][task].items():
            require(Path(name).name == name, 'unsafe model path')
            require(sha(evidence / 'models' / task / name) == digest, 'frozen model changed')
    oracle_hashes = {}
    for name, digest in protocol['compiled'].items():
        require(Path(name).name == name, 'unsafe compact path')
        path = evidence / 'compiled' / name
        require(sha(path) == digest, 'original compact file changed')
        header = HEADER.unpack_from(path.read_bytes())
        oracle_hashes[name] = header[-1].hex()
    require(sha(library) == protocol['library_sha256'] and
            sha(official) == protocol['official_sha256'], 'native library changed')
    records = [loads(line) for line in (study / 'observations.jsonl').read_text().splitlines()]
    require(len(records) == len(jobs), 'missing study attempt')
    for number, (record, job) in enumerate(zip(records, jobs)):
        task, phase, bits, backend, repeat = job
        require(record['job'] == number and type(record['job']) is int, 'job index changed')
        require(tuple(record[k] for k in ('task', 'phase', 'bits', 'backend', 'repeat')) == job,
                'job order or identity changed')
        require(type(record['bits']) is int and type(record['repeat']) is int, 'invalid job integer')
        require(record['status'] == 'PASS', 'unaccepted attempt')
        process = read(study / f'process-{number:03d}.json')
        require(process['returncode'] == 0 and type(process['returncode']) is int,
                'process did not succeed')
        command = process['command']
        require(type(command) is list and command[:3] == ['taskset', '-c', str(protocol['cpu_core'])]
                and command[-8:] == ['--task', task, '--phase', phase, '--bits', str(bits),
                                      '--backend', backend], 'executed command differs')
        literal = loads(process['stdout'])
        require(type(literal) is dict and set(literal) == set(record) -
                {'job', 'repeat', 'process_wall_ns'}, 'process observation inventory differs')
        require(all(record.get(k) == v and type(record.get(k)) is type(v)
                    for k, v in literal.items()), 'process observation differs from summary')
        for field in ('elapsed_ns', 'cpu_ns', 'process_wall_ns', 'VmHWM_KiB', 'VmRSS_KiB'):
            require(type(record[field]) is int and record[field] > 0, 'invalid resource observation')
        require(record['elapsed_ns'] <= record['process_wall_ns'] and
                process['wall_ns'] == record['process_wall_ns'], 'phase exceeds process duration')
        require(record['VmHWM_KiB'] >= record['VmRSS_KiB'], 'invalid memory range')
        require(record['affinity'] == [protocol['cpu_core']], 'CPU pinning changed')
        name = f'{task}-{bits}.sct'
        require(record['source_sha256'] == protocol['models'][task]['model.json'] and
                record['packed_sha256'] == protocol['compiled'][name], 'receipt model identity changed')
        require(record['oracle_sha256'] == (None if phase == 'official' else oracle_hashes[name]),
                'canonical certificate digest mismatch')
        require(record['library_sha256'] == protocol['library_sha256'] and
                record['official_sha256'] == protocol['official_sha256'], 'receipt library identity changed')
    result = {}
    for task in TASKS:
        task_result = {}
        for phase in ('verify', 'session'):
            task_result[phase] = {}
            for bits in (8, 16):
                med = {}
                for backend in ('reference', 'dyadic'):
                    cell = [r for r in records if (r['task'], r['phase'], r['bits'], r['backend']) ==
                            (task, phase, bits, backend)]
                    require(len(cell) == 3, 'wrong repetition count')
                    med[backend] = {
                        'median_ms': statistics.median(r['elapsed_ns'] for r in cell) / 1e6,
                        'median_cpu_ms': statistics.median(r['cpu_ns'] for r in cell) / 1e6,
                        'process_median_ms': statistics.median(r['process_wall_ns'] for r in cell) / 1e6,
                        'peak_KiB_range': [min(r['VmHWM_KiB'] for r in cell), max(r['VmHWM_KiB'] for r in cell)]}
                med['speedup'] = med['reference']['median_ms'] / med['dyadic']['median_ms']
                med['peak_ratio'] = statistics.mean(med['dyadic']['peak_KiB_range']) / statistics.mean(med['reference']['peak_KiB_range'])
                task_result[phase][str(bits)] = med
        official_rows = [r for r in records if r['task'] == task and r['phase'] == 'official']
        task_result['official_load_median_ms'] = statistics.median(r['elapsed_ns'] for r in official_rows) / 1e6
        task_result['official_peak_KiB_range'] = [min(r['VmHWM_KiB'] for r in official_rows), max(r['VmHWM_KiB'] for r in official_rows)]
        result[task] = task_result
    return {'status': 'PASS', 'attempts': len(records), 'tasks': result,
            'scope': 'complete source/model/observation identities and recomputed medians; no clock authenticity, new classification accuracy or cross-platform inference claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'source', 'evidence', 'library', 'official', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.study, args.source, args.evidence, args.library, args.official)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
