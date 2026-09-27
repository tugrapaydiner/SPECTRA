"""Matched cost of request leases; no numerical-source/model change or speed gate."""
from __future__ import annotations
from array import array
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('chess', 'penguins', 'titanic', 'wdbc', 'wine', 'zoo')
SCOPES = ('buffer1', 'buffer32', 'fused_trace')
SEED = 2026092709


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def package(name, directory):
    spec = importlib.util.spec_from_file_location(name, directory / 'spectra/__init__.py',
                 submodule_search_locations=[str(directory / 'spectra')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(name + '.svm_pipeline')


def files(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted((root / 'spectra').rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.cpp', '.hpp', '.h')}


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    old, new = package('spectra_old', args.before), package('spectra_new', ROOT)
    metadata = {'schema': 'spectra.lifetime.cost.v1', 'seed': SEED, 'repeats': 15,
        'before_commit': 'e6f3d6b44ec53c158b9fa8021ddbadded482f1bb',
        'scopes': SCOPES, 'model_files': {}, 'before_source': files(args.before),
        'after_source': files(ROOT), 'libraries': {p.name: sha(p) for p in (args.library, args.preprocessor)},
        'affinity': sorted(os.sched_getaffinity(0)), 'python': sys.version,
        'scope': 'all-row jobs; same native binaries; fresh labels; no loading/build or verifier cost',
        'cpu': next(l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))}
    models = [f'{t}-{seed}' for t in TASKS for seed in (101, 202, 303)]
    for name in models:
        for file in ('model.srt', 'preprocessing.json', 'cases.json', 'transformed.f64'):
            metadata['model_files'][name + '/' + file] = sha(args.inputs / name / file)
    write(args.out / 'protocol.json', metadata)
    rng = random.Random(SEED)
    observations = []
    with (args.out / 'rows.jsonl').open('x') as log:
        for ordinal, name in enumerate(models):
            folder = args.inputs / name
            case = json.loads((folder / 'cases.json').read_text())
            values = array('d'); values.frombytes((folder / 'transformed.f64').read_bytes())
            with old.PreparedPipeline(folder, args.library, preprocessor_library=args.preprocessor) as om, \
                 new.PreparedPipeline(folder, args.library, preprocessor_library=args.preprocessor) as nm, \
                 om.session() as ow, nm.session() as nw:
                rows, expected = case['rows'], case['expected']
                dimensions = nm.preprocessor.features
                if len(values) != len(rows) * dimensions:
                    raise ValueError('feature inventory mismatch')
                if om.preprocessor.transform(rows).tobytes() != values.tobytes() or nm.preprocessor.transform(rows).tobytes() != values.tobytes():
                    raise ValueError('transformed feature mismatch')
                buffers = {n: [values[i * dimensions:min(i + n, len(rows)) * dimensions]
                                 for i in range(0, len(rows), n)] for n in (1, 32)}
                order = list(range(len(rows))); random.Random(SEED + ordinal).shuffle(order)
                batches = []; start = 0
                while start < len(rows):
                    count = min((1, 1, 8, 32, 128)[len(batches) % 5], len(rows) - start)
                    batches.append([rows[i] for i in order[start:start + count]]); start += count
                wanted = {'buffer1': expected, 'buffer32': expected,
                          'fused_trace': [expected[i] for i in order]}
                def invoke(worker, scope):
                    output = []
                    if scope == 'fused_trace':
                        for batch in batches: output.extend(worker.predict_fused(batch, schedule='binary_stream'))
                    else:
                        for batch in buffers[int(scope[6:])]: output.extend(worker._worker.predict_buffer(batch, schedule='binary_stream'))
                    return output
                for worker in (ow, nw):
                    for scope in SCOPES:
                        if invoke(worker, scope) != wanted[scope]: raise ValueError('warmup mismatch')
                for repeat in range(15):
                    jobs = [(arm, scope) for arm in ('before', 'after') for scope in SCOPES]
                    rng.shuffle(jobs)
                    for arm, scope in jobs:
                        worker = ow if arm == 'before' else nw
                        start = time.perf_counter_ns(); answer = invoke(worker, scope); ns = time.perf_counter_ns() - start
                        if answer != wanted[scope]: raise ValueError('timed prediction mismatch')
                        record = {'model': name, 'repeat': repeat, 'arm': arm, 'scope': scope,
                                  'ns': ns, 'rows': len(rows),
                                  'output_sha256': hashlib.sha256(json.dumps(answer).encode()).hexdigest()}
                        observations.append(record); log.write(json.dumps(record) + '\n')
            print(name, 'complete', flush=True)
    groups = {}
    for record in observations:
        groups.setdefault((record['model'], record['scope'], record['arm']), []).append(record['ns'])
    medians = {'|'.join(key): statistics.median(value) for key, value in groups.items()}
    tasks = {}
    for task in TASKS:
        task_models = [m for m in models if m.startswith(task + '-')]
        tasks[task] = {scope: statistics.geometric_mean(medians[m + '|' + scope + '|after'] / medians[m + '|' + scope + '|before'] for m in task_models) for scope in SCOPES}
    summary = {'observations': len(observations), 'checked_predictions': sum(r['rows'] for r in observations),
        'task_after_over_before': tasks, 'medians_ns': medians,
        'equal_task_ratio': {scope: statistics.geometric_mean(tasks[t][scope] for t in TASKS) for scope in SCOPES},
        'regressed_models': {scope: sum(medians[m + '|' + scope + '|after'] > medians[m + '|' + scope + '|before'] for m in models) for scope in SCOPES}}
    write(args.out / 'summary.json', summary); print(json.dumps(summary['equal_task_ratio']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'inputs', 'library', 'preprocessor', 'out'): p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
