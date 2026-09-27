"""Independent stdlib audit of the complete ownership-cost experiment.

Reconstructs the frozen grid and numerical aggregates, verifies original input
identity, package snapshots and binaries. This checks recorded evidence, not
hardware timer authenticity or independent researcher reproduction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import statistics

TASKS = ('chess', 'penguins', 'titanic', 'wdbc', 'wine', 'zoo')
SCOPES = ('buffer1', 'buffer32', 'fused_trace')
MODELS = tuple(f'{t}-{s}' for t in TASKS for s in (101, 202, 303))
SEED = 2026092709
ANCHOR = '017f63aa30ee9f2db773ab86234b24b5eada3a9f2fcc97049d0802dbede90f4e'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError('duplicate key')
        out[key] = value
    return out

def reject(value): raise ValueError('nonfinite token')
def loads(text): return json.loads(text, object_pairs_hook=unique, parse_constant=reject)
def read(path): return loads(Path(path).read_text(encoding='utf-8'))

def member(root, name):
    if type(name) is not str or '\\' in name: raise ValueError('invalid member')
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts: raise ValueError('escaping member')
    path = Path(root)
    for part in rel.parts:
        path = path / part
        if path.is_symlink(): raise ValueError('symlinked member')
    if not path.is_file(): raise ValueError('missing member')
    return path

def same(actual, expected):
    if type(actual) is not type(expected): raise ValueError('aggregate type mismatch')
    if isinstance(expected, dict):
        if set(actual) != set(expected): raise ValueError('aggregate fields mismatch')
        for k, v in expected.items(): same(actual[k], v)
    elif isinstance(expected, float):
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-13, abs_tol=1e-13):
            raise ValueError('aggregate value mismatch')
    elif actual != expected: raise ValueError('aggregate value mismatch')


def audit(run, before, after, inputs, manifest, libraries):
    run, before, after, inputs = map(Path, (run, before, after, inputs))
    if sha(manifest) != ANCHOR: raise ValueError('original input anchor differs')
    anchor = read(manifest)
    p = read(run / 'protocol.json')
    if p['schema'] != 'spectra.lifetime.cost.v1' or p['scopes'] != list(SCOPES) or type(p['seed']) is not int or p['seed'] != SEED or type(p['repeats']) is not int or p['repeats'] != 15:
        raise ValueError('fixed protocol differs')
    if p['before_commit'] != 'e6f3d6b44ec53c158b9fa8021ddbadded482f1bb':
        raise ValueError('baseline differs')
    for name, folder in (('before_source', before), ('after_source', after)):
        files = {q.relative_to(folder).as_posix() for q in (folder / 'spectra').rglob('*')
                 if q.is_file() and q.suffix in ('.py', '.cpp', '.hpp', '.h')}
        if files != set(p[name]): raise ValueError('package inventory differs')
        for filename, expected in p[name].items():
            if sha(member(folder, filename)) != expected: raise ValueError('source digest differs')
    actual_libraries = {Path(lib).name: sha(lib) for lib in libraries}
    if actual_libraries != p['libraries']: raise ValueError('native binary identity differs')
    files = ('model.srt', 'preprocessing.json', 'cases.json', 'transformed.f64')
    expected_files = {f'{m}/{f}' for m in MODELS for f in files}
    if set(p['model_files']) != expected_files: raise ValueError('model inventory differs')
    for name in expected_files:
        path = member(inputs, name)
        expected = anchor['models/' + name]
        if sha(path) != p['model_files'][name] or sha(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError('input identity differs')
    expected = []
    rng = random.Random(SEED)
    for ordinal, model in enumerate(MODELS):
        cases = read(inputs / model / 'cases.json')
        rows, labels = cases['rows'], cases['expected']
        if len(rows) != len(labels): raise ValueError('case counts differ')
        order = list(range(len(rows))); random.Random(SEED + ordinal).shuffle(order)
        hashes = {scope: hashlib.sha256(json.dumps([labels[i] for i in order] if scope == 'fused_trace' else labels).encode()).hexdigest() for scope in SCOPES}
        for repeat in range(15):
            jobs = [(arm, scope) for arm in ('before', 'after') for scope in SCOPES]
            rng.shuffle(jobs)
            for arm, scope in jobs:
                expected.append({'model': model, 'repeat': repeat, 'arm': arm, 'scope': scope,
                                 'rows': len(rows), 'output_sha256': hashes[scope]})
    records = [loads(line) for line in (run / 'rows.jsonl').read_text().splitlines()]
    if len(records) != len(expected): raise ValueError('timing inventory differs')
    groups = {}
    for observed, wanted in zip(records, expected):
        if set(observed) != set(wanted) | {'ns'}: raise ValueError('timing fields differ')
        for key, value in wanted.items():
            if type(observed[key]) is not type(value) or observed[key] != value: raise ValueError('timing order/outcome differs')
        if type(observed['ns']) is not int or observed['ns'] <= 0: raise ValueError('invalid timing')
        groups.setdefault('|'.join(observed[k] for k in ('model', 'scope', 'arm')), []).append(observed['ns'])
    medians = {key: statistics.median(values) for key, values in groups.items()}
    tasks = {task: {scope: statistics.geometric_mean(medians[m + '|' + scope + '|after'] / medians[m + '|' + scope + '|before'] for m in MODELS if m.startswith(task + '-')) for scope in SCOPES} for task in TASKS}
    derived = {'observations': len(records), 'checked_predictions': sum(r['rows'] for r in records),
        'medians_ns': medians, 'task_after_over_before': tasks,
        'equal_task_ratio': {scope: statistics.geometric_mean(tasks[t][scope] for t in TASKS) for scope in SCOPES},
        'regressed_models': {scope: sum(medians[m + '|' + scope + '|after'] > medians[m + '|' + scope + '|before'] for m in MODELS) for scope in SCOPES}}
    same(read(run / 'summary.json'), derived)
    return {'status': 'PASS', 'timing_records': len(records), 'checked_predictions': derived['checked_predictions'],
            'model_files': len(expected_files), 'equal_task_ratio': derived['equal_task_ratio'],
            'regressed_models': derived['regressed_models'], 'scope': __doc__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'before', 'after', 'inputs', 'manifest', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--library', type=Path, action='append', required=True)
    a = parser.parse_args()
    result = audit(a.run, a.before, a.after, a.inputs, a.manifest, a.library)
    with a.out.open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
