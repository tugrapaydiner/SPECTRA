"""Synthetic recorded-data fixtures; no timings, model quality or proof claims."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import random
import struct

import pytest

from .audit import ARMS, TASKS, audit


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def evidence(tmp_path):
    """Small, internally consistent observations for the recorded-data auditor.

    Model/source placeholders are inert and are never executed or mathematically
    verified. This fixture exercises record completeness, not the numerical proof.
    """
    root = tmp_path / 'synthetic-evidence'
    run = root / 'results/benchmark'
    run.mkdir(parents=True)
    before, after = b'synthetic original source\n', b'synthetic resumed source\n'
    (run / 'benchmark_initial.py').write_bytes(before)
    source = root / 'source/experiments/tree_total/benchmark.py'
    source.parent.mkdir(parents=True)
    source.write_bytes(after)
    sources = {'experiments/tree_total/benchmark.py': 'results/benchmark/benchmark_initial.py'}
    source_digests = {'experiments/tree_total/benchmark.py': digest(before)}
    replay_sources = {'benchmark.py': digest(after)}
    for name in ('build.py', 'compiler.py', 'exact.py', 'replay.py', 'runtime.cpp', 'session.py'):
        relative = 'experiments/tree_total/' + name
        target = root / 'source' / relative
        target.write_bytes(('inert source: ' + name).encode())
        sources[relative] = 'source/' + relative
        source_digests[relative] = digest(target.read_bytes())
        replay_sources[name] = source_digests[relative]
    for name in ('tree_residual', 'certified_trees'):
        relative = f'experiments/{name}/runtime.cpp'
        target = root / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(('inert source: ' + relative).encode())
        sources[relative] = 'source/' + relative
        source_digests[relative] = digest(target.read_bytes())
    paths, files, proof, replay_models = {}, {}, [], {}
    def bind(mapped, raw):
        original = '/synthetic/' + mapped
        target = root / mapped
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        paths[original], files[original] = mapped, digest(raw)
        return digest(raw)
    for mapped in ('native/total.so', 'residual/residual.so', 'baseline_evidence/native-register/trees.so',
                   'baseline_evidence/official/libcatboostmodel-linux-x86_64-1.2.8.so',
                   'baseline_evidence/official/libcatboostmodel-linux-x86_64-1.2.10.so'):
        bind(mapped, ('inert library: ' + mapped).encode())
    rng, jobs = random.Random(2026092967), []
    for task in TASKS:
        for repeat in range(7):
            sequence = list(itertools.product((1, 32, 256), ARMS))
            rng.shuffle(sequence)
            jobs.extend([task, repeat, n, arm] for n, arm in sequence)
    summary = {'cells': 756, 'geometric_ratio': 1., 'max_task_ratio': 1.,
               'performance_gate': True, 'repeated_predictions': 756 * 2 * 10,
               'tasks': {}}
    for task in TASKS:
        indices = struct.pack('<ii', 0, 1)
        replay_models[task] = {}
        for name, raw in (('model.json', b'inert source model'), ('model-16.sct', b'inert compact'),
                          ('model.cbm', b'inert official'), ('input.u8', b'\x00\x01'),
                          ('indices.i32', indices)):
            identity = bind(f'baseline_sdk/models/{task}/{name}', raw)
            field = {'model.json': 'source', 'input.u8': 'inputs', 'indices.i32': 'expected'}.get(name)
            if field:
                replay_models[task][field] = identity
        bind(f'residual_sdk/models/{task}/model.scr', b'inert residual')
        bind(f'baseline_evidence/replay/{task}/cpp/export.so', b'inert exported library')
        records = [{'task': t, 'repeat': rep, 'chunk': chunk, 'arm': arm,
                    'rows': 2, 'cycles': 10, 'wall_ns': 20000, 'cpu_ns': 20000,
                    'indices_sha256': digest(indices)}
                   for t, rep, chunk, arm in jobs if t == task]
        lines = [(json.dumps(row) + '\n').encode() for row in records]
        (run / (task + '.jsonl')).write_bytes(b''.join(lines))
        if task in ('letter', 'satellite'):
            (run / (task + '_prefix.jsonl')).write_bytes(lines[0])
            amendment = ({'original_sha256': digest(before), 'resumed_sha256': digest(after),
                          'preserved_prefix_cells': 1, 'preserved_prefix_sha256': digest(lines[0])}
                         if task == 'letter' else
                         {'prefix_cells': 1, 'prefix_sha256': digest(lines[0])})
            write_json(run / ('RESUMPTION.json' if task == 'letter' else 'satellite_resumption.json'), amendment)
        summary['tasks'][task] = {
            'us_per_row': {clock: {str(n): {arm: 1. for arm in ARMS} for n in (1, 32, 256)}
                          for clock in ('wall', 'cpu')},
            'batch32_total_over_prior_full16': 1., 'batch32_paired_ratios': [1.] * 7,
            'batch32_paired_regressions': 0,
        }
        folder = root / 'results/replay' / task
        folder.mkdir(parents=True)
        scores = struct.pack('<dddd', 1., 0., 0., 1.)
        (folder / 'reference_scores.f64').write_bytes(scores)
        models = {}
        for layout in ('flat', 'interned'):
            raw = (task + '/' + layout + ': inert fixture').encode()
            bind(f'results/replay/{task}/{layout}.sctt', raw)
            proof.append({'task': task, 'layout': layout, 'bytes': len(raw), 'sha256': digest(raw)})
            models[layout] = {'sha256': digest(raw), 'info': {'features': 1, 'maximum': 1, 'classes': 2},
                              'work': {'total': {'unresolved': 0, 'coarse_certified': 2,
                                                 'exact_completed': 0}}}
        stress = {}
        for kind in ('uniform', 'boundary'):
            stress_scores, stress_indices = scores * 1024, indices * 1024
            (folder / (kind + '.u8')).write_bytes(b'\x00\x01' * 1024)
            (folder / (kind + '-scores.f64')).write_bytes(stress_scores)
            (folder / (kind + '-indices.i32')).write_bytes(stress_indices)
            stress[kind] = {'rows': 2048, 'source_scores_sha256': digest(stress_scores),
                            'indices_sha256': digest(stress_indices),
                            'work': {'unresolved': 0, 'coarse_certified': 2048, 'exact_completed': 0}}
        write_json(folder / 'result.json', {'rows': 2, 'classes': 2, 'score_values': 4,
                                            'models': models, 'stress': stress})
    write_json(root / 'PATHS.json', paths)
    write_json(root / 'SOURCES.json', sources)
    write_json(run / 'LOCK.json', {'tasks': list(TASKS), 'arms': list(ARMS), 'chunks': [1, 32, 256],
                                  'repeats': 7, 'cycles': 10, 'seed': 2026092967, 'jobs': jobs,
                                  'files': files, 'source': source_digests,
                                  'gate': {'batch32_geometric_ratio_limit': 1.10,
                                           'batch32_max_task_ratio_limit': 1.25, 'baseline': 'full16'}})
    write_json(run / 'SUMMARY.json', summary)
    write_json(root / 'results/independent.json', {'status': 'PASS', 'files': proof})
    write_json(root / 'results/replay/LOCK.json', {'source_files': replay_sources,
                                                  'models': replay_models,
                                                  'library_sha256': files['/synthetic/native/total.so'],
                                                  'stress_rows_per_kind': 2048,
                                                  'seed_rule': '2026092961+d+D'})
    resource_jobs = list(itertools.product(TASKS, ('flat', 'interned', 'residual', 'official'), range(3)))
    records = [{'task': task, 'policy': policy, 'repeat': repeat, 'matched': True,
                'numerical_frameworks': [], 'setup_ns': 100, 'model_bytes': 32,
                'memory_kib': {'VmHWM': 200, 'VmRSS': 100}, 'runtime_info': {},
                'model_sha256': digest(b'synthetic model'), 'library_sha256': digest(b'synthetic library'),
                'scope': 'synthetic bookkeeping fixture; no measurement'}
               for task, policy, repeat in resource_jobs]
    save_resources(root, resource_jobs, records)
    return root


def save_resources(root, jobs, records):
    folder = root / 'results/resources'
    write_json(folder / 'LOCK.json', {'isolated': True, 'jobs': jobs})
    (folder / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    for index, record in enumerate(records):
        write_json(folder / f'process-{index}.json', {'returncode': 0, 'stdout': json.dumps(record)})


def resources(root):
    folder = root / 'results/resources'
    jobs = json.loads((folder / 'LOCK.json').read_text())['jobs']
    records = [json.loads(line) for line in (folder / 'rows.jsonl').read_text().splitlines()]
    return jobs, records


def test_complete_recorded_matrix_passes(evidence):
    report = audit(evidence)
    assert report['status'] == 'PASS'
    assert report['isolated_resource_processes'] == 48
    assert report['timing_cells'] == 756 and report['retained_rows'] == 8


@pytest.mark.parametrize('kind', ['empty', 'short', 'long', 'duplicate', 'wrong-task', 'wrong-policy', 'bool-repeat'])
def test_incomplete_or_invalid_resource_schedule_rejected(evidence, kind):
    jobs, records = resources(evidence)
    if kind == 'empty':
        jobs = []
    elif kind == 'short':
        jobs.pop()
    elif kind == 'long':
        jobs.append(jobs[0])
    elif kind == 'duplicate':
        jobs[-1], records[-1] = jobs[0], records[0]
    else:
        field, index, value = {'wrong-task': ('task', 0, 'missing-task'),
                               'wrong-policy': ('policy', 1, 'missing-policy'),
                               'bool-repeat': ('repeat', 2, False)}[kind]
        jobs[0][index], records[0][field] = value, value
    save_resources(evidence, jobs, records)
    with pytest.raises(ValueError):
        audit(evidence)


def test_empty_schedule_cannot_hide_failed_resource_process(evidence):
    folder = evidence / 'results/resources'
    write_json(folder / 'LOCK.json', {'isolated': True, 'jobs': []})
    write_json(folder / 'process-0.json', {'returncode': 1, 'stdout': ''})
    with pytest.raises(ValueError):
        audit(evidence)


def test_complete_matrix_can_preserve_a_different_recorded_order(evidence):
    jobs, records = resources(evidence)
    save_resources(evidence, list(reversed(jobs)), list(reversed(records)))
    assert audit(evidence)['isolated_resource_processes'] == 48


def test_boolean_process_returncode_is_not_a_success_status(evidence):
    path = evidence / 'results/resources/process-0.json'
    record = json.loads(path.read_text())
    record['returncode'] = False
    write_json(path, record)
    with pytest.raises(ValueError):
        audit(evidence)


@pytest.mark.parametrize('payload', [{}, {'task': 'letter', 'policy': 'flat', 'matched': True,
                                       'numerical_frameworks': []}])
def test_missing_literal_measurements_cannot_pass(evidence, payload):
    path = evidence / 'results/resources/process-0.json'
    write_json(path, {'returncode': 0, 'stdout': json.dumps(payload)})
    with pytest.raises(ValueError):
        audit(evidence)


def test_negative_setup_time_cannot_pass_when_both_records_agree(evidence):
    jobs, records = resources(evidence)
    records[0]['setup_ns'] = -1
    save_resources(evidence, jobs, records)
    with pytest.raises(ValueError):
        audit(evidence)


def test_negative_auditor_rejects_all_disposable_fixture_mutations(evidence, tmp_path):
    from .negative_audit import run

    report = run(evidence, tmp_path / 'negative-results')
    assert report['status'] == 'PASS' and report['rejected'] == 16
    assert all(case['rejected'] for case in report['cases'])
    assert audit(evidence)['status'] == 'PASS'
