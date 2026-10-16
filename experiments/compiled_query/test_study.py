"""Contracts for costs, original-input decisions, failure retention and resume."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import random
import subprocess

import pytest

from experiments.compiled_query.audit import audit_study, original_feasible
from experiments.compiled_query.binary_list import BinaryListProblem
from experiments.compiled_query.cases import Case, canonical, digest, load_case, synthetic_cases, write_case
from experiments.compiled_query.sessions import ARMS, decode_output, preload, query_order, run_session
from experiments.compiled_query.study import Config, create_study, read_manifest, run_study, schedule
from spectra.cnf.quotient_query import QuotientRuntime, build_quotient_runtime


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    explicit = os.environ.get('SPECTRA_QUOTIENT_LIBRARY')
    return Path(explicit) if explicit else build_quotient_runtime(tmp_path_factory.mktemp('native'))


@pytest.fixture
def case():
    p = BinaryListProblem.create(4, 3, ((0, 1), (1, 2)), (3, 6, 5, 3))
    return Case('tiny', 'original-graph-tiny', 'synthetic, no filtering', p,
                ((), ((0, 1),), ((0, 1), (0, 2)), ((2, 4),), ((3, 2),)))


def brute(p, q):
    choices = [tuple(c for c in range(p.palette) if m >> c & 1) for m in p.masks]
    return {bytes(row) for row in itertools.product(*choices) if p.check(bytes(row), q)}


def test_case_roundtrip_does_not_contain_precomputed_answers(case):
    assert Case.decode(case.payload()) == case
    assert set(json.loads(case.payload())) == {
        'schema', 'name', 'cluster', 'provenance', 'n', 'palette', 'edges', 'masks', 'queries'}


@pytest.mark.parametrize('change', [dict(n=True), dict(n=-1), dict(palette=65),
    dict(masks=[3, 6, 5]), dict(masks=[3, 6, 5, 0]), dict(queries=[]),
    dict(queries=[[[0, True]]]), dict(queries=[[[4, 1]]]), dict(answer=[0, 1, 2, 0]),
    dict(cluster=''), dict(schema='other')])
def test_case_rejects_invalid_inventory_or_values(case, change):
    value = json.loads(case.payload()); value.update(change)
    with pytest.raises(ValueError):
        Case.decode(canonical(value))


def test_duplicate_json_keys_and_nan_rejected(case):
    raw = case.payload()
    with pytest.raises(ValueError, match='duplicate'):
        Case.decode(raw.replace(b'"n":4', b'"n":4,"n":4'))
    with pytest.raises(ValueError):
        Case.decode(raw.replace(b'"n":4', b'"n":NaN'))


def test_read_is_bounded_before_json(tmp_path, monkeypatch):
    monkeypatch.setattr('experiments.compiled_query.cases.MAX_INPUT_BYTES', 64)
    path = tmp_path / 'huge'; path.write_bytes(b' ' * 65)
    with pytest.raises(ValueError, match='input limit'):
        load_case(path)


def test_synthetic_generation_is_deterministic_and_keeps_bad_requests():
    first = synthetic_cases(40, 20)
    assert first == synthetic_cases(40, 20)
    assert len({x.cluster for x in first}) == 5
    for case in first:
        assert any(not original_feasible(case.problem, q) for q in case.queries)


def test_independent_non_sat_observer_against_original_enumeration():
    rng = random.Random(94811)
    for _ in range(160):
        n = rng.randint(1, 5)
        p = BinaryListProblem.create(n, 3,
                ((u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < .4),
                (rng.choice((1, 2, 3, 4, 5, 6)) for _ in range(n)))
        for _ in range(24):
            q = tuple((rng.randrange(n), rng.randrange(8)) for _ in range(rng.randrange(5)))
            assert original_feasible(p, q) == bool(brute(p, q))


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('order', ('forward', 'reverse', 'shuffle'))
def test_complete_costs_full_outputs_and_unknown_not_promoted(tmp_path, library, case, arm, order):
    path = tmp_path / 'input.json'; write_case(path, case)
    preload(arm)
    record, output = run_session(path, QuotientRuntime(library), arm, count=5, order=order, seed=12)
    rows = decode_output(output, 4, 5)
    for (status, labels), index in zip(rows, query_order(5, order, 12)):
        expected = brute(case.problem, case.queries[index])
        if expected:
            assert status == 1 and labels in expected
        else:
            assert status == (2 if arm in ('scc', 'none') else 0)
    fields = ('input_ns', 'setup_ns', 'query_loop_ns', 'materialize_ns', 'diagnostics_ns', 'dispose_ns')
    assert sum(record[f] for f in fields) == record['complete_ns']
    assert record['materialize_ns'] > 0 and record['dispose_ns'] > 0
    assert record['query_ns'] <= record['query_loop_ns']
    assert record['output_sha256'] == digest(output)
    assert record['status'] == ('INCOMPLETE' if arm in ('scc', 'none') else 'COMPLETE')


@pytest.mark.parametrize('config', [Config(arms=('oracle',)), Config(arms=('scc', 'scc')),
    Config(prefixes=(0,)), Config(prefixes=(1, 1)), Config(repeats=True),
    Config(seed=True), Config(wall_seconds=float('nan')), Config(wall_seconds=0),
    Config(address_space_bytes=1), Config(bdd_nodes=False)])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        config.validate()


def test_paired_schedule_uses_same_requests_and_replicates_not_queries(case):
    config = Config(prefixes=(1, 5), repeats=4)
    cases = [dict(query_count=5, sha256=digest(case.payload()),
                  inputs={str(n): dict(sha256=str(n)) for n in (1, 5)})]
    jobs = schedule(cases, config)
    assert jobs == schedule(cases, config)
    groups = defaultdict(list)
    for job in jobs:
        groups[job['case_index'], job['count'], job['repeat']].append(job)
    for rows in groups.values():
        assert {j['arm'] for j in rows} == set(ARMS)
        assert len({(j['order'], j['order_seed']) for j in rows}) == 1
    with pytest.raises(ValueError, match='exceeds'):
        schedule(cases, replace(config, prefixes=(6,)))


def make_study(tmp_path, library, case, **changes):
    path = tmp_path / 'input.json'; write_case(path, case)
    root = tmp_path / 'study'
    config = replace(Config(arms=('scc', 'minicard'), prefixes=(1, 5), repeats=1), **changes)
    create_study([path], library, root, config)
    return root


def test_real_process_replay_resumes_without_rerunning_and_withholds_false_wins(tmp_path, library, case):
    root = make_study(tmp_path, library, case)
    assert len(run_study(root, 1)) == 1
    first = (root / 'sessions/000000/controller.json').read_bytes()
    assert len(run_study(root)) == 3
    assert run_study(root) == []
    assert first == (root / 'sessions/000000/controller.json').read_bytes()
    audit = audit_study(root)
    assert audit['scheduled_jobs'] == 4
    assert audit['outcomes'] == {'COMPLETE': 3, 'INCOMPLETE': 1}
    assert next(x for x in audit['comparisons'] if x['count'] == 1)['eligible']
    incomplete = next(x for x in audit['comparisons'] if x['count'] == 5)
    assert not incomplete['eligible'] and 'geomean_complete_ratio' not in incomplete


def test_interrupted_job_is_preserved_not_retried(tmp_path, library, case):
    root = make_study(tmp_path, library, case, arms=('scc',), prefixes=(1,))
    (root / 'sessions/000000').mkdir()
    assert run_study(root)[0]['status'] == 'INTERRUPTED'
    assert run_study(root) == []
    assert audit_study(root)['outcomes'] == {'INTERRUPTED': 1}


def test_timeout_kills_process_group_and_is_never_a_solver_answer(tmp_path, library, case, monkeypatch):
    root = make_study(tmp_path, library, case, arms=('scc',), prefixes=(1,))
    killed = []
    class Process:
        pid = 123456
        def __init__(self, *args, **kwargs):
            assert kwargs['start_new_session'] is True
        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired('test', timeout)
            return -9
    monkeypatch.setattr('experiments.compiled_query.study.subprocess.Popen', Process)
    monkeypatch.setattr('experiments.compiled_query.study.os.killpg', lambda pid, sig: killed.append(pid))
    assert run_study(root)[0]['status'] == 'TIMEOUT'
    assert killed == [123456]
    assert audit_study(root)['outcomes'] == {'TIMEOUT': 1}
    assert run_study(root) == []


def test_compilation_limit_is_retained_as_resource_failure(tmp_path, library, case):
    root = make_study(tmp_path, library, case, arms=('cudd-bfs',), prefixes=(1,), bdd_nodes=1)
    assert run_study(root)[0]['status'] == 'RESOURCE_LIMIT'
    assert audit_study(root)['outcomes'] == {'RESOURCE_LIMIT': 1}


@pytest.mark.parametrize('target', ('source', 'environment', 'case', 'library', 'manifest'))
def test_resume_rejects_changed_identity(tmp_path, library, case, target, monkeypatch):
    root = make_study(tmp_path, library, case)
    manifest = read_manifest(root)
    if target == 'source':
        monkeypatch.setattr('experiments.compiled_query.study.sources', lambda: {})
    elif target == 'environment':
        monkeypatch.setattr('experiments.compiled_query.study.execution_identity', lambda: {})
    else:
        name = {'case': manifest['cases'][0]['file'], 'library': manifest['library'],
                'manifest': 'manifest.json'}[target]
        with (root / name).open('ab') as stream:
            stream.write(b'!')
    with pytest.raises(ValueError, match='changed'):
        run_study(root)


def rewrite_record(root, edit):
    folder = root / 'sessions/000000'
    record = json.loads((folder / 'record.json').read_bytes())
    output = bytearray((folder / 'outputs.bin').read_bytes())
    edit(record, output)
    (folder / 'outputs.bin').write_bytes(output)
    record['output_sha256'] = digest(output)
    (folder / 'record.json').write_bytes(canonical(record))
    controller = json.loads((folder / 'controller.json').read_bytes())
    controller['record_sha256'] = digest((folder / 'record.json').read_bytes())
    (folder / 'controller.json').write_bytes(canonical(controller))


@pytest.mark.parametrize('attack', ('labels', 'unsat', 'cost'))
def test_independent_audit_catches_semantics_even_with_recomputed_hashes(tmp_path, library, case, attack):
    root = make_study(tmp_path, library, case, arms=('minicard',), prefixes=(1,))
    run_study(root)
    def edit(record, output):
        if attack == 'labels':
            output[17] = 63
        elif attack == 'unsat':
            output[16:] = bytes(5)
            record['counts'] = {'UNSAT': 1}
        else:
            record['complete_ns'] -= record['materialize_ns']
    rewrite_record(root, edit)
    with pytest.raises(ValueError):
        audit_study(root)


def test_resume_checks_previously_written_output_hash(tmp_path, library, case):
    root = make_study(tmp_path, library, case, arms=('minicard',), prefixes=(1,))
    run_study(root)
    path = root / 'sessions/000000/outputs.bin'
    path.write_bytes(path.read_bytes() + b'!')
    with pytest.raises(ValueError, match='output changed'):
        run_study(root)


def test_one_query_session_receives_one_query_not_an_entire_bank(tmp_path, library, case):
    root = make_study(tmp_path, library, case)
    manifest = read_manifest(root)
    for count in (1, 5):
        entry = manifest['cases'][0]['inputs'][str(count)]
        actual, identity = load_case(root / entry['file'])
        assert identity == entry['sha256']
        assert len(actual.queries) == count
        assert actual.queries == case.queries[:count]
    path = tmp_path / 'all.json'; write_case(path, case)
    with pytest.raises(ValueError, match='exactly'):
        run_session(path, QuotientRuntime(library), 'scc', count=1)
