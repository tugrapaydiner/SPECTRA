"""Original-input replay and failure-aware, graph-clustered descriptive summaries.

The audit is outside the timed endpoint. SAT witnesses are checked directly in
Python. Non-SAT decisions use a separately written original-list implication
checker, not the benchmark's CNF encoder, SCC quotient or BDD. No portable UNSAT
certificate or independently conducted replication is claimed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from .cases import digest, load_case
from .sessions import EXACT_ARMS, decode_output, decode_certified_output, query_order
from .proof import verify_answer
from .study import read_manifest


def original_feasible(problem, query) -> bool:
    """Independent exact binary-domain feasibility via iterative graph traversal."""
    n = problem.n
    choices = [tuple(c for c in range(problem.palette) if (mask >> c) & 1)
               for mask in problem.masks]
    choices = [row if len(row) == 2 else (row[0], row[0]) for row in choices]
    graph = [[] for _ in range(2 * n)]
    reverse = [[] for _ in graph]

    def arc(u, v):
        graph[u].append(v)
        reverse[v].append(u)

    for left, right in problem.edges:
        for a in range(2):
            for b in range(2):
                if choices[left][a] == choices[right][b]:
                    arc(2 * left + a, 2 * right + 1 - b)
                    arc(2 * right + b, 2 * left + 1 - a)
    for vertex, allowed in query:
        for a in range(2):
            if not ((allowed >> choices[vertex][a]) & 1):
                arc(2 * vertex + a, 2 * vertex + 1 - a)
    seen, order = set(), []
    for root in range(2 * n):
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(graph[root]))]
        while stack:
            u, neighbors = stack[-1]
            v = next(neighbors, None)
            if v is None:
                order.append(u)
                stack.pop()
            elif v not in seen:
                seen.add(v)
                stack.append((v, iter(graph[v])))
    component = [-1] * (2 * n)
    for root in reversed(order):
        if component[root] != -1:
            continue
        component[root] = root
        stack = [root]
        while stack:
            for v in reverse[stack.pop()]:
                if component[v] == -1:
                    component[v] = root
                    stack.append(v)
    return all(component[2*v] != component[2*v+1] for v in range(n))


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def audit_study(root: Path) -> dict:
    manifest = read_manifest(root, check_sources=False)
    cases = [load_case(root / c['file'])[0] for c in manifest['cases']]
    statuses, counts = Counter(), Counter()
    decision_cache = {}
    certified_counts = Counter()
    cells = defaultdict(list)
    raw_rows = []
    for job in manifest['jobs']:
        folder = root / 'sessions' / job['id']
        terminal = folder / 'controller.json'
        if not terminal.exists():
            statuses['MISSING'] += 1
            cells[(job['case_index'], job['count'], job['arm'])].append(None)
            continue
        receipt = json.loads(terminal.read_bytes())
        if receipt['job'] != job:
            raise ValueError('controller job differs')
        status = receipt['status']
        statuses[status] += 1
        record = None
        if receipt['record_sha256'] is not None:
            payload = (folder / 'record.json').read_bytes()
            if digest(payload) != receipt['record_sha256']:
                raise ValueError('worker record digest differs')
            record = json.loads(payload)
            if record['job'] != job:
                raise ValueError('worker job differs')
        if status not in ('COMPLETE', 'INCOMPLETE'):
            cells[(job['case_index'], job['count'], job['arm'])].append(None)
            raw_rows.append(dict(job=job['id'], arm=job['arm'], status=status))
            continue
        if record is None or record['status'] != status or receipt['returncode'] != 0:
            raise ValueError('successful controller/worker outcome differs')
        case = cases[job['case_index']]
        for key, expected in [('case_sha256', job['case_sha256']), ('arm', job['arm']),
                              ('query_count', job['count']), ('order', job['order']),
                              ('order_seed', job['order_seed']), ('cluster', case.cluster)]:
            if record[key] != expected:
                raise ValueError('session identity differs: ' + key)
        stages = ('input_ns', 'setup_ns', 'query_loop_ns', 'materialize_ns',
                  'diagnostics_ns', 'dispose_ns')
        if (any(type(record[k]) is not int or record[k] < 0 for k in stages)
                or sum(record[k] for k in stages) != record['complete_ns']
                or record['query_ns'] != sum(record['per_query_ns'])
                or len(record['per_query_ns']) != job['count']
                or any(type(v) is not int or v < 0 for v in record['per_query_ns'])
                or record['query_ns'] > record['query_loop_ns']
                or receipt['cold_wall_ns'] < record['complete_ns']):
            raise ValueError('timing accounting differs')
        output = (folder / 'outputs.bin').read_bytes()
        if digest(output) != record['output_sha256'] or len(output) != record['output_bytes']:
            raise ValueError('output digest/size differs')
        certified = job['arm'] in EXACT_ARMS
        if bool(record['unsat_certificate']) != certified:
            raise ValueError('certificate contract differs from selected arm')
        if certified:
            rows = decode_certified_output(output, case.problem.n, job['count'])
            if sum(len(row[2]) for row in rows) != record['proof_bytes']:
                raise ValueError('proof payload accounting differs')
        else:
            rows = tuple((code, labels, b'') for code, labels in decode_output(output, case.problem.n, job['count']))
        order = query_order(job['count'], job['order'], job['order_seed'])
        observed = Counter()
        for (decision, labels, proof), query_index in zip(rows, order):
            query = case.queries[query_index]
            label = ('UNSAT', 'SAT_VERIFIED', 'UNKNOWN')[decision]
            observed[label] += 1
            if certified:
                name = 'SAT_VERIFIED' if decision else 'UNSAT_VERIFIED'
                if not verify_answer(case.problem, query, name, labels if decision else b'', proof):
                    raise ValueError('proof/witness fails original-input Python verification')
                certified_counts[name] += 1
            if decision == 1:
                if not case.problem.check(labels, query):
                    raise ValueError('full witness fails independent original-input replay')
            else:
                key = (job['case_sha256'], query_index)
                if key not in decision_cache:
                    decision_cache[key] = original_feasible(case.problem, query)
                if decision == 0 and decision_cache[key]:
                    raise ValueError('false UNSAT decision')
        if dict(observed) != record['counts']:
            raise ValueError('decision counts differ')
        if (status == 'COMPLETE') != (observed['UNKNOWN'] == 0):
            raise ValueError('UNKNOWN session incorrectly classified')
        counts.update(observed)
        cells[(job['case_index'], job['count'], job['arm'])].append(record if status == 'COMPLETE' else None)
        raw_rows.append(dict(job=job['id'], arm=job['arm'], status=status,
                             complete_ns=record['complete_ns'], cold_wall_ns=receipt['cold_wall_ns']))

    summaries, comparisons = [], []
    repeats = manifest['config']['repeats']
    for (index, count, arm), records in sorted(cells.items()):
        complete = len(records) == repeats and all(r is not None for r in records)
        row = dict(case=cases[index].name, cluster=cases[index].cluster, count=count,
                   arm=arm, eligible=complete, scheduled=repeats,
                   complete_samples=sum(r is not None for r in records))
        if complete:
            for key in ('complete_ns', 'input_ns', 'setup_ns', 'query_ns',
                        'materialize_ns', 'dispose_ns'):
                row['mean_' + key] = statistics.fmean(r[key] for r in records)
            row['p95_complete_ns'] = percentile([r['complete_ns'] for r in records], .95)
        summaries.append(row)
    lookup = {(r['case'], r['count'], r['arm']): r for r in summaries}
    for candidate in ('scc', *EXACT_ARMS):
        if candidate not in manifest['config']['arms']:
            continue
        for prefix in manifest['config']['prefixes']:
            for baseline in manifest['config']['arms']:
                if baseline == candidate:
                    continue
                by_cluster = defaultdict(list)
                missing = []
                for case in cases:
                    a = lookup[(case.name, prefix, candidate)]
                    b = lookup[(case.name, prefix, baseline)]
                    if not (a['eligible'] and b['eligible']):
                        missing.append(case.name)
                    else:
                        by_cluster[case.cluster].append(a['mean_complete_ns'] / b['mean_complete_ns'])
                comparison = dict(candidate=candidate, baseline=baseline, count=prefix,
                                  eligible=not missing, excluded_or_incomplete=missing,
                                  matching_certificate_contract=(candidate in EXACT_ARMS)==(baseline in EXACT_ARMS),
                                  graph_clusters=len(by_cluster), descriptive_only=True)
                if not missing:
                    # A graph is one cluster even when multiple cases share it.
                    cluster_ratios = {key: math.exp(statistics.fmean(map(math.log, ratios)))
                                      for key, ratios in by_cluster.items()}
                    comparison['cluster_ratios'] = cluster_ratios
                    comparison['geomean_complete_ratio'] = math.exp(statistics.fmean(
                        map(math.log, cluster_ratios.values())))
                comparisons.append(comparison)
    return dict(schema='spectra.compiled_query.audit.v1', classification='development-only',
                manifest_sha256=digest((root / 'manifest.json').read_bytes()),
                scheduled_jobs=len(manifest['jobs']), outcomes=dict(statuses),
                audited_decisions=dict(counts), original_non_sat_requests_checked=len(decision_cache),
                all_scheduled_decisions_complete=statuses['COMPLETE'] == len(manifest['jobs']),
                portable_unsat_certificate=bool(certified_counts), certified_decisions=dict(certified_counts),
                mixed_output_contracts=any(a in EXACT_ARMS for a in manifest['config']['arms']) and
                    any(a not in EXACT_ARMS for a in manifest['config']['arms']), outside_replication=False,
                session_summaries=summaries, comparisons=comparisons, raw_rows=raw_rows)
