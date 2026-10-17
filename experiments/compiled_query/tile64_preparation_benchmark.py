"""Paired development benchmark for tile64 request preparation.

This uses the already exposed 48-case structural panel. It is not a new holdout,
external replication, deployment trace, or universal speed claim. The benchmark
compares two current same-contract engines: explicit SCC tile64 and explicit raw
traversal. Every call emits and independently checks complete SCQ2 evidence.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

from spectra.exact import (
    ExactProblem,
    ExactRuntime,
    decode_certified_output,
    pack_query_bank,
    verify_certified_output,
)
from .transfer_panel import FAMILIES, make_case, seed_for

SCHEMA = 'spectra.tile64_preparation_benchmark.v1'


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _gmean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def _upper_graph_bootstrap(values: list[float], *, draws: int = 20000) -> float:
    rng = random.Random(20261017)
    samples = []
    for _ in range(draws):
        chosen = [values[rng.randrange(len(values))] for _ in values]
        samples.append(_gmean(chosen))
    samples.sort()
    return samples[math.ceil(.95 * len(samples)) - 1]


def run(library: Path, destination: Path, *, repeats: int = 31,
        max_bytes: int = 512 << 20) -> dict:
    if type(repeats) is not int or repeats < 9 or repeats % 2 == 0:
        raise ValueError('repeats must be an odd exact integer of at least nine')
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    runtime = ExactRuntime(Path(library).resolve())
    try:
        if hasattr(os, 'sched_getaffinity'):
            cpus = sorted(os.sched_getaffinity(0))
            if cpus:
                os.sched_setaffinity(0, {cpus[0]})
    except OSError:
        pass

    records = []
    gc.disable()
    try:
        for family in FAMILIES:
            for replica in range(6):
                case = make_case(family, replica)
                original = case.problem
                problem = ExactProblem.create(
                    original.n, original.palette, original.edges, original.masks)
                queries = case.queries
                packet = pack_query_bank(problem, queries)
                with runtime.prepare(problem, mode='raw', max_bytes=max_bytes) as raw, \
                        runtime.prepare(problem, mode='scc', max_bytes=max_bytes) as scc:
                    raw_first = raw.solve_bank(packet)
                    scc_first = scc.solve_bank(packet)
                    if (raw_first.sat_count, raw_first.unsat_count) != (
                            scc_first.sat_count, scc_first.unsat_count):
                        raise AssertionError('raw/SCC decision counts differ')
                    raw_codes = tuple(row[0] for row in decode_certified_output(
                        raw_first.output, problem.n, len(queries)))
                    scc_codes = tuple(row[0] for row in decode_certified_output(
                        scc_first.output, problem.n, len(queries)))
                    if raw_codes != scc_codes:
                        raise AssertionError('raw/SCC decisions differ')
                    if (not verify_certified_output(problem, queries, raw_first.output)
                            or not verify_certified_output(problem, queries, scc_first.output)):
                        raise AssertionError('independent Python replay failed')
                    for _ in range(5):
                        raw.solve_bank(packet)
                        scc.solve_bank(packet)
                    samples = {'raw': [], 'scc': []}
                    rng = random.Random(seed_for(case.name + '/tile64-preparation-v1'))
                    for _ in range(repeats):
                        order = ['raw', 'scc']
                        rng.shuffle(order)
                        for mode in order:
                            session = raw if mode == 'raw' else scc
                            start = time.perf_counter_ns()
                            session.solve_bank(packet)
                            samples[mode].append(time.perf_counter_ns() - start)
                    medians = {mode: statistics.median(values)
                               for mode, values in samples.items()}
                    info = scc.info
                record = {
                    'case': case.name,
                    'family': family,
                    'replica': replica,
                    'vertices': problem.n,
                    'edges': len(problem.edges),
                    'queries': len(queries),
                    'pins': sum(map(len, queries)),
                    'sat': scc_first.sat_count,
                    'unsat': scc_first.unsat_count,
                    'base_satisfiable': bool(info['base_satisfiable']),
                    'raw_median_ns': medians['raw'],
                    'scc_median_ns': medians['scc'],
                    'scc_raw_ratio': medians['scc'] / medians['raw'],
                    'raw_samples_ns': samples['raw'],
                    'scc_samples_ns': samples['scc'],
                    'raw_output_sha256': _digest(raw_first.output),
                    'scc_output_sha256': _digest(scc_first.output),
                }
                records.append(record)
                print(case.name, f"ratio={record['scc_raw_ratio']:.6f}", flush=True)
    finally:
        gc.enable()

    feasible = [record['scc_raw_ratio'] for record in records
                if record['base_satisfiable']]
    infeasible = [record['scc_raw_ratio'] for record in records
                  if not record['base_satisfiable']]
    if not feasible or not infeasible:
        raise AssertionError('panel must retain both feasible and base-infeasible graphs')
    feasible_upper95 = _upper_graph_bootstrap(feasible)
    summary = {
        'schema': SCHEMA,
        'classification': 'exposed synthetic development panel; not a holdout',
        'source_commit': os.environ.get('GITHUB_SHA'),
        'library_sha256': _digest(Path(library).read_bytes()),
        'python': sys.version,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'repeats': repeats,
        'cases': len(records),
        'feasible_cases': len(feasible),
        'base_infeasible_cases': len(infeasible),
        'feasible_geometric_mean_ratio': _gmean(feasible),
        'feasible_upper95_graph_bootstrap_ratio': feasible_upper95,
        'feasible_max_ratio': max(feasible),
        'feasible_wins_below_0_90': sum(value < .90 for value in feasible),
        'base_infeasible_geometric_mean_ratio': _gmean(infeasible),
        'gate': {
            'all_48_complete_and_replayed': len(records) == 48,
            'all_feasible_cases_below_0_90': max(feasible) < .90,
            'feasible_upper95_below_0_75': feasible_upper95 < .75,
        },
    }
    summary['gate_pass'] = all(summary['gate'].values())
    payload = {'summary': summary, 'records': records}
    destination.joinpath('results.json').write_bytes(_canonical(payload))
    destination.joinpath('SHA256.txt').write_text(
        _digest(destination.joinpath('results.json').read_bytes()) + '  results.json\n')
    if not summary['gate_pass']:
        raise AssertionError('tile64 preparation development gate failed')
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=31)
    args = parser.parse_args(argv)
    payload = run(args.library, args.out, repeats=args.repeats)
    print(json.dumps(payload['summary'], sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
