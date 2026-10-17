"""Paired first-use benchmark for the public ExactProblem -> SBC1 -> SCQ2 path.

The two WAP inputs are already exposed development cases.  This benchmark is not
new confirmation.  It charges ordinary Python validation and SBC1 packing plus
one complete native construction/solve/check/serialization/disposal call.
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
    pack_session_packet,
    verify_session_output,
)
from .cases import Case

SCHEMA = 'spectra.session_packet_cold_benchmark.v1'
CASES = ('wap05a.col', 'wap06a.col')


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _gmean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def run(library: Path, cases: Path, destination: Path, *, repeats: int = 31) -> dict:
    if type(repeats) is not int or repeats < 9 or repeats % 2 == 0:
        raise ValueError('repeats must be an odd exact integer of at least nine')
    destination.mkdir(parents=True, exist_ok=False)
    runtime = ExactRuntime(library)
    records = []
    try:
        if hasattr(os, 'sched_getaffinity'):
            cpus = sorted(os.sched_getaffinity(0))
            if cpus:
                os.sched_setaffinity(0, {cpus[0]})
    except OSError:
        pass

    gc.disable()
    try:
        for name in CASES:
            source = cases / (name + '.json')
            case = Case.decode(source.read_bytes())
            problem = ExactProblem.create(
                case.problem.n, case.problem.palette,
                case.problem.edges, case.problem.masks,
            )
            rows = case.queries
            frozen = pack_session_packet(problem, rows)
            first = {
                mode: runtime.solve_session_packet(frozen, mode=mode)
                for mode in ('raw', 'scc')
            }
            if first['raw'].output != first['scc'].output:
                raise AssertionError('raw and SCC certified outputs differ')
            if not verify_session_output(frozen, first['scc'].output):
                raise AssertionError('independent SBC1/SCQ2 replay failed')
            for _ in range(5):
                for mode in ('raw', 'scc'):
                    packet = pack_session_packet(problem, rows)
                    runtime.solve_session_packet(packet, mode=mode)

            samples = {'raw': [], 'scc': []}
            pack_samples = {'raw': [], 'scc': []}
            solve_samples = {'raw': [], 'scc': []}
            packet_hashes = set()
            output_hashes = {'raw': set(), 'scc': set()}
            rng = random.Random(int.from_bytes(
                hashlib.sha256(('SPECTRA-cold-packet-v1/' + name).encode()).digest()[:8], 'big'))
            for _ in range(repeats):
                order = ['raw', 'scc']
                rng.shuffle(order)
                for mode in order:
                    start = time.perf_counter_ns()
                    packet = pack_session_packet(problem, rows)
                    packed = time.perf_counter_ns()
                    answer = runtime.solve_session_packet(packet, mode=mode)
                    solved = time.perf_counter_ns()
                    samples[mode].append(solved - start)
                    pack_samples[mode].append(packed - start)
                    solve_samples[mode].append(solved - packed)
                    packet_hashes.add(_digest(packet))
                    output_hashes[mode].add(_digest(answer.output))
            if len(packet_hashes) != 1 or any(len(values) != 1 for values in output_hashes.values()):
                raise AssertionError('packet or output bytes changed across repetitions')
            medians = {mode: statistics.median(values) for mode, values in samples.items()}
            record = {
                'case': name,
                'vertices': problem.n,
                'edges': len(problem.edges),
                'queries': len(rows),
                'pins': sum(map(len, rows)),
                'packet_bytes': len(frozen),
                'packet_sha256': _digest(frozen),
                'output_sha256': _digest(first['scc'].output),
                'raw_samples_ns': samples['raw'],
                'scc_samples_ns': samples['scc'],
                'raw_pack_samples_ns': pack_samples['raw'],
                'scc_pack_samples_ns': pack_samples['scc'],
                'raw_solve_samples_ns': solve_samples['raw'],
                'scc_solve_samples_ns': solve_samples['scc'],
                'raw_median_ns': medians['raw'],
                'scc_median_ns': medians['scc'],
                'raw_pack_median_ns': statistics.median(pack_samples['raw']),
                'scc_pack_median_ns': statistics.median(pack_samples['scc']),
                'raw_solve_median_ns': statistics.median(solve_samples['raw']),
                'scc_solve_median_ns': statistics.median(solve_samples['scc']),
                'scc_raw_ratio': medians['scc'] / medians['raw'],
            }
            records.append(record)
            print(name, f"ratio={record['scc_raw_ratio']:.6f}", flush=True)
    finally:
        gc.enable()

    ratios = [record['scc_raw_ratio'] for record in records]
    summary = {
        'schema': SCHEMA,
        'classification': 'exposed WAP development cases; not a holdout',
        'python': sys.version,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'repeats': repeats,
        'cases': len(records),
        'geometric_mean_ratio': _gmean(ratios),
        'max_case_ratio': max(ratios),
        'gate': {
            'all_cases_complete_and_replayed': len(records) == len(CASES),
            'geometric_mean_below_0_75': _gmean(ratios) < .75,
            'every_case_below_0_80': max(ratios) < .80,
        },
    }
    summary['gate_pass'] = all(summary['gate'].values())
    payload = {'summary': summary, 'records': records}
    destination.joinpath('results.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if not summary['gate_pass']:
        raise AssertionError('first-use session packet gate failed')
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=31)
    args = parser.parse_args(argv)
    payload = run(args.library.resolve(), args.cases.resolve(), args.out, repeats=args.repeats)
    print(json.dumps(payload['summary'], indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
