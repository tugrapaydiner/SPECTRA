"""Paired first-use benchmark for the public ExactProblem -> SBC1 -> SCQ2 path.

The two WAP inputs are already exposed development cases. This benchmark is not
new confirmation. It measures both the normal admitted-problem bank boundary and
a secondary one-bank boundary that also reconstructs the immutable problem.
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

SCHEMA = 'spectra.session_packet_cold_benchmark.v2'
CASES = ('wap05a.col', 'wap06a.col')
BANK_MEAN_LIMIT = .60
BANK_CASE_LIMIT = .65
APPLICATION_MEAN_LIMIT = .75
APPLICATION_CASE_LIMIT = .80


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

            def admit() -> ExactProblem:
                return ExactProblem.create(
                    case.problem.n, case.problem.palette,
                    case.problem.edges, case.problem.masks,
                )

            admitted_at = time.perf_counter_ns()
            problem = admit()
            admitted_ns = time.perf_counter_ns() - admitted_at
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
                    fresh = admit()
                    packet = pack_session_packet(fresh, rows)
                    runtime.solve_session_packet(packet, mode=mode)

            bank_samples = {'raw': [], 'scc': []}
            pack_samples = {'raw': [], 'scc': []}
            solve_samples = {'raw': [], 'scc': []}
            application_samples = {'raw': [], 'scc': []}
            admission_samples = {'raw': [], 'scc': []}
            packet_hashes = set()
            output_hashes = {'raw': set(), 'scc': set()}
            rng = random.Random(int.from_bytes(
                hashlib.sha256(('SPECTRA-first-use-v2/' + name).encode()).digest()[:8],
                'big',
            ))
            for _ in range(repeats):
                order = ['raw', 'scc']
                rng.shuffle(order)
                for mode in order:
                    start = time.perf_counter_ns()
                    packet = pack_session_packet(problem, rows)
                    packed = time.perf_counter_ns()
                    answer = runtime.solve_session_packet(packet, mode=mode)
                    solved = time.perf_counter_ns()
                    bank_samples[mode].append(solved - start)
                    pack_samples[mode].append(packed - start)
                    solve_samples[mode].append(solved - packed)
                    packet_hashes.add(_digest(packet))
                    output_hashes[mode].add(_digest(answer.output))

                    application_start = time.perf_counter_ns()
                    fresh = admit()
                    application_admitted = time.perf_counter_ns()
                    fresh_packet = pack_session_packet(fresh, rows)
                    fresh_answer = runtime.solve_session_packet(fresh_packet, mode=mode)
                    application_end = time.perf_counter_ns()
                    application_samples[mode].append(application_end - application_start)
                    admission_samples[mode].append(
                        application_admitted - application_start
                    )
                    packet_hashes.add(_digest(fresh_packet))
                    output_hashes[mode].add(_digest(fresh_answer.output))

            if len(packet_hashes) != 1 or any(
                    len(values) != 1 for values in output_hashes.values()):
                raise AssertionError('packet or output bytes changed across repetitions')
            if output_hashes['raw'] != output_hashes['scc']:
                raise AssertionError('raw and SCC output identities differ')

            bank_medians = {
                mode: statistics.median(values)
                for mode, values in bank_samples.items()
            }
            application_medians = {
                mode: statistics.median(values)
                for mode, values in application_samples.items()
            }
            record = {
                'case': name,
                'vertices': problem.n,
                'edges': len(problem.edges),
                'queries': len(rows),
                'pins': sum(map(len, rows)),
                'problem_admission_ns': admitted_ns,
                'fixed_problem_bytes': len(problem._sbc1_fixed),
                'packet_bytes': len(frozen),
                'packet_sha256': _digest(frozen),
                'output_sha256': _digest(first['scc'].output),
                'raw_bank_samples_ns': bank_samples['raw'],
                'scc_bank_samples_ns': bank_samples['scc'],
                'raw_pack_samples_ns': pack_samples['raw'],
                'scc_pack_samples_ns': pack_samples['scc'],
                'raw_solve_samples_ns': solve_samples['raw'],
                'scc_solve_samples_ns': solve_samples['scc'],
                'raw_application_samples_ns': application_samples['raw'],
                'scc_application_samples_ns': application_samples['scc'],
                'raw_admission_samples_ns': admission_samples['raw'],
                'scc_admission_samples_ns': admission_samples['scc'],
                'raw_bank_median_ns': bank_medians['raw'],
                'scc_bank_median_ns': bank_medians['scc'],
                'raw_pack_median_ns': statistics.median(pack_samples['raw']),
                'scc_pack_median_ns': statistics.median(pack_samples['scc']),
                'raw_solve_median_ns': statistics.median(solve_samples['raw']),
                'scc_solve_median_ns': statistics.median(solve_samples['scc']),
                'raw_application_median_ns': application_medians['raw'],
                'scc_application_median_ns': application_medians['scc'],
                'raw_admission_median_ns': statistics.median(admission_samples['raw']),
                'scc_admission_median_ns': statistics.median(admission_samples['scc']),
                'bank_scc_raw_ratio': bank_medians['scc'] / bank_medians['raw'],
                'application_scc_raw_ratio': (
                    application_medians['scc'] / application_medians['raw']
                ),
            }
            records.append(record)
            print(
                name,
                f"bank_ratio={record['bank_scc_raw_ratio']:.6f}",
                f"application_ratio={record['application_scc_raw_ratio']:.6f}",
                flush=True,
            )
    finally:
        gc.enable()

    bank_ratios = [record['bank_scc_raw_ratio'] for record in records]
    application_ratios = [
        record['application_scc_raw_ratio'] for record in records
    ]
    summary = {
        'schema': SCHEMA,
        'classification': 'exposed WAP development cases; not a holdout',
        'python': sys.version,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'repeats': repeats,
        'cases': len(records),
        'bank_geometric_mean_ratio': _gmean(bank_ratios),
        'bank_max_case_ratio': max(bank_ratios),
        'application_geometric_mean_ratio': _gmean(application_ratios),
        'application_max_case_ratio': max(application_ratios),
        'limits': {
            'bank_geometric_mean': BANK_MEAN_LIMIT,
            'bank_per_case': BANK_CASE_LIMIT,
            'application_geometric_mean': APPLICATION_MEAN_LIMIT,
            'application_per_case': APPLICATION_CASE_LIMIT,
        },
        'gate': {
            'all_cases_complete_and_replayed': len(records) == len(CASES),
            'bank_geometric_mean_below_limit': (
                _gmean(bank_ratios) < BANK_MEAN_LIMIT
            ),
            'bank_every_case_below_limit': max(bank_ratios) < BANK_CASE_LIMIT,
            'application_geometric_mean_below_limit': (
                _gmean(application_ratios) < APPLICATION_MEAN_LIMIT
            ),
            'application_every_case_below_limit': (
                max(application_ratios) < APPLICATION_CASE_LIMIT
            ),
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
    payload = run(args.library.resolve(), args.cases.resolve(), args.out,
                  repeats=args.repeats)
    print(json.dumps(payload['summary'], indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
