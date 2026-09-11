#!/usr/bin/env python3
"""Frozen-checkpoint CPU proposal experiment on consumed M17 development only.

No new confirmation, model fitting, threshold selection or historical rewriting.
The retained protocol predates this execution locally, not external preregistration.
"""
from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

if __name__ == '__main__':
    os.environ.update(ATEN_CPU_CAPABILITY='avx2', MKL_ENABLE_INSTRUCTIONS='AVX2',
        ONEDNN_MAX_CPU_ISA='AVX2', DNNL_MAX_CPU_ISA='AVX2', CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from data.ancestry import digest
from deploy.m10_native import load_extension
from eval.symmetry_search import grid_views, symmetry_solve, POLICY_ID
from scripts.m17_cross_task import symbolic_solve
from scripts.m17_models import tensor_digest
from scripts.verify_fixed_pool_replay import (Evidence, AcceptedM16, M17_INVENTORY_SHA,
    M17_ZIP_SHA, FAMILY_SPECS, frozen_inputs, load_sources, read_bound_archive, source_identity)
from scripts.m16_evidence import write_json

ARMS = ('identity_4', 'identity_32', 'repeat_identity_32', 'dihedral_32', 'symbolic')
ARM_CONFIGS = {
    'identity_4': {'policy':'identity', 'transition_budget':4},
    'identity_20': {'policy':'identity', 'transition_budget':20},
    'identity_32': {'policy':'identity', 'transition_budget':32},
    'repeat_identity_32': {'policy':'repeat_identity', 'transition_budget':32},
    'dihedral_32': {'policy':'dihedral', 'transition_budget':32},
    'support4_prefix8': {'policy':'dihedral', 'transition_budget':20, 'identity_cycles':8, 'view_limit':4},
}
REFINED_ARMS = ('identity_4', 'identity_20', 'identity_32', 'support4_prefix8', 'symbolic')


def summarize(rows: list[dict], *, rounds: int = 3, arms: tuple[str, ...] = ARMS) -> dict:
    """Reject incomplete/misaligned/repeated/nonfinite observations before reporting."""
    if type(rounds) is not int or rounds < 1 or not rows:
        raise ValueError('nonempty observations and positive integer round count required')
    if arms not in (ARMS, REFINED_ARMS):
        raise ValueError('unsupported fixed comparison inventory')
    groups = defaultdict(list)
    for r in rows:
        if (r['family'] not in FAMILY_SPECS or type(r['core_seed']) is not int
            or r['core_seed'] < 0 or not isinstance(r['example_id'], str) or not r['example_id']
            or not isinstance(r['work'], dict) or type(r['work'].get('valid')) is not bool
            or r['work']['valid'] != r['valid'] or type(r['work'].get('transitions')) is not int
            or r['work']['transitions'] < 0 or (r['valid'] and r['answer'] is None)):
            raise ValueError('invalid model/example/work identity')
        if (r['arm'] not in arms or type(r['round']) is not int or not 0 <= r['round'] < rounds
            or type(r['valid']) is not bool or isinstance(r['latency_ms'], bool)
            or not isinstance(r['latency_ms'], (float, int))
            or not math.isfinite(r['latency_ms']) or r['latency_ms'] <= 0):
            raise ValueError('malformed timing/outcome row')
        groups[(r['family'], r['core_seed'], r['example_id'], r['arm'])].append(r)
    if not groups:
        raise ValueError('no observations')
    collapsed = {}
    for key, rr in groups.items():
        if len(rr) != rounds or {r['round'] for r in rr} != set(range(rounds)):
            raise ValueError('missing or duplicate timing rounds')
        if any((r['answer'], r['valid'], r['work']) != (rr[0]['answer'], rr[0]['valid'], rr[0]['work']) for r in rr):
            raise ValueError('deterministic outcomes/work changed across repeats')
        collapsed[key] = {**rr[0], 'latency_ms': float(np.median([r['latency_ms'] for r in rr]))}
    identities = {k[:3] for k in collapsed}
    if set(collapsed) != {(*key, arm) for key in identities for arm in arms}:
        raise ValueError('arms are not paired on the same model-examples')
    for key in identities:
        anchor = collapsed[(*key, 'identity_4')]
        if 'repeat_identity_32' in arms:
            repeat = collapsed[(*key, 'repeat_identity_32')]
            if repeat['valid'] != anchor['valid'] or repeat['answer'] != anchor['answer']:
                raise ValueError('identical restarts changed deterministic four-cycle outcome')
        if anchor['valid'] and any(not collapsed[(*key, arm)]['valid'] for arm in arms):
            raise ValueError('a valid identity-prefix answer was lost')
    report = {}
    for family in sorted({k[0] for k in identities}):
        selected_ids = {k for k in identities if k[0] == family}
        arm_report = {}
        for arm in arms:
            rr = [collapsed[(*key, arm)] for key in sorted(selected_ids)]
            times = np.array([r['latency_ms'] for r in rr])
            arm_report[arm] = {
                'model_example_pairs': len(rr), 'valid_answers': sum(r['valid'] for r in rr),
                'success': sum(r['valid'] for r in rr)/len(rr),
                'mean_ms_after_round_medians': float(times.mean()),
                'median_ms_after_round_medians': float(np.median(times)),
                'p95_ms_after_round_medians': float(np.quantile(times, .95)),
                'mean_transitions': float(np.mean([r['work']['transitions'] for r in rr])),
                'new_solves_vs_identity_4': sum(r['valid'] and not collapsed[(family, r['core_seed'], r['example_id'], 'identity_4')]['valid'] for r in rr),
                'new_solves_vs_identity_32': sum(r['valid'] and not collapsed[(family, r['core_seed'], r['example_id'], 'identity_32')]['valid'] for r in rr),
                'regressions_vs_identity_32': sum(not r['valid'] and collapsed[(family, r['core_seed'], r['example_id'], 'identity_32')]['valid'] for r in rr),
                'by_model_seed': {str(seed): {'valid': sum(r['valid'] for r in rr if r['core_seed'] == seed),
                    'examples': sum(r['core_seed'] == seed for r in rr)} for seed in sorted({r['core_seed'] for r in rr})}}
        report[family] = {'arms': arm_report, 'unique_examples': len({k[2] for k in selected_ids}),
            'core_seeds': sorted({k[1] for k in selected_ids}), 'rounds': rounds,
            'scope': 'consumed development, paired descriptive results; equal transition caps are not equal CPU time'}
    return report


def seal_rows(out: Path, rows: list[dict]) -> str:
    """Require the complete in-memory observations to equal the closed raw log.

    Keep a separately checksummed compressed mirror before declaring completion.
    This does not excuse mutations after publication: later verification must
    still compare either representation against the retained content hash.
    """
    if not rows:
        raise ValueError('cannot seal an empty run')
    expected = ''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in rows).encode()
    if (out/'rows.jsonl').read_bytes() != expected:
        raise ValueError('closed raw log is incomplete or differs from executed observations')
    compressed = gzip.compress(expected, mtime=0)
    with (out/'rows.jsonl.gz').open('xb') as stream:
        stream.write(compressed)
        stream.flush()
        os.fsync(stream.fileno())
    if gzip.decompress((out/'rows.jsonl.gz').read_bytes()) != expected:
        raise ValueError('compressed raw-log mirror differs')
    return digest(expected)


def run(out: Path, protocol_path: Path) -> dict:
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    arms = tuple(protocol['arms'])
    if (arms not in (ARMS, REFINED_ARMS) or protocol['timing_rounds'] != 3
        or protocol['gpu_permitted'] or protocol['new_confirmation_permitted']
        or protocol['training_updates'] != 0 or protocol['device'] != 'cpu'
        or protocol['families'] != ['sudoku_shift', 'maze']
        or protocol['surfaces'] != ['previously_consumed_development']
        or protocol['cycles_per_view'] != 4 or protocol['threads'] != 1
        or protocol['candidate_reference_answers_permitted'] is not False
        or protocol['maze_endpoint_swap_views'] != ['rotate180', 'anti_transpose']):
        raise ValueError('protocol does not match this fixed pilot')
    source_before = source_identity()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError('CPU-only PyTorch required')
    affinity = None
    if hasattr(os, 'sched_getaffinity'):
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpus[0]})
        affinity = sorted(os.sched_getaffinity(0))
    cold_start = time.perf_counter()
    load_extension()
    cold_native_load_s = time.perf_counter()-cold_start
    historical = Evidence(ROOT/'results/m16/sources')
    accepted = AcceptedM16(ROOT/'results/m16/runs/accepted-34522192590.tar.gz', historical)
    m17_path = ROOT/'results/m17/runs/34534009702-33e548d3e0f52d0664c00929316b2198d5fcffe4.tar.gz'
    members = read_bound_archive(m17_path, inventory_sha=M17_INVENTORY_SHA, zip_sha=M17_ZIP_SHA)
    rows, identities = [], {}
    order = np.random.default_rng(protocol['timing_order_seed'])
    with tempfile.TemporaryDirectory(prefix='spectra-symmetry-proposals-') as td, (out/'rows.jsonl').open('x') as stream:
        for family in protocol['families']:
            spec, seed, _ = FAMILY_SPECS[family]
            x, ids, manifest_sha = frozen_inputs(members, f'experiment/{family}/', seed, spec)
            sources = load_sources(family, members, historical, accepted, Path(td))
            if list(sources) != protocol['model_seeds'][family] or len(ids) != protocol['examples_per_family']:
                raise ValueError('source model/example inventory mismatch')
            views = grid_views(spec)
            if [v.name for v in views] != protocol['view_order']:
                raise ValueError('view ordering differs from protocol')
            identities[family] = {'manifest_sha256': manifest_sha, 'input_tensor_sha256': tensor_digest({'x':x}),
                'views': [v.metadata() for v in views], 'cores': {}}
            for core_seed, (core, _) in sources.items():
                before = tensor_digest(core.state_dict())
                identities[family]['cores'][str(core_seed)] = before
                def execute(arm, inp):
                    if arm == 'symbolic':
                        return symbolic_solve(inp, spec, native=spec.task == 'maze')
                    return symmetry_solve(core, inp, spec, **ARM_CONFIGS[arm], cycles_per_view=4)
                # Warmup reuses consumed development inputs. It never reads targets.
                for arm in arms:
                    for i in range(2): execute(arm, x[i:i+1])
                for round_id in range(protocol['timing_rounds']):
                    for i in order.permutation(len(x)):
                        inp = x[i:i+1]
                        for ai in order.permutation(len(arms)):
                            arm = arms[int(ai)]
                            start = time.perf_counter_ns()
                            answer, work = execute(arm, inp)
                            elapsed = (time.perf_counter_ns()-start)/1e6
                            valid = spec.independent_correct(inp, answer)
                            if valid != bool(work['valid']):
                                raise AssertionError('native decision differs from independent semantic checker')
                            row = {'family': family, 'core_seed': core_seed, 'example_id': ids[int(i)],
                                   'arm': arm, 'round': round_id, 'latency_ms': elapsed,
                                   'answer': None if answer is None else answer.flatten().tolist(),
                                   'valid': bool(valid), 'work': work}
                            rows.append(row)
                            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                    stream.flush()
                    print(f'{family} core={core_seed} round={round_id} complete', flush=True)
                if tensor_digest(core.state_dict()) != before:
                    raise RuntimeError('model changed during inference audit')
    if len(rows) != 7680:
        raise ValueError('incomplete fixed pilot')
    if source_identity() != source_before or protocol_path.read_bytes() != protocol_raw:
        raise RuntimeError('source/protocol changed during the experiment')
    rows_sha256 = seal_rows(out, rows)
    return {'execution_status': 'COMPLETE', 'scientific_status': 'DEVELOPMENT_ONLY_NO_CONFIRMATION',
        'policy_id': POLICY_ID, 'families': summarize(rows, arms=arms), 'arms':list(arms), 'timing_rows': len(rows),
        'source': source_before, 'protocol_sha256': digest(protocol_raw),
        'm17_inventory_sha256': M17_INVENTORY_SHA, 'm17_container_sha256': digest(m17_path.read_bytes()),
        'm16_identity': accepted.identity(), 'identities': identities,
        'rows_sha256': rows_sha256,
        'cold_native_load_seconds_not_in_warm_timings': cold_native_load_s,
        'environment': {'device':'cpu', 'torch':str(torch.__version__), 'numpy':np.__version__,
            'python':platform.python_version(), 'torch_threads':torch.get_num_threads(), 'affinity':affinity,
            'cpu_model': next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines()
                               if l.startswith('model name')), 'unknown'),
            'dispatch':{k:os.environ.get(k) for k in ('ATEN_CPU_CAPABILITY','MKL_ENABLE_INSTRUCTIONS','MKL_CBWR',
                'ONEDNN_MAX_CPU_ISA','DNNL_MAX_CPU_ISA')}},
        'training_updates':0, 'new_confirmation_generated':False, 'reference_answer_used_for_inference':False,
        'physical_energy_joules':None, 'energy_reason':'not_measured',
        'claim_boundary':protocol['claims']}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, default=ROOT/'docs/SYMMETRY_PROPOSAL_PROTOCOL.json')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        report = run(args.out, args.protocol)
        write_json(args.out/'summary.json', report)
        print(json.dumps({'execution_status':report['execution_status'], 'families':report['families']}, indent=2))
    except Exception as exc:
        write_json(args.out/'failure.json', {'status':'FAIL', 'exception':type(exc).__name__,
            'message':str(exc), 'traceback':traceback.format_exc()})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
