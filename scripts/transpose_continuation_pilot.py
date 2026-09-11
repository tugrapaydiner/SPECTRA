#!/usr/bin/env python3
"""Adaptive static follow-up after the four-cycle candidate ceiling diagnosis."""
from __future__ import annotations
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
import time

if __name__ == '__main__':
    os.environ.update(ATEN_CPU_CAPABILITY='avx2', MKL_ENABLE_INSTRUCTIONS='AVX2',
        ONEDNN_MAX_CPU_ISA='AVX2', DNNL_MAX_CPU_ISA='AVX2',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from eval.failure_information import first_success, simulate, fixed_order
from eval.paired_frontier import paired_frontier
from eval.symmetry_search import symmetry_solve, grid_views
from scripts.failure_information_pilot import (seal_source, assert_source, setup, sources,
    load_cases, trajectory, write_json, sha, FAMILY_SPECS, load_sources, tensor_digest)
from scripts.m17_cross_task import symbolic_solve

PROTOCOL = ROOT/'docs/TRANSPOSE_CONTINUATION_PROTOCOL.json'
ARMS = ('original_20', 'prefix_transpose_12', 'prefix_transpose_20', 'transpose_20',
        'identity_20', 'identity_32', 'symbolic')


def predict(case, extended, arm):
    if arm == 'original_20':
        return simulate(case, fixed_order((0,1,2)))
    if arm in ('prefix_transpose_12', 'prefix_transpose_20'):
        t = first_success(case['identity'], 8)
        if t is not None:
            return {'valid':True, 'transitions':t, 'answer':case['identity'][t-1]['answer']}
        cap = 4 if arm.endswith('_12') else 12
        t = first_success(extended, cap)
        return {'valid':t is not None, 'transitions':8+(t or cap), 'answer':extended[(t or cap)-1]['answer']}
    if arm == 'transpose_20':
        steps, cap = extended, 20
    elif arm in ('identity_20','identity_32'):
        steps, cap = case['identity'], int(arm[9:])
    else:
        raise ValueError('unknown trajectory arm')
    t = first_success(steps, cap)
    return {'valid':t is not None, 'transitions':t or cap, 'answer':steps[(t or cap)-1]['answer']}


@torch.inference_mode()
def execute(core, inp, spec, arm):
    if arm == 'symbolic':
        return symbolic_solve(inp, spec, native=spec.task=='maze')
    if arm.startswith('identity_'):
        return symmetry_solve(core, inp, spec, policy='identity', transition_budget=int(arm[9:]))
    if arm == 'original_20':
        return symmetry_solve(core, inp, spec, transition_budget=20, identity_cycles=8, view_limit=4)
    if arm in ('prefix_transpose_12', 'prefix_transpose_20'):
        cap = 12 if arm.endswith('_12') else 20
        return symmetry_solve(core, inp, spec, transition_budget=cap, identity_cycles=8,
                              cycles_per_view=cap-8, view_limit=2)
    if arm != 'transpose_20':
        raise ValueError('unknown deployed arm')
    original = spec.native_problem(inp)
    view = grid_views(spec)[1]
    transformed = view.apply(inp)
    answer, work = symmetry_solve(core, transformed, spec, policy='identity', transition_budget=20)
    answer = view.restore(answer)
    work['checker_constructions'] += 1
    work['input_transforms'] += 1
    work['answer_inverse_transforms'] += 1
    if work['valid']:
        work['checks'] += 1
        work['restoration_checks'] += 1
        if not bool(original.check(answer)):
            raise RuntimeError('transpose solution failed original-input verification')
    return answer, work


def analyze(rows, collection_summary):
    if len(rows) != 512*7*3 or any(r['family'] not in FAMILY_SPECS or r['arm'] not in ARMS for r in rows):
        raise ValueError('unexpected continuation timing inventory')
    reports = {}
    for family, identity in collection_summary['identities'].items():
        reports[family] = {}
        for candidate in ('prefix_transpose_12', 'prefix_transpose_20', 'transpose_20'):
            reports[family][candidate] = {}
            for comparator in ARMS:
                if comparator == candidate:
                    continue
                reports[family][candidate][comparator] = paired_frontier(
                    [r for r in rows if r['family']==family and r['arm'] in (candidate, comparator)],
                    family=family, candidate=candidate, comparator=comparator,
                    model_seeds=tuple(map(int, identity['cores'])), example_ids=tuple(identity['ids']),
                    example_groups=tuple(identity['groups']), seed=2026091122, replicates=2000)
    return reports


def run(out, collection):
    source = seal_source(out, PROTOCOL)
    cases, summary = load_cases(collection)
    historical, accepted, members = sources()
    rng = np.random.default_rng(2026091124)
    rows, extended_rows = [], []
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as td, (out/'rows.jsonl').open('x') as stream, \
            (out/'extended.jsonl').open('x') as trajectories:
        for family, (spec, _, _) in FAMILY_SPECS.items():
            cores = load_sources(family, members, historical, accepted, Path(td))
            for seed, (core, _) in cores.items():
                before = tensor_digest(core.state_dict())
                selected = [c for c in cases if c['family']==family and c['core_seed']==seed]
                extended = {}
                for case in selected:
                    inp = torch.tensor([case['input']], dtype=torch.int64)
                    steps = trajectory(core, inp, spec, grid_views(spec)[1], 32)
                    if steps[:4] != case['views'][0]:
                        raise AssertionError('extended trajectory changed its measured prefix')
                    extended[case['example_id']] = steps
                    row = {**{k:case[k] for k in ('family','core_seed','example_id')}, 'steps':steps}
                    extended_rows.append(row)
                    trajectories.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                trajectories.flush()
                print(f'{family} model={seed} extended 128 trajectories', flush=True)
                for arm in ARMS:
                    execute(core, torch.tensor([selected[0]['input']], dtype=torch.int64), spec, arm)
                for round_id in range(3):
                    for ci in rng.permutation(len(selected)):
                        case = selected[int(ci)]
                        inp = torch.tensor([case['input']], dtype=torch.int64)
                        for ai in rng.permutation(len(ARMS)):
                            arm = ARMS[int(ai)]
                            begin = time.perf_counter_ns()
                            answer, work = execute(core, inp, spec, arm)
                            elapsed = (time.perf_counter_ns()-begin)/1e6
                            valid = bool(spec.independent_correct(inp, answer))
                            if valid != work['valid']:
                                raise AssertionError('native answer failed independent agreement')
                            if arm != 'symbolic':
                                expected = predict(case, extended[case['example_id']], arm)
                                if any([valid != expected['valid'], work['transitions'] != expected['transitions'],
                                        answer.flatten().tolist() != expected['answer']]):
                                    raise AssertionError('live continuation disagrees with trajectory prediction')
                            row = {**{k:case[k] for k in ('family','core_seed','example_id')}, 'arm':arm,
                                'round':round_id, 'latency_ms':elapsed, 'valid':valid,
                                'answer':answer.flatten().tolist(), 'work':work}
                            rows.append(row)
                            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                    stream.flush()
                    print(f'{family} model={seed} round={round_id} complete', flush=True)
                if tensor_digest(core.state_dict()) != before:
                    raise RuntimeError('frozen core changed')
    if len(rows) != 512*7*3 or len(extended_rows) != 512:
        raise ValueError('incomplete follow-up inventory')
    digests = {}
    for name in ('rows','extended'):
        raw = (out/f'{name}.jsonl').read_bytes()
        (out/f'{name}.jsonl.gz').write_bytes(gzip.compress(raw, mtime=0))
        digests[name+'_sha256'] = sha(raw)
    comparisons = analyze(rows, summary)
    assert_source(source)
    result = {'status':'COMPLETE', 'scope':'ADAPTIVELY_SELECTED_CONSUMED_DEVELOPMENT',
        **digests, 'timing_rows':len(rows), 'independently_checked_extended_answers':512*32,
        'collection_sha256':summary['cases_sha256'], 'comparisons':comparisons,
        'seconds':time.perf_counter()-start, 'new_confirmation':False, 'core_training_updates':0}
    write_json(out/'summary.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--collection', type=Path, default=ROOT/'results/failure_information/collection')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'environment.json', setup())
    try:
        run(args.out, args.collection)
        print('FOLLOWUP_COMPLETE', flush=True)
    except Exception as error:
        write_json(args.out/'failure.json', {'type':type(error).__name__, 'message':str(error)})
        raise


if __name__ == '__main__':
    main()
