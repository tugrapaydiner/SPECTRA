#!/usr/bin/env python3
"""Collect complete frozen trajectories, then fit and measure a bounded pilot.

Every input was already consumed by M17 development. No new task data or frozen
confirmation surfaces are used. Core parameters never receive training updates.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
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
from data.sudoku_symmetry import sudoku4_orbit_key
from eval.failure_information import (Observation, LinearSelector, MODES, coverage,
    first_success, simulate, fixed_order, input_random_order, fit_selector,
    best_static_order, solve_with_selector)
from eval.symmetry_search import grid_views, symmetry_solve
from eval.paired_frontier import paired_frontier
from scripts.m17_cross_task import symbolic_solve
from scripts.m17_models import tensor_digest
from scripts.verify_fixed_pool_replay import (Evidence, AcceptedM16, read_bound_archive,
    M17_INVENTORY_SHA, M17_ZIP_SHA, FAMILY_SPECS, load_sources, frozen_inputs, source_identity)

PROTOCOL = ROOT/'docs/FAILURE_INFORMATION_PROTOCOL.json'
ARMS = ('original_order', 'best_static', 'hash_random', *MODES, 'identity_20', 'identity_32', 'symbolic')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def seal_source(out, extra_protocol=None):
    """Archive executable bytes before execution; detect mutation afterwards."""
    source = source_identity()
    files = dict(source['files'])
    files[str(PROTOCOL.relative_to(ROOT))] = sha(PROTOCOL.read_bytes())
    if extra_protocol is not None:
        files[str(extra_protocol.relative_to(ROOT))] = sha(extra_protocol.read_bytes())
    with tarfile.open(out/'source.tar.gz', 'w:gz') as archive:
        for name in sorted(files):
            raw = (ROOT/name).read_bytes()
            if sha(raw) != files[name]:
                raise RuntimeError('source changed before sealing')
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(raw), 0, 0o644
            archive.addfile(info, io.BytesIO(raw))
    receipt = {'files': files, 'archive_sha256': sha((out/'source.tar.gz').read_bytes()),
               'protocol_sha256': files[str(PROTOCOL.relative_to(ROOT))], 'source': source}
    write_json(out/'pre_execution_source.json', receipt)
    return receipt


def assert_source(receipt):
    if any(sha((ROOT/name).read_bytes()) != digest for name, digest in receipt['files'].items()):
        raise RuntimeError('source/protocol changed during execution')


def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return {'torch': str(torch.__version__), 'numpy': np.__version__,
        'python': platform.python_version(), 'device': 'cpu', 'threads': 1,
        'gpu_available': torch.cuda.is_available(), 'physical_energy_joules': None}


def sources():
    historical = Evidence(ROOT/'results/m16/sources')
    accepted = AcceptedM16(ROOT/'results/m16/runs/accepted-34522192590.tar.gz', historical)
    path = ROOT/'results/m17/runs/34534009702-33e548d3e0f52d0664c00929316b2198d5fcffe4.tar.gz'
    members = read_bound_archive(path, inventory_sha=M17_INVENTORY_SHA, zip_sha=M17_ZIP_SHA)
    return historical, accepted, members


def group_key(inp, spec):
    if spec.task == 'sudoku':
        return sudoku4_orbit_key(inp.numpy().ravel())
    # All 8 maps x either endpoint labeling. The subset used for inference need
    # not itself be a group; the exclusion group includes both labelings.
    variants = []
    for view in grid_views(spec):
        v = view.apply(inp).numpy().ravel()
        variants.extend([v.astype(np.uint8).tobytes(),
                         np.array([0, 1, 3, 2, 4], dtype=np.uint8)[v].tobytes()])
    return sha(b'spectra.maze.d4.endpoint.v1\0'+min(variants))


def fold_for(group):
    return int(sha(group.encode()), 16) % 5


@torch.inference_mode()
def trajectory(core, inp, spec, view, cycles):
    transformed = view.apply(inp)
    problem = spec.native_problem(transformed)
    embedded = core.token_embed(transformed)+core.encode_positions(transformed, spec.height, spec.width)
    y, z = torch.zeros_like(embedded), torch.zeros_like(embedded)
    result = []
    for _ in range(cycles):
        y, z = core.recursive_cycle(embedded, y, z)
        answer, valid = problem.decode(core.out_head(y).contiguous())
        restored = view.restore(answer)
        independent = bool(spec.independent_correct(inp, restored))
        if bool(valid) != independent:
            raise AssertionError('native/transformed decision differs from independent original-input checker')
        result.append({'answer': restored.flatten().tolist(), 'valid': independent})
    return result


def collect(out):
    source = seal_source(out)
    start = time.perf_counter()
    historical, accepted, members = sources()
    identities, cases = {}, []
    protocol = json.loads(PROTOCOL.read_text())
    with tempfile.TemporaryDirectory() as td, (out/'cases.jsonl').open('x') as stream:
        for family, (spec, seed, _) in FAMILY_SPECS.items():
            inputs, ids, manifest_sha = frozen_inputs(members, f'experiment/{family}/', seed, spec)
            cores = load_sources(family, members, historical, accepted, Path(td))
            if len(ids) != 128 or list(cores) != protocol['models'][family]:
                raise ValueError('source inventory differs from the protocol')
            views = grid_views(spec)
            if [v.name for v in views[1:]] != protocol['alternative_views']:
                raise ValueError('view inventory changed')
            groups = [group_key(x[None], spec) for x in inputs]
            identities[family] = {'ids': ids, 'groups': groups, 'manifest_sha256': manifest_sha,
                'input_sha256': tensor_digest({'x': inputs}), 'cores': {}}
            for core_seed, (core, _) in cores.items():
                before = tensor_digest(core.state_dict())
                identities[family]['cores'][str(core_seed)] = before
                for i, example_id in enumerate(ids):
                    inp = inputs[i:i+1]
                    case = {'family': family, 'core_seed': core_seed, 'example_id': example_id,
                        'input': inp.flatten().tolist(), 'group': groups[i], 'fold': fold_for(groups[i]),
                        'identity': trajectory(core, inp, spec, views[0], 32),
                        'views': [trajectory(core, inp, spec, view, 4) for view in views[1:]]}
                    cases.append(case)
                    stream.write(json.dumps(case, sort_keys=True, allow_nan=False)+'\n')
                    if (i+1) % 32 == 0:
                        stream.flush()
                        print(f'{family} model={core_seed} complete={i+1}/128', flush=True)
                if tensor_digest(core.state_dict()) != before:
                    raise RuntimeError('frozen model changed')
    raw = (out/'cases.jsonl').read_bytes()
    (out/'cases.jsonl.gz').write_bytes(gzip.compress(raw, mtime=0))
    assert_source(source)
    summary = {'status': 'COMPLETE', 'scope': protocol['status'], 'cases': len(cases),
        'independently_checked_answers': len(cases)*60, 'cases_sha256': sha(raw),
        'identities': identities, 'core_training_updates': 0, 'new_confirmation': False,
        'collection_seconds': time.perf_counter()-start,
        'coverage': {f: coverage([c for c in cases if c['family']==f]) for f in FAMILY_SPECS}}
    write_json(out/'summary.json', summary)
    return summary


def load_cases(directory):
    summary = json.loads((directory/'summary.json').read_text())
    raw = gzip.decompress((directory/'cases.jsonl.gz').read_bytes())
    if sha(raw) != summary['cases_sha256']:
        raise ValueError('trajectory record digest mismatch')
    cases = [json.loads(line) for line in raw.splitlines()]
    if summary['status'] != 'COMPLETE' or len(cases) != 512:
        raise ValueError('incomplete trajectory inventory')
    seen = set()
    for case in cases:
        family, seed, eid = (case[k] for k in ('family', 'core_seed', 'example_id'))
        identity = summary['identities'][family]
        if (type(seed) is not int or str(seed) not in identity['cores'] or eid not in identity['ids']
                or (family, seed, eid) in seen):
            raise ValueError('unexpected or repeated trajectory identity')
        seen.add((family, seed, eid))
        spec = FAMILY_SPECS[family][0]
        inp = torch.tensor([case['input']], dtype=torch.int64)
        expected_group = group_key(inp, spec)
        if (case['group'] != expected_group or case['fold'] != fold_for(expected_group)
                or expected_group != identity['groups'][identity['ids'].index(eid)]
                or len(case['identity']) != 32 or len(case['views']) != 7
                or any(len(v) != 4 for v in case['views'])):
            raise ValueError('invalid input group, fold or trajectory lengths')
        for step in case['identity'] + [s for view in case['views'] for s in view]:
            Observation(tuple(case['input']), tuple(step['answer']))
            if type(step['valid']) is not bool:
                raise ValueError('trajectory validity must be boolean')
    _, _, members = sources()
    for family, identity in summary['identities'].items():
        spec, seed, _ = FAMILY_SPECS[family]
        expected_models = {1401,2402} if family=='sudoku_shift' else {1701,2702}
        if set(identity['cores']) != {str(s) for s in expected_models} or len(identity['ids']) != 128:
            raise ValueError('declared model/example inventory differs from frozen source slots')
        original, ids, manifest_sha = frozen_inputs(members, f'experiment/{family}/', seed, spec)
        if (identity['ids'] != ids or identity['manifest_sha256'] != manifest_sha
                or identity['input_sha256'] != tensor_digest({'x': original})):
            raise ValueError('declared inventory differs from pinned M17 source')
        expected = {(family, int(seed), eid) for seed in identity['cores'] for eid in identity['ids']}
        if expected != {key for key in seen if key[0] == family}:
            raise ValueError('missing declared model/example pair')
        for core_seed in identity['cores']:
            actual = {c['example_id']: c['input'] for c in cases
                      if c['family']==family and c['core_seed']==int(core_seed)}
            x = torch.tensor([actual[eid] for eid in identity['ids']], dtype=torch.int64)
            if tensor_digest({'x': x}) != identity['input_sha256']:
                raise ValueError('input tensor differs from retained identity')
    return cases, summary


def crossfit_partition(cases, fold):
    if type(fold) is not int or fold not in range(5):
        raise ValueError('invalid cross-fit fold')
    train = [c for c in cases if c['fold'] != fold]
    test = [c for c in cases if c['fold'] == fold]
    if not train or not test or {c['group'] for c in train} & {c['group'] for c in test}:
        raise ValueError('empty or overlapping cross-fit partition')
    return train, test


def fit(out, collection):
    source = seal_source(out)
    cases, collection_summary = load_cases(collection)
    weights, folds, predictions = {}, {}, []
    start = time.perf_counter()
    for family in FAMILY_SPECS:
        selected = [c for c in cases if c['family']==family]
        for fold in range(5):
            train, test = crossfit_partition(selected, fold)
            key = f'{family}_{fold}'
            order = best_static_order(train)
            policies = {mode: fit_selector(train, mode) for mode in MODES}
            for mode, policy in policies.items():
                weights[f'{key}_{mode}'] = policy.weights
            folds[key] = {'order': list(order),
                'train': [[c['core_seed'], c['example_id']] for c in train],
                'test': [[c['core_seed'], c['example_id']] for c in test],
                'train_groups': sorted({c['group'] for c in train}),
                'test_groups': sorted({c['group'] for c in test}),
                'residual_training_pairs': sum(first_success(c['identity'], 8) is None for c in train)}
            for case in test:
                choices = {'original_order': fixed_order((0, 1, 2)),
                    'best_static': fixed_order(order), 'hash_random': fixed_order(input_random_order(case['input'])),
                    **{mode: policy.choose for mode, policy in policies.items()}}
                for arm, choose in choices.items():
                    for budget in (12, 16, 20):
                        predictions.append({**{k:case[k] for k in ('family','core_seed','example_id','group','fold')},
                            'arm': arm, 'budget': budget, **simulate(case, choose, budget)})
            print(f'{family} fold={fold} train={len(train)} test={len(test)} order={order}', flush=True)
    np.savez_compressed(out/'policies.npz', **weights)
    raw = ''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in predictions).encode()
    (out/'predictions.jsonl.gz').write_bytes(gzip.compress(raw, mtime=0))
    assert_source(source)
    summary = {'status': 'COMPLETE', 'scope': 'GROUP_CROSSFIT_ON_CONSUMED_DEVELOPMENT',
        'collection_sha256': collection_summary['cases_sha256'], 'folds': folds,
        'policies_sha256': sha((out/'policies.npz').read_bytes()), 'predictions_sha256': sha(raw),
        'fitting_seconds': time.perf_counter()-start, 'controller_fits': len(weights),
        'core_training_updates': 0, 'new_confirmation': False,
        'curves': {family: {arm: {str(budget): {'valid': sum(r['valid'] for r in predictions
            if r['family']==family and r['arm']==arm and r['budget']==budget),
            'transitions': sum(r['transitions'] for r in predictions
            if r['family']==family and r['arm']==arm and r['budget']==budget)}
            for budget in (12,16,20)} for arm in ARMS[:6]} for family in FAMILY_SPECS}}
    write_json(out/'summary.json', summary)
    return summary


def load_policies(directory, cases, collection_summary):
    summary = json.loads((directory/'summary.json').read_text())
    if summary['status'] != 'COMPLETE' or summary['collection_sha256'] != collection_summary['cases_sha256']:
        raise ValueError('policy fit is not bound to this complete collection')
    raw = (directory/'policies.npz').read_bytes()
    if sha(raw) != summary['policies_sha256']:
        raise ValueError('policy coefficient digest mismatch')
    policies = {}
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        expected = {f'{f}_{fold}_{mode}' for f in FAMILY_SPECS for fold in range(5) for mode in MODES}
        if set(arrays.files) != expected:
            raise ValueError('missing or extra policy coefficients')
        for family in FAMILY_SPECS:
            for fold in range(5):
                key = f'{family}_{fold}'
                train, test = crossfit_partition([c for c in cases if c['family']==family], fold)
                record = summary['folds'][key]
                if (record['train'] != [[c['core_seed'], c['example_id']] for c in train]
                        or record['test'] != [[c['core_seed'], c['example_id']] for c in test]
                        or record['train_groups'] != sorted({c['group'] for c in train})
                        or record['test_groups'] != sorted({c['group'] for c in test})):
                    raise ValueError('policy fit membership differs from the group exclusion contract')
                policies[key] = {mode: LinearSelector(mode, arrays[f'{key}_{mode}']) for mode in MODES}
    return policies, summary


def selector_for(arm, case, policies, fit_summary):
    key = f"{case['family']}_{case['fold']}"
    if arm == 'best_static':
        return fixed_order(fit_summary['folds'][key]['order'])
    if arm == 'original_order':
        return fixed_order((0, 1, 2))
    if arm == 'hash_random':
        return fixed_order(input_random_order(case['input']))
    return policies[key][arm].choose


def timed_analysis(rows, cases, collection_summary):
    # paired_frontier checks every arm's declared inventory and deterministic
    # round agreement. Groups are resampled whole, jointly across model seeds.
    if len(rows) != 512*9*3 or any(r['family'] not in FAMILY_SPECS or r['arm'] not in ARMS for r in rows):
        raise ValueError('unexpected complete timing inventory')
    reports = {}
    for family, identity in collection_summary['identities'].items():
        reports[family] = {}
        for comparator in ARMS:
            if comparator == 'failure_aware':
                continue
            reports[family][comparator] = paired_frontier(
                [r for r in rows if r['family']==family and r['arm'] in ('failure_aware', comparator)],
                family=family, candidate='failure_aware', comparator=comparator,
                model_seeds=tuple(map(int, identity['cores'])), example_ids=tuple(identity['ids']),
                example_groups=tuple(identity['groups']), replicates=2000, seed=2026091122)
    return reports


def benchmark(out, collection, fit_dir):
    source = seal_source(out)
    cases, collection_summary = load_cases(collection)
    policies, fit_summary = load_policies(fit_dir, cases, collection_summary)
    historical, accepted, members = sources()
    rng = np.random.default_rng(2026091121)
    rows = []
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as td, (out/'rows.jsonl').open('x') as stream:
        for family, (spec, _, _) in FAMILY_SPECS.items():
            cores = load_sources(family, members, historical, accepted, Path(td))
            for seed, (core, _) in cores.items():
                before = tensor_digest(core.state_dict())
                selected = [c for c in cases if c['family']==family and c['core_seed']==seed]
                def execute(arm, case, inp):
                    if arm == 'symbolic':
                        return symbolic_solve(inp, spec, native=spec.task=='maze')
                    if arm.startswith('identity_'):
                        return symmetry_solve(core, inp, spec, policy='identity', transition_budget=int(arm[9:]))
                    if arm == 'original_order':
                        return symmetry_solve(core, inp, spec, transition_budget=20, identity_cycles=8, view_limit=4)
                    # Hashing and selector construction belong inside measured
                    # solves. Loading the frozen policy coefficients does not.
                    return solve_with_selector(core, inp, spec, selector_for(arm, case, policies, fit_summary))
                for arm in ARMS:
                    execute(arm, selected[0], torch.tensor([selected[0]['input']], dtype=torch.int64))
                for round_id in range(3):
                    for ci in rng.permutation(len(selected)):
                        case = selected[int(ci)]
                        inp = torch.tensor([case['input']], dtype=torch.int64)
                        for ai in rng.permutation(len(ARMS)):
                            arm = ARMS[int(ai)]
                            begin = time.perf_counter_ns()
                            answer, work = execute(arm, case, inp)
                            latency = (time.perf_counter_ns()-begin)/1e6
                            valid = bool(spec.independent_correct(inp, answer))
                            if valid != work['valid']:
                                raise AssertionError('deployed validity differs from independent checker')
                            if arm in ARMS[:6]:
                                expected = simulate(case, selector_for(arm, case, policies, fit_summary))
                                if (valid != expected['valid'] or answer.flatten().tolist() != expected['answer']
                                        or work['transitions'] != expected['transitions']):
                                    raise AssertionError('actual inference differs from counterfactual prediction')
                                if arm != 'original_order' and work['actions'] != expected['actions']:
                                    raise AssertionError('actual failure-conditioned decisions differ')
                            row = {**{k: case[k] for k in ('family','core_seed','example_id')}, 'arm': arm,
                                'round': round_id, 'latency_ms': latency, 'valid': valid,
                                'answer': answer.flatten().tolist(), 'work': work}
                            rows.append(row)
                            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                    stream.flush()
                    print(f'{family} model={seed} round={round_id} complete', flush=True)
                if tensor_digest(core.state_dict()) != before:
                    raise RuntimeError('frozen model changed during benchmark')
    raw = (out/'rows.jsonl').read_bytes()
    if len(rows) != 512*9*3 or len(raw.splitlines()) != len(rows):
        raise ValueError('incomplete timing inventory')
    (out/'rows.jsonl.gz').write_bytes(gzip.compress(raw, mtime=0))
    comparisons = timed_analysis(rows, cases, collection_summary)
    assert_source(source)
    primary_pass = all(reports['prefix_only']['descriptive_gate_pass'] for reports in comparisons.values())
    summary = {'status': 'COMPLETE', 'scope': 'CONSUMED_DEVELOPMENT_ONLY',
        'timing_rows': len(rows), 'rows_sha256': sha(raw),
        'collection_sha256': collection_summary['cases_sha256'],
        'policies_sha256': fit_summary['policies_sha256'], 'comparisons': comparisons,
        'benchmark_seconds': time.perf_counter()-start,
        'primary_pilot_gate_pass': primary_pass, 'new_confirmation': False,
        'core_training_updates': 0,
        'bootstrap_scope': 'Conditional on these fitted cross-fit policies. No controller refitting inside resampling; no correction for prior adaptive development or multiple comparisons.'}
    write_json(out/'summary.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['collect', 'fit', 'benchmark'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--collection', type=Path, default=ROOT/'results/failure_information/collection')
    ap.add_argument('--fit', type=Path, default=ROOT/'results/failure_information/fit')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    environment = setup()
    write_json(args.out/'environment.json', environment)
    try:
        if args.stage == 'collect':
            result = collect(args.out)
        elif args.stage == 'fit':
            result = fit(args.out, args.collection)
        else:
            result = benchmark(args.out, args.collection, args.fit)
        print(json.dumps(result.get('coverage', result.get('curves', {'status':result['status']})), indent=2))
    except Exception as error:
        write_json(args.out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
