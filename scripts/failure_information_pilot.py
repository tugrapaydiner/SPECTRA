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


def seal_source(out):
    """Archive executable bytes before execution; detect mutation afterwards."""
    source = source_identity()
    files = dict(source['files'])
    files[str(PROTOCOL.relative_to(ROOT))] = sha(PROTOCOL.read_bytes())
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
    for family, identity in summary['identities'].items():
        expected = {(family, int(seed), eid) for seed in identity['cores'] for eid in identity['ids']}
        if expected != {key for key in seen if key[0] == family}:
            raise ValueError('missing declared model/example pair')
        actual = {c['example_id']: c['input'] for c in cases if c['family']==family}
        x = torch.tensor([actual[eid] for eid in identity['ids']], dtype=torch.int64)
        if tensor_digest({'x': x}) != identity['input_sha256']:
            raise ValueError('input tensor differs from retained identity')
    return cases, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['collect'])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    environment = setup()
    write_json(args.out/'environment.json', environment)
    try:
        result = collect(args.out)
        print(json.dumps(result.get('coverage', result), indent=2))
    except Exception as error:
        write_json(args.out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
