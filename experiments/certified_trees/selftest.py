"""Frozen SDK replay: source-verified compact inference and real native fallback.

Run with -I -S. No Python numerical frameworks, training, downloading or compilation.
Output must be outside the SDK so its exact file-closure manifest remains stable.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_manifest(root):
    manifest_path = root / 'SDK_MANIFEST.json'
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError('invalid SDK manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format') != 'spectra.tree.sdk.v1' or not manifest.get('files'):
        raise ValueError('unknown or empty SDK inventory')
    actual = set()
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError('symlink in SDK')
        if p.is_file() and p != manifest_path:
            actual.add(p.relative_to(root).as_posix())
    if actual != set(manifest['files']):
        raise ValueError('missing or extra SDK file; place output outside SDK')
    for name, record in manifest['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:
            raise ValueError('unsafe SDK path')
        p = root / path
        if p.stat().st_size != record['bytes'] or sha(p) != record['sha256']:
            raise ValueError('changed SDK member: ' + name)
    return manifest


def run(root, target):
    root = Path(root).resolve(strict=True)
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise ValueError('this SDK contains Linux x86-64 builds only')
    if target not in ('portable', 'avx2'):
        raise ValueError('unsupported native target')
    if target == 'avx2' and 'avx2' not in Path('/proc/cpuinfo').read_text().lower():
        raise ValueError('AVX2 target needs a compatible processor')
    manifest = verify_manifest(root)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    from experiments.certified_trees.session import VerifiedCompact, TreeSession
    library = root / 'native' / target / 'trees.so'
    official = root / 'official' / 'libcatboostmodel-linux-x86_64-1.2.10.so'
    results = []
    start = time.process_time()
    for task, entry in manifest['models'].items():
        folder = root / 'models' / task
        raw = bytearray((folder / 'input.u8').read_bytes())
        n, d = entry['rows'], entry['features']
        if len(raw) != n * d:
            raise ValueError('input size mismatch')
        expected = list(struct.unpack('<' + str(n) + 'i', (folder / 'indices.i32').read_bytes()))
        proofs = {b: VerifiedCompact.from_files(folder / 'model.json', folder / f'model-{b}.sct') for b in (8, 16)}
        model_result = {'task': task, 'rows': n, 'compact': {}, 'full_coverage': {}}
        for bits in (8, 16):
            with TreeSession(library, first=proofs[bits]) as runner:
                result = runner.inspect_buffer(raw)
                if any(p != -1 and p != e for p, e in zip(result['indices'], expected)):
                    raise AssertionError('incorrect certified class')
                if result['work']['certified_first'] != entry[f'certified_{bits}']:
                    raise AssertionError('changed certified count')
                if runner.predict_buffer(raw, mode='scalar') != result['indices']:
                    raise AssertionError('scalar/selected-target disagreement')
                model_result['compact'][str(bits)] = result['work']
        for name, bits, refine in (('16_first', 16, False), ('8_to_16', 8, True)):
            with TreeSession(library, first=proofs[bits], second=proofs[16] if refine else None,
                             official_model=folder / 'model.cbm', official_library=official) as runner:
                result = runner.inspect_buffer(raw, refine=refine, fallback=True)
                if result['indices'] != expected or result['work']['unresolved_rows']:
                    raise AssertionError('full-coverage source disagreement')
                if result['work']['official_rows'] != n - entry['certified_16']:
                    raise AssertionError('fallback count changed')
                model_result['full_coverage'][name] = result['work']
                # Actual new-label result, not just a receipt parser.
                one = runner.predict_buffer(array('B', raw[:d]), refine=refine, fallback=True)
                if one != expected[:1]:
                    raise AssertionError('single input disagrees')
        results.append(model_result)
        print(task + ': source-verified compact and actual fallback replay PASS', flush=True)
    forbidden = {'numpy', 'scipy', 'sklearn', 'catboost', 'torch', 'pandas'} & sys.modules.keys()
    if forbidden:
        raise AssertionError('numerical Python framework imported: ' + repr(forbidden))
    verify_manifest(root)
    return {'status': 'PASS', 'target': target, 'source_models': len(results),
            'underlying_rows': sum(r['rows'] for r in results),
            'compact16_certified': sum(r['compact']['16']['certified_first'] for r in results),
            'official_fallback_rows': sum(r['full_coverage']['16_first']['official_rows'] for r in results),
            'cpu_seconds_including_exact_source_verification': time.process_time() - start,
            'sdk_manifest_sha256': sha(root / 'SDK_MANIFEST.json'),
            'runner_sha256': sha(__file__), 'python': sys.version,
            'platform': platform.platform(), 'results': results,
            'scope': 'same frozen source-model replay, not new accuracy, timing or independent researcher evidence'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--target', choices=('portable', 'avx2'), default='portable')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.resolve().is_relative_to(args.sdk.resolve()):
        parser.error('use a new result path outside the SDK directory')
    result = run(args.sdk, args.target)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, indent=2))
