"""Explicit strict-arithmetic local build; never compile as an import side effect."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time


def build(out, target='portable', sanitize=False):
    if target not in ('portable', 'avx2') or type(sanitize) is not bool:
        raise ValueError('invalid target/settings')
    if sys.platform != 'linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise ValueError('accepted build scope is Linux x86-64')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('runtime.cpp')
    base = source.parent.parent / 'certified_trees/runtime.cpp'
    library = out / 'total.so'
    command = ['g++', '-std=c++17', '-O3', '-fno-fast-math', '-ffp-contract=off',
               '-fPIC', '-shared', '-Wl,-z,defs']
    if target == 'avx2':
        command += ['-mavx2', '-DTC_FAST_END', '-DTC_VECTOR_ROUTING']
    if sanitize:
        command += ['-O1', '-g', '-fsanitize=undefined', '-fno-sanitize-recover=all']
    command += [f'-DTT_BASE_RUNTIME="{base}"', str(source), '-ldl', '-o', str(library)]
    receipt = {'command': command, 'target': target, 'sanitize': sanitize,
               'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, base)}}
    start = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        receipt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        receipt.update(returncode=None, error=str(error))
    receipt['elapsed_seconds'] = time.perf_counter() - start
    if receipt['returncode'] == 0:
        receipt['library_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
    (out / 'build.json').write_text(json.dumps(receipt, indent=2))
    if receipt['returncode'] != 0:
        raise RuntimeError('explicit build failed; see build.json')
    return library


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--target', choices=['portable', 'avx2'], default='portable')
    p.add_argument('--ubsan', action='store_true')
    args = p.parse_args()
    print(build(args.out, args.target, args.ubsan))
