"""Apply a bounded patch to a COPY of pinned upstream source, never site-packages.

stock: no changes; empty: cheaper impossible lookup on an empty private cache;
guard: previous type-filter algorithm; single: filter plus one lookup per hit.
Unknown source geometry fails before writing any file. This script executes no
model or upstream installer. Destination must not exist.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

LOOKUP = ('        if expr in self._cached_expr_results:\n'
          '            return self._cached_expr_results[expr].var_name')
SINGLE = ('        cached = self._cached_expr_results.get(expr, CACHE_MISS)\n'
          '        if cached is not CACHE_MISS:\n'
          '            return cached.var_name')


def apply(source: Path, destination: Path, variant: str):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if variant not in ('stock', 'empty', 'guard', 'single'):
        raise ValueError('unsupported variant')
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('use a fresh separate destination')
    package = source / 'm2cgen'
    if not (package / 'ast.py').is_file():
        raise ValueError('source must contain the original m2cgen package')
    interpreter = package / 'interpreters/interpreter.py'
    identity = (hashlib.sha256((package/'ast.py').read_bytes()).hexdigest(),
                hashlib.sha256(interpreter.read_bytes()).hexdigest())
    known = {
        ('fc4712516653f672c5e33eaa813e9655dec74f1e13a0a8b09b01c93da671e62c',
         'e6bb10e359e3ccb0f5e8973b51cbeb7b3eb96d660525243b052ac941fa05a778'),
        ('70da81a16119e9579e56f5bd2e3fd912f6eb8da5c483a9cefc71f307180054ba',
         'fa893e24d26fd702d871240a1bfb1c0ab76258affac3a5d216ba68630c7b8400'),
    }
    if identity not in known:
        raise ValueError('unvalidated AST/interpreter source; refusing guessed semantics')
    original = interpreter.read_text(encoding='utf-8')
    if original.count('self._cached_expr_results = {}') != 2 or original.count(LOOKUP) != 2:
        raise ValueError('unexpected cache construction or lookup sites')
    changes = {}
    sites = {}
    if variant != 'stock':
        for path in sorted((package / 'interpreters').rglob('*.py')):
            text = path.read_text(encoding='utf-8')
            n = text.count(LOOKUP)
            if n:
                sites[str(path.relative_to(source))] = n
                if variant == 'single':
                    text = 'from m2cgen.interpreters.type_guarded_cache import CACHE_MISS\n' + text.replace(LOOKUP, SINGLE)
                elif variant == 'empty':
                    text = text.replace(LOOKUP, LOOKUP.replace('if expr in', 'if self._cached_expr_results and expr in'))
            if path == interpreter and variant in ('guard', 'single'):
                text = ('from m2cgen.interpreters.type_guarded_cache import TypeGuardedCache\n' +
                        text.replace('self._cached_expr_results = {}', 'self._cached_expr_results = TypeGuardedCache()'))
            if text != path.read_text(encoding='utf-8'):
                changes[path.relative_to(source).as_posix()] = text.encode('utf-8')
        # Both pinned source snapshots have five lookup sites.
        if sum(sites.values()) != 5:
            raise ValueError('unexpected cache lookup inventory')
        if variant in ('guard', 'single'):
            raw = Path(__file__).with_name('cache.py').read_bytes()
            changes['m2cgen/interpreters/type_guarded_cache.py'] = raw
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '.git'))
    before = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
              for name in changes if (source/name).exists()}
    for name, raw in changes.items():
        (destination/name).write_bytes(raw)
    result = {'variant': variant, 'lookup_sites': sites, 'changed_files': len(changes),
              'original_sha256': before,
              'patched_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in changes.items()}}
    (destination/'SPECTRA_PATCH_RECEIPT.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--variant', choices=['stock', 'empty', 'guard', 'single'], required=True)
    a = p.parse_args()
    print(json.dumps(apply(a.source, a.destination, a.variant), indent=2))
