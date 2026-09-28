"""Apply only to a fresh copy of one completely validated PR36 package snapshot."""
import hashlib,json,shutil
from pathlib import Path

ACCEPTED={
    'a64ac72e81febad4473731f8c21c5d3cfe3d1bbd2c8cf06713da9fa6243dc6c0',
    '6e2710594c46ca284444a6b2be94fc4aacffedbf10b94f88021cd1b351f112e8',
}


def inventory(package):
    return {p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob('*.py')) if '__pycache__' not in p.parts}


def apply(source,dest):
    source=Path(source).resolve(strict=True);dest=Path(dest).resolve()
    if dest.exists() or dest.is_relative_to(source):raise ValueError('fresh separate destination required')
    if any(p.is_symlink() for p in source.rglob('*')):raise ValueError('source symlinks not accepted')
    package=source/'m2cgen';before=inventory(package)
    digest=hashlib.sha256(json.dumps(before,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest not in ACCEPTED:raise ValueError('unvalidated source inventory')
    interpreter=package/'interpreters/interpreter.py';text=interpreter.read_text()
    if text.count('TypeGuardedCache()')!=2:raise ValueError('unexpected interpreter structure')
    shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git'))
    target=dest/'m2cgen/interpreters/interpreter.py'
    target.write_text(text.replace('type_guarded_cache import TypeGuardedCache','structural_cache import StructuralCache').replace('TypeGuardedCache()','StructuralCache()'),encoding='utf-8')
    raw=Path(__file__).with_name('cache.py').read_bytes()
    (dest/'m2cgen/interpreters/structural_cache.py').write_bytes(raw)
    result={'original_inventory_sha256':digest,'before':before,'after':inventory(dest/'m2cgen'),
            'scope':'isolated experimental exporter copy; quiescent exact stock ASTs; no model/runtime change'}
    (dest/'STRUCTURAL_PATCH.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('dest',type=Path);a=p.parse_args();print(json.dumps(apply(a.source,a.dest),indent=2))
