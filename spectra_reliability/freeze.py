"""Exclusive-create experiment freezes and tamper-detecting confirmation entry.

This binds declared code, data and fitted artifacts. It is not a cryptographic
proof of chronology by itself; commit the freeze to Git before opening holdout.
Documentation/results may be added without changing scientific execution code.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .identity import canonical_json, file_sha256, strict_json, write_json

CODE_DIRECTORIES=('common','config','data','deploy','eval','kernel','kernels','model','scripts','train','spectra_reliability')
CODE_SUFFIXES={'.py','.cpp','.cc','.c','.h','.hpp','.yaml','.yml','.toml'}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def scientific_source_inventory(root:Path) -> dict[str,str]:
    result={}
    for name in CODE_DIRECTORIES:
        directory=root/name
        if not directory.exists():continue
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in CODE_SUFFIXES and '__pycache__' not in path.parts:
                if path.is_symlink():raise ValueError('scientific code symlinks are not supported')
                result[path.relative_to(root).as_posix()]=file_sha256(path)
    for name in ('setup.py','pyproject.toml','requirements.txt','requirements-cpu.lock'):
        path=root/name
        if path.is_file():result[name]=file_sha256(path)
    if not any(key.startswith('spectra_reliability/') for key in result):
        raise ValueError('reliability source inventory is empty')
    return result


def artifact_inventory(output:Path) -> dict[str,str]:
    result={}
    for name in ('recipe.json','prepare_complete.json','fit_complete.json','development_complete.json'):
        path=output/name
        if not path.is_file():raise ValueError(f'missing completed development evidence: {name}')
        result[name]=file_sha256(path)
    for name in ('sources','lineage','auxiliary'):
        directory=output/name
        if not directory.is_dir():raise ValueError(f'missing source directory: {name}')
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                if path.is_symlink():raise ValueError('artifact symlinks are not supported')
                result[path.relative_to(output).as_posix()]=file_sha256(path)
    for stage in ('fit','validation','development'):
        for suffix in ('.json','.npz','_audit.json'):
            path=output/'data'/f'{stage}{suffix}'
            if not path.is_file():raise ValueError(f'missing data artifact: {path.name}')
            result[path.relative_to(output).as_posix()]=file_sha256(path)
    history=output/'development_repair_history'
    if history.exists():
        for path in sorted(history.rglob('*')):
            if path.is_file():result[path.relative_to(output).as_posix()]=file_sha256(path)
    # Every raw development row is bound, not only the favorable aggregate.
    for surface in ('validation','development'):
        directory=output/'evaluation'/surface
        if not (directory/'aggregate.json').is_file():raise ValueError('development surface incomplete')
        for path in sorted(directory.rglob('*')):
            if path.is_file():result[path.relative_to(output).as_posix()]=file_sha256(path)
    return result


def create_freeze(root:Path,output:Path,source_commit:str) -> dict[str,Any]:
    if not isinstance(source_commit,str) or re.fullmatch(r'[0-9a-f]{40}',source_commit) is None:
        raise ValueError('a full published source commit SHA is required')
    for name in ('confirmation','shift'):
        if any((output/'data').glob(f'{name}*')) or (output/'evaluation'/name).exists():
            raise ValueError('holdout content exists before freeze; this attempt cannot be declared unopened')
    if (output/'confirmation_opened.json').exists():raise ValueError('confirmation already opened')
    source=scientific_source_inventory(root);artifacts=artifact_inventory(output)
    record={'schema':'spectra.m16_confirmation_freeze.v1','created_utc':utc_now(),
        'source_commit':source_commit,'scientific_source_files':source,'artifacts':artifacts,
        'scientific_source_inventory_sha256':hashlib.sha256(canonical_json(source)).hexdigest(),
        'artifact_inventory_sha256':hashlib.sha256(canonical_json(artifacts)).hexdigest(),
        'candidate':'baseline_first_best_first_survival','candidate_selection':'predeclared_before_fitting',
        'confirmation_seed':160104,'shift_seed':160105,'confirmation_generated':False,
        'no_post_confirmation_selection':True,'scope':'conditional two-fixed-core 4x4 diagnostic',
        'chronology_note':'commit this record before confirmation; a local timestamp alone is not independent chronology evidence'}
    write_json(output/'confirmation_freeze.json',record,exclusive=True)
    return record


def verify_freeze(root:Path,output:Path) -> dict[str,Any]:
    path=output/'confirmation_freeze.json';record=strict_json(path.read_bytes())
    if record.get('schema')!='spectra.m16_confirmation_freeze.v1':raise ValueError('unknown freeze schema')
    actual=scientific_source_inventory(root)
    if actual!=record['scientific_source_files']:
        changed=sorted(key for key in set(actual)|set(record['scientific_source_files'])
                       if actual.get(key)!=record['scientific_source_files'].get(key))
        raise ValueError('scientific implementation changed after freeze: '+', '.join(changed[:10]))
    for relative,expected in record['artifacts'].items():
        path=output/relative
        if path.resolve().parent!=output.resolve() and output.resolve() not in path.resolve().parents:
            raise ValueError('unsafe artifact inventory path')
        if not path.is_file() or file_sha256(path)!=expected:raise ValueError('frozen artifact changed: '+relative)
    if hashlib.sha256(canonical_json(record['scientific_source_files'])).hexdigest()!=record['scientific_source_inventory_sha256']:
        raise ValueError('source inventory digest mismatch')
    if hashlib.sha256(canonical_json(record['artifacts'])).hexdigest()!=record['artifact_inventory_sha256']:
        raise ValueError('artifact inventory digest mismatch')
    return record


def open_confirmation(root:Path,output:Path,freeze_commit:str) -> dict[str,Any]:
    record=verify_freeze(root,output)
    if not isinstance(freeze_commit,str) or re.fullmatch(r'[0-9a-f]{40}',freeze_commit) is None:
        raise ValueError('a full published freeze commit SHA is required')
    for name in ('confirmation','shift'):
        if any((output/'data').glob(f'{name}*')) or (output/'evaluation'/name).exists():
            raise ValueError('confirmation content already exists; do not silently reuse or overwrite')
    opened={'schema':'spectra.m16_confirmation_opened.v1','opened_utc':utc_now(),
        'freeze_sha256':file_sha256(output/'confirmation_freeze.json'),'freeze_commit':freeze_commit,
        'source_commit':record['source_commit'],'no_post_confirmation_selection':True}
    write_json(output/'confirmation_opened.json',opened,exclusive=True)
    return opened
