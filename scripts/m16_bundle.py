#!/usr/bin/env python3
"""Build a self-contained source/evidence ZIP; never publish or mutate the library."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectra_reliability.freeze import verify_freeze
from spectra_reliability.identity import file_sha256,strict_json

EXCLUDE={'.git','__pycache__','.pytest_cache','.mypy_cache','.venv','node_modules','build','dist','outputs'}
SOURCE_DIRS=('common','config','data','deploy','docs','eval','kernel','kernels','model','scripts',
             'train','spectra_reliability','tests','.github','assets')
SOURCE_FILES=('README.md','LICENSE','requirements.txt','requirements-cpu.lock','pyproject.toml','setup.py','.gitignore')


def checked_name(name:str)->str:
    path=PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in ('','..','.') for p in path.parts) or '\\' in name:
        raise ValueError('unsafe archive member name')
    return path.as_posix()


def build_bundle(source:Path,experiment:Path,benchmarks:Path,report:Path,archives:Path,destination:Path):
    verify_freeze(source,experiment)
    if not (experiment/'confirmation_complete.json').is_file():raise ValueError('confirmation incomplete')
    if not strict_json((experiment/'independent_replay.json').read_bytes())['pass']:raise ValueError('replay failed')
    if destination.exists():raise FileExistsError(destination)
    entries={}
    def add(path,name):
        name=checked_name(name)
        if path.is_symlink() or not path.is_file():raise ValueError('only regular non-symlink files may be bundled')
        if name in entries:raise ValueError('duplicate member name')
        entries[name]=path
    def tree(root,prefix,source_only=False):
        if not root.is_dir():raise ValueError(f'missing input directory: {root}')
        for p in sorted(root.rglob('*')):
            relative=p.relative_to(root)
            if any(part in EXCLUDE for part in relative.parts):continue
            if not p.is_file():continue
            if source_only and (p.suffix in ('.so','.pyc','.pyo','.o') or '.egg-info' in p.as_posix()):continue
            add(p,f'{prefix}/{relative.as_posix()}')
    for name in SOURCE_DIRS:
        if (source/name).is_dir():tree(source/name,'source/'+name,True)
    for name in SOURCE_FILES:
        if (source/name).is_file():add(source/name,'source/'+name)
    tree(experiment,'evidence/experiment');tree(benchmarks,'evidence/native');tree(report,'results/m16')
    for name in ('spectra_m14_evidence.zip','spectra_m14_source_evidence.zip','spectra_m15_evidence.zip','spectra_m10_evidence.zip'):
        add(archives/name,'historical_archives/'+name)
    inventory={name:{'sha256':file_sha256(p),'bytes':p.stat().st_size} for name,p in entries.items()}
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.tmp')
    if temporary.exists():raise FileExistsError(temporary)
    try:
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for name,path in entries.items():z.write(path,name)
            z.writestr('MANIFEST.json',json.dumps({'schema':'spectra.m16_bundle.v1','files':inventory},sort_keys=True,indent=2)+'\n')
            z.writestr('BUNDLE_README.md','# SPECTRA M16 source and evidence\n\n'
                'Read results/m16/REPORT.md first. Source code is in source/. Historical input archives are retained unchanged. '
                'Evidence includes fitted heads, all generated input arrays, predictions, timing rows, rejected development exports and source manifests. '
                'MANIFEST.json binds every payload file by SHA-256 and byte length.\n\n'
                'CPU-only work. No 100/100, hiring, broad-task or physical-energy superiority is implied. '
                'Compile native code locally using the included C++ source; no native binary is distributed.\n')
        verify_bundle(temporary)
        temporary.replace(destination)
    finally:temporary.unlink(missing_ok=True)
    return {'file':str(destination),'sha256':file_sha256(destination),'bytes':destination.stat().st_size,'payload_files':len(entries)}


def verify_bundle(path:Path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('duplicate ZIP members')
        for name in names:checked_name(name)
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        manifest=strict_json(z.read('MANIFEST.json'))
        expected=set(manifest['files'])|{'MANIFEST.json','BUNDLE_README.md'}
        if set(names)!=expected:raise ValueError('ZIP inventory mismatch')
        for name,record in manifest['files'].items():
            raw=z.read(name)
            if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
                raise ValueError('payload hash mismatch: '+name)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    for arg in ('experiment','benchmarks','report','archives','out'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build_bundle(a.source,a.experiment,a.benchmarks,a.report,a.archives,a.out),sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
