"""Explicit immutable deployment bundle for four retained source models.

The bundle contains no CatBoost model/API and never trains. A trusted manifest
binds artifacts; source reconstruction separately verifies numerical content.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
from .compiler import VerifiedTotal
from .session import TotalSession
from .deployment import stream

TASKS=('letter','pendigits','satellite','optdigits')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root=Path(root).resolve(strict=True)
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if manifest.get('format')!='spectra.total-tree.kit.v1':raise ValueError('unknown bundle')
    for name,wanted in manifest['files'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('unsafe bundle path')
        file=root
        for part in p.parts:
            file/=part
            if file.is_symlink():raise ValueError('symlink in bundle')
        if not file.is_file() or file.stat().st_size!=wanted['bytes'] or sha(file)!=wanted['sha256']:
            raise ValueError('bundle member differs: '+name)
    return manifest


def assemble(source,models,parent_sdk,native,out):
    source,models,parent_sdk,native,out=map(Path,(source,models,parent_sdk,native,out))
    out.mkdir(parents=True,exist_ok=False)
    for name in ('LICENSE',):shutil.copy2(source/name,out/name)
    shutil.copy2(parent_sdk/'ATTRIBUTION.md',out/'ATTRIBUTION.md')
    paths=['spectra/__init__.py','spectra/svm_lifetime.py','spectra/svm_stream.py',
           'experiments/certified_trees/runtime.cpp','experiments/certified_trees/packed.py',
           'experiments/certified_trees/reference/certificate_oracle.py']
    paths+=['experiments/tree_total/'+name for name in ('compiler.py','exact.py','runtime.cpp','session.py','build.py','deployment.py','kit.py','CONTRACT.md','README.md')]
    for name in paths:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
    metadata={}
    parent=json.loads((parent_sdk/'SDK_MANIFEST.json').read_text())
    for task in TASKS:
        folder=out/'models'/task;folder.mkdir(parents=True)
        for name in ('model.json','input.u8','input.jsonl','indices.i32'):
            shutil.copy2(parent_sdk/'models'/task/name,folder/name)
        shutil.copy2(models/task/'interned.sctt',folder/'model.sctt')
        metadata[task]=parent['models'][task]
    for target in ('portable','avx2'):
        folder=out/'native'/target;folder.mkdir(parents=True)
        shutil.copy2(native/('native-'+target)/'total.so',folder/'total.so')
        shutil.copy2(native/('native-'+target)/'build.json',folder/'build.json')
    boot='import sys\nfrom pathlib import Path\nsys.dont_write_bytecode=True\nsys.path.insert(0,str(Path(__file__).resolve().parent))\n'
    (out/'selftest.py').write_text(boot+'from experiments.tree_total.kit import selftest_main\nselftest_main(Path(__file__).resolve().parent)\n')
    (out/'run.py').write_text(boot+'from experiments.tree_total.kit import run_main\nrun_main(Path(__file__).resolve().parent)\n')
    (out/'README.md').write_text('''# Total source-arithmetic tree kit

Run `python -I -S selftest.py --target portable --out ../replay.json`.
On a compatible AVX2 CPU, use `--target avx2`.

Run `python -I -S run.py --model satellite --input models/satellite/input.jsonl
--output ../predictions.jsonl` as one command. Output must be new and outside
this immutable directory. Labels are class indices; their original mapping is
in MANIFEST.json. Numeric sources, valid integer inputs and the specified FP
arithmetic are the contract, not all possible CatBoost reduction backends.

Every source proof is reconstructed at load. Existing output is never replaced;
errors before completion do not publish partial final results. Source JSON and
exact leaves add storage. No CatBoost, pretrained fitting or numerical Python
framework is needed. Linux x86-64 and matching system C++ libraries are required.
See experiments/tree_total/CONTRACT.md for scope and limitations.
''')
    files={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'MANIFEST.json').write_text(json.dumps({'format':'spectra.total-tree.kit.v1','models':metadata,'files':files},indent=2))
    return verify(out)


def target_library(root,target):
    if target=='avx2' and 'avx2' not in Path('/proc/cpuinfo').read_text().lower():
        raise ValueError('AVX2 requires compatible hardware')
    return root/'native'/target/'total.so'


def selftest_main(root):
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['portable','avx2'],default='portable');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();manifest=verify(root);records=[]
    for task,meta in manifest['models'].items():
        folder=root/'models'/task;proof=VerifiedTotal.from_files(folder/'model.json',folder/'model.sctt')
        q=bytearray((folder/'input.u8').read_bytes());expected=list(struct.unpack('<'+'i'*meta['rows'],(folder/'indices.i32').read_bytes()))
        with TotalSession(proof,target_library(root,a.target)) as engine:
            for policy in ('total','exact','audit'):
                got=engine.inspect_buffer(q,policy=policy)
                if got['indices']!=expected or got['work']['unresolved']:raise ValueError('replay disagreement')
            records.append({'task':task,'rows':len(expected),'work':engine.inspect_buffer(q)['work']})
    if {'catboost','numpy','scipy','torch','sklearn'}&sys.modules.keys():raise ValueError('numerical framework imported')
    if 'libcatboostmodel' in Path('/proc/self/maps').read_text():raise ValueError('external engine loaded')
    result={'status':'PASS','target':a.target,'models':records,'rows':sum(r['rows'] for r in records),
            'repeated_predictions':3*sum(r['rows'] for r in records),'external_engine_loaded':False}
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


def run_main(root):
    p=argparse.ArgumentParser();p.add_argument('--model',choices=TASKS,required=True)
    p.add_argument('--target',choices=['portable','avx2'],default='portable')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.resolve().is_relative_to(root) or os.path.lexists(a.output):p.error('new output outside immutable bundle required')
    verify(root);folder=root/'models'/a.model
    proof=VerifiedTotal.from_files(folder/'model.json',folder/'model.sctt')
    with TotalSession(proof,target_library(root,a.target)) as engine,a.input.open('rb') as src:
        result=stream(engine,src,a.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','models','parent-sdk','native','out'):p.add_argument('--'+name,type=Path,required=True)
    assemble(**vars(p.parse_args()))
