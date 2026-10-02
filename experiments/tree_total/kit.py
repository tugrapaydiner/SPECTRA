"""Explicit immutable deployment bundle for four retained source models.

The bundle contains no CatBoost model/API and never trains. A trusted manifest
binds artifacts; source reconstruction separately verifies numerical content.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
from .compiler import VerifiedTotal
from .session import TotalSession
from .deployment import stream

TASKS=('letter','pendigits','satellite','optdigits')
SOURCE_FILES=(
    'spectra/__init__.py','spectra/svm_lifetime.py','spectra/svm_stream.py',
    'experiments/certified_trees/runtime.cpp','experiments/certified_trees/packed.py',
    'experiments/certified_trees/reference/certificate_oracle.py',
    *('experiments/tree_total/'+name for name in (
        'compiler.py','exact.py','runtime.cpp','session.py','build.py',
        'deployment.py','kit.py','CONTRACT.md','README.md')),
)
MODEL_FILES=('model.json','model.sctt','input.u8','input.jsonl','indices.i32')
TARGETS=('portable','avx2')
REQUIRED_FILES=frozenset((
    'LICENSE','ATTRIBUTION.md','README.md','selftest.py','run.py',*SOURCE_FILES,
    *(f'models/{task}/{name}' for task in TASKS for name in MODEL_FILES),
    *(f'native/{target}/{name}' for target in TARGETS for name in ('total.so','build.json')),
))
MAX_MANIFEST_BYTES=1024*1024


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _object(pairs):
    result={}
    for name,value in pairs:
        if name in result:raise ValueError('duplicate manifest field: '+name)
        result[name]=value
    return result


def _finite(value):
    number=float(value)
    if not math.isfinite(number):raise ValueError('nonfinite manifest number')
    return number


def _model_metadata(models):
    if not isinstance(models,dict) or set(models)!=set(TASKS):
        raise ValueError('complete four-model inventory required')
    for meta in models.values():
        if not isinstance(meta,dict):raise ValueError('invalid model metadata')
        for name,limit in (('rows',65536),('features',256),('maximum',255)):
            value=meta.get(name)
            if type(value) is not int or not 1<=value<=limit:
                raise ValueError('invalid model '+name)
        if meta['rows']*meta['features']>8000000:raise ValueError('replay input too large')
        labels=meta.get('classes')
        if (not isinstance(labels,list) or not 2<=len(labels)<=64
                or not (all(type(v) is int for v in labels) or all(type(v) is str for v in labels))
                or len(set(labels))!=len(labels)):
            raise ValueError('invalid class mapping')


def verify(root):
    root=Path(root).resolve(strict=True)
    manifest_path=root/'MANIFEST.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():raise ValueError('invalid manifest file')
    with manifest_path.open('rb') as source:raw=source.read(MAX_MANIFEST_BYTES+1)
    if len(raw)>MAX_MANIFEST_BYTES:raise ValueError('manifest exceeds byte limit')
    try:
        manifest=json.loads(raw.decode('utf-8'),object_pairs_hook=_object,
                            parse_constant=_finite,parse_float=_finite)
    except (UnicodeError,RecursionError) as error:
        raise ValueError('invalid manifest encoding or nesting') from error
    if not isinstance(manifest,dict) or manifest.get('format')!='spectra.total-tree.kit.v1':
        raise ValueError('unknown bundle')
    _model_metadata(manifest.get('models'))
    files=manifest.get('files')
    if not isinstance(files,dict) or not REQUIRED_FILES<=files.keys():
        raise ValueError('incomplete bundle inventory')
    for name,wanted in files.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or p.as_posix()!=name or name=='.':
            raise ValueError('unsafe bundle path')
        if (not isinstance(wanted,dict) or set(wanted)!= {'bytes','sha256'}
                or type(wanted['bytes']) is not int or wanted['bytes']<0
                or not isinstance(wanted['sha256'],str) or len(wanted['sha256'])!=64
                or any(c not in '0123456789abcdef' for c in wanted['sha256'])):
            raise ValueError('invalid member metadata: '+name)
    actual=set()
    for file in root.rglob('*'):
        if file.is_symlink() or not (file.is_file() or file.is_dir()):
            raise ValueError('nonregular member in bundle')
        if file.is_file() and file!=manifest_path:actual.add(file.relative_to(root).as_posix())
    if actual!=set(files):raise ValueError('bundle inventory differs')
    for task,meta in manifest['models'].items():
        for name,size in (('input.u8',meta['rows']*meta['features']),('indices.i32',4*meta['rows'])):
            if files[f'models/{task}/{name}']['bytes']!=size:raise ValueError('replay dimensions differ')
    for name,wanted in files.items():
        file=root/name
        if file.stat().st_size!=wanted['bytes'] or sha(file)!=wanted['sha256']:
            raise ValueError('bundle member differs: '+name)
    return manifest


def assemble(source,models,parent_sdk,native,out):
    source,models,parent_sdk,native,out=map(Path,(source,models,parent_sdk,native,out))
    out.mkdir(parents=True,exist_ok=False)
    for name in ('LICENSE',):shutil.copy2(source/name,out/name)
    shutil.copy2(parent_sdk/'ATTRIBUTION.md',out/'ATTRIBUTION.md')
    for name in SOURCE_FILES:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
    metadata={}
    parent=json.loads((parent_sdk/'SDK_MANIFEST.json').read_text())
    for task in TASKS:
        folder=out/'models'/task;folder.mkdir(parents=True)
        for name in ('model.json','input.u8','input.jsonl','indices.i32'):
            shutil.copy2(parent_sdk/'models'/task/name,folder/name)
        shutil.copy2(models/task/'interned.sctt',folder/'model.sctt')
        metadata[task]=parent['models'][task]
    for target in TARGETS:
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


def _verified_model(root,task,meta):
    folder=root/'models'/task
    proof=VerifiedTotal.from_files(folder/'model.json',folder/'model.sctt')
    if any(proof.info[key]!=meta[key] for key in ('features','maximum')) or proof.info['classes']!=len(meta['classes']):
        raise ValueError('source and replay metadata differ')
    return proof


def selftest_main(root):
    root=Path(root).resolve(strict=True)
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['portable','avx2'],default='portable');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.resolve().is_relative_to(root) or os.path.lexists(a.out):p.error('new output outside immutable bundle required')
    manifest=verify(root);records=[]
    for task,meta in manifest['models'].items():
        folder=root/'models'/task;proof=_verified_model(root,task,meta)
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
    root=Path(root).resolve(strict=True)
    p=argparse.ArgumentParser();p.add_argument('--model',choices=TASKS,required=True)
    p.add_argument('--target',choices=['portable','avx2'],default='portable')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.resolve().is_relative_to(root) or os.path.lexists(a.output):p.error('new output outside immutable bundle required')
    manifest=verify(root)
    proof=_verified_model(root,a.model,manifest['models'][a.model])
    with TotalSession(proof,target_library(root,a.target)) as engine,a.input.open('rb') as src:
        result=stream(engine,src,a.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','models','parent-sdk','native','out'):p.add_argument('--'+name,type=Path,required=True)
    assemble(**vars(p.parse_args()))
