"""Assemble a replayable tree SDK and check the real CBM/source relationship.

This is an explicit packaging step, not an import/install side effect. It needs
CatBoost to re-export the trusted fallback model; the finished SDK does not need
CatBoost's Python package. Hashes protect identity, not malicious authoring.
"""
from __future__ import annotations
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any
from .packed import verify, metadata, MAX_BYTES
from .reference import certificate_oracle as oracle
from .selftest import verify_manifest

TASKS = ('letter', 'pendigits', 'satellite', 'optdigits')
ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FILES = (
    'spectra/__init__.py', 'spectra/svm_lifetime.py',
    'experiments/certified_trees/session.py',
    'experiments/certified_trees/packed.py',
    'experiments/certified_trees/reference/certificate_oracle.py',
    'experiments/certified_trees/reference/test_oracle.py',
    'experiments/certified_trees/selftest.py',
    'experiments/certified_trees/deployment.py',
    'experiments/certified_trees/verify_output.py',
    'experiments/certified_trees/runtime.cpp',
    'experiments/certified_trees/build.py',
    'experiments/certified_trees/CONTRACT.md',
    'LICENSE',
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_model(path: Path) -> bytes:
    if path.is_symlink():
        raise ValueError('model must not be a symlink')
    with path.open('rb') as stream:
        data = stream.read(MAX_BYTES + 1)
    if not 0 < len(data) <= MAX_BYTES:
        raise ValueError('model exceeds byte limits')
    return data


def verify_fallback_pair(source: Path, cbm: Path, *, features: int,
                         maximum: int, labels: list[int | str]) -> dict[str, Any]:
    """Compare every interpreted split/leaf/scale, not only sample predictions.

    The saved original must have the same numeric oblivious-tree function and
    class mapping over the declared integer domain. CBM re-export may differ in
    JSON metadata/order; those differences do not alter the interpreted function.
    The same trusted CatBoost exporter is part of this packaging check's boundary.
    """
    import catboost
    original_bytes = read_model(source)
    read_model(cbm)
    original = oracle.parse_source(original_bytes, maximum, features=features)
    model = catboost.CatBoostClassifier()
    model.load_model(str(cbm))
    actual_labels = model.classes_.tolist()
    if (len(labels) != original.classes or len(labels) != len(actual_labels) or
        any(type(a) is not type(b) or a != b for a, b in zip(labels, actual_labels))):
        raise ValueError('fallback/source class mapping disagrees')
    with tempfile.TemporaryDirectory(prefix='spectra-cbm-binding-') as work:
        exported_path = Path(work) / 'source.json'
        model.save_model(str(exported_path), format='json')
        exported_bytes = read_model(exported_path)
    exported = oracle.parse_source(exported_bytes, maximum, features=features)
    for field in fields(original):
        if field.name != 'digest' and getattr(original, field.name) != getattr(exported, field.name):
            raise ValueError('fallback/source numerical structure disagrees: ' + field.name)
    return {
        'status': 'PASS', 'source_sha256': hashlib.sha256(original_bytes).hexdigest(),
        'cbm_sha256': sha(cbm), 'reexport_sha256': hashlib.sha256(exported_bytes).hexdigest(),
        'features': features, 'maximum': maximum, 'labels': labels,
        'trees': len(original.trees),
        'leaf_scalars': sum(len(t.leaves) * original.classes for t in original.trees),
        'catboost': catboost.__version__, 'verifier_sha256': sha(Path(__file__)),
        'scope': 'all interpreted integer-domain splits, leaves, scale, bias and labels; not library authentication',
    }


def make_sdk(models: Path, compiled: Path, evaluation: Path, replay: Path,
             native: dict[str, Path], official: Path, out: Path) -> dict[str, Any]:
    """Package the four fixed models; no fitting, downloading or hidden compilation."""
    for name in TASKS:
        if not (models / name / 'FIT.json').is_file():
            raise ValueError('missing fixed model inventory')
    if set(native) != {'portable', 'avx2'}:
        raise ValueError('supply both explicit native targets')
    lock = oracle.loads((models / 'MODEL_LOCK.json').read_bytes())
    for name, digest in lock['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:
            raise ValueError('unsafe frozen inventory')
        if sha(models / name) != digest:
            raise ValueError('frozen model changed: ' + name)
    out = out.absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError('SDK destination must be new')
    out.mkdir(parents=False)
    completed = False
    try:
        def copy(src: Path, name: str) -> None:
            if src.is_symlink() or not src.is_file():
                raise ValueError('bundle member must be a regular file')
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with src.open('rb') as f, target.open('xb') as g:
                shutil.copyfileobj(f, g)
        for name in RUNTIME_FILES:
            copy(ROOT / name, name)
        bindings = []
        entries = {}
        quality = oracle.loads((evaluation / 'QUALITY.json').read_bytes())
        for task in TASKS:
            fit = oracle.loads((models / task / 'FIT.json').read_bytes())
            domain = quality[task]
            if any(fit[k] != domain[k] for k in ('features', 'maximum', 'classes')):
                raise ValueError('evaluation/model metadata differs')
            for name, digest in domain['files'].items():
                if Path(name).name != name or sha(evaluation / task / name) != digest:
                    raise ValueError('evaluation content changed')
            source = models / task / 'model.json'
            binding = verify_fallback_pair(source, models / task / 'model.cbm',
                features=fit['features'], maximum=fit['maximum'], labels=fit['classes'])
            bindings.append({'task': task, **binding})
            for name in ('model.json', 'model.cbm', 'model.cpp', 'FIT.json'):
                copy(models / task / name, f'models/{task}/{name}')
            for name in ('input.u8', 'indices.i32'):
                copy(evaluation / task / name, f'models/{task}/{name}')
            n, d = domain['rows'], domain['features']
            raw = (evaluation / task / 'input.u8').read_bytes()
            indices = (evaluation / task / 'indices.i32').read_bytes()
            if len(raw) != n*d or len(indices) != 4*n:
                raise ValueError('reference input/output size differs')
            entry = {k: domain[k] for k in ('rows', 'features', 'maximum', 'classes')}
            for bits in (8, 16):
                data = read_model(compiled / f'{task}-{bits}.sct')
                verify(read_model(source), data)
                copy(compiled / f'{task}-{bits}.sct', f'models/{task}/model-{bits}.sct')
                status_path = replay / task / f'{bits}-tiled-0.indices'
                statuses = status_path.read_bytes()
                if len(statuses) != 4*n:
                    raise ValueError('replay size differs')
                statuses = struct.unpack('<' + str(n) + 'i', statuses)
                expected = struct.unpack('<' + str(n) + 'i', indices)
                if any(s != -1 and s != e for s, e in zip(statuses, expected)):
                    raise ValueError('wrong certified replay')
                entry[f'certified_{bits}'] = sum(s >= 0 for s in statuses)
            sample = b''.join((json.dumps(list(raw[i*d:(i+1)*d]))+'\n').encode()
                              for i in range(min(n, 32)))
            (out / 'models' / task / 'sample.jsonl').write_bytes(sample)
            entries[task] = entry
        for target, path in native.items():
            build_receipt = oracle.loads(path.with_name('build.json').read_bytes())
            if build_receipt.get('library_sha256') != sha(path):
                raise ValueError('native build receipt disagrees')
            if build_receipt.get('source_sha256') != sha(ROOT/'experiments/certified_trees/runtime.cpp'):
                raise ValueError('native library was built from different source')
            copy(path, f'native/{target}/trees.so')
            copy(path.with_name('build.json'), f'native/{target}/build.json')
        if sha(official/'libcatboostmodel-linux-x86_64-1.2.10.so') != 'de32f8e147ee8f969599a00b7145e8561b943e916b0132b811037f8f71952a0e':
            raise ValueError('official library differs from the accepted upstream asset')
        copy(official/'libcatboostmodel-linux-x86_64-1.2.10.so',
             'official/libcatboostmodel-linux-x86_64-1.2.10.so')
        copy(official/'catboost-LICENSE', 'official/LICENSE')
        for name, module in (('run.py', 'deployment'), ('selftest.py', 'selftest')):
            (out/name).write_text(
                'from pathlib import Path\nimport runpy,sys\n'
                'root=Path(__file__).resolve().parent\n'
                'sys.dont_write_bytecode=True\nsys.path.insert(0,str(root))\n'
                'sys.argv[1:1]=["--sdk",str(root)]\n'
                f'runpy.run_module("experiments.certified_trees.{module}",run_name="__main__")\n')
        (out/'SOURCE_BINDING.json').write_bytes(oracle.canonical({'status':'PASS','models':bindings}))
        (out/'README.md').write_text(
            '# SPECTRA verified tree replay kit\n\n'
            'Linux x86-64, Python 3.11+. The portable library is the default. '
            'No numerical Python framework, network, fitting or compilation is needed to replay.\n\n'
            'From this directory, choose a new output path OUTSIDE the kit:\n\n'
            '```bash\npython -I -S selftest.py --out ../tree-replay.json\n'
            'python -I -S run.py --model letter --input models/letter/sample.jsonl --output ../predictions.jsonl\n```\n\n'
            'Add `--target avx2` only on compatible hardware. `--compact-only` allows '
            'explicit UNRESOLVED outputs; default execution calls the bundled official '
            'CatBoost library for unresolved decisions. `--refine` uses 8-bit then 16-bit.\n\n'
            'Source verification happens on loading and can dominate short jobs. '
            'Full coverage retains original models and the official library: compact '
            'leaf storage is not total-deployment compression. Certificates concern '
            'the stated source arithmetic, not ground-truth labels or arbitrary CatBoost backends.\n\n'
            'The fixed public benchmark tests were already exposed; this is not a new '
            'accuracy study. Dataset sources: UCI Letter 59 (David Slate), Pendigits 81 '
            '(Ethem Alpaydin, Fevzi Alimoglu), Statlog Satellite 146 (Ashwin Srinivasan), '
            'and OptDigits 80 (Ethem Alpaydin, Cevdet Kaynak). Preserve their CC BY 4.0 '
            'attribution; see https://archive.ics.uci.edu/. CatBoost is Apache-2.0; '
            'its license is included. System Python/C++ runtimes are not redistributed.\n')
        manifest = {'format':'spectra.tree.sdk.v1','models':entries,
                    'source_pair_validation':'SOURCE_BINDING.json',
                    'files':{p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
                        for p in sorted(out.rglob('*')) if p.is_file()}}
        (out/'SDK_MANIFEST.json').write_bytes(oracle.canonical(manifest))
        verify_manifest(out)
        completed = True
        return {'status':'PASS','models':len(entries),'files':len(manifest['files']),
                'manifest_sha256':sha(out/'SDK_MANIFEST.json'),
                'scope':'artifact creation and source-pair checks; run the included selftest for native replay'}
    finally:
        if not completed:
            shutil.rmtree(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('models','compiled','evaluation','replay','portable','avx2','official','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(make_sdk(a.models,a.compiled,a.evaluation,a.replay,
          {'portable':a.portable,'avx2':a.avx2},a.official,a.out),indent=2))
