"""Replay real frozen SDK inputs through the isolated public CLI and output auditor.

No model fitting, numerical frameworks or accuracy selection. Every output is
checked against independently supplied frozen class indices and source identities.
The output directory must be new and outside the immutable SDK.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys


def run(sdk, out):
    sdk, out = Path(sdk).resolve(strict=True), Path(out).absolute()
    if out.is_relative_to(sdk):
        raise ValueError('write acceptance evidence outside the immutable SDK')
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('independent_output_verifier',
        Path(__file__).with_name('verify_output.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads((sdk / 'SDK_MANIFEST.json').read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    results = []
    refusals = []
    for task, entry in sorted(manifest['models'].items()):
        folder = sdk / 'models' / task
        raw = (folder / 'input.u8').read_bytes()
        d, n = entry['features'], entry['rows']
        if len(raw) != n*d:
            raise ValueError('frozen input dimensions differ')
        expected = list(struct.unpack('<' + str(n) + 'i', (folder / 'indices.i32').read_bytes()))
        input_path = out / (task + '-input.jsonl')
        input_path.write_bytes(b''.join((json.dumps(list(raw[i*d:(i+1)*d])) + '\n').encode()
                                      for i in range(n)))
        for target in ('portable', 'avx2'):
            for policy in ('full16', 'full_refined', 'compact16'):
                refine, compact = policy == 'full_refined', policy == 'compact16'
                destination = out / f'{task}-{target}-{policy}.jsonl'
                cmd = [sys.executable, '-I', '-S', str(sdk/'run.py'), '--model', task,
                       '--target', target, '--input', str(input_path), '--output', str(destination)]
                if refine:
                    cmd.append('--refine')
                if compact:
                    cmd.append('--compact-only')
                process = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                call = {'command': cmd, 'returncode': process.returncode,
                        'stdout': process.stdout, 'stderr': process.stderr}
                (out / (destination.stem + '-process.json')).write_text(json.dumps(call, indent=2))
                if process.returncode:
                    raise RuntimeError('isolated CLI failed: ' + destination.stem)
                bits = 8 if refine else 16
                identity = {'source_sha256': sha(folder/'model.json'),
                            'compact_sha256': sha(folder/f'model-{bits}.sct'),
                            'bundle_manifest_sha256': sha(sdk/'SDK_MANIFEST.json'),
                            'native_sha256': sha(sdk/'native'/target/'trees.so')}
                if not compact:
                    identity.update(official_model_sha256=sha(folder/'model.cbm'),
                        official_library_sha256=sha(sdk/'official/libcatboostmodel-linux-x86_64-1.2.10.so'))
                checked = module.verify_output(destination, expected_indices=expected,
                    labels=entry['classes'], input_sha256=sha(input_path), identity=identity,
                    features=d, maximum=entry['maximum'], compact_only=compact, refine=refine)
                expected_missing = n-entry['certified_16'] if compact else 0
                if checked['unresolved_rows'] != expected_missing:
                    raise ValueError('changed expected abstention count')
                results.append({'task': task, 'target': target, 'policy': policy, **checked})
        # Actual repeated invocation must not replace an existing completed file.
        previous = destination.read_bytes()
        again = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if not again.returncode or destination.read_bytes() != previous:
            raise ValueError('completed output replaced')
        refusals.append({'task':task,'case':'existing-output','returncode':again.returncode,
                         'stdout':again.stdout,'stderr':again.stderr,'preserved_sha256':sha(destination)})
        invalid = out / (task + '-invalid.jsonl')
        invalid.write_bytes(input_path.read_bytes()+b'{}\n')
        failed = out / (task + '-must-not-publish.jsonl')
        bad_cmd = [sys.executable,'-I','-S',str(sdk/'run.py'),'--model',task,
                   '--input',str(invalid),'--output',str(failed)]
        bad = subprocess.run(bad_cmd,capture_output=True,text=True,timeout=60)
        if not bad.returncode or failed.exists() or list(out.glob('.'+failed.name+'.*.partial')):
            raise ValueError('late invalid input published output or leaked a partial file')
        refusals.append({'task':task,'case':'late-invalid-input','returncode':bad.returncode,
                         'command':bad_cmd,'stdout':bad.stdout,'stderr':bad.stderr})
        print(task + ': 6 complete file jobs and 2 refusal cases PASS', flush=True)
    if {'numpy','scipy','catboost','torch','sklearn'} & sys.modules.keys():
        raise ValueError('unexpected numerical Python import')
    report={'status':'PASS','file_jobs':len(results),'refusal_cases':len(refusals),
            'repeated_input_rows':sum(r['rows'] for r in results),'results':results,'refusals':refusals,
            'sdk_manifest_sha256':sha(sdk/'SDK_MANIFEST.json'),'script_sha256':sha(__file__),
            'scope':'isolated public CLI and independent output audit; unchanged source-model predictions, not new model accuracy'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    report=run(a.sdk,a.out)
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','refusals')},indent=2))
