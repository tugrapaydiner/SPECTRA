"""Installed-only, framework-free full frozen-corpus and JSONL deployment replay.

Run with isolated Python from an actual installed wheel. The manifest and original
corpus are outside the package and are validated before executing any model.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(args):
    import spectra
    from spectra.svm_pipeline import PreparedPipeline
    from spectra.svm_receipt import create_receipt, verify_receipt
    from spectra.svm_stream import Limits, run_stream
    package=Path(spectra.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError('acceptance requires the actual installed package, not this checkout')
    forbidden={'numpy','pandas','sklearn','torch','jax'}
    if forbidden & sys.modules.keys() or any(importlib.util.find_spec(n) for n in forbidden):
        raise ValueError('acceptance requires an environment without numerical frameworks')
    spec=importlib.util.spec_from_file_location('independent_stream_auditor',args.auditor)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest['format']!='spectra.stream.corpus.v1' or len(manifest['models'])!=18:
        raise ValueError('unexpected retained corpus')
    args.out.mkdir(parents=True,exist_ok=False)
    details=[];predictions=features=receipts=0
    for name,entry in sorted(manifest['models'].items()):
        folder=args.corpus/name
        for filename,wanted in entry['files'].items():
            path=folder/filename
            if path.is_symlink() or path.stat().st_size!=wanted['bytes'] or sha(path)!=wanted['sha256']:
                raise ValueError('retained corpus identity mismatch: '+name+'/'+filename)
        rows=[json.loads(line) for line in (folder/'input.jsonl').read_text(encoding='utf-8').splitlines()]
        expected=json.loads((folder/'expected.json').read_text(encoding='utf-8'))
        if len(rows)!=entry['rows'] or len(expected)!=entry['rows']:raise ValueError('corpus size mismatch')
        report={'model':name,'rows':len(rows),'runs':[]}
        with PreparedPipeline(folder,args.library,preprocessor_library=args.preprocessor) as p,p.session() as w:
            transformed=p.preprocessor.transform(rows)
            if transformed.tobytes()!=(folder/'transformed.f64').read_bytes():raise ValueError('feature bytes differ: '+name)
            features+=len(transformed)
            for engine in ('two_stage','fused'):
                for mode in ('exhaustive','beretta_cert','binary_stream'):
                    destination=args.out/f'{name}-{engine}-{mode}.jsonl'
                    with (folder/'input.jsonl').open('rb') as source:
                        result=run_stream(source,destination,p,engine=engine,schedule=mode,limits=Limits(batch_rows=17))
                    observed=audit.verify_result(destination,expected=expected,input_identity=entry['files']['input.jsonl'],
                         model_sha256=entry['files']['model.srt']['sha256'],plan_sha256=entry['files']['preprocessing.json']['sha256'])
                    if result['output_sha256']!=observed['output_sha256']:raise ValueError('output receipt mismatch')
                    predictions+=len(rows);report['runs'].append({'engine':engine,'mode':mode,**observed})
            d=p.preprocessor.features
            for i in range(min(4,len(rows))):
                row=list(transformed[i*d:(i+1)*d]);receipt=create_receipt(w._worker,row)
                if not verify_receipt(folder/'model.srt',receipt,expected_input=row,input_dtype='float64')['verified']:
                    raise ValueError('independent numerical replay failed: '+name)
                receipts+=1
            p.close()
            if w.predict_fused(rows[:1])!=expected[:1]:raise ValueError('owner-close fidelity')
        details.append(report)
    # Exercise the public CLI from the installed wheel, including binary stdin.
    name=sorted(manifest['models'])[0];folder=args.corpus/name
    command=[sys.executable,'-I','-m','spectra','svm','run',str(folder),'--library',str(args.library),
             '--preprocessor',str(args.preprocessor),'--output',str(args.out/'cli.jsonl')]
    process=subprocess.run(command,input=(folder/'input.jsonl').read_bytes(),capture_output=True)
    (args.out/'cli-command.json').write_text(json.dumps({'command':command,'returncode':process.returncode,
             'stdout':process.stdout.decode('utf-8'),'stderr':process.stderr.decode('utf-8')},indent=2),encoding='utf-8')
    if process.returncode:raise RuntimeError('public CLI failed')
    expected=json.loads((folder/'expected.json').read_text())
    audit.verify_result(args.out/'cli.jsonl',expected=expected,input_identity=manifest['models'][name]['files']['input.jsonl'],
                        model_sha256=manifest['models'][name]['files']['model.srt']['sha256'],
                        plan_sha256=manifest['models'][name]['files']['preprocessing.json']['sha256'])
    if forbidden & sys.modules.keys():raise ValueError('numerical framework imported')
    summary={'status':'PASS','python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
         'package':str(package),'models':len(details),'model_input_pairs':sum(x['rows'] for x in details),
         'streamed_predictions':predictions,'matched_feature_values':features,'independent_receipts':receipts,
         'owner_close_checks':len(details),'cli_rows':len(expected),'manifest_sha256':sha(args.manifest),
         'auditor_sha256':sha(args.auditor),'runtime_sha256':sha(args.library),'preprocessor_sha256':sha(args.preprocessor),
         'scope':'same retained corpus on this native platform; repeated paths are not independent accuracy examples',
         'details':details}
    (args.out/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','manifest','auditor','library','preprocessor','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(check(args),indent=2))
