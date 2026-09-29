"""Complete file-job comparison and output validation for a frozen replay SDK.

No model training or policy tuning. Timing includes JSON decoding/encoding, hashes,
file I/O, native inference and publication, but excludes model/proof preparation.
The separately reported native benchmark is not an end-to-end application speedup.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import struct
import time
from .deployment import stream_predict
from .selftest import verify_manifest, sha
from .session import VerifiedCompact, TreeSession
from .verify_output import verify_output


def run(sdk: Path, official_128: Path, out: Path) -> dict:
    sdk = sdk.resolve(strict=True)
    out.mkdir(parents=False, exist_ok=False)
    manifest = verify_manifest(sdk)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    arms = ('official_128', 'official_1210', 'certified_16')
    seed, repetitions, chunk = 2026092941, 7, 128
    protocol = {'arms': arms, 'seed': seed, 'repetitions': repetitions,
                'chunk_rows': chunk, 'sdk_manifest_sha256': sha(sdk/'SDK_MANIFEST.json'),
                'source_sha256': sha(Path(__file__)),
                'official128_sha256': sha(official_128),
                'scope': 'warm full-file jobs including fsync/create-if-absent; proof and model loading excluded'}
    (out/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2))
    rng = random.Random(seed)
    observations, validations = [], []
    library = sdk/'native/avx2/trees.so'
    lib1210 = sdk/'official/libcatboostmodel-linux-x86_64-1.2.10.so'
    for task, entry in manifest['models'].items():
        folder = sdk/'models'/task
        raw = (folder/'input.u8').read_bytes()
        n, d = entry['rows'], entry['features']
        expected = struct.unpack('<'+str(n)+'i', (folder/'indices.i32').read_bytes())
        wire = b''.join((json.dumps(list(raw[i*d:(i+1)*d]))+'\n').encode() for i in range(n))
        input_path = out/(task+'-input.jsonl')
        input_path.write_bytes(wire)
        input_hash = hashlib.sha256(wire).hexdigest()
        proof = VerifiedCompact.from_files(folder/'model.json', folder/'model-16.sct')
        with ExitStack() as stack:
            engines = {}
            for arm, lib in [('official_128',official_128),('official_1210',lib1210)]:
                engines[arm] = stack.enter_context(TreeSession(library,
                    official_model=folder/'model.cbm', official_library=lib,
                    features=d, maximum=entry['maximum'], classes=len(entry['classes'])))
            engines['certified_16'] = stack.enter_context(TreeSession(library,first=proof,
                    official_model=folder/'model.cbm', official_library=lib1210))
            for repeat in range(repetitions):
                order = list(arms); rng.shuffle(order)
                for arm in order:
                    output = out/f'{task}-{repeat}-{arm}.jsonl'
                    identity = {'source_sha256': proof.info['source_sha256'], 'execution': arm}
                    start = time.perf_counter_ns()
                    with input_path.open('rb') as source:
                        result = stream_predict(engines[arm], entry['classes'], source,
                            output, identity=identity, chunk_rows=chunk)
                    ns = time.perf_counter_ns()-start
                    validated = verify_output(output, expected_indices=expected,
                        labels=entry['classes'], input_sha256=input_hash, identity=identity,
                        features=d, maximum=entry['maximum'], compact_only=False, refine=False)
                    if validated['output_sha256'] != result['output_sha256']:
                        raise ValueError('independent output digest differs')
                    row = {'task':task, 'arm':arm, 'repeat':repeat, 'rows':n, 'ns':ns,
                           'input_sha256':input_hash, 'output':output.name,
                           'output_sha256':validated['output_sha256'],
                           'certified_rows':result['certified_rows'],
                           'official_rows':result['official_rows']}
                    observations.append(row)
                    with (out/'rows.jsonl').open('a') as log: log.write(json.dumps(row)+'\n')
            # Explicit abstention remains an observed outcome, not a final class.
            output = out/f'{task}-compact-only.jsonl'
            identity = {'source_sha256':proof.info['source_sha256'],'execution':'compact-only'}
            with input_path.open('rb') as source:
                stream_predict(engines['certified_16'],entry['classes'],source,output,
                               identity=identity,compact_only=True,chunk_rows=chunk)
            validated = verify_output(output,expected_indices=expected,labels=entry['classes'],
                input_sha256=input_hash,identity=identity,features=d,maximum=entry['maximum'],
                compact_only=True,refine=False)
            if validated['unresolved_rows'] != n-entry['certified_16']:
                raise ValueError('compact file coverage changed')
            validations.append({'task':task,**validated})
        print(task+': full-file and explicit abstention outputs verified',flush=True)
    table = {}
    for task,entry in manifest['models'].items():
        us = {arm:statistics.median(r['ns'] for r in observations if r['task']==task and r['arm']==arm)/entry['rows']/1000 for arm in arms}
        table[task] = {'microseconds_per_row':us,
                      'certified_over_faster_official':us['certified_16']/min(us['official_128'],us['official_1210'])}
    report = {'status':'PASS','cells':len(observations),'models':len(table),
              'table':table,'compact_only':validations,
              'scope':protocol['scope']}
    (out/'REPORT.json').write_text(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('sdk','official-128','out'): parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    print(json.dumps(run(a.sdk,a.official_128,a.out),indent=2))
