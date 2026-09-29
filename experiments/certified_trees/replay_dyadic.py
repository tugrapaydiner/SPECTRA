"""Whole-corpus equivalence of both exact verifiers and both official fallbacks.

Both compilers reconstruct the existing bytes independently. The unchanged native
engine executes those verified objects; work, indices and routing counts must
agree as well as the final class. No training or numerical Python dependency.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import struct
import sys


def main(a):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.certified_trees.session import VerifiedCompact, TreeSession
    a.out.mkdir(parents=True, exist_ok=False)
    records = []
    task_counts = {}
    for task in ('letter', 'pendigits', 'satellite', 'optdigits'):
        source = a.evidence/'models'/task/'model.json'
        raw = array('B', (a.evidence/'evaluation'/task/'input.u8').read_bytes())
        truth = array('i'); truth.frombytes((a.evidence/'evaluation'/task/'indices.i32').read_bytes())
        if sys.byteorder != 'little' or truth.itemsize != 4:
            raise ValueError('replay requires little-endian int32 indices')
        expected = truth.tolist()
        proofs = {backend: {bits: VerifiedCompact.from_files(source,
                  a.evidence/'compiled'/f'{task}-{bits}.sct', backend=backend)
                  for bits in (8, 16)} for backend in ('reference', 'dyadic')}
        for bits in (8, 16):
            if proofs['reference'][bits].raw != proofs['dyadic'][bits].raw:
                raise AssertionError('canonical compact bytes differ')
        original_results = {}
        for backend in ('reference', 'dyadic'):
            def record(name, result):
                if any(i != -1 and i != e for i,e in zip(result['indices'], expected)) or len(result['indices']) != len(expected):
                    raise AssertionError('certified or fallback output disagrees with original source')
                if backend == 'reference':
                    original_results[name] = result
                elif result != original_results[name]:
                    raise AssertionError('verified-object results or work differ')
                packed = struct.pack('<'+str(len(expected))+'i', *result['indices'])
                used = struct.pack('<'+str(len(expected))+'I', *result['trees_evaluated'])
                records.append({'task':task,'backend':backend,'case':name,'rows':len(expected),
                    'indices_sha256':hashlib.sha256(packed).hexdigest(),
                    'trees_evaluated_sha256':hashlib.sha256(used).hexdigest(),
                    'work':result['work']})
            for bits in (8, 16):
                with TreeSession(a.library, first=proofs[backend][bits]) as runner:
                    for mode in ('scalar', 'tiled'):
                        for checkpoint in (0, 16):
                            result = runner.inspect_buffer(raw, mode=mode, checkpoint=checkpoint)
                            record(f'compact{bits}/{mode}/{checkpoint}',result)
            with TreeSession(a.library, first=proofs[backend][8], second=proofs[backend][16]) as runner:
                for mode in ('scalar', 'tiled'):
                    for checkpoint in (0, 16):
                        result = runner.inspect_buffer(raw, mode=mode, checkpoint=checkpoint, refine=True)
                        record(f'refined/{mode}/{checkpoint}',result)
            for version in ('1.2.8', '1.2.10'):
                official = a.evidence/'official'/f'libcatboostmodel-linux-x86_64-{version}.so'
                for refine in (False, True):
                    with TreeSession(a.library,first=proofs[backend][8 if refine else 16],
                        second=proofs[backend][16] if refine else None,
                        official_model=a.evidence/'models'/task/'model.cbm',official_library=official) as runner:
                        result=runner.inspect_buffer(raw,refine=refine,fallback=True)
                        if result['indices'] != expected:
                            raise AssertionError('official fallback did not resolve source output')
                        record(f'full/{version}/refine={refine}',result)
        task_counts[task] = {'rows':len(expected),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'compact_sha256':{str(bits):hashlib.sha256(proofs['dyadic'][bits].raw).hexdigest() for bits in (8,16)}}
        print(task+': exact bytes, complete outputs, work and both official fallbacks PASS',flush=True)
    if {'numpy','scipy','catboost','torch','sklearn'} & sys.modules.keys():
        raise ValueError('numerical Python module imported')
    result={'status':'PASS','source_models':len(task_counts),'underlying_rows':sum(t['rows'] for t in task_counts.values()),
        'execution_cases':len(records),'repeated_rows':sum(r['rows'] for r in records),'models':task_counts,'cases':records,
        'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'unchanged source-classifier fidelity; no fitting, new accuracy or inference-speed measurement'}
    (a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','library','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();report=main(args)
    print(json.dumps({k:v for k,v in report.items() if k not in ('models','cases')},indent=2))
