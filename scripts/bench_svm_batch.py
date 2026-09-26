"""Compare the public scalar and batch APIs on retained JSON input rows.

Timing includes input conversion, native execution, and fresh output lists. This
is dispatch amortization, not a new model or a numerical-kernel speed claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectra.svm import Session


def run(model: Path, inputs: Path, library: Path, out: Path, tables: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    rows = json.loads(inputs.read_text())
    if len(rows) != 512 or any(len(row) != 16 for row in rows):
        raise ValueError('benchmark requires exactly 512 sixteen-feature rows')
    names = ['scalar', 'batch1', 'batch8', 'batch64', 'batch512']
    source_paths = [ROOT/'spectra/svm.py', Path(__file__)] + sorted((ROOT/'spectra/_native/ovo').rglob('*.cpp')) + sorted((ROOT/'spectra/_native/ovo').rglob('*.hpp'))
    metadata = {'schema':'spectra.svm_batch_api.v1','rows':512,'repeats':31,'seed':20260926,
                'arms':names,'tables':tables,'host':platform.platform(),'python':sys.version,
                'affinity':sorted(os.sched_getaffinity(0)),
                'scope':'complete 512-row callable, including Python conversion and fresh results; model loading/build and JSON parsing excluded',
                'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [model,inputs,library]+source_paths}}
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    observations=[]
    with Session(model,library,tables=tables) as session:
        expected = session.predict_many(rows, schedule='exhaustive')
        def call(name):
            if name=='scalar':return [session.predict(row) for row in rows]
            count=int(name[5:]); result=[]
            for start in range(0,512,count):
                result.extend(session.predict_many(rows[start:start+count]))
            return result
        for name in names:assert call(name)==expected
        rng=random.Random(metadata['seed'])
        with (out/'rows.jsonl').open('x') as stream:
            for repeat in range(31):
                order=list(names);rng.shuffle(order)
                for name in order:
                    begin=time.perf_counter_ns();output=call(name);elapsed=time.perf_counter_ns()-begin
                    if output!=expected:raise AssertionError(f'class mismatch in {name}')
                    record={'arm':name,'repeat':repeat,'ns':elapsed,'matched':True}
                    observations.append(record);stream.write(json.dumps(record)+'\n')
    medians={name:statistics.median([r['ns'] for r in observations if r['arm']==name])/512/1000 for name in names}
    report={'median_us_per_row':medians,'scalar_over_batch64':medians['scalar']/medians['batch64'],
            'scalar_over_batch512':medians['scalar']/medians['batch512'],
            'calls_checked':len(observations)*512,'rows':512,'interpretation':'retained-data API benchmark, not independent generalization'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','inputs','library','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--tables',action='store_true');args=parser.parse_args()
    print(json.dumps(run(args.model,args.inputs,args.library,args.out,args.tables),indent=2))
