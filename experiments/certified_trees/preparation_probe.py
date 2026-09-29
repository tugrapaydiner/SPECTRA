"""One fresh-process preparation measurement, including optional source proof.

No fitting, numerical Python frameworks or inherited high-water statistic. The
model and code are already fixed. This is not a cold-filesystem measurement.
"""
from __future__ import annotations
from array import array
import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time
from .selftest import sha
from .session import TreeSession, VerifiedCompact


def probe(sdk: Path, task: str, arm: str) -> dict:
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    manifest = json.loads((sdk/'SDK_MANIFEST.json').read_text())
    if task not in manifest['models'] or arm not in ('official','verified16'):
        raise ValueError('explicit known model and preparation arm required')
    entry = manifest['models'][task]
    folder = sdk/'models'/task
    with (folder/'input.u8').open('rb') as f: row = array('B', f.read(entry['features']))
    with (folder/'indices.i32').open('rb') as f: expected = struct.unpack('<i',f.read(4))[0]
    begin = time.perf_counter_ns()
    proof = VerifiedCompact.from_files(folder/'model.json',folder/'model-16.sct') if arm=='verified16' else None
    verified = time.perf_counter_ns()
    with TreeSession(sdk/'native/avx2/trees.so', first=proof,
                     official_model=folder/'model.cbm',
                     official_library=sdk/'official/libcatboostmodel-linux-x86_64-1.2.10.so',
                     features=entry['features'], maximum=entry['maximum'],
                     classes=len(entry['classes'])) as model:
        loaded = time.perf_counter_ns()
        result = model.predict_buffer(row, fallback=True)
        if result != [expected]:
            raise ValueError('preparation probe prediction differs')
        status = Path('/proc/self/status').read_text().splitlines()
        memory = {key:int(next(line.split()[1] for line in status if line.startswith(key+':')))
                  for key in ('VmHWM','VmRSS')}
        info = model.info
    if {'catboost','numpy','scipy','sklearn','torch','pandas'} & sys.modules.keys():
        raise ValueError('numerical Python framework imported')
    return {'task':task,'arm':arm,'proof_ns':verified-begin,
            'native_preparation_ns':loaded-verified,'total_preparation_ns':loaded-begin,
            'memory_kib':memory,'runtime_info':info,'matched':True,
            'model_sha256':sha(folder/'model.cbm'),'source_sha256':sha(folder/'model.json'),
            'probe_sha256':sha(Path(__file__)),
            'scope':'source verification and native loading; imports/whole-SDK hash check excluded; warm filesystem possible'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk',type=Path,required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--arm',required=True,choices=('official','verified16'))
    a=parser.parse_args()
    print(json.dumps(probe(a.sdk,a.task,a.arm)))
