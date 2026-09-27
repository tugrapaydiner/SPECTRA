"""Synthetic resource stress only: 512 features, 15000 rows, trivial known voting."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import zlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from spectra.svm_pipeline import PreparedPipeline,SCHEMA


def make_model(folder):
    folder.mkdir(parents=True,exist_ok=False)
    d=512;c=3;nsv=3
    labels=['a','b','c'];meta=json.dumps({'labels':labels},separators=(',',':')).encode()
    payload=struct.pack('<dIII',.5,1,1,1)+struct.pack('<'+'d'*(d*nsv+2*nsv+3),*([0.]*(d*nsv+2*nsv+3)))
    body=meta+payload
    raw=struct.pack('<8sIIIIII',b'SPCSVM02',c,nsv,d,len(meta),len(payload),zlib.crc32(body))+body
    (folder/'model.srt').write_bytes(raw)
    plan={'schema':SCHEMA,'columns':[f'x{i}' for i in range(d)],'features':d,
          'model_sha256':hashlib.sha256(raw).hexdigest(),
          'operations':[{'kind':'numeric','column':i,'fill':(0.).hex(),'mean':(0.).hex(),'scale':(1.).hex()} for i in range(d)]}
    (folder/'preprocessing.json').write_text(json.dumps(plan))


def status():
    lines=Path('/proc/self/status').read_text().splitlines()
    return {key:int(next(s.split()[1] for s in lines if s.startswith(key+':'))) for key in ('VmRSS','VmHWM')}


def main(a):
    rows=[[0.]*512]*15000  # Raw objects deliberately reused: isolate feature materialization.
    with PreparedPipeline(a.model,a.library,preprocessor_library=a.preprocessor) as owner,owner.session() as w:
        # Equal one-row warmup initializes either route before recording baseline.
        if a.arm=='fused':assert w.predict_fused(rows[:1])==['c']
        else:assert w.predict_many(rows[:1])==['c']
        before=status();start=time.perf_counter_ns()
        result=w.predict_fused(rows) if a.arm=='fused' else w.predict_many(rows)
        elapsed=time.perf_counter_ns()-start
        assert result==['c']*15000
        measured=status()
        print(json.dumps({'arm':a.arm,'rows':15000,'features':512,'before_kib':before,
                          'after_kib':measured,'elapsed_ns':elapsed,'matched':True,
                          'expected_transformed_buffer_bytes':131072 if a.arm=='fused' else 61440000,
                          'scope':'synthetic allocation stress; trivial known all-zero pair scores, repeated raw objects; NOT workload performance or accuracy',
                          'numerical_frameworks_loaded':sorted({'numpy','pandas','sklearn','torch'}&sys.modules.keys())}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--make-model',type=Path)
    for name in ('model','library','preprocessor'):p.add_argument('--'+name,type=Path)
    p.add_argument('--arm',choices=['fused','compiled'])
    a=p.parse_args()
    if a.make_model:make_model(a.make_model)
    else:main(a)
