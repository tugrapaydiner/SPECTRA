"""One fresh process: source verification, native load, one checked prediction.

Own-process Linux VmHWM includes imports/verification temporaries. In-scope setup
starts after imports and includes files, verification and native loading. Read RSS
while the session is live, not after closing it. Filesystem cache may be warm.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
from .compiler import VerifiedTotal
from .session import TotalSession
from ..tree_residual.compile import VerifiedResidual
from ..tree_residual.session import ResidualSession
from ..certified_trees.session import TreeSession


def run(a):
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    parent=a.root/'SPECTRA-residual-evidence/baseline_sdk/models'/a.task
    results=a.root/'results'
    shape=json.loads((results/'replay'/a.task/'result.json').read_text())['models']['interned']['info']
    d,D,c=shape['features'],shape['maximum'],shape['classes']
    raw=bytearray((parent/'input.u8').read_bytes()[:d])
    expected=struct.unpack('<i',(parent/'indices.i32').read_bytes()[:4])[0]
    start=time.perf_counter_ns()
    if a.policy=='residual':
        model=a.root/'SPECTRA_residual_trees/models'/a.task/'model.scr'
        proof=VerifiedResidual.from_files(parent/'model.json',model)
        lib=a.root/'SPECTRA-residual-evidence/native-avx2/residual.so'
        session=ResidualSession(proof,lib)
        del proof
    elif a.policy=='official':
        model=parent/'model.cbm'
        lib=a.root/'SPECTRA-residual-evidence/baseline_evidence/native-register/trees.so'
        official=a.root/'SPECTRA-residual-evidence/baseline_evidence/official/libcatboostmodel-linux-x86_64-1.2.10.so'
        session=TreeSession(lib,official_model=model,official_library=official,features=d,maximum=D,classes=c)
    else:
        model=results/'replay'/a.task/(a.policy+'.sctt')
        proof=VerifiedTotal.from_files(parent/'model.json',model)
        lib=results/'native-avx2/total.so';session=TotalSession(proof,lib)
        del proof
    elapsed=time.perf_counter_ns()-start
    with session as engine:
        if engine.predict_buffer(raw)!=[expected]:raise ValueError('resource probe output mismatch')
        gc.collect()
        lines=Path('/proc/self/status').read_text().splitlines()
        memory={key:int(next(s.split()[1] for s in lines if s.startswith(key+':'))) for key in ('VmHWM','VmRSS')}
        return {'task':a.task,'policy':a.policy,'setup_ns':elapsed,'memory_kib':memory,
                'runtime_info':engine.info,'model_bytes':model.stat().st_size,
                'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
                'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),
                'matched':True,'numerical_frameworks':sorted({'numpy','scipy','sklearn','torch','catboost'}&sys.modules.keys()),
                'scope':'fresh process, own high-water and live-session RSS; not maximum-batch or cold-filesystem evidence'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',choices=['letter','pendigits','satellite','optdigits'],required=True)
    p.add_argument('--policy',choices=['flat','interned','residual','official'],required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True))
