"""Matched bounded feature-JSONL processing; no raw-sensor feature extraction.

This is a reproduction/example harness, not an extension of `spectra svm run`'s
accepted bundle format. All backends share parser, packing, writer and publication.
Use installed isolated Python for no-framework acceptance; --source permits a
recorded source-checkout experiment explicitly.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def memory_kib():
    return {key:int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith(key+':'))) for key in ('VmHWM','VmRSS')}


def main(a):
    if a.source is not None:sys.path.insert(0,str(a.source.resolve()))
    if a.dependency_site is not None:sys.path.append(str(a.dependency_site.resolve()))
    from spectra.svm_stream import _row,_Writer,_staged_output
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    loaded=time.perf_counter_ns();owner=None
    if a.backend=='native_linear':
        from spectra.linear import CompiledLinear
        model=CompiledLinear(a.model,a.library);labels=list(model.labels);identity=model.sha256
        predict=model.predict_buffer
    elif a.backend=='native_rbf':
        from spectra.svm_shared import PreparedModel
        owner=PreparedModel(a.model,a.library,input_dtype='float64');model=owner.session()
        labels=list(model.labels);identity=model.sha256
        predict=model.predict_buffer
    elif a.backend=='numpy_linear':
        import numpy as np
        from threadpoolctl import threadpool_limits
        controller=threadpool_limits(1)
        meta=json.loads((a.model/'model.json').read_text())
        raw=(a.model/'parameters.f64').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=meta['parameters_sha256']:raise ValueError('parameter identity')
        parameters=np.frombuffer(raw,dtype='<f8');weights=parameters[:3366].reshape(6,561);bias=parameters[3366:]
        labels=meta['labels'];class_values=np.asarray(labels);identity=meta['model_sha256']
        def predict(packed):
            x=np.frombuffer(packed,dtype=np.float64).reshape(-1,561)
            if not np.isfinite(x).all():raise ValueError('nonfinite')
            return class_values[(x@weights.T+bias).argmax(axis=1)].tolist()
    else:
        import joblib,numpy as np
        from threadpoolctl import threadpool_limits
        # Environment also pins BLAS before imports; keep returned controller alive.
        controller=threadpool_limits(1);model=joblib.load(a.model)
        labels=model.classes_.tolist();identity=hashlib.sha256(a.model.read_bytes()).hexdigest()
        def predict(packed):
            x=np.frombuffer(packed,dtype=np.float64).reshape(-1,561)
            if a.backend=='numpy_linear':
                if not np.isfinite(x).all():raise ValueError('nonfinite')
                return model.classes_[(x@model.coef_.T+model.intercept_).argmax(axis=1)].tolist()
            return model.predict(x).tolist()
    load_ns=time.perf_counter_ns()-loaded
    started=time.perf_counter_ns();ih=hashlib.sha256();input_bytes=count=batches=0
    with a.input.open('rb') as source,_staged_output(a.output) as output:
        writer=_Writer(output,268435456)
        writer.write({'record':'header','format':'spectra.har.application.v1','backend':a.backend,'model_sha256':identity,'features':561,'labels':labels})
        packed=array('d');n=0
        def flush():
            nonlocal count,batches,n
            if not n:return
            result=predict(packed)
            if len(result)!=n:raise ValueError('invalid count')
            for value in result:writer.write({'record':'prediction','index':count,'label':value});count+=1
            packed.clear();n=0;batches+=1
        for raw in iter(lambda:source.readline(1048577),b''):
            if len(raw)>1048576:raise ValueError('oversized input')
            row=_row(raw,561,count+n+1)
            # Dataset features have no missing values or categories. Validate this
            # additional fixed application schema before buffer conversion.
            if any(type(x) not in (int,float) for x in row):raise ValueError('numeric features required')
            ih.update(raw);input_bytes+=len(raw);packed.extend(row);n+=1
            if n==128:flush()
        flush()
        writer.write({'record':'complete','rows':count,'batches':batches,'input_bytes':input_bytes,'input_sha256':ih.hexdigest(),'output_prefix_sha256':writer.digest.hexdigest()})
        output_bytes=writer.bytes
    end=time.perf_counter_ns();mem=memory_kib()
    if owner is not None:model.close();owner.close()
    forbidden=sorted({'numpy','pandas','sklearn','torch','jax'} & sys.modules.keys())
    if a.backend.startswith('native') and forbidden:raise ValueError('framework imported')
    report={'backend':a.backend,'load_ns':load_ns,'processing_ns':end-started,'rows':count,'output_bytes':output_bytes,'memory_kib':mem,
       'frameworks_loaded':forbidden,'input_sha256':ih.hexdigest(),'python':sys.version,'platform':platform.platform(),
       'model_sha256':identity,'scope':'same strict JSONL parsing, numerical buffer conversion, inference, label encoding, file flush/fsync/link; supplied HAR feature extraction excluded'}
    print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['native_linear','native_rbf','sklearn_linear','sklearn_rbf','numpy_linear'],required=True)
    for n in ('model','library','input','output','source','dependency-site'):p.add_argument('--'+n,type=Path)
    main(p.parse_args())
