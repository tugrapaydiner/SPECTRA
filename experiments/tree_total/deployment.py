"""Bounded total-source JSONL runner: no external model or unresolved predictions.

Uses the existing no-replacement staged-file primitive. Output directories and
native binaries must be trusted. File completion is not a crash-durability claim.
"""
from __future__ import annotations
from array import array
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from spectra.svm_stream import _staged_output
from .compiler import VerifiedTotal
from .session import TotalSession


def _row(raw, width, maximum):
    def reject(_):raise ValueError('nonfinite input')
    try:value=json.loads(raw.decode('utf-8'),parse_constant=reject)
    except (UnicodeError,RecursionError,json.JSONDecodeError) as error:
        raise ValueError('strict UTF-8 JSON array required') from error
    if type(value) is not list or len(value)!=width:
        raise ValueError('wrong feature count')
    if any(type(v) is not int or not 0<=v<=maximum for v in value):
        raise ValueError('original bounded integer features required')
    return value


def stream(engine, source, destination, *, chunk_rows=128, max_rows=1_000_000,
           max_line_bytes=65536, max_output_bytes=128*1024**2):
    if type(engine) is not TotalSession:
        raise ValueError('verified total session required')
    for name,value,upper in [('chunk_rows',chunk_rows,65536),('max_rows',max_rows,10_000_000),
                             ('max_line_bytes',max_line_bytes,1024**2),('max_output_bytes',max_output_bytes,2**31)]:
        if type(value) is not int or not 1<=value<=upper:
            raise ValueError('invalid '+name)
    if chunk_rows*engine.features>8_000_000:
        raise ValueError('native batch element cap')
    count=coarse=exact=encoded=0
    source_hash=hashlib.sha256();output_hash=hashlib.sha256()
    with _staged_output(Path(destination)) as output:
        def write(record):
            nonlocal encoded
            raw=(json.dumps(record,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')
            if encoded+len(raw)>max_output_bytes:
                raise ValueError('output byte cap')
            if output.write(raw)!=len(raw):
                raise OSError('incomplete output write')
            encoded+=len(raw);output_hash.update(raw)
        write({'format':'spectra.tree.total.v1','type':'header','features':engine.features,
               'maximum':engine.maximum,'classes':engine.classes,'model_sha256':engine.model_sha256,
               'source_sha256':engine.source_sha256,'claim':'specified source arithmetic; not ground-truth correctness'})
        data=array('B');pending=0
        def flush():
            nonlocal count,coarse,exact,data,pending
            if not pending:return
            result=engine.inspect_buffer(data);indices=result['indices'];work=result['work']
            if len(indices)!=pending or any(type(v) is not int or not 0<=v<engine.classes for v in indices):
                raise ValueError('total engine returned an invalid or unresolved index')
            for key in ('coarse_certified','exact_completed','unresolved'):
                if type(work.get(key)) is not int or not 0<=work[key]<=pending:
                    raise ValueError('invalid native counter')
            if work['unresolved'] or work['coarse_certified']+work['exact_completed']!=pending:
                raise ValueError('native coverage inventory differs')
            for index in indices:
                write({'type':'prediction','row':count,'class_index':index});count+=1
            coarse+=work['coarse_certified'];exact+=work['exact_completed']
            data=array('B');pending=0
        while True:
            raw=source.readline(max_line_bytes+1)
            if type(raw) is not bytes:raise ValueError('binary input stream required')
            if not raw:break
            if len(raw)>max_line_bytes:raise ValueError('input line byte cap')
            if count+pending>=max_rows:raise ValueError('input row cap')
            row=_row(raw,engine.features,engine.maximum)
            source_hash.update(raw);data.extend(row);pending+=1
            if pending==chunk_rows:flush()
        flush()
        complete={'type':'complete','rows':count,'coarse_certified':coarse,'exact_completed':exact,
                  'unresolved':0,'input_sha256':source_hash.hexdigest(),'prefix_sha256':output_hash.hexdigest()}
        write(complete)
        report={**complete,'output_bytes':encoded,'output_sha256':output_hash.hexdigest()}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','model','library','input','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--chunk-rows',type=int,default=128)
    a=p.parse_args()
    if os.path.lexists(a.output):p.error('output already exists')
    try:
        proof=VerifiedTotal.from_files(a.source,a.model)
        with TotalSession(proof,a.library) as engine,a.input.open('rb') as source:
            result=stream(engine,source,a.output,chunk_rows=a.chunk_rows)
    except (ValueError,OSError) as error:
        p.exit(2,str(error)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
