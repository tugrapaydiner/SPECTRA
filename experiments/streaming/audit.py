"""Independent output-file integrity/label audit; not a native math verifier."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('duplicate output key')
        result[key]=value
    return result


def verify_result(path, *, expected, input_identity, model_sha256, plan_sha256):
    prefix=hashlib.sha256();full=hashlib.sha256();count=0;footer=None
    with Path(path).open('rb') as stream:
        first=stream.readline(4*1024*1024+1)
        if len(first)>4*1024*1024:raise ValueError('oversized header')
        header=json.loads(first,object_pairs_hook=_unique)
        if (header.get('format')!='spectra.svm.stream.v1' or header.get('record')!='header'
            or header.get('model_sha256')!=model_sha256 or header.get('preprocessing_sha256')!=plan_sha256
            or header.get('input_dtype')!='float64'):
            raise ValueError('invalid output binding')
        prefix.update(first);full.update(first)
        for line in stream:
            if len(line)>1_048_576:raise ValueError('oversized output row')
            obj=json.loads(line,object_pairs_hook=_unique);full.update(line)
            if obj.get('record')=='complete':
                footer=obj
                if stream.read(1):raise ValueError('trailing output')
                break
            if (set(obj)!={'record','index','label'} or obj['record']!='prediction'
                or type(obj['index']) is not int or obj['index']!=count or count>=len(expected)
                or type(obj['label']) is not type(expected[count]) or obj['label']!=expected[count]):
                raise ValueError('prediction index or label mismatch')
            count+=1;prefix.update(line)
    if footer is None or set(footer)!={'record','rows','input_bytes','input_sha256','output_prefix_sha256','batches'}:
        raise ValueError('missing or invalid completion record')
    if (type(footer['rows']) is not int or footer['rows']!=len(expected) or count!=len(expected)
        or type(footer['input_bytes']) is not int or footer['input_bytes']!=input_identity['bytes']
        or footer['input_sha256']!=input_identity['sha256']
        or footer['output_prefix_sha256']!=prefix.hexdigest()
        or type(footer['batches']) is not int or not 0<=footer['batches']<=count):
        raise ValueError('completion integrity mismatch')
    return {'status':'PASS','rows':count,'output_sha256':full.hexdigest()}
