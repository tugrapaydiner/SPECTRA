"""Independent JSONL result audit rejects changed/missing completion and labels."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

p=Path(__file__).resolve().parents[2]/'experiments/streaming/audit.py'
spec=importlib.util.spec_from_file_location('stream_audit',p)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture_bytes():
    header={'format':'spectra.svm.stream.v1','record':'header','model_sha256':'1'*64,
            'preprocessing_sha256':'2'*64,'input_dtype':'float64'}
    rows=[header,{'record':'prediction','index':0,'label':'a'},{'record':'prediction','index':1,'label':'b'}]
    prefix=b''.join((json.dumps(v)+'\n').encode() for v in rows)
    footer={'record':'complete','rows':2,'input_bytes':8,'input_sha256':'3'*64,
            'output_prefix_sha256':hashlib.sha256(prefix).hexdigest(),'batches':1}
    return prefix+(json.dumps(footer)+'\n').encode()


def check(path):return audit.verify_result(path,expected=['a','b'],input_identity={'bytes':8,'sha256':'3'*64},model_sha256='1'*64,plan_sha256='2'*64)


def test_valid_result(tmp_path):
    path=tmp_path/'result';raw=fixture_bytes();path.write_bytes(raw)
    assert check(path)=={'status':'PASS','rows':2,'output_sha256':hashlib.sha256(raw).hexdigest()}


@pytest.mark.parametrize('damage',['label','index','missing','trailing','duplicate','input-hash','prefix-hash','model','plan','bool-row','boolean-index','row-count'])
def test_corrupt_result(tmp_path,damage):
    path=tmp_path/'result';raw=fixture_bytes();lines=raw.splitlines(keepends=True)
    if damage=='label':raw=raw.replace(b'"label": "b"',b'"label": "c"')
    elif damage=='index':raw=raw.replace(b'"index": 1',b'"index": 9')
    elif damage=='missing':raw=b''.join(lines[:-1])
    elif damage=='trailing':raw+=b'{}\n'
    elif damage=='duplicate':raw=raw.replace(b'"record": "prediction"',b'"record": "x","record": "prediction"',1)
    elif damage=='input-hash':raw=raw.replace(b'3'*64,b'4'*64)
    elif damage=='prefix-hash':
        obj=json.loads(lines[-1]);obj['output_prefix_sha256']='0'*64;raw=b''.join(lines[:-1])+(json.dumps(obj)+'\n').encode()
    elif damage=='model':raw=raw.replace(b'1'*64,b'4'*64)
    elif damage=='plan':raw=raw.replace(b'2'*64,b'4'*64)
    elif damage=='bool-row':raw=raw.replace(b'"rows": 2',b'"rows": true')
    elif damage=='boolean-index':raw=raw.replace(b'"index": 0',b'"index": false')
    else:raw=raw.replace(b'"rows": 2',b'"rows": 3')
    path.write_bytes(raw)
    with pytest.raises(ValueError):check(path)
