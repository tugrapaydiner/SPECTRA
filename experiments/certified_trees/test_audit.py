"""Small parser/identity tests; synthetic inputs are not benchmark observations."""
from pathlib import Path
import json,struct
import pytest
from .audit import loads,truth,bindings,indices,child

@pytest.mark.parametrize('raw',[b'{"a":1,"a":2}',b'[NaN]',b'[Infinity]',b'[1e999]'])
def test_strict_json(raw):
    with pytest.raises(ValueError):loads(raw)

@pytest.mark.parametrize('value',[{},[],{'a':'0'},{'a':'g'*64}])
def test_missing_or_bad_bindings(tmp_path,value):
    with pytest.raises(ValueError):bindings(value,tmp_path,('a',))

@pytest.mark.parametrize('name',['../escape','/tmp/escape','a\\b'])
def test_escaping_paths(tmp_path,name):
    with pytest.raises(ValueError):child(tmp_path,name)

def test_index_length(tmp_path):
    p=tmp_path/'a';p.write_bytes(struct.pack('<ii',1,-1))
    assert indices(p,2)==[1,-1]
    with pytest.raises(ValueError):indices(p,1)

def test_truth_rejects_pickle_and_bad_geometry(tmp_path):
    p=tmp_path/'y';p.write_bytes(b'pickle data')
    with pytest.raises(ValueError):truth(p,1)
    h=str({'descr':'<i8','fortran_order':False,'shape':(2,)}).encode();p.write_bytes(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(h))+h+struct.pack('<qq',10,20))
    assert truth(p,2)==(10,20)
    with pytest.raises(ValueError):truth(p,1)
