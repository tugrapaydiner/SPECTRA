"""Stream publication tests; fake engines isolate transport, not certificate proof."""
import hashlib
import io
import json
from pathlib import Path
import pytest
from .deployment import stream_predict

class Engine:
    features, maximum, classes = 2, 3, 2
    def __init__(self, missing=False, broken=False, failure=False):
        self.missing, self.broken, self.failure = missing, broken, failure
    def inspect_buffer(self, data, *, refine, fallback):
        if self.failure: raise RuntimeError('native failure fixture')
        n = len(data)//2
        indices = [int(data[2*i] > data[2*i+1]) for i in range(n)]
        absent = 1 if self.missing and n and not fallback else 0
        if absent: indices[-1] = -1
        return {'indices': indices, 'work': {'certified_first': n-absent+(1 if self.broken else 0),
                'certified_second': 0, 'official_rows': 0, 'unresolved_rows': absent}}

def execute(tmp_path, raw=b'[0,1]\n[3,0]\n', engine=None, **kwargs):
    out=tmp_path/'result.jsonl'
    result=stream_predict(engine or Engine(), [7,99], io.BytesIO(raw), out, identity={'source':'fixture'}, **kwargs)
    return out,result

@pytest.mark.parametrize('chunk', [1,2,3,128])
def test_outputs_and_hashes(tmp_path,chunk):
    out,result=execute(tmp_path,chunk_rows=chunk)
    raw=out.read_bytes(); lines=raw.splitlines(keepends=True); rows=[json.loads(x) for x in lines]
    assert [r['label'] for r in rows[1:-1]]==[7,99]
    assert result['rows']==2 and result['unresolved_rows']==0
    assert result['prefix_sha256']==hashlib.sha256(b''.join(lines[:-1])).hexdigest()
    assert result['output_sha256']==hashlib.sha256(raw).hexdigest()
    assert not list(tmp_path.glob('*.partial'))

@pytest.mark.parametrize('bad',[b'',b'\n',b'[true,0]\n',b'[1.0,0]\n',b'[4,0]\n',b'[-1,0]\n',
                                b'[0]\n',b'{}\n',b'[NaN,0]\n',b'\xff\n'])
def test_late_bad_input_no_publication(tmp_path,bad):
    if bad==b'': bad=b'['
    with pytest.raises(ValueError): execute(tmp_path,b'[0,1]\n'+bad,chunk_rows=1)
    assert not (tmp_path/'result.jsonl').exists() and not list(tmp_path.glob('.*.partial'))

def test_blank_file_is_complete_zero_rows(tmp_path):
    out,result=execute(tmp_path,b'')
    assert result['rows']==0 and len(out.read_bytes().splitlines())==2

def test_last_line_without_newline(tmp_path):
    _,result=execute(tmp_path,b'[0,1]')
    assert result['rows']==1

@pytest.mark.parametrize('kwargs',[{'max_rows':1},{'max_line_bytes':16},{'max_output_bytes':256}])
def test_late_limits(tmp_path,kwargs):
    raw=b'[0,1]\n'+(b' '*40+b'[0,1]\n' if 'max_line_bytes' in kwargs else b'[0,1]\n')
    with pytest.raises(ValueError): execute(tmp_path,raw,chunk_rows=1,**kwargs)
    assert not (tmp_path/'result.jsonl').exists()

def test_existing_output_and_broken_symlink_untouched(tmp_path):
    out=tmp_path/'result.jsonl';out.write_bytes(b'preserve')
    with pytest.raises(FileExistsError): execute(tmp_path)
    assert out.read_bytes()==b'preserve';out.unlink();out.symlink_to('missing-target')
    with pytest.raises(FileExistsError): execute(tmp_path)
    assert out.is_symlink()

def test_racing_writer_is_not_overwritten(tmp_path,monkeypatch):
    from . import deployment
    link=deployment.os.link
    def race(src,dst):
        Path(dst).write_bytes(b'other writer')
        return link(src,dst)
    monkeypatch.setattr(deployment.os,'link',race)
    with pytest.raises(FileExistsError): execute(tmp_path)
    assert (tmp_path/'result.jsonl').read_bytes()==b'other writer'

def test_unresolved_is_not_last_class(tmp_path):
    out,result=execute(tmp_path,engine=Engine(missing=True),compact_only=True)
    records=[json.loads(s) for s in out.read_bytes().splitlines()]
    assert records[-2]['label'] is None and records[-2]['class_index'] is None
    assert records[-2]['status']=='UNRESOLVED' and result['unresolved_rows']==1

@pytest.mark.parametrize('engine',[Engine(broken=True),Engine(failure=True)])
def test_native_failure_no_publication(tmp_path,engine):
    with pytest.raises((ValueError,RuntimeError)):execute(tmp_path,engine=engine)
    assert not (tmp_path/'result.jsonl').exists()

@pytest.mark.parametrize('value',[0,-1,True,65537])
def test_bad_chunks(tmp_path,value):
    with pytest.raises(ValueError):execute(tmp_path,chunk_rows=value)
