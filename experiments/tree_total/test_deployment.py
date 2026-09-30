"""Total JSONL contracts: hard cases, no replacement, no partial publication."""
from array import array
import hashlib
import io
import json
import pytest
from .test_total import library,verified
from ..certified_trees.test_native import source
from .session import TotalSession
from .deployment import stream

@pytest.fixture
def engine(library):
    # Ordinary/residual compact certificates cannot settle this source tie.
    raw=source([[-100.,100.],[1e-20,100.],[100.,-100.]])
    with TotalSession(verified(raw,D=1),library) as model:yield model


def test_hard_input_publishes_original_answer(engine,tmp_path):
    raw=b'[0]\n[1]\n'*40;dest=tmp_path/'result'
    result=stream(engine,io.BytesIO(raw),dest,chunk_rows=7)
    lines=dest.read_bytes().splitlines(keepends=True);doc=[json.loads(v) for v in lines]
    assert len(doc)==82 and [r['class_index'] for r in doc[1:-1]]==[0,1]*40
    assert result['rows']==80 and result['exact_completed']==40 and result['unresolved']==0
    assert result['input_sha256']==hashlib.sha256(raw).hexdigest()
    assert result['prefix_sha256']==hashlib.sha256(b''.join(lines[:-1])).hexdigest()
    assert result['output_sha256']==hashlib.sha256(dest.read_bytes()).hexdigest()


@pytest.mark.parametrize('bad',[b'\n',b'{}\n',b'[true]\n',b'[2]\n',b'[1.0]\n',b'[NaN]\n',b'[[1]]\n',b'[1,0]\n',b'["private"]\n',b'\xff\n'])
def test_late_errors_leave_no_output(engine,tmp_path,bad):
    dest=tmp_path/'out'
    with pytest.raises(ValueError):stream(engine,io.BytesIO(b'[0]\n'*4+bad),dest,chunk_rows=1)
    assert not dest.exists() and not list(tmp_path.glob('*.partial'))
    assert engine.predict_buffer(array('B',[0]))==[0]


def test_existing_path_unread(engine,tmp_path):
    dest=tmp_path/'out';dest.write_bytes(b'old')
    class Unread:
        def readline(self,*args):raise AssertionError('existing output should fail first')
    with pytest.raises(FileExistsError):stream(engine,Unread(),dest)
    assert dest.read_bytes()==b'old'


@pytest.mark.parametrize('limit',[{'max_rows':1},{'max_output_bytes':2},{'max_line_bytes':2},{'chunk_rows':True}])
def test_limits_fail_without_publication(engine,tmp_path,limit):
    dest=tmp_path/'out'
    with pytest.raises(ValueError):stream(engine,io.BytesIO(b'[0]\n[1]\n'),dest,**limit)
    assert not dest.exists()


def test_bad_native_counter(engine,tmp_path,monkeypatch):
    original=engine.inspect_buffer
    def changed(*args,**kwargs):
        r=original(*args,**kwargs);r['work']['exact_completed']=True;return r
    monkeypatch.setattr(engine,'inspect_buffer',changed)
    with pytest.raises(ValueError):stream(engine,io.BytesIO(b'[0]\n'),tmp_path/'out')
    assert not (tmp_path/'out').exists()
