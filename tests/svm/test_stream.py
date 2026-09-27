"""Offline streaming contracts: bounded reads, atomic publication and exact labels."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from spectra.svm_pipeline import PreparedPipeline
from spectra.svm_preprocess_native import build_preprocessor
from spectra.svm_stream import Limits, run_stream
from test_fused_pipeline import fixture_folder


@pytest.fixture(scope='module')
def prelib(tmp_path_factory):
    value = os.environ.get('SPECTRA_PREPROCESS_LIBRARY')
    return Path(value) if value else build_preprocessor(tmp_path_factory.mktemp('stream-pre')/'build')


@pytest.fixture
def prepared(tmp_path, library, prelib):
    with PreparedPipeline(fixture_folder(tmp_path/'bundle'), library,
                          preprocessor_library=prelib) as model:
        yield model


def encode(rows):
    return b''.join((json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8') for row in rows)


def check_file(path, raw, expected):
    pieces = path.read_bytes().splitlines(keepends=True)
    doc = [json.loads(line) for line in pieces]
    assert doc[0]['record'] == 'header' and doc[0]['format'] == 'spectra.svm.stream.v1'
    assert doc[-1]['record'] == 'complete'
    assert doc[-1]['input_sha256'] == hashlib.sha256(raw).hexdigest()
    assert doc[-1]['input_bytes'] == len(raw)
    assert doc[-1]['output_prefix_sha256'] == hashlib.sha256(b''.join(pieces[:-1])).hexdigest()
    assert doc[1:-1] == [{'record':'prediction', 'index':i, 'label':v} for i,v in enumerate(expected)]
    assert doc[-1]['rows'] == len(expected)
    return doc


@pytest.mark.parametrize('engine', ['two_stage', 'fused'])
@pytest.mark.parametrize('rows', [[], [[1.,'α',None]], [[1.,'b',2.], [None,'unknown',-0.]]*257],
                         ids=['empty', 'single', 'multiple-chunks'])
def test_complete_output_and_hashes(prepared, tmp_path, engine, rows):
    with prepared.session() as w:
        expected = w.predict_many(rows, schedule='exhaustive')
    raw=encode(rows); destination=tmp_path/'result.jsonl'
    result=run_stream(io.BytesIO(raw), destination, prepared,
                      limits=Limits(batch_rows=3), engine=engine, schedule='binary_stream')
    check_file(destination,raw,expected)
    assert result['status']=='COMPLETE'
    assert result['output_sha256']==hashlib.sha256(destination.read_bytes()).hexdigest()
    assert result['batches']==(len(rows)+2)//3
    assert not list(tmp_path.glob('.spectra-*.partial'))


@pytest.mark.parametrize('raw', [b'', b'[1,"b",2]', b'[1,"b",2]\r\n'], ids=['eof', 'no-final-newline','crlf'])
def test_eof_conventions(prepared,tmp_path,raw):
    dest=tmp_path/'out'
    result=run_stream(io.BytesIO(raw),dest,prepared)
    assert result['rows']==(1 if raw else 0)
    assert result['input_bytes']==len(raw)


@pytest.mark.parametrize('bad', [
    b'\n', b'{}\n', b'{"a":1,"a":2}\n', b'[1,"b",2,3]\n',
    b'[1,"b"]\n', b'[NaN,"b",2]\n', b'[Infinity,"b",2]\n',
    b'[1e999,"b",2]\n', b'[true,"b",2]\n', b'[[1],"b",2]\n',
    b'[1,{"b":1},2]\n', b'[1,"\xff",2]\n', b'[1,"\\ud800",2]\n',
    b'[1,"'+b'x'*4097+b'",2]\n', b'["SENSITIVE","b",2]\n',
    b'['*2000+b']'*2000+b'\n', b'[1,"b",2]extra\n',
], ids=['blank','object','duplicate','wide','short','nan','infinity','overflow',
        'boolean','nested-list','nested-object','utf8','surrogate','long-string',
        'wrong-column-type','nesting','extra-text'])
def test_late_errors_leave_no_result(prepared,tmp_path,bad):
    dest=tmp_path/'out'
    with pytest.raises(ValueError) as failure:
        run_stream(io.BytesIO(encode([[1.,'b',2.]])*3+bad),dest,prepared,
                   limits=Limits(batch_rows=1))
    assert 'SENSITIVE' not in str(failure.value)
    assert not dest.exists() and not list(tmp_path.glob('.spectra-*.partial'))
    with prepared.session() as w:
        assert w.predict_many([[1.,'b',2.]])


@pytest.mark.parametrize('name', ['batch_rows','line_bytes','batch_bytes','total_rows','output_bytes'])
@pytest.mark.parametrize('value',[0,-1,True,1.5,sys.maxsize])
def test_invalid_limits(name,value):
    with pytest.raises(ValueError): Limits(**{name:value})


def test_inconsistent_limits():
    with pytest.raises(ValueError):Limits(batch_rows=65537)
    with pytest.raises(ValueError):Limits(line_bytes=101,batch_bytes=100)


@pytest.mark.parametrize('kwargs', [dict(total_rows=2),dict(output_bytes=10),
                                   dict(line_bytes=10,batch_bytes=10)])
def test_resource_failures_do_not_publish(prepared,tmp_path,kwargs):
    dest=tmp_path/'out'
    with pytest.raises(ValueError):run_stream(io.BytesIO(encode([[1.,'b',2.]]*3)),dest,prepared,limits=Limits(**kwargs))
    assert not dest.exists() and not list(tmp_path.glob('.spectra-*.partial'))


def test_batch_byte_limit_and_reads_are_bounded(prepared,tmp_path,monkeypatch):
    from spectra.svm_pipeline import PipelineSession
    source_bytes=encode([[1.,'b',2.]]*8)
    requests=[];sizes=[]
    class Source(io.BytesIO):
        def readline(self,size=-1):
            assert size==33
            requests.append(size)
            return super().readline(size)
    original=PipelineSession.predict_fused
    def watch(self,rows,**kwargs):
        sizes.append(len(rows));assert not (tmp_path/'out').exists()
        return original(self,rows,**kwargs)
    monkeypatch.setattr(PipelineSession,'predict_fused',watch)
    run_stream(Source(source_bytes),tmp_path/'out',prepared,limits=Limits(line_bytes=32,batch_bytes=32))
    assert sizes==[2,2,2,2] and len(requests)==9


def test_existing_output_refused_without_read(prepared,tmp_path):
    dest=tmp_path/'out';dest.write_bytes(b'original')
    class Source:
        def readline(self,*args):raise AssertionError('must not read')
    with pytest.raises(FileExistsError):run_stream(Source(),dest,prepared)
    assert dest.read_bytes()==b'original'


def test_concurrent_creator_is_not_overwritten(prepared,tmp_path,monkeypatch):
    import spectra.svm_stream as stream
    original=stream.os.link
    def competing(source,destination):
        Path(destination).write_bytes(b'another writer')
        return original(source,destination)
    monkeypatch.setattr(stream.os,'link',competing)
    dest=tmp_path/'out'
    with pytest.raises(FileExistsError):run_stream(io.BytesIO(encode([[1.,'b',2.]])),dest,prepared)
    assert dest.read_bytes()==b'another writer'
    assert not list(tmp_path.glob('.spectra-*.partial'))


def test_unsupported_link_fails_without_replacement(prepared,tmp_path,monkeypatch):
    import spectra.svm_stream as stream
    def unsupported(*args):raise OSError('link unsupported')
    monkeypatch.setattr(stream.os,'link',unsupported)
    with pytest.raises(OSError):run_stream(io.BytesIO(b''),tmp_path/'out',prepared)
    assert not (tmp_path/'out').exists() and not list(tmp_path.glob('.spectra-*.partial'))


def test_interrupt_cleanup(prepared,tmp_path):
    class Interrupted(io.BytesIO):
        def readline(self,*a):raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):run_stream(Interrupted(),tmp_path/'out',prepared)
    assert not (tmp_path/'out').exists() and not list(tmp_path.glob('.spectra-*.partial'))


def test_bad_engine_schedule_and_source(prepared,tmp_path):
    for kwargs in ({'engine':'auto'}, {'schedule':'maybe'},{'limits':{}}):
        with pytest.raises(ValueError):run_stream(io.BytesIO(b''),tmp_path/'out',prepared,**kwargs)
    with pytest.raises(ValueError):run_stream(io.StringIO(''),tmp_path/'out',prepared)
    assert not (tmp_path/'out').exists()


def test_compiled_explicit_requirement(tmp_path,library):
    with PreparedPipeline(fixture_folder(tmp_path/'bundle'),library) as p:
        with pytest.raises(ValueError,match='explicitly'):run_stream(io.BytesIO(b''),tmp_path/'out',p)
        assert run_stream(io.BytesIO(b''),tmp_path/'out',p,engine='two_stage')['rows']==0


def test_cli_without_frameworks(prepared,tmp_path,library,prelib):
    rows=[[1.,'b',2.],[None,'α',-0.]]*6
    inp=tmp_path/'input.jsonl';inp.write_bytes(encode(rows))
    out=tmp_path/'predictions.jsonl'
    root=Path(__file__).resolve().parents[2]
    code="import sys;sys.path.insert(0,sys.argv.pop(1));from spectra.cli import main;r=main();assert not ({'numpy','pandas','sklearn','torch'}&sys.modules.keys());raise SystemExit(r)"
    args=[sys.executable,'-I','-S','-c',code,str(root),'svm','run',str(tmp_path/'bundle'),
          '--library',str(library),'--preprocessor',str(prelib),'--input',str(inp),'--output',str(out),'--batch-rows','3']
    p=subprocess.run(args,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['rows']==len(rows)
    old=out.read_bytes()
    p=subprocess.run(args,capture_output=True,text=True)
    assert p.returncode==2 and 'already exists' in p.stderr and out.read_bytes()==old
    out.unlink();inp.write_bytes(encode(rows)+b'["private value","b",2]\n')
    p=subprocess.run(args,capture_output=True,text=True)
    assert p.returncode==2 and not out.exists() and 'private value' not in p.stderr
