"""Process-level evidence and accounting for the additive proof-carrying arms."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import struct

import pytest

from .audit import audit_study
from .cases import Case, canonical, digest, write_case
from .exact import ExactRuntime, build_exact_runtime, pack_session_packet
from .binary_list import BinaryListProblem
from .proof import verify_answer
from .sessions import (decode_certified_output, run_prepared_session,
                       run_session, query_order)
from .study import Config, create_study, read_manifest, run_study
from . import study
from spectra.cnf.quotient_query import QuotientRuntime, build_quotient_runtime


@pytest.fixture(scope='module')
def libraries(tmp_path_factory):
    q = os.environ.get('SPECTRA_QUOTIENT_LIBRARY')
    e = os.environ.get('SPECTRA_EXACT_LIBRARY')
    return (Path(q) if q else build_quotient_runtime(tmp_path_factory.mktemp('quotient')),
            Path(e) if e else build_exact_runtime(tmp_path_factory.mktemp('exact')))


@pytest.fixture
def example():
    p = BinaryListProblem.create(4, 2, ((0, 1), (1, 2), (2, 3)), (3,)*4)
    return Case('mixed', 'one-graph', 'synthetic without filtering', p,
                ((), ((0, 1), (3, 1)), ((1, 0),), ((0, 2),), ((0, 1), (0, 2))))


@pytest.mark.parametrize('shared_source', ('spectra/exact_proof.py', 'spectra/_exact_wire.py'))
def test_shared_source_change_rejects_frozen_study(tmp_path, monkeypatch, shared_source):
    source = tmp_path / 'source'
    checker = Path(shared_source)
    paths = {Path(name) for name in study.sources()} | {checker}
    for path in paths:
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((study.ROOT / path).read_bytes())
    monkeypatch.setattr(study, 'ROOT', source)
    monkeypatch.setattr(study, 'execution_identity', lambda: {'fixture': 'unchanged'})
    root = tmp_path / 'study'
    root.mkdir()
    native = b'identity-only fixture; never loaded'
    (root / 'native.fixture').write_bytes(native)
    payload = canonical(dict(sources=study.sources(),
        execution_identity=study.execution_identity(), config=asdict(Config()),
        library='native.fixture', library_sha256=digest(native), cases=[], jobs=[]))
    (root / 'manifest.json').write_bytes(payload)
    (root / 'manifest.sha256').write_text(digest(payload))
    assert read_manifest(root) == json.loads(payload)

    with (source / checker).open('ab') as stream:
        stream.write(b'\n# changed shared source\n')
    with pytest.raises(ValueError, match='benchmark source changed'):
        read_manifest(root)


@pytest.mark.parametrize('arm', ('exact-raw', 'exact-scc', 'exact-sparse', 'exact-route'))
@pytest.mark.parametrize('order', ('forward', 'reverse', 'shuffle'))
def test_complete_costs_include_proofs_and_disposal(tmp_path, libraries, example, arm, order):
    q, e = libraries
    path = tmp_path / 'case.json'; write_case(path, example)
    record, payload = run_session(path, QuotientRuntime(q), arm, count=5, order=order,
                                  seed=13, exact_runtime=ExactRuntime(e))
    assert record['status'] == 'COMPLETE'
    assert record['counts'] == {'SAT_VERIFIED': 2, 'UNSAT': 3}
    assert record['decision_contract'] == 'checked-SAT-and-UNSAT'
    fields = ('input_ns', 'setup_ns', 'query_loop_ns', 'materialize_ns', 'diagnostics_ns', 'dispose_ns')
    assert record['complete_ns'] == sum(record[k] for k in fields)
    assert record['query_loop_ns'] >= record['query_ns'] == sum(record['per_query_ns'])
    rows = decode_certified_output(payload, 4, 5)
    assert sum(len(p) for _, _, p in rows) == record['proof_bytes']
    assert len(payload) == 16 + 5*(4+5) + record['proof_bytes']
    assert record['unsat_certificate'] and record['output_sha256'] == digest(payload)
    for (decision, labels, proof), index in zip(rows, query_order(5, order, 13)):
        assert verify_answer(example.problem, example.queries[index],
                             'SAT_VERIFIED' if decision else 'UNSAT_VERIFIED',
                             labels if decision else b'', proof)


@pytest.mark.parametrize('mutation', ('trailing', 'truncate', 'magic', 'count', 'decision', 'length', 'padding'))
def test_scq2_transport_is_strict(tmp_path, libraries, example, mutation):
    q, e = libraries
    path=tmp_path/'case.json'; write_case(path, example)
    _, payload=run_session(path, QuotientRuntime(q), 'exact-scc', count=5, exact_runtime=ExactRuntime(e))
    out=bytearray(payload)
    # First row is SAT with no proof; second row is UNSAT.
    if mutation=='trailing': out+=b'!'
    elif mutation=='truncate': out=out[:-1]
    elif mutation=='magic': out[3]=ord('1')
    elif mutation=='count': out[8]=1
    elif mutation=='decision': out[16]=2
    elif mutation=='length': out[21:25]=b'\xff'*4
    elif mutation=='padding': out[26]=1
    with pytest.raises(ValueError): decode_certified_output(bytes(out),4,5)


def make(tmp_path, libraries, example, arms=('exact-scc', 'exact-raw', 'scc', 'minicard')):
    q, e=libraries
    path=tmp_path/'case.json';write_case(path,example)
    root=tmp_path/'study'
    create_study([path],q,root,Config(arms=arms,prefixes=(5,),repeats=1),exact_library=e)
    return root


def test_worker_exact_and_legacy_contracts_stay_separate(tmp_path, libraries, example):
    root=make(tmp_path,libraries,example)
    assert len(run_study(root))==4
    report=audit_study(root)
    assert report['outcomes']=={'COMPLETE':3,'INCOMPLETE':1}
    assert report['certified_decisions']=={'SAT_VERIFIED':4,'UNSAT_VERIFIED':6}
    assert report['mixed_output_contracts'] is True
    assert report['portable_unsat_certificate'] is True
    assert run_study(root)==[]
    for row in report['comparisons']:
        if row['candidate']=='exact-scc' and row['baseline']=='exact-raw':
            assert row['eligible']
        if row['candidate']=='exact-scc' and row['baseline']=='scc':
            assert not row['eligible'] and 'geomean_complete_ratio' not in row


def test_ready_session_packet_is_frozen_run_and_independently_audited(tmp_path, libraries, example):
    q, e = libraries
    path = tmp_path / 'case.json'; write_case(path, example)
    root = tmp_path / 'prepared-study'
    create_study([path], q, root,
                 Config(arms=('exact-packet-route', 'exact-packet-raw'),
                        prefixes=(5,), repeats=1), exact_library=e)
    manifest = read_manifest(root)
    assert manifest['prepared_packet_schema'] == 'SBC1'
    assert all(job['prepared_input']['preparation_charged'] is False
               for job in manifest['jobs'])
    assert len(run_study(root)) == 2
    report = audit_study(root)
    assert report['outcomes'] == {'COMPLETE': 2}
    assert report['prepared_packet_contracts'] is True
    assert report['certified_decisions'] == {'SAT_VERIFIED': 4, 'UNSAT_VERIFIED': 6}
    comparison = next(row for row in report['comparisons']
                      if row['candidate'] == 'exact-packet-route'
                      and row['baseline'] == 'exact-packet-raw')
    assert comparison['eligible'] and comparison['matching_certificate_contract']
    assert run_study(root) == []


def test_ready_packet_direct_endpoint_charges_read_hash_and_native_disposal(
        tmp_path, libraries, example):
    _, e = libraries
    packet = pack_session_packet(example.problem, example.queries)
    path = tmp_path / 'case.sbc1'; path.write_bytes(packet)
    record, output = run_prepared_session(
        path, ExactRuntime(e), 'exact-packet-route', count=5, order='forward', seed=0,
        case_name=example.name, cluster=example.cluster, provenance=example.provenance,
        case_sha256='original-json-digest', packet_sha256=digest(packet))
    assert record['timing_granularity'] == 'session-packet'
    assert record['packet_preparation_charged'] is False
    assert record['complete_ns'] == sum(record[key] for key in
        ('input_ns', 'setup_ns', 'query_loop_ns', 'materialize_ns',
         'diagnostics_ns', 'dispose_ns'))
    assert record['setup_ns'] == 0 and record['per_query_ns'] == []
    assert record['counts'] == {'SAT_VERIFIED': 2, 'UNSAT': 3}
    assert len(decode_certified_output(output, 4, 5)) == 5


def test_exact_library_receipt_tamper_rejected(tmp_path,libraries,example):
    root=make(tmp_path,libraries,example)
    manifest=read_manifest(root)
    with (root/manifest['exact_runtime']['library']).open('ab') as stream: stream.write(b'!')
    with pytest.raises(ValueError,match='exact native library changed'): run_study(root)


def test_runtime_must_be_explicit(tmp_path,libraries,example):
    q,_=libraries
    path=tmp_path/'case.json';write_case(path,example)
    with pytest.raises(ValueError,match='explicit'):
        create_study([path],q,tmp_path/'bad',Config(arms=('exact-scc',),prefixes=(5,)))
    with pytest.raises(ValueError,match='explicit'):
        run_session(path,QuotientRuntime(q),'exact-scc',count=5)


def test_forged_proof_rejected_even_with_new_outer_hashes(tmp_path,libraries,example):
    root=make(tmp_path,libraries,example,arms=('exact-scc',))
    run_study(root)
    folder=root/'sessions/000000'
    output=bytearray((folder/'outputs.bin').read_bytes())
    # SCQ2 header 16, first SAT row 9, second row proof begins at 34.
    # Replace its first literal by an impossible original literal, preserving size.
    output[34+16:34+20]=struct.pack('<I',999)
    (folder/'outputs.bin').write_bytes(output)
    record=json.loads((folder/'record.json').read_bytes());record['output_sha256']=digest(output)
    (folder/'record.json').write_bytes(canonical(record))
    control=json.loads((folder/'controller.json').read_bytes())
    control['record_sha256']=digest((folder/'record.json').read_bytes())
    (folder/'controller.json').write_bytes(canonical(control))
    with pytest.raises(ValueError,match='proof/witness'):audit_study(root)


def test_unfiltered_request_generation_keeps_original_domain_failures(example):
    from .mixed_cases import unfiltered_requests
    rows = unfiltered_requests(example.problem, 100)
    assert rows == unfiltered_requests(example.problem, 100)
    assert len(rows) == 100 and rows[0] == ()
    assert rows[17][0][0] == rows[17][-1][0]
    assert rows[17][0][1] & rows[17][-1][1] == 0
    assert rows[23][-1][1] == 0
    assert rows[29][0][1] & (1 << 63)
