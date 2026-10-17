"""Installed proof-carrying exact-query API contracts.

These are correctness and packaging contracts, not performance claims. Native
compilation is explicit and skipped only when the platform has no supported local
compiler; the dedicated exact-public workflow requires it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys

import pytest

from spectra.cli import main as cli_main
from spectra.exact import (
    EXACT_API_VERSION,
    EXACT_NATIVE_ABI,
    ExactProblem,
    ExactRuntime,
    build_exact_runtime,
    decode_certified_output,
    pack_query_bank,
    pack_session_packet,
    unpack_query_bank,
    unpack_session_packet,
    verify_answer,
    verify_certified_output,
    verify_session_output,
)

ROOT = Path(__file__).resolve().parents[2]


def tiny_problem() -> ExactProblem:
    return ExactProblem.create(2, 2, ((0, 1),), (0b11, 0b11))


def test_public_contract_versions_and_packaged_source_identity():
    assert EXACT_API_VERSION == 1
    assert EXACT_NATIVE_ABI == 3
    packaged = ROOT / 'spectra/_native/exact_binary.cpp'
    research = ROOT / 'experiments/compiled_query/exact_native.cpp'
    assert packaged.read_bytes() == research.read_bytes()
    assert hashlib.sha256(packaged.read_bytes()).hexdigest() == hashlib.sha256(
        research.read_bytes()).hexdigest()


def test_problem_canonicalizes_inputs_but_packets_require_canonical_bytes():
    problem = ExactProblem.create(3, 3, ((2, 1), (0, 1), (1, 0)), (0b011, 0b110, 0b101))
    assert problem.edges == ((0, 1), (1, 2))
    packet = pack_session_packet(problem, ((), ((0, 1),)))
    decoded, rows = unpack_session_packet(packet)
    assert decoded == problem and rows == ((), ((0, 1),))

    # Reverse the first canonical edge inside SBC1. The strict decoder must reject
    # rather than silently sort attacker-controlled prepared bytes.
    damaged = bytearray(packet)
    edge_at = struct.Struct('<4s7I').size + 8 * problem.n
    damaged[edge_at:edge_at + 8] = struct.pack('<II', 1, 0)
    with pytest.raises(ValueError, match='sorted unique'):
        unpack_session_packet(bytes(damaged))


def test_query_bank_round_trip_and_strict_types():
    problem = tiny_problem()
    queries = ((), ((0, 1),), ((0, 1), (1, 1)))
    packet = pack_query_bank(problem, queries)
    assert unpack_query_bank(packet, problem.n) == queries
    with pytest.raises(ValueError):
        pack_query_bank(problem, (((True, 1),),))
    with pytest.raises(ValueError):
        ExactProblem.create(True, 2, (), ())


def test_standard_library_verifier_accepts_sat_and_empty_domain_unsat():
    problem = tiny_problem()
    assert verify_answer(problem, (), 'SAT_VERIFIED', b'\x00\x01', b'')
    proof = struct.pack('<4sIII I', b'SBP1', 1, 1, 0, 0)
    assert verify_answer(problem, ((0, 0),), 'UNSAT_VERIFIED', b'', proof)
    assert not verify_answer(problem, (), 'UNSAT_VERIFIED', b'', proof)
    assert not verify_answer(problem, (), 'SAT_VERIFIED', b'\x00\x00', b'')


def _compiler_available() -> bool:
    return bool(shutil.which('cl' if sys.platform == 'win32' else 'g++'))


@pytest.fixture(scope='session')
def exact_build(tmp_path_factory):
    supplied = os.environ.get('SPECTRA_EXACT_LIBRARY')
    if supplied is not None:
        return Path(supplied).resolve()
    if sys.platform not in ('linux', 'win32') or not _compiler_available():
        pytest.skip('supported native compiler is unavailable')
    return build_exact_runtime(tmp_path_factory.mktemp('public-exact-runtime'))


@pytest.fixture(scope='session')
def exact_runtime(exact_build):
    return ExactRuntime(exact_build)


def test_receipt_bound_build_rejects_tampering(tmp_path, exact_build):
    build = tmp_path / 'build'
    build.mkdir()
    library = build / exact_build.name
    shutil.copy2(exact_build, library)
    shutil.copy2(exact_build.with_name('build.json'), build / 'build.json')
    record_path = library.with_name('build.json')
    record = json.loads(record_path.read_text())
    record['source_sha256'] = '0' * 64
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='source hash'):
        ExactRuntime(library)


def test_native_session_returns_checked_sat_and_unsat(exact_runtime):
    problem = tiny_problem()
    unsat_query = ((0, 1), (1, 1))
    with exact_runtime.prepare(problem, mode='scc') as session:
        sat = session.solve(())
        unsat = session.solve(unsat_query)
        assert sat.status == 'SAT_VERIFIED'
        assert problem.check(sat.labels)
        assert verify_answer(problem, (), sat.status, sat.labels, sat.proof)
        assert unsat.status == 'UNSAT_VERIFIED'
        assert verify_answer(problem, unsat_query, unsat.status, unsat.labels, unsat.proof)
        assert session.check((), sat)
        assert session.check(unsat_query, unsat)
        assert session.info['exact_api_version'] == EXACT_API_VERSION
        assert session.info['native_abi'] == EXACT_NATIVE_ABI


def test_one_call_default_uses_structural_auto_policy(exact_runtime):
    problem = tiny_problem()
    packet = pack_session_packet(problem, ((),) * 3)
    result = exact_runtime.solve_session_packet(packet)
    assert result.info['mode'] == 'auto'
    assert result.info['auto_policy'] == 'structural-v2'
    assert result.info['selected_mode'] == 'raw'
    assert result.info['auto_policy_reason'] == 'bank_below_tile64'
    assert verify_session_output(packet, result.output)


def test_one_call_packet_output_is_independently_replayable(exact_runtime):
    problem = tiny_problem()
    queries = ((), ((0, 1),), ((0, 1), (1, 1)))
    packet = pack_session_packet(problem, queries)
    result = exact_runtime.solve_session_packet(packet, mode='auto')
    assert result.query_count == 3
    assert result.sat_count == 2 and result.unsat_count == 1
    rows = decode_certified_output(result.output, problem.n, len(queries))
    assert tuple(code for code, _, _ in rows) == (1, 1, 0)
    assert verify_certified_output(problem, queries, result.output)
    assert verify_session_output(packet, result.output)

    damaged = bytearray(result.output)
    damaged[-1] ^= 1
    assert not verify_session_output(packet, bytes(damaged))


def test_runtime_close_is_idempotent_and_fail_closed(exact_runtime):
    session = exact_runtime.prepare(tiny_problem())
    session.close()
    session.close()
    with pytest.raises(RuntimeError):
        session.solve(())


def test_cli_replays_certified_output_without_native_code(tmp_path, capsys):
    problem = tiny_problem()
    query = ((0, 0),)
    packet = pack_session_packet(problem, (query,))
    proof = struct.pack('<4sIII I', b'SBP1', 1, 1, 0, 0)
    output = struct.pack('<4sIQ', b'SCQ2', problem.n, 1) + b'\x00' + bytes(problem.n) + struct.pack('<I', len(proof)) + proof
    packet_path = tmp_path / 'request.sbc1'
    output_path = tmp_path / 'answer.scq2'
    packet_path.write_bytes(packet)
    output_path.write_bytes(output)
    assert cli_main(['exact', 'verify', str(packet_path), str(output_path)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record == {
        'exact_api_version': 1, 'native_abi': 3, 'queries': 1,
        'sat': 0, 'unsat': 1, 'verified': True,
    }


def test_cli_exact_verify_enforces_file_limit(tmp_path, capsys):
    packet = tmp_path / 'request.sbc1'
    output = tmp_path / 'answer.scq2'
    packet.write_bytes(b'x' * 9)
    output.write_bytes(b'')
    assert cli_main(['exact', 'verify', str(packet), str(output), '--max-bytes', '8']) == 2
    assert 'exceeds max-bytes' in capsys.readouterr().err


def test_doctor_reports_packaged_exact_contract(capsys):
    assert cli_main(['doctor']) == 0
    record = json.loads(capsys.readouterr().out)
    assert record['exact_api_version'] == EXACT_API_VERSION
    assert record['exact_native_abi'] == EXACT_NATIVE_ABI
    assert record['exact_source_packaged'] is True
