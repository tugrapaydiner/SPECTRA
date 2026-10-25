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
    from experiments.compiled_query.exact import SOURCE as research_source
    from experiments.compiled_query.study import sources

    assert EXACT_API_VERSION == 1
    assert EXACT_NATIVE_ABI == 3
    packaged = ROOT / 'spectra/_native/exact_binary.cpp'
    assert research_source.resolve() == packaged.resolve()
    assert sources()[str(packaged.relative_to(ROOT))] == hashlib.sha256(packaged.read_bytes()).hexdigest()


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


def test_bank_replay_revalidates_caller_supplied_inputs():
    problem = tiny_problem()
    output = struct.pack('<4sIQ', b'SCQ2', 2, 1) + b'\x01\x00\x01' + struct.pack('<I', 0)
    assert verify_certified_output(problem, ((),), output)
    for query in (((True, 2),), ((0, True),), ((-1, 2),)):
        assert not verify_certified_output(problem, (query,), output)

    object.__setattr__(problem, 'masks', (True, 0b11))
    assert not verify_certified_output(problem, ((),), output)


def test_session_replay_rejects_malformed_packet_inputs():
    problem = ExactProblem.create(3, 2, ((0, 1), (1, 2)), (0b11,) * 3)
    packet = pack_session_packet(problem, (((0, 1),),))
    output = struct.pack('<4sIQ', b'SCQ2', 3, 1) + b'\x01\x00\x01\x00' + struct.pack('<I', 0)
    assert verify_session_output(packet, output)
    masks_at = struct.calcsize('<4s7I')
    edges_at = masks_at + 8 * problem.n
    offsets_at = edges_at + 8 * len(problem.edges)
    vertices_at = offsets_at + 8
    for offset, data in (
        (24, struct.pack('<I', 1)),
        (masks_at, struct.pack('<Q', 0)),
        (masks_at, struct.pack('<Q', 7)),
        (edges_at, struct.pack('<II', 1, 0)),
        (edges_at + 8, struct.pack('<II', 0, 1)),
        (offsets_at + 4, struct.pack('<I', 0)),
        (vertices_at, struct.pack('<I', problem.n)),
    ):
        damaged = bytearray(packet)
        damaged[offset:offset + len(data)] = data
        assert not verify_session_output(bytes(damaged), output), (offset, data)
    for damaged in (packet[:-1], packet + b'\x00', bytearray(packet)):
        assert not verify_session_output(damaged, output)


@pytest.mark.parametrize('palette,mask', [
    (1, 1), (2, 1), (2, 3), (8, (1 << 2) | (1 << 7)),
    (64, 1 << 63), (64, 1 | (1 << 63)),
])
def test_bank_domain_checks_match_scalar_verifier_for_every_byte(palette, mask):
    problem = ExactProblem.create(1, palette, (), (mask,))
    queries = ((), ((0, 0),), ((0, mask),), ((0, 1),),
               ((0, (1 << 64) - 1),), ((0, 1), (0, 1 << 63)))
    header = struct.pack('<4sIQ', b'SCQ2', 1, 1)
    for color in range(256):
        labels = bytes((color,))
        output = header + b'\x01' + labels + struct.pack('<I', 0)
        for query in queries:
            expected = verify_answer(problem, query, 'SAT_VERIFIED', labels, b'')
            assert verify_certified_output(problem, (query,), output) == expected


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


@pytest.mark.parametrize('interrupted_link', (1, 2))
def test_native_publication_rolls_back_links_published_before_interruption(
        tmp_path, monkeypatch, interrupted_link):
    from spectra import _native_build

    staging = tmp_path / 'staging'
    staging.mkdir()
    target = staging / 'native.so'
    target.write_bytes(b'compiled bytes')
    library = tmp_path / target.name
    link = _native_build.os.link
    calls = 0

    def interrupted(source, destination):
        nonlocal calls
        link(source, destination)
        calls += 1
        if calls == interrupted_link:
            raise OSError('interrupted publication')

    monkeypatch.setattr(_native_build.os, 'link', interrupted)
    with pytest.raises(OSError, match='interrupted publication'):
        _native_build.publish_build(target, library, {'complete': True})
    assert not library.exists()
    assert not library.with_name('build.json').exists()


@pytest.mark.parametrize('replacement', ('file', 'dangling-symlink'))
def test_native_publication_preserves_other_writer_outputs(tmp_path, monkeypatch, replacement):
    from spectra import _native_build

    if replacement == 'dangling-symlink' and os.name != 'posix':
        pytest.skip('unprivileged symlink creation is not portable on Windows')
    staging = tmp_path / 'staging'
    staging.mkdir()
    target = staging / 'native.so'
    target.write_bytes(b'compiled bytes')
    library = tmp_path / target.name
    receipt = tmp_path / 'build.json'
    receipt.write_bytes(b'existing receipt')
    link = _native_build.os.link

    def replace_before_receipt(source, destination):
        if Path(destination) == receipt:
            library.unlink()
            if replacement == 'file':
                library.write_bytes(b'another writer\'s library')
            else:
                library.symlink_to(tmp_path / 'missing-library')
        link(source, destination)

    monkeypatch.setattr(_native_build.os, 'link', replace_before_receipt)
    with pytest.raises(FileExistsError):
        _native_build.publish_build(target, library, {'complete': True})
    assert receipt.read_bytes() == b'existing receipt'
    if replacement == 'file':
        assert library.read_bytes() == b'another writer\'s library'
    else:
        assert library.is_symlink() and not library.exists()


@pytest.mark.parametrize('existing_output', ('library', 'receipt'))
def test_native_publication_preserves_a_preexisting_link_to_staged_inode(
        tmp_path, monkeypatch, existing_output):
    from spectra import _native_build

    staging = tmp_path / 'staging'
    staging.mkdir()
    target = staging / 'native.so'
    target.write_bytes(b'compiled bytes')
    library = tmp_path / target.name
    receipt = tmp_path / 'build.json'
    link = _native_build.os.link
    preexisting = library if existing_output == 'library' else receipt

    def race(source, destination):
        if Path(destination) == preexisting:
            link(source, destination)
        link(source, destination)  # The second link must fail with EEXIST.

    monkeypatch.setattr(_native_build.os, 'link', race)
    with pytest.raises(FileExistsError):
        _native_build.publish_build(target, library, {'complete': True})
    assert preexisting.is_file()
    if existing_output == 'receipt':
        assert json.loads(receipt.read_text()) == {'complete': True}
        assert not library.exists()


def test_native_load_guard_rechecks_receipt_and_remembers_failed_loads(tmp_path):
    from spectra._native_build import native_load_guard

    library = tmp_path / 'not-loaded.so'
    library.write_bytes(b'first identity')
    identity = hashlib.sha256(library.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='hash differs'):
        with native_load_guard(library, '0' * 64):
            pytest.fail('a mismatching receipt must not enter the loader')
    with pytest.raises(RuntimeError, match='injected loader failure'):
        with native_load_guard(library, identity):
            raise RuntimeError('injected loader failure')
    with native_load_guard(library, identity):
        pass
    library.write_bytes(b'second identity')
    with pytest.raises(ValueError, match='already loaded with different bytes'):
        with native_load_guard(library):
            pytest.fail('a previous failed loader may have retained its handle')


@pytest.mark.skipif(os.name != 'posix', reason='replacing a loaded extension requires POSIX unlink')
@pytest.mark.parametrize('research', (False, True))
def test_exact_runtime_rejects_changed_bytes_at_a_loaded_path(exact_build, tmp_path, research):
    runtime_type = ExactRuntime
    if research:
        from experiments.compiled_query.exact import ExactRuntime as runtime_type
    library = tmp_path / exact_build.name
    receipt = tmp_path / 'build.json'
    shutil.copy2(exact_build, library)
    shutil.copy2(exact_build.with_name('build.json'), receipt)
    runtime_type(library)
    runtime_type(library)
    changed = library.read_bytes() + b'\x00'
    library.unlink()
    library.write_bytes(changed)
    record = json.loads(receipt.read_text())
    record['library_sha256'] = hashlib.sha256(changed).hexdigest()
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='already loaded with different bytes'):
        runtime_type(library)


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
    assert result.info['auto_policy'] == 'structural-v11'
    assert result.info['selected_mode'] == 'raw'
    assert result.info['auto_policy_reason'] == 'short_weak_compression'
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


@pytest.mark.parametrize('mode', ('raw', 'scc', 'auto'))
def test_direct_forced_conflict_uses_short_original_edge_proof(
        exact_runtime, mode):
    problem = tiny_problem()
    query = ((0, 1), (1, 1))
    packet = pack_session_packet(problem, (query,))
    result = exact_runtime.solve_session_packet(packet, mode=mode)
    assert (result.sat_count, result.unsat_count) == (0, 1)
    code, labels, proof = decode_certified_output(
        result.output, problem.n, 1)[0]
    assert code == 0 and not any(labels)
    assert struct.unpack_from('<4sIII', proof) == (b'SBP1', 2, 2, 1)
    assert result.visited == 0
    assert result.arcs_scanned <= 4
    assert verify_session_output(packet, result.output)


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
