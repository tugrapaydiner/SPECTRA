from __future__ import annotations

import itertools
import os
from pathlib import Path
import shutil
import sys

import pytest

from spectra.exact import (
    ExactProblem,
    ExactRuntime,
    build_exact_runtime,
    decode_certified_output,
    pack_session_packet,
    verify_session_output,
)
from spectra.exact_proof import decode_proof


def _compiler_available() -> bool:
    return bool(shutil.which('cl' if sys.platform == 'win32' else 'g++'))


@pytest.fixture(scope='session')
def runtime(tmp_path_factory):
    supplied=os.environ.get('SPECTRA_EXACT_LIBRARY')
    if supplied:
        return ExactRuntime(Path(supplied).resolve())
    if sys.platform not in ('linux','win32') or not _compiler_available():
        pytest.skip('supported native compiler is unavailable')
    return ExactRuntime(build_exact_runtime(
        tmp_path_factory.mktemp('edgeless-exact')))


def _expected(problem, row):
    domains=list(problem.masks)
    for vertex,allowed in row:
        domains[vertex]&=allowed
    return int(all(domains))


def test_auto_edgeless_kernel_matches_exhaustive_two_vertex_oracle(runtime):
    rows=tuple(
        ((0,left),(1,right))
        for left in range(8)
        for right in range(8)
    )
    for masks in itertools.product((1,2,3,4,5,6),repeat=2):
        problem=ExactProblem.create(2,3,(),masks)
        packet=pack_session_packet(problem,rows)
        result=runtime.solve_session_packet(packet,mode='auto')
        assert result.info['bank_kernel']=='edgeless'
        assert result.info['selected_mode']=='scc'
        assert result.info['auto_policy_reason']=='scc_long_bank'
        assert verify_session_output(packet,result.output)
        decoded=decode_certified_output(result.output,problem.n,len(rows))
        assert tuple(code for code,_labels,_proof in decoded)==tuple(
            _expected(problem,row) for row in rows)
        assert all(
            code or decode_proof(proof,problem.n)[0]==1
            for code,_labels,proof in decoded
        )


def test_edgeless_kernel_covers_high_colours_duplicates_and_short_policy(runtime):
    problem=ExactProblem.create(
        4,64,(),
        ((1<<63)|(1<<5),(1<<62)|(1<<7),1<<61,(1<<60)|(1<<2)),
    )
    rows=(
        (),
        ((0,1<<63),),
        ((0,1<<5),(0,1<<63)),
        ((1,1<<7),),
        ((2,0),),
        ((3,1<<2),),
        ((3,(1<<60)|(1<<2)),),
        ((0,(1<<63)|(1<<5)),(1,1<<62)),
    )
    packet=pack_session_packet(problem,rows)
    result=runtime.solve_session_packet(packet,mode='auto')
    assert result.info['bank_kernel']=='edgeless'
    assert result.info['selected_mode']=='raw'
    assert result.info['auto_policy_reason']=='short_weak_compression'
    assert verify_session_output(packet,result.output)
    decoded=decode_certified_output(result.output,problem.n,len(rows))
    assert tuple(code for code,_labels,_proof in decoded)==(
        1,1,0,1,0,1,1,1
    )
    assert [
        decode_proof(proof,problem.n)[0]
        for code,_labels,proof in decoded if not code
    ]==[1,1]


def test_explicit_edgeless_modes_remain_available_as_controls(runtime):
    problem=ExactProblem.create(3,2,(),(3,3,3))
    packet=pack_session_packet(problem,((),)*64)
    auto=runtime.solve_session_packet(packet,mode='auto')
    raw=runtime.solve_session_packet(packet,mode='raw')
    scc=runtime.solve_session_packet(packet,mode='scc')
    assert auto.info['bank_kernel']=='edgeless'
    assert raw.info['bank_kernel']!='edgeless'
    assert scc.info['bank_kernel']!='edgeless'
    assert all(verify_session_output(packet,row.output)
               for row in (auto,raw,scc))
