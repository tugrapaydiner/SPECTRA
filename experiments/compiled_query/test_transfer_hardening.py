"""Pre-freeze transport/provenance contracts; no held-out data are generated."""
from pathlib import Path
import json
import struct
import sys
import sysconfig

import pytest

from .binary_list import BinaryListProblem
from .exact import (ExactRuntime, build_exact_runtime, pack_session_packet,
                    unpack_session_packet)


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    import os
    explicit = os.environ.get('SPECTRA_EXACT_LIBRARY')
    path = Path(explicit) if explicit else build_exact_runtime(tmp_path_factory.mktemp('transfer-native'))
    return ExactRuntime(path)


@pytest.mark.parametrize('kind', ['reverse', 'duplicate', 'unsorted'])
def test_reference_and_native_reject_noncanonical_edges(runtime, kind):
    problem = BinaryListProblem.create(3, 2, ((0, 1), (1, 2)), (3, 3, 3))
    data = bytearray(pack_session_packet(problem, [()]))
    at = 32 + 8 * problem.n
    if kind == 'reverse':
        struct.pack_into('<II', data, at, 1, 0)
    elif kind == 'duplicate':
        data[at+8:at+16] = data[at:at+8]
    else:
        data[at:at+8], data[at+8:at+16] = data[at+8:at+16], data[at:at+8]
    with pytest.raises(ValueError):
        unpack_session_packet(bytes(data))
    with pytest.raises(ValueError):
        runtime.solve_session_packet(bytes(data))


@pytest.mark.parametrize('cap', [True, -1, 1 << 64, 1.5])
def test_caps_match_native_uint64_domain(runtime, cap):
    p = BinaryListProblem.create(1, 2, (), (3,))
    with pytest.raises(ValueError):
        runtime.prepare(p, max_bytes=cap)
    with pytest.raises(ValueError):
        runtime.solve_session_packet(pack_session_packet(p, [()]), max_bytes=cap)


def test_receipt_binds_interpreter(tmp_path):
    library = build_exact_runtime(tmp_path / 'bound')
    record = json.loads(library.with_name('build.json').read_text())
    assert record['python_executable'] == sys.executable
    assert record['python_version'] == sys.version
    assert record['soabi'] == sysconfig.get_config_var('SOABI')
    assert record['python_implementation'] == 'CPython'


def test_failed_receipt_removes_published_binary(tmp_path, monkeypatch):
    original = Path.open
    def fail(self, *args, **kwargs):
        if self.name == 'build.json' and args and args[0] == 'x':
            raise OSError('injected publication failure')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail)
    with pytest.raises(OSError, match='injected'):
        build_exact_runtime(tmp_path / 'failed')
    assert not list((tmp_path / 'failed').glob('*.so'))
    assert not (tmp_path / 'failed' / 'build.json').exists()


def test_output_handoff_admission_is_native(runtime):
    # The old single-envelope formula admits this cap; the corrected native
    # boundary must reserve for both owners of returned immutable output.
    n, count = 1001, 512
    p = BinaryListProblem.create(n, 2, tuple((v, v+1) for v in range(n-1)), (3,)*n)
    rows = [((0, 1), (n-1, 2))] * count
    packet = pack_session_packet(p, rows)
    envelope = 16 + count * (25 + 17*n)
    cap = envelope + 2048*(n+1) + 64*(2*count) + 64*count + len(packet) + (1 << 20)
    with pytest.raises(MemoryError, match='admission'):
        runtime._module.solve_session_packet(packet, 3, cap)
