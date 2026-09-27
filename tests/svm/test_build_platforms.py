"""Build contract tests; command inspection is NOT remote platform execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest
from spectra import svm_build as build


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(build.sys, 'platform', 'win32')
    monkeypatch.setattr(build.platform, 'machine', lambda: 'AMD64')
    for name in ('CL', '_CL_', 'LINK', '_LINK_'):
        monkeypatch.delenv(name, raising=False)


def test_windows_strict_export_command(windows, tmp_path):
    args = build.command_for(tmp_path/'source.cpp', tmp_path/'file.dll', target='avx2',
                             compiler=None, exports=('symbol',))
    assert args[0] == 'cl'
    assert '/fp:strict' in args and '/arch:AVX2' in args and '/EXPORT:symbol' in args
    assert not any(s in args for s in ('/fp:fast', '/fp:contract', '-ffast-math'))
    assert args.index('/link') > args.index('/MD')


@pytest.mark.parametrize('name', ('CL', '_CL_', 'LINK', '_LINK_'))
def test_windows_ambient_flags_rejected(windows, monkeypatch, tmp_path, name):
    monkeypatch.setenv(name, '/fp:fast')
    with pytest.raises(ValueError, match='overrides'):
        build.command_for(tmp_path/'x.cpp', tmp_path/'x.dll', target='portable', compiler=None)


def test_windows_requires_msvc(windows, tmp_path):
    with pytest.raises(ValueError, match='MSVC'):
        build.command_for(tmp_path/'x.cpp', tmp_path/'x.dll', target='portable', compiler='gcc')


def test_arm_rejects_avx2(monkeypatch):
    monkeypatch.setattr(build.sys, 'platform', 'linux')
    monkeypatch.setattr(build.platform, 'machine', lambda: 'aarch64')
    assert build.host_target('portable') == 'aarch64'
    with pytest.raises(ValueError, match='AVX2'): build.host_target('avx2')


@pytest.mark.parametrize('machine', ('i686', 'ppc64', 's390x', 'mips'))
def test_unsupported_linux_machine_rejected(monkeypatch, machine):
    monkeypatch.setattr(build.sys, 'platform', 'linux')
    monkeypatch.setattr(build.platform, 'machine', lambda: machine)
    with pytest.raises(ValueError): build.host_target('portable')


def test_missing_compiler_receipt(tmp_path):
    lib = tmp_path/'missing.so'
    with pytest.raises(RuntimeError, match='build.json'):
        build.execute([str(tmp_path/'compiler-not-there')], lib, {'test': True})
    receipt = json.loads((tmp_path/'build.json').read_text())
    assert receipt['status'] == 'BUILD_FAILED' and receipt['exception'] == 'FileNotFoundError'


def test_failed_compiler_receipt(tmp_path):
    with pytest.raises(RuntimeError):
        build.execute([sys.executable, '-c', 'raise SystemExit(3)'], tmp_path/'bad.so', {})
    assert json.loads((tmp_path/'build.json').read_text())['returncode'] == 3


def test_success_without_library_is_failure(tmp_path):
    with pytest.raises(RuntimeError):
        build.execute([sys.executable, '-c', 'pass'], tmp_path/'absent.so', {})
    assert json.loads((tmp_path/'build.json').read_text())['status'] == 'BUILD_FAILED'


def test_timeout_receipt(tmp_path, monkeypatch):
    def fail(*a, **k): raise subprocess.TimeoutExpired(['compiler'], 180)
    monkeypatch.setattr(build.subprocess, 'run', fail)
    with pytest.raises(RuntimeError): build.execute(['compiler'], tmp_path/'x.so', {})
    assert json.loads((tmp_path/'build.json').read_text())['exception'] == 'TimeoutExpired'


def test_all_declared_exports_exist(library):
    import ctypes
    lib = ctypes.CDLL(str(library))
    assert all(hasattr(lib, name) for name in build.SVM_EXPORTS)


def test_shared_model_unicode_path(tmp_path, library):
    from test_shared_model import encoded
    from spectra.svm_shared import PreparedModel
    folder = tmp_path/'Türkçe 日本語'; folder.mkdir()
    model = encoded(folder/'model.srt')
    with PreparedModel(model, library) as m, m.session() as w:
        assert w.predict([0.]*16) in m.labels


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows narrow legacy file-path contract')
def test_legacy_windows_rejects_unicode(tmp_path, library):
    from spectra.svm import Session
    with pytest.raises(ValueError, match='ASCII'):
        Session(tmp_path/'Türkçe.srt', library)
