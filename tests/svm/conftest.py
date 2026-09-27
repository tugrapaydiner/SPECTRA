from pathlib import Path
import os
import pytest
from spectra.svm import build_runtime


@pytest.fixture(scope='session')
def library(tmp_path_factory):
    value = os.environ.get('SPECTRA_SVM_LIBRARY')
    if value:
        return Path(value).resolve(strict=True)
    return build_runtime(tmp_path_factory.mktemp('native') / 'build')


@pytest.fixture(scope='session')
def rounding_library(tmp_path_factory):
    """Read the target C runtime's FE_* constants, never Linux/x86 literals."""
    import ctypes
    from spectra.svm_build import command_for, execute
    folder = tmp_path_factory.mktemp('target-fenv')
    source = folder / 'fenv.cpp'
    source.write_text('#include <cfenv>\nextern "C" {\n'
                      'int test_get_round(){return std::fegetround();}\n'
                      'int test_set_round(int n){return std::fesetround(n);}\n'
                      'int test_downward(){return FE_DOWNWARD;}\n}\n', encoding='utf-8')
    import sys
    path = folder / ('fenv.dll' if sys.platform == 'win32' else 'fenv.so')
    command = command_for(source, path, target='portable', compiler=None,
                          exports=('test_get_round', 'test_set_round', 'test_downward'))
    execute(command, path, {'scope': 'test-only target fenv constants'})
    lib = ctypes.CDLL(str(path))
    lib.test_get_round.argtypes = []; lib.test_get_round.restype = ctypes.c_int
    lib.test_downward.argtypes = []; lib.test_downward.restype = ctypes.c_int
    lib.test_set_round.argtypes = [ctypes.c_int]; lib.test_set_round.restype = ctypes.c_int
    return lib
