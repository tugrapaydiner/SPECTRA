"""Explicit SVM builds for Linux x86-64/ARM64 and Windows x64.

No installation/import side effects, bundled compiler, or emulated target.
Windows requires a configured MSVC x64 developer environment. Numerical source
is shared; distinct system libm implementations need per-host validation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig

# Export the existing C ABI on Windows without editing retained numerical files.
SVM_EXPORTS = (
    'lo_error', 'lo_create', 'lo_destroy', 'lo_info', 'lo_run', 'lo_certificate',
    'lo_kernels', 'lo_tournament', 'et_create', 'et_destroy', 'et_error', 'et_run',
    'et_info', 'et_cache_info', 'et_table_info', 'et_certificate', 'et_kernels',
    'et_tournament', 'et_feature_probe', 'sp_svm_abi', 'sp_svm_batch', 'sp_svm_single',
    'sp_shared_abi', 'sp_model_create', 'sp_model_destroy', 'sp_worker_create',
    'sp_worker_destroy', 'sp_model_info', 'sp_worker_info', 'sp_worker_run',
    'sp_worker_certificate',
)


def host_target(target: str) -> str:
    """Reject unsupported hosts before creating a build directory."""
    machine = platform.machine().lower()
    if target not in ('portable', 'avx2') or type(target) is not str:
        raise ValueError('target must be portable or avx2')
    if sys.maxsize <= 2**32 or sys.byteorder != 'little':
        raise ValueError('64-bit little-endian Python is required')
    if sys.platform == 'win32':
        if machine not in ('amd64', 'x86_64'):
            raise ValueError('Windows builds currently require x64 Python')
    elif sys.platform == 'linux':
        if machine not in ('amd64', 'x86_64', 'aarch64', 'arm64'):
            raise ValueError('Linux builds require x86-64 or ARM64')
    else:
        raise ValueError('supported build hosts are Linux and Windows x64')
    if target == 'avx2' and machine not in ('amd64', 'x86_64'):
        raise ValueError('AVX2 requires x86-64; use portable on ARM64')
    return machine


def command_for(source: Path, library: Path, *, target: str, compiler: str | None,
                include: Path | None = None, import_library: Path | None = None,
                exports: tuple[str, ...] = ()) -> list[str]:
    """Build argv only; no shell parsing or implicit compilation flags."""
    host_target(target)
    compiler = compiler or ('cl' if sys.platform == 'win32' else 'g++')
    if not isinstance(compiler, str) or not compiler or '\0' in compiler:
        raise ValueError('compiler must be a path to one executable')
    if sys.platform == 'win32':
        if Path(compiler).name.lower() not in ('cl', 'cl.exe'):
            raise ValueError('Windows requires MSVC cl.exe from an x64 developer environment')
        if any(os.environ.get(key) for key in ('CL', '_CL_', 'LINK', '_LINK_')):
            raise ValueError('clear CL/_CL_/LINK/_LINK_ overrides for a recorded strict build')
        command = [compiler, '/nologo', '/std:c++17', '/O2', '/fp:strict', '/EHsc',
                   '/MD', '/LD', '/utf-8', '/DNOMINMAX']
        if target == 'avx2':
            command.append('/arch:AVX2')
        if include is not None:
            command.append('/I' + str(include))
        command += [str(source), '/link', '/OUT:' + str(library), '/INCREMENTAL:NO']
        if import_library is not None:
            command.append(str(import_library))
        command += ['/EXPORT:' + name for name in exports]
        return command
    command = [compiler, '-std=c++17', '-O3', '-fno-fast-math', '-ffp-contract=off',
               '-fPIC', '-shared']
    if target == 'avx2':
        command.append('-mavx2')
    if include is not None:
        command.append('-I' + str(include))
    return command + [str(source), '-o', str(library)]


def execute(command: list[str], library: Path, receipt: dict) -> Path:
    """Keep a failure receipt, including missing compilers and build timeouts."""
    receipt = {**receipt, 'command': command, 'platform': sys.platform,
               'machine': platform.machine(), 'python': sys.version,
               'source_order_contract': 'separate binary64 operations; no contraction'}
    try:
        process = subprocess.run(command, cwd=library.parent, capture_output=True,
                                 text=True, encoding='utf-8', errors='replace',
                                 check=False, timeout=180)
        receipt.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        receipt.update(returncode=None, stdout='', stderr=str(error), exception=type(error).__name__)
    success = receipt['returncode'] == 0 and library.is_file()
    if success:
        receipt['library_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
    else:
        receipt['status'] = 'BUILD_FAILED'
    with (library.parent / 'build.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=True)
        stream.write('\n')
    if not success:
        raise RuntimeError(f'native build failed; see {library.parent / "build.json"}')
    return library


def runtime_build(out: str | Path, source_root: Path, *, target: str = 'portable',
                  compiler: str | None = None) -> Path:
    host_target(target)
    folder = Path(out).resolve()
    library = folder / ('spectra_svm.dll' if sys.platform == 'win32' else 'libspectra_svm.so')
    source_root = source_root.resolve()
    command = command_for(source_root/'runtime.cpp', library, target=target,
                          compiler=compiler, exports=SVM_EXPORTS)
    hashes = {p.relative_to(source_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source_root.rglob('*')) if p.suffix in ('.cpp', '.hpp')}
    folder.mkdir(parents=True, exist_ok=False)
    return execute(command, library, {'target': target, 'source_sha256': hashes})


def preprocessor_build(out: str | Path, source: Path, *, compiler: str | None = None) -> Path:
    host_target('portable')
    if sys.implementation.name != 'cpython' or sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise ValueError('GIL-enabled CPython is required')
    include = Path(sysconfig.get_path('include'))
    if not (include / 'Python.h').is_file():
        raise ValueError('CPython development headers are required')
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if not isinstance(suffix, str) or not suffix.endswith(('.so', '.pyd')):
        raise ValueError('unsupported CPython extension suffix')
    import_library = None
    if sys.platform == 'win32':
        if sysconfig.get_config_var('Py_DEBUG'):
            raise ValueError('Windows debug Python is not supported')
        name = f'python{sys.version_info.major}{sys.version_info.minor}.lib'
        import_library = Path(sys.base_prefix) / 'libs' / name
        if not import_library.is_file():
            raise ValueError('CPython import library missing: ' + str(import_library))
    folder = Path(out).resolve()
    library = folder / ('_spectra_preprocess' + suffix)
    command = command_for(source.resolve(), library, target='portable', compiler=compiler,
                          include=include, import_library=import_library)
    folder.mkdir(parents=True, exist_ok=False)
    return execute(command, library, {'soabi': sysconfig.get_config_var('SOABI'),
                    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
