"""Explicitly built optional CPython preprocessing; no numerical dependencies.

The validated Python plan remains the reference. The common built-in scalar path
uses C++ loops with identical subtract/divide order and no intermediate row tuples. Other scalar objects fall back
to the reference on the same bounded materialized rows; no labels are cached.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import itertools
from pathlib import Path
import subprocess
import sys
import sysconfig

SOURCE = Path(__file__).resolve().parent / '_native' / 'ovo' / 'preprocess.cpp'


def build_preprocessor(out: str | Path, *, compiler: str = 'g++') -> Path:
    """Build for this CPython ABI, not abi3 or a universal binary.

    Requires the running interpreter's development headers and a C++17 compiler.
    Import and installation never compile. Failed build logs are retained.
    Linux and GIL-enabled CPython are the currently accepted build scope.
    """
    if sys.implementation.name != 'cpython' or sys.platform != 'linux':
        raise ValueError('compiled preprocessing requires CPython on Linux')
    if sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise ValueError('free-threaded CPython is not supported')
    include = Path(sysconfig.get_path('include'))
    if not (include / 'Python.h').is_file():
        raise ValueError('CPython development headers are required')
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if not isinstance(suffix, str) or not suffix.endswith('.so'):
        raise ValueError('unsupported CPython extension suffix')
    folder = Path(out).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    library = folder / ('_spectra_preprocess' + suffix)
    command = [compiler, '-std=c++17', '-O3', '-fno-fast-math', '-ffp-contract=off',
               '-fPIC', '-shared', '-I' + str(include), str(SOURCE), '-o', str(library)]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    receipt = {'command': command, 'returncode': process.returncode,
               'stdout': process.stdout, 'stderr': process.stderr, 'python': sys.version,
               'soabi': sysconfig.get_config_var('SOABI'),
               'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    if process.returncode == 0:
        receipt['library_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
    (folder / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    if process.returncode:
        raise RuntimeError(f'preprocessing build failed; see {folder / "build.json"}')
    return library


class NativePreprocessor:
    """Owned native snapshot of an already validated Preprocessor.

    A fresh writable memoryview of binary64 values is returned on the accelerated
    path. Non-built-in real/string objects retain Python reference behavior and
    may return array('d'). Both support the same buffer/iteration interface. The
    GIL remains held during C++ traversal; no multi-thread throughput claim.
    Supplied libraries are executable code and must be trusted.
    """
    def __init__(self, reference, library: str | Path):
        from .svm_pipeline import Preprocessor
        if type(reference) is not Preprocessor:
            raise ValueError('expected a validated stock Preprocessor')
        if sys.implementation.name != 'cpython' or sysconfig.get_config_var('Py_GIL_DISABLED'):
            raise ValueError('GIL-enabled CPython required')
        path = Path(library).resolve(strict=True)
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        if not suffix or not path.name.endswith(suffix):
            raise ValueError('preprocessing library must target the running CPython ABI')
        spec = importlib.util.spec_from_file_location('_spectra_preprocess', path)
        if spec is None or spec.loader is None:
            raise ValueError('cannot load preprocessing extension')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.abi() != 2:
            raise ValueError('unsupported preprocessing ABI')
        self._reference, self._module = reference, module
        numerical = tuple((o.column, o.output, o.fill, o.mean, o.scale) for o in reference._numerical)
        categorical = tuple((o.column, o.offset, tuple(o.lookup), o.unknown == 'error')
                            for o in reference._categorical)
        self._plan = module.prepare(len(reference.columns), reference.features,
                                    reference._row_cap, numerical, categorical)

    @property
    def columns(self):
        return self._reference.columns

    @property
    def features(self):
        return self._reference.features

    @property
    def model_sha256(self):
        return self._reference.model_sha256

    @property
    def sha256(self):
        return self._reference.sha256

    def transform(self, rows, *, columns=None):
        if columns is not None and tuple(itertools.islice(iter(columns), len(self.columns)+1)) != self.columns:
            raise ValueError('input schema/order mismatch')
        result = self._module.transform_raw(self._plan, rows)
        if result is not NotImplemented:
            return memoryview(result).cast('d')
        materialized = self._reference._materialize(rows)
        result = self._module.transform(self._plan, materialized)
        if result is NotImplemented:
            return self._reference._transform_rows(materialized)
        return memoryview(result).cast('d')
