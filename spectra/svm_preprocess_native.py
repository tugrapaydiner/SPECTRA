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


def build_preprocessor(out: str | Path, *, compiler: str | None = None) -> Path:
    """Explicit build for the running GIL-enabled CPython ABI, not abi3.

    Supports Linux x86-64/ARM64 and Windows x64. Requires development headers;
    Windows also requires its matching CPython import library and MSVC.
    """
    from .svm_build import preprocessor_build
    return preprocessor_build(out, SOURCE, compiler=compiler)


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
        if module.abi() != 4:
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


class _FusedRunner:
    """Private integration: each invocation leases the native worker allocation.

    No persistent capsule contains an address that can outlive explicit close.
    The invocation capsule holds the same owning resource as the operation lease.
    """
    def __init__(self, preprocessor, worker):
        import ctypes
        from .svm_shared import SharedSession
        if type(preprocessor) is not NativePreprocessor or type(worker) is not SharedSession:
            raise ValueError('fused execution requires stock compiled preprocessing and worker')
        self._preprocessor, self._worker = preprocessor, worker
        with worker._lock:
            if not worker._handle or worker._closing:
                raise ValueError('closed worker')
            if worker.features != preprocessor.features or worker.sha256 != preprocessor.model_sha256:
                raise ValueError('fused preprocessing/model binding mismatch')
            if worker.input_dtype != 'float64':
                raise ValueError('fused execution requires explicit float64 precision')
            self._run_address = ctypes.cast(worker._lib.sp_worker_run, ctypes.c_void_p).value
            self._error_address = ctypes.cast(worker._lib.et_error, ctypes.c_void_p).value

    def predict(self, rows, *, columns=None, schedule='beretta_cert', hint=-1, tile_rows=128):
        from .svm_shared import SHARED_SCHEDULES
        if type(tile_rows) is not int or not 1 <= tile_rows <= 128:
            raise ValueError('tile_rows must be an integer between 1 and 128')
        prep, worker = self._preprocessor, self._worker
        if columns is not None and tuple(itertools.islice(iter(columns), len(prep.columns)+1)) != prep.columns:
            raise ValueError('input schema/order mismatch')
        with worker._operation() as handle:
            if type(schedule) is not str or schedule not in SHARED_SCHEDULES:
                raise ValueError('invalid schedule')
            if type(hint) is not int or not -1 <= hint < len(worker.labels):
                raise ValueError('hint must be a valid class INDEX or -1')
            capsule = prep._module._bind_worker(handle, self._run_address, self._error_address,
                                               worker.labels, worker._resource)
            result = prep._module.predict_fused(prep._plan, capsule, rows,
                                               SHARED_SCHEDULES[schedule], hint, tile_rows)
        if result is NotImplemented:
            # No custom conversion/iteration happened in the eligibility pass.
            # The next operation acquires its own live lease (or rejects close).
            return worker.predict_buffer(prep.transform(rows), schedule=schedule, hint=hint)
        return result
