"""Shared immutable SVM preparation with independent worker state.

SPCSVM01 compatibility and SPC SVM02 generic feature/label mapping. Inference uses
only the standard library. The original spectra.svm.Session is unchanged.
"""
from __future__ import annotations
from array import array
import ctypes as C
from dataclasses import dataclass
import hashlib
import json
import math
import itertools
import os
from pathlib import Path
import struct
import sys
import threading
from typing import Iterable
import zlib
from .svm import SCHEDULES, build_runtime, verify_certificate

MAX_BYTES = 64 * 1024 * 1024
MAX_ELEMENTS = 8_000_000
# Opt-in prototype; an ordering heuristic, never an acceptance rule.
SHARED_SCHEDULES = {**SCHEDULES, 'cost_aware': 6}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate metadata key')
        result[key] = value
    return result


def _labels_valid(labels: object, count: int) -> bool:
    if not isinstance(labels, list) or len(labels) != count:
        return False
    if not all(type(v) is int and -(2**63) <= v < 2**63 for v in labels):
        if not all(type(v) is str and len(v.encode('utf-8')) <= 256 for v in labels):
            return False
    return len(set(labels)) == count


def _decode(raw: bytes) -> tuple[int, tuple[int | str, ...]]:
    if not 24 <= len(raw) <= MAX_BYTES:
        raise ValueError('invalid model byte count')
    magic = raw[:8]
    if magic == b'SPCSVM01':
        _, classes, supports, size, crc = struct.unpack('<8sIIII', raw[:24])
        header, features, label_size = 24, 16, 0
        labels = list(range(classes)) if classes <= 128 else []
    elif magic == b'SPCSVM02' and len(raw) >= 32:
        _, classes, supports, features, label_size, size, crc = struct.unpack('<8sIIIIII', raw[:32])
        header = 32
        if label_size > 65536 or len(raw) != header + label_size + size:
            raise ValueError('invalid metadata inventory')
        try:
            metadata = json.loads(raw[header:header + label_size].decode('utf-8'), object_pairs_hook=_unique_object)
            if type(metadata) is not dict or set(metadata) != {'labels'}:
                raise ValueError('expected labels metadata')
            labels = metadata['labels']
        except (UnicodeError, RecursionError, TypeError) as error:
            raise ValueError('invalid labels metadata') from error
    else:
        raise ValueError('unsupported model format')
    if not 2 <= classes <= 128 or not 1 <= supports <= 100000 or not 1 <= features <= 4096:
        raise ValueError('unsupported model geometry')
    if supports * features > MAX_ELEMENTS:
        raise ValueError('support feature bank exceeds element limit')
    expected = 8 * (1 + features*supports + (classes-1)*supports + classes*(classes-1)//2) + 4*classes
    if size != expected or len(raw) != header + label_size + size:
        raise ValueError('invalid model inventory')
    if zlib.crc32(raw[header:]) != crc:
        raise ValueError('model CRC mismatch')
    try:
        if not _labels_valid(labels, classes):
            raise ValueError('labels must be unique bounded integers or UTF-8 strings of one type')
    except (UnicodeError, TypeError) as error:
        raise ValueError('invalid label encoding') from error
    return features, tuple(labels)


def _bind(path):
    lib = C.CDLL(str(Path(path).resolve(strict=True)))
    lib.sp_shared_abi.restype = C.c_int
    if lib.sp_shared_abi() != 1:
        raise ValueError('unsupported shared-model ABI')
    lib.et_error.restype = C.c_char_p
    lib.sp_model_create.argtypes = [C.c_void_p, C.c_uint64, C.c_int]
    lib.sp_model_create.restype = C.c_void_p
    lib.sp_model_destroy.argtypes = [C.c_void_p]
    lib.sp_model_destroy.restype = None
    lib.sp_worker_create.argtypes = [C.c_void_p]
    lib.sp_worker_create.restype = C.c_void_p
    lib.sp_worker_destroy.argtypes = [C.c_void_p]
    lib.sp_worker_destroy.restype = None
    for name in ('sp_model_info', 'sp_worker_info'):
        getattr(lib, name).argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.c_int]
    lib.sp_worker_run.argtypes = [C.c_void_p, C.POINTER(C.c_double), C.c_int, C.c_int, C.c_int, C.c_int,
                                  C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_uint64), C.c_int, C.c_int]
    lib.sp_worker_certificate.argtypes = [C.c_void_p, C.POINTER(C.c_int8), C.c_int]
    return lib


def _check(lib, status):
    if status:
        raise ValueError(lib.et_error().decode('utf-8', errors='replace'))


class PreparedModel:
    """One immutable native model; sessions allocate private caches, not weights.

    Labels are integers or strings and preserve exporter class ordering. Input
    precision is explicit: float32 reproduces the old input rounding; float64
    preserves the provided Python binary64 values. Workers outlive owner.close().
    Closing the owner prevents NEW workers, not work by existing ones. A raw native
    library is executable code and must be trusted. This is not a process sandbox.
    """
    def __init__(self, model: str | Path, library: str | Path, *, tables: bool = False,
                 input_dtype: str = 'float32'):
        self._lock = threading.RLock()
        self._handle = None
        if type(tables) is not bool or input_dtype not in ('float32', 'float64'):
            raise ValueError('expected tables bool and input_dtype float32 or float64')
        self._input_dtype = input_dtype
        with Path(model).open('rb') as stream:
            raw = stream.read(MAX_BYTES + 1)
        self._features, self._labels = _decode(raw)
        self._sha256 = hashlib.sha256(raw).hexdigest()
        self._lib = _bind(library)
        # Decode labels and native weights from the SAME owned bytes: no reopen race.
        blob = C.create_string_buffer(raw, len(raw))
        self._handle = self._lib.sp_model_create(blob, len(raw), int(tables))
        if not self._handle:
            raise ValueError(self._lib.et_error().decode('utf-8', errors='replace'))
        try:
            values = (C.c_uint64 * 7)()
            _check(self._lib, self._lib.sp_model_info(self._handle, values, 7))
            self._info = dict(zip(('model_id', 'features', 'classes', 'support_vectors',
                                   'shared_prepared_bytes', 'tables_enabled', 'dictionary_values'), values))
            if self._info['features'] != self.features or self._info['classes'] != len(self.labels):
                raise ValueError('native/Python model geometry mismatch')
        except BaseException:
            self.close()
            raise

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def input_dtype(self):
        return self._input_dtype

    @property
    def sha256(self):
        return self._sha256

    @property
    def info(self) -> dict:
        return {**self._info, 'sha256': self.sha256, 'input_dtype': self.input_dtype, 'labels': self.labels}

    def session(self) -> 'SharedSession':
        with self._lock:
            if not self._handle:
                raise ValueError('closed prepared model')
            handle = self._lib.sp_worker_create(self._handle)
            if not handle:
                raise ValueError(self._lib.et_error().decode('utf-8', errors='replace'))
            return SharedSession(self._lib, handle, self.features, self.labels, self.input_dtype, self.sha256)

    def close(self):
        with self._lock:
            if self._handle:
                self._lib.sp_model_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        with self._lock:
            if not self._handle:
                raise ValueError('closed prepared model')
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, '_lock'):
            self.close()


@dataclass(frozen=True)
class CertifiedPrediction:
    label: int | str
    class_index: int
    pair_outcomes: tuple[int, ...]
    evaluated_kernels: int
    evaluated_pairs: int
    evaluated_terms: int
    certificate_checks: int
    cost_scan_terms: int


class SharedSession:
    """Create via PreparedModel.session(). Native caches are private and locked."""
    def __init__(self, lib, handle, features, labels, input_dtype, sha256):
        self._lock = threading.RLock()
        self._lib, self._handle = lib, handle
        self._fused_active = False
        self._close_after_fused = False
        self._features, self._labels, self._input_dtype, self._sha256 = features, labels, input_dtype, sha256
        try:
            values = (C.c_uint64 * 3)()
            _check(lib, lib.sp_worker_info(handle, values, 3))
            self._info = dict(zip(('model_id', 'shared_prepared_bytes', 'worker_scratch_bytes'), values))
        except BaseException:
            self.close()
            raise

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def input_dtype(self):
        return self._input_dtype

    @property
    def sha256(self):
        return self._sha256

    @property
    def info(self):
        return {**self._info, 'features': self.features, 'classes': len(self.labels), 'sha256': self.sha256}

    def _pack(self, rows: Iterable[Iterable[float]]):
        packed = array('d')
        count = 0
        cap = min(65536, MAX_ELEMENTS // self.features)
        for row in rows:
            if count == cap:
                raise ValueError('batch exceeds row/element limit')
            # Consume at most d+1 values: malformed/infinite row iterators are bounded.
            values = list(itertools.islice(iter(row), self.features + 1))
            if len(values) != self.features:
                raise ValueError(f'exactly {self.features} features required')
            try:
                if any(isinstance(x, (str, bytes, bool)) or not math.isfinite(x) for x in values):
                    raise ValueError('finite numeric input required')
                rounded = array('f' if self.input_dtype == 'float32' else 'd', values)
            except (TypeError, OverflowError) as error:
                raise ValueError('unsupported input number') from error
            if any(not math.isfinite(x) for x in rounded):
                raise ValueError('input outside selected floating-point range')
            packed.extend(iter(rounded))
            count += 1
        return packed, count

    def _run(self, rows, schedule, hint, certificate):
        packed, count = self._pack(rows)
        with self._lock:
            if self._fused_active:
                raise ValueError('reentrant inference during fused execution')
            if not self._handle:
                raise ValueError('closed worker')
            if type(schedule) is not str or schedule not in SHARED_SCHEDULES:
                raise ValueError('invalid schedule')
            if type(hint) is not int or not -1 <= hint < len(self.labels):
                raise ValueError('hint must be a class INDEX or -1, not a label')
            data = (C.c_double * len(packed)).from_buffer(packed) if count else None
            output, stats = (C.c_int * count)(), (C.c_uint64 * 5)()
            _check(self._lib, self._lib.sp_worker_run(self._handle, data, count, self.features,
                SHARED_SCHEDULES[schedule], hint, output, count, stats, 5, int(certificate)))
            if not certificate:
                return list(output)
            trace = (C.c_int8 * (len(self.labels)*(len(self.labels)-1)//2))()
            _check(self._lib, self._lib.sp_worker_certificate(self._handle, trace, len(trace)))
            known = tuple(trace)
            if not verify_certificate(len(self.labels), output[0], known):
                raise RuntimeError('independent vote certificate check failed')
            return CertifiedPrediction(self.labels[output[0]], output[0], known, *map(int, stats))

    def predict(self, row, *, schedule='beretta_cert', hint=-1):
        return self.labels[self._run([row], schedule, hint, False)[0]]

    def predict_index(self, row, *, schedule='beretta_cert', hint=-1):
        return self._run([row], schedule, hint, False)[0]

    def predict_many(self, rows, *, schedule='beretta_cert', hint=-1):
        return [self.labels[i] for i in self._run(rows, schedule, hint, False)]

    def predict_buffer(self, values, *, schedule='beretta_cert', hint=-1):
        """Borrow a contiguous binary64 buffer and return a fresh list of labels.

        Use a float64 session and a writable, aligned, native-endian buffer:
        array('d') flat rows or a C-contiguous 2D float64 matrix. Inputs are NOT
        modified. The caller must not mutate them concurrently. The exporting
        object stays alive and resizing is blocked while its view is held.
        Readonly, noncontiguous or float32 buffers use predict_many instead.
        Native code validates the entire batch's finite values before any output.
        This amortizes conversion only; numerical work and per-row resets stay
        identical. Successful calls do not expose a last-row certificate.
        """
        if self.input_dtype != 'float64':
            raise ValueError('predict_buffer requires explicit float64 session precision')
        try:
            view = memoryview(values)
        except TypeError as error:
            raise ValueError('a native binary64 buffer is required') from error
        try:
            native = '<d' if sys.byteorder == 'little' else '>d'
            if view.format not in ('d', '@d', '=d', native) or view.itemsize != 8:
                raise ValueError('native-endian binary64 buffer required; no implicit casting')
            if view.readonly or not view.c_contiguous or view.ndim not in (1, 2):
                raise ValueError('writable C-contiguous one- or two-dimensional buffer required')
            if view.ndim == 2 and view.shape[1] != self.features:
                raise ValueError('buffer feature dimension mismatch')
            elements = view.nbytes // 8
            if elements % self.features:
                raise ValueError('flat buffer must contain complete rows')
            count = elements // self.features
            if count > 65536 or elements > MAX_ELEMENTS:
                raise ValueError('batch exceeds row/element limit')
            data = (C.c_double * elements).from_buffer(view) if elements else None
            if data is not None and C.addressof(data) % C.alignment(C.c_double):
                raise ValueError('binary64 buffer must be naturally aligned')
            with self._lock:
                if self._fused_active:
                    raise ValueError('reentrant inference during fused execution')
                if not self._handle:
                    raise ValueError('closed worker')
                if type(schedule) is not str or schedule not in SHARED_SCHEDULES:
                    raise ValueError('invalid schedule')
                if type(hint) is not int or not -1 <= hint < len(self.labels):
                    raise ValueError('hint must be a valid class INDEX or -1')
                output, stats = (C.c_int * count)(), (C.c_uint64 * 5)()
                _check(self._lib, self._lib.sp_worker_run(self._handle, data, count, self.features,
                    SHARED_SCHEDULES[schedule], hint, output, count, stats, 5, 0))
                return [self.labels[i] for i in output]
        finally:
            # ctypes can hold a second export; dropping it before release permits
            # the caller to resize its array once this method returns.
            if 'data' in locals():
                del data
            view.release()

    def predict_with_certificate(self, row, *, schedule='beretta_cert', hint=-1):
        return self._run([row], schedule, hint, True)

    def close(self):
        with self._lock:
            if self._fused_active:
                # A signal handler can re-enter an RLock on the same thread.
                # Keep the borrowed pointer alive until the C request unwinds.
                self._close_after_fused = True
                return
            if self._handle:
                self._lib.sp_worker_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        with self._lock:
            if not self._handle:
                raise ValueError('closed worker')
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, '_lock'):
            self.close()
