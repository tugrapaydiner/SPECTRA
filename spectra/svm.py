"""Optional framework-free RBF SVM execution. Build explicitly, never on import.

Supports the SPC SVM01 format, 16 finite binary32 features and first-index OvO
voting. Certificates establish fidelity to computed pair decisions, not labels.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
from typing import Iterable
from .svm_lifetime import _NativeOwner

SOURCE = Path(__file__).resolve().parent / '_native' / 'ovo'
SCHEDULES = {'exhaustive': 0, 'lazy': 1, 'knockout_cert': 2,
             'beretta_array': 3, 'beretta_ordered': 4, 'beretta_cert': 5}
MAX_ROWS = 65536


def build_runtime(out: str | Path, *, target: str = 'portable', compiler: str = 'g++') -> Path:
    """Build locally with strict arithmetic; require a fresh output directory.

    Linux is the tested platform. AVX2 is an explicit hardware-specific opt-in.
    A failed build retains its log; no binary fallback is substituted.
    """
    if sys.platform != 'linux' or target not in ('portable', 'avx2'):
        raise ValueError('supported build: Linux, target portable or avx2')
    if target == 'avx2' and platform.machine().lower() not in ('x86_64', 'amd64'):
        raise ValueError('AVX2 requires x86-64')
    folder = Path(out).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    library = folder / 'libspectra_svm.so'
    entry = SOURCE / 'runtime.cpp'
    command = [compiler, '-std=c++17', '-O3', '-fno-fast-math', '-ffp-contract=off',
               '-fPIC', '-shared']
    if target == 'avx2':
        command.append('-mavx2')
    command += [str(entry), '-o', str(library)]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    receipt = {'command': command, 'returncode': process.returncode,
               'stdout': process.stdout, 'stderr': process.stderr, 'target': target,
               'source_sha256': {p.relative_to(SOURCE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(SOURCE.rglob('*')) if p.suffix in ('.cpp', '.hpp')}}
    if process.returncode == 0:
        receipt['library_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
    (folder / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    if process.returncode:
        raise RuntimeError(f'native build failed; see {folder / "build.json"}')
    return library


def verify_certificate(classes: int, winner: int, outcomes: Iterable[int]) -> bool:
    """Check an untrusted partial tournament with first-index tie handling.

    This checks the discrete witness only. Kernel evaluation, model identity and
    the truth of individual supplied pair outcomes are separate obligations.
    """
    if type(classes) is not int or not 2 <= classes <= 128:
        return False
    if type(winner) is not int or not 0 <= winner < classes:
        return False
    entries = list(outcomes)
    if len(entries) != classes * (classes - 1) // 2:
        return False
    wins = [0] * classes
    remaining = [classes - 1] * classes
    k = 0
    for i in range(classes):
        for j in range(i + 1, classes):
            value = entries[k]
            k += 1
            if type(value) is not int or value not in (-1, i, j):
                return False
            if value != -1:
                wins[value] += 1
                remaining[i] -= 1
                remaining[j] -= 1
    return all(wins[winner] > wins[j] + remaining[j] or
               (winner < j and wins[winner] == wins[j] + remaining[j])
               for j in range(classes) if j != winner)


@dataclass(frozen=True)
class Prediction:
    class_index: int
    pair_outcomes: tuple[int, ...]
    evaluated_pairs: int
    evaluated_kernels: int


def _input(values: Iterable[float]) -> array:
    row = list(values)
    if len(row) != 16:
        raise ValueError('exactly sixteen features required')
    try:
        if any(isinstance(x, (str, bytes, bool)) or not math.isfinite(x) for x in row):
            raise ValueError('numeric finite features required')
        packed = array('f', row)
    except (TypeError, OverflowError) as error:
        raise ValueError('finite binary32 features required') from error
    if any(not math.isfinite(x) for x in packed):
        raise ValueError('feature outside finite binary32 range')
    return packed


class Session(_NativeOwner):
    """Owning session; calls and close are serialized with one lock.

    ctypes releases the GIL during native calls. Do not remove this lock: the
    kernel cache and certificate are mutable. Use separate sessions for parallel
    execution. The caller must trust the supplied native library.
    """
    def __init__(self, model: str | Path, library: str | Path, *, tables: bool = False):
        self._init_lifetime('session')
        if type(tables) is not bool:
            raise ValueError('tables must be bool')
        path = os.fsencode(model)
        if b'\0' in path:
            raise ValueError('NUL in model path')
        self._lib = C.CDLL(str(Path(library).resolve(strict=True)))
        lib = self._lib
        lib.sp_svm_abi.restype = C.c_int
        if lib.sp_svm_abi() != 1:
            raise ValueError('unsupported SPECTRA SVM ABI')
        lib.et_error.restype = C.c_char_p
        lib.et_create.argtypes = [C.c_char_p, C.c_int]
        lib.et_create.restype = C.c_void_p
        lib.et_destroy.argtypes = [C.c_void_p]
        lib.et_destroy.restype = None
        lib.et_info.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.c_int]
        lib.sp_svm_single.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int, C.c_int, C.c_int,
                              C.POINTER(C.c_int), C.POINTER(C.c_uint64), C.c_int]
        lib.sp_svm_batch.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int, C.c_int,
                                      C.c_int, C.c_int, C.POINTER(C.c_int), C.c_int]
        lib.et_certificate.argtypes = [C.c_void_p, C.POINTER(C.c_int8), C.c_int]
        pointer = lib.et_create(path, int(tables))
        if not pointer:
            raise ValueError(lib.et_error().decode('utf-8', errors='replace'))
        self._adopt(pointer, lib, 'et_destroy')
        try:
            raw = (C.c_uint64 * 8)()
            with self._operation() as handle:
                self._check(lib.et_info(handle, raw, 8))
            self._info = dict(zip(('classes', 'supports', 'tables_enabled', 'dictionary_values',
                                   'prepared_bytes', 'scratch_bytes', 'code_bytes', 'value_bytes'), raw))
        except BaseException:
            self.close()
            raise

    @property
    def info(self) -> dict[str, int]:
        return dict(self._info)

    def _check(self, status: int) -> None:
        if status:
            raise ValueError(self._lib.et_error().decode('utf-8', errors='replace'))

    def _settings(self, schedule: str, hint: int) -> int:
        if not self._handle:
            raise ValueError('closed session')
        if type(schedule) is not str or schedule not in SCHEDULES:
            raise ValueError('invalid schedule')
        if type(hint) is not int or not -1 <= hint < self._info['classes']:
            raise ValueError('invalid class hint')
        return SCHEDULES[schedule]

    def _run(self, packed: array, schedule: str, hint: int, certificate: bool):
        with self._operation() as handle:
            mode = self._settings(schedule, hint)
            data = (C.c_float * 16).from_buffer(packed)
            out, stats = C.c_int(), (C.c_uint64 * 10)()
            self._check(self._lib.sp_svm_single(handle, data, 16, mode, hint,
                                         C.byref(out), stats, 10))
            if not certificate:
                return out.value
            classes = self._info['classes']
            trace = (C.c_int8 * (classes * (classes - 1) // 2))()
            self._check(self._lib.et_certificate(handle, trace, len(trace)))
            known = tuple(trace)
            if not verify_certificate(classes, out.value, known):
                raise RuntimeError('native vote certificate failed independent check')
            return Prediction(out.value, known, int(stats[1]), int(stats[0]))

    def predict(self, values: Iterable[float], *, schedule: str = 'beretta_cert', hint: int = -1) -> int:
        return self._run(_input(values), schedule, hint, False)

    def predict_with_certificate(self, values: Iterable[float], *, schedule: str = 'beretta_cert', hint: int = -1) -> Prediction:
        return self._run(_input(values), schedule, hint, True)

    def predict_many(self, rows: Iterable[Iterable[float]], *, schedule: str = 'beretta_cert', hint: int = -1) -> list[int]:
        """One native call, fresh model state per row; at most 65,536 rows.

        Input conversion and output materialization are included in this API.
        No cross-row kernel reuse, matrix batching or new threading is implied.
        """
        packed = array('f')
        count = 0
        for row in rows:
            if count == MAX_ROWS:
                raise ValueError('batch exceeds 65,536 rows')
            packed.extend(_input(row))
            count += 1
        with self._operation() as handle:
            mode = self._settings(schedule, hint)
            output = (C.c_int * count)()
            data = (C.c_float * len(packed)).from_buffer(packed) if count else None
            self._check(self._lib.sp_svm_batch(handle, data, count, 16,
                                               mode, hint, output, count))
            return list(output)
