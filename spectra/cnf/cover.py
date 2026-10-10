"""Opt-in native cover/exclusion solver, with original-CNF witness checking.

ChoiceProblem inputs retain covering/exclusion groups without pairwise expansion.
Supported normalized clauses are positive clauses and negative units/pairs.
Tautologies and literal repetitions preserve their original semantics. Other
clauses raise UnsupportedStructure. No implicit compiler invocation, fallback,
UNSAT certificate, neural model, or hard execution-time/memory guarantee.
"""
from __future__ import annotations
from array import array
from dataclasses import asdict, dataclass, replace
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
import time

from data.cnf import CNF

SOURCE = Path(__file__).resolve().parents[1] / '_native' / 'cover_search.cpp'
DEFAULT_INDEX_BYTES = 64 * 1024 * 1024
DEFAULT_STATE_BYTES = 64 * 1024 * 1024

class UnsupportedStructure(ValueError):
    """The input is not in the declared cover/exclusion clause class."""


def _integer(value: int, name: str, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f'{name} must be a nonnegative integer <= {maximum}')
    return value


def build_cover_runtime(directory: str | Path, *, compiler: str = 'g++',
                        sanitize: bool = False) -> Path:
    """Explicit Linux shared-library build; never overwrite an existing build.

    Only Linux is admitted by this builder. An available C++17 compiler is
    required here, not for importing or installing SPECTRA. The output library
    is trusted executable code, not an untrusted data file.
    """
    if platform.system() != 'Linux':
        raise NotImplementedError('cover runtime builder is currently tested only on Linux')
    if type(sanitize) is not bool:
        raise TypeError('sanitize must be bool')
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError(f'C++ compiler not found: {compiler}')
    dest = Path(directory).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    library, receipt = dest / 'spectra_cover.so', dest / 'build.json'
    if library.exists() or receipt.exists():
        raise FileExistsError('cover build destination is already occupied')
    flags = ['-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
             '-fno-fast-math', '-ffp-contract=off']
    flags += (['-O1', '-g', '-fsanitize=undefined', '-fno-sanitize-recover=all']
              if sanitize else ['-O3', '-DNDEBUG'])
    start = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix='.cover-', dir=dest) as temp:
        target = Path(temp) / library.name
        command = [executable, *flags, str(SOURCE), '-o', str(target)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if completed.returncode:
            raise RuntimeError(f'cover compilation failed:\n{completed.stdout}\n{completed.stderr}')
        metadata = {'schema': 'spectra.cover.build.v1', 'command': command,
                    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    'library_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                    'compiler': subprocess.run([executable, '--version'], check=True,
                        capture_output=True, text=True, timeout=10).stdout,
                    'platform': platform.platform(), 'sanitize': sanitize,
                    'elapsed_ns': time.perf_counter_ns()-start,
                    'stdout': completed.stdout, 'stderr': completed.stderr}
        # Create-if-absent publication. The build directory must be trusted.
        os.link(target, library)
        with receipt.open('x') as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
    return library


@dataclass(frozen=True)
class ChoiceProblem:
    """One-based Boolean variables: each cover needs >=1, each exclusive <=1.

    Groups contain distinct variables. Empty covers are contradictory; empty
    exclusive groups are vacuous. Direct group input avoids materializing every
    pairwise CNF exclusion. This is exact constraint encoding, not learning.
    """
    nvars: int
    covers: tuple[tuple[int, ...], ...]
    exclusive: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        _integer(self.nvars, 'nvars', 65536)
        for groups in (self.covers, self.exclusive):
            if type(groups) is not tuple:
                raise TypeError('constraint collections must be immutable tuples')
            for group in groups:
                if type(group) is not tuple:
                    raise TypeError('groups must be immutable tuples')
                if any(type(v) is not int or not 1 <= v <= self.nvars for v in group):
                    raise ValueError('group variable outside declared range')
                if len(group)!=len(set(group)):
                    raise ValueError('group variables must be distinct')

    def violated(self, witness: tuple[bool, ...]) -> tuple[int, ...]:
        if (type(witness) is not tuple or len(witness)!=self.nvars
                or any(type(v) is not bool for v in witness)):
            raise ValueError('complete immutable Boolean witness required')
        violations=[i for i,g in enumerate(self.covers) if not any(witness[v-1] for v in g)]
        # Independent direct constraint checker; does not read the native index.
        for i,group in enumerate(self.exclusive,start=len(self.covers)):
            seen=False
            for v in group:
                if witness[v-1]:
                    if seen:
                        violations.append(i)
                        break
                    seen=True
        return tuple(violations)


@dataclass(frozen=True)
class CoverResult:
    status: str
    witness: tuple[bool, ...]
    unsatisfied: tuple[int, ...]
    reason: str
    nodes: int
    decisions: int
    propagations: int
    backtracks: int
    max_nodes: int
    max_state_bytes: int
    index_payload_bytes: int
    state_word_bytes_peak: int
    trace_fingerprint: str
    elapsed_ns: int

    def record(self) -> dict:
        return {**asdict(self), 'witness': list(self.witness),
                'unsatisfied': list(self.unsatisfied),
                'schema': 'spectra.cnf.cover.v1', 'algorithm': 'native_cover_exclusion',
                'learned': False}


class PreparedCover:
    """Immutable native index and serialized, independently owned search states.

    Reuse excludes preparation from solve.elapsed_ns. close/solve share a lock,
    so a concurrent close cannot release a live index. max_index_bytes bounds
    the admitted bit matrices, NOT Python input storage, temporary buffers,
    allocator overhead, search stack, process RSS or the caller's whole job.
    """
    def __init__(self, problem: CNF | ChoiceProblem, library: str | Path, *,
                 max_index_bytes: int = DEFAULT_INDEX_BYTES, incremental: bool = True):
        if type(problem) not in (CNF, ChoiceProblem):
            raise TypeError('CNF or ChoiceProblem required')
        direct=isinstance(problem, ChoiceProblem)
        clauses=problem.covers+problem.exclusive if direct else problem.clauses
        _integer(max_index_bytes, 'max_index_bytes')
        if type(incremental) is not bool:
            raise TypeError('incremental must be bool')
        if problem.nvars > 65536 or len(clauses) > 1000000:
            raise MemoryError('cover input exceeds geometry limits')
        nl = sum(map(len, clauses))
        if nl > 16000000:
            raise MemoryError('cover input exceeds literal limit')
        self._problem = problem
        self._nvars = problem.nvars
        self._lock = threading.RLock()
        self._handle = C.c_void_p()
        self._library = C.CDLL(str(Path(library).resolve()))
        lib = self._library
        lib.spectra_cover_abi.argtypes = []
        lib.spectra_cover_abi.restype = C.c_uint32
        if lib.spectra_cover_abi() != 3:
            raise ValueError('unsupported cover runtime ABI')
        lib.spectra_cover_create.argtypes = [C.c_uint32, C.c_uint64,
            C.POINTER(C.c_uint64), C.POINTER(C.c_int32), C.c_uint64, C.c_uint64, C.c_uint32,
            C.POINTER(C.c_void_p), C.c_char_p, C.c_size_t]
        lib.spectra_cover_create.restype = C.c_int
        lib.spectra_choices_create.argtypes = [C.c_uint32,C.c_uint64,C.c_uint64,
            C.POINTER(C.c_uint64),C.POINTER(C.c_int32),C.c_uint64,C.c_uint64,C.c_uint32,
            C.POINTER(C.c_void_p),C.c_char_p,C.c_size_t]
        lib.spectra_choices_create.restype = C.c_int
        lib.spectra_cover_solve.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64,
            C.POINTER(C.c_uint8), C.POINTER(C.c_uint64), C.c_char_p, C.c_size_t]
        lib.spectra_cover_solve.restype = C.c_int
        lib.spectra_cover_destroy.argtypes = [C.c_void_p]
        lib.spectra_cover_destroy.restype = None
        offsets, literals = array('Q', [0]), array('i')
        if offsets.itemsize != 8 or literals.itemsize != 4:
            raise RuntimeError('unsupported native integer widths')
        for clause in clauses:
            literals.extend(clause)
            offsets.append(len(literals))
        offbuf = (C.c_uint64 * len(offsets)).from_buffer(offsets)
        litbuf = (C.c_int32 * len(literals)).from_buffer(literals)
        err = C.create_string_buffer(512)
        args=(offbuf,litbuf,len(literals),max_index_bytes,int(incremental),C.byref(self._handle),err,len(err))
        code=(lib.spectra_choices_create(problem.nvars,len(clauses),len(problem.covers),*args)
              if direct else lib.spectra_cover_create(problem.nvars,len(clauses),*args))
        self._check(code, err)

    @property
    def problem(self):
        return self._problem

    @staticmethod
    def _check(code, err):
        if not code:
            return
        message = err.value.decode('utf-8', errors='replace')
        if code == 2:
            raise UnsupportedStructure(message)
        if code == 3:
            raise MemoryError(message)
        raise ValueError(message)

    def solve(self, *, max_nodes: int = 100000,
              max_state_bytes: int = DEFAULT_STATE_BYTES) -> CoverResult:
        _integer(max_nodes, 'max_nodes')
        _integer(max_state_bytes, 'max_state_bytes')
        start = time.perf_counter_ns()
        with self._lock:
            if not self._handle.value:
                raise RuntimeError('cover index is closed')
            output = (C.c_uint8 * self._nvars)()
            stats = (C.c_uint64 * 8)()
            err = C.create_string_buffer(512)
            code = self._library.spectra_cover_solve(self._handle, max_nodes, max_state_bytes, output,
                                                     stats, err, len(err))
            self._check(code, err)
            if any(value not in (0, 1) for value in output):
                raise AssertionError('native cover returned a non-Boolean witness')
            witness = tuple(bool(value) for value in output)
            unsatisfied = self.problem.violated(witness)
            reason = ('satisfied', 'budget', 'exhausted', 'state_budget')[stats[4]]
            if reason == 'satisfied' and unsatisfied:
                raise AssertionError('native cover witness fails original signed clauses')
            if stats[0] > max_nodes:
                raise AssertionError('native cover exceeded node cap')
            values = tuple(stats)
            del output, stats, err
        return CoverResult('UNKNOWN' if unsatisfied else 'SAT_VERIFIED', witness,
            unsatisfied, reason, *values[:4], max_nodes, max_state_bytes, values[5], values[6],
            f'{values[7]:016x}', time.perf_counter_ns()-start)

    def close(self) -> None:
        with self._lock:
            if self._handle.value:
                self._library.spectra_cover_destroy(self._handle)
                self._handle.value = None

    def __enter__(self):
        with self._lock:
            if not self._handle.value:
                raise RuntimeError('cover index is closed')
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except (AttributeError, RuntimeError):
            pass


def solve_cover(problem: CNF | ChoiceProblem, library: str | Path, *, max_nodes: int = 100000,
                max_index_bytes: int = DEFAULT_INDEX_BYTES,
                max_state_bytes: int = DEFAULT_STATE_BYTES,
                incremental: bool = True) -> CoverResult:
    """Cold parsed-CNF call: preparation, search, original checks and disposal.

    Imports, native build and input parsing are excluded. Use an external worker
    deadline to bound a whole job. The default SPECTRA solver is not changed.
    """
    _integer(max_nodes, 'max_nodes')
    _integer(max_state_bytes, 'max_state_bytes')
    start = time.perf_counter_ns()
    with PreparedCover(problem, library, max_index_bytes=max_index_bytes, incremental=incremental) as index:
        result = index.solve(max_nodes=max_nodes, max_state_bytes=max_state_bytes)
    return replace(result, elapsed_ns=time.perf_counter_ns()-start)
