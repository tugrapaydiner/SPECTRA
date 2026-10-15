"""Opt-in exact binary-domain service; no fallback, no changed legacy decisions.

Both SAT colourings and UNSAT original-literal paths are checked on every call.
Native payload admission is not a process RSS limit or a real-time guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sysconfig
import tempfile
import threading
import time

from .binary_list import BinaryListProblem

SOURCE = Path(__file__).with_name('exact_native.cpp')
DEFAULT_BYTES = 512 << 20


def build_exact_runtime(directory: Path, *, compiler: str = 'g++',
                        sanitizer: str = 'none') -> Path:
    """Build explicitly; receipts bind source, flags, interpreter, and binary."""
    if platform.system() != 'Linux':
        raise NotImplementedError('native builder currently validates Linux only')
    if sanitizer not in ('none', 'undefined', 'address,undefined'):
        raise ValueError('unsupported sanitizer')
    cc = shutil.which(compiler)
    if cc is None:
        raise FileNotFoundError(compiler)
    dest = Path(directory).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    library = dest / ('_spectra_exact_binary' + sysconfig.get_config_var('EXT_SUFFIX'))
    receipt = dest / 'build.json'
    if library.exists() or receipt.exists():
        raise FileExistsError('build outputs already exist')
    flags = ['-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
             '-fno-fast-math', '-ffp-contract=off']
    flags += ['-O3', '-DNDEBUG'] if sanitizer == 'none' else [
        '-O1', '-g', '-fno-omit-frame-pointer', '-fsanitize=' + sanitizer,
        '-fno-sanitize-recover=all']
    start = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        target = Path(tmp) / library.name
        command = [cc, *flags, '-I' + sysconfig.get_path('include'), str(SOURCE), '-o', str(target)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr)
        record = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      library_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                      command=command, sanitizer=sanitizer,
                      elapsed_ns=time.perf_counter_ns()-start,
                      compiler=subprocess.run([cc, '--version'], capture_output=True,
                                              text=True, check=True).stdout,
                      stdout=result.stdout, stderr=result.stderr)
        os.link(target, library)
        with receipt.open('x') as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.write('\n')
    return library


@dataclass(frozen=True)
class ExactAnswer:
    status: str
    labels: bytes
    proof: bytes
    visited: int
    arcs_scanned: int


class ExactRuntime:
    def __init__(self, library: Path):
        spec = importlib.util.spec_from_file_location('_spectra_exact_binary', Path(library).resolve())
        if spec is None or spec.loader is None:
            raise ValueError('invalid trusted native extension path')
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        if self._module.abi() != 2:
            raise ValueError('unsupported exact binary ABI')

    def prepare(self, problem: BinaryListProblem, *, mode: str = 'scc',
                max_bytes: int = DEFAULT_BYTES) -> 'ExactSession':
        return ExactSession(self._module, problem, mode=mode, max_bytes=max_bytes)


class ExactSession:
    def __init__(self, module, problem: BinaryListProblem, *, mode: str,
                 max_bytes: int):
        if mode not in ('scc', 'raw', 'sparse', 'route'):
            raise ValueError('mode must be scc, raw, sparse or route')
        if type(max_bytes) is not int or max_bytes < 1 << 20:
            raise ValueError('max_bytes must be an exact integer of at least 1 MiB')
        self.problem = BinaryListProblem.create(problem.n, problem.palette, problem.edges, problem.masks)
        self._module, self.mode = module, mode
        self._lock = threading.RLock()
        self._solver = self._observer = None
        self._max_bytes = max_bytes
        self._owned_bytes = 0
        p = self.problem
        try:
            # The two native owners parse original inputs separately.
            self._observer = module.observer_create(p.n, p.palette, p.edges, p.masks, max_bytes)
            observer_bytes = module.observer_bytes(self._observer)
            self._solver = module.create(p.n, p.palette, p.edges, p.masks,
                                         {'raw': 0, 'scc': 1, 'sparse': 2, 'route': 3}[mode], max_bytes - observer_bytes)
            self._owned_bytes = observer_bytes + module.info(self._solver)['retained_array_bytes']
        except BaseException:
            self.close()
            raise

    def _require_open(self) -> None:
        if self._solver is None or self._observer is None:
            raise RuntimeError('exact binary session is closed')

    def _admit_query(self, query) -> None:
        # Conservatively include both native owners, active scratch and returned
        # proof/label payload. Python input objects and allocator overhead are not
        # bounded by this check; a process cap remains a separate concern.
        bound = self._owned_bytes + 512*(self.problem.n+1) + 32*len(query) + 8192
        if len(query) > 1_000_000 or bound > self._max_bytes:
            raise MemoryError('combined native query payload admission exceeds cap')

    def solve(self, restrictions=()) -> ExactAnswer:
        query = self.problem.freeze_query(restrictions)
        with self._lock:
            self._require_open()
            self._admit_query(query)
            decision, labels, proof, visits, arcs = self._module.solve(self._solver, query)
            if not self._module.verify(self._observer, query, decision, labels, proof):
                raise AssertionError('exact answer failed independent original-input observer')
            return ExactAnswer('SAT_VERIFIED' if decision else 'UNSAT_VERIFIED',
                               labels, proof, visits, arcs)

    def check(self, restrictions, answer: ExactAnswer) -> bool:
        query = self.problem.freeze_query(restrictions)
        with self._lock:
            self._require_open()
            self._admit_query(query)
            if type(answer) is not ExactAnswer or answer.status not in ('SAT_VERIFIED', 'UNSAT_VERIFIED'):
                return False
            return self._module.verify(self._observer, query, int(answer.status == 'SAT_VERIFIED'),
                                       answer.labels, answer.proof)

    @property
    def info(self) -> dict:
        with self._lock:
            self._require_open()
            return {**self._module.info(self._solver), 'backend': 'native-exact-binary',
                    'mode': self.mode, 'query_answer_cache': False,
                    'portable_unsat_certificate': True, 'proof_schema': 'SBP1',
                    'native_observer_array_bytes': self._module.observer_bytes(self._observer),
                    'resource_limit_kind': 'conservative native array admission, not RSS',
                    'exact_scope': 'one/two-colour original domains, fixed conflict graph'}

    def close(self) -> None:
        with self._lock:
            self._solver = self._observer = None

    def __enter__(self):
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, *_args):
        self.close()
