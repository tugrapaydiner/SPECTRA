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
import struct
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from typing import Iterable, Sequence

from .binary_list import BinaryListProblem

SOURCE = Path(__file__).with_name('exact_native.cpp')
DEFAULT_BYTES = 512 << 20
MAX_BYTES = (1 << 64) - 1
QUERY_BANK_HEADER = struct.Struct('<4sIII')
QUERY_BANK_MAGIC = b'SBQ1'
SESSION_PACKET_HEADER = struct.Struct('<4s7I')
SESSION_PACKET_MAGIC = b'SBC1'


def build_exact_runtime(directory: Path, *, compiler: str | None = None,
                        sanitizer: str = 'none') -> Path:
    """Build on Linux or Windows x64; bind source, flags, Python ABI and binary.

    Windows needs a configured MSVC x64 developer prompt and CPython import
    libraries. Sanitizer builds remain Linux-only. No compiler is installed here.
    """
    windows = sys.platform == 'win32'
    if sys.platform not in ('linux', 'win32'):
        raise NotImplementedError('native builder supports Linux and Windows x64')
    if sys.implementation.name != 'cpython' or sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise ValueError('GIL-enabled CPython is required')
    if sanitizer not in ('none', 'undefined', 'address,undefined'):
        raise ValueError('unsupported sanitizer')
    if windows and sanitizer != 'none':
        raise ValueError('sanitizer builds currently require Linux')
    compiler = compiler or ('cl' if windows else 'g++')
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
        if windows:
            from spectra.svm_build import command_for
            if sysconfig.get_config_var('Py_DEBUG'):
                raise ValueError('Windows debug Python is not supported')
            name = f'python{sys.version_info.major}{sys.version_info.minor}.lib'
            import_library = Path(sys.base_prefix) / 'libs' / name
            if not import_library.is_file():
                raise ValueError('CPython import library missing: ' + str(import_library))
            command = command_for(SOURCE.resolve(), target, target='portable', compiler=cc,
                include=Path(sysconfig.get_path('include')), import_library=import_library)
        else:
            command = [cc, *flags, '-I' + sysconfig.get_path('include'), str(SOURCE), '-o', str(target)]
        result = subprocess.run(command, cwd=tmp, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=120)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        version = subprocess.run([cc] if windows else [cc, '--version'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', check=not windows)
        record = dict(python_executable=sys.executable, python_version=sys.version,
                      python_implementation=platform.python_implementation(),
                      soabi=sysconfig.get_config_var('SOABI'), platform=platform.platform(),
                      source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      library_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                      command=command, sanitizer=sanitizer,
                      elapsed_ns=time.perf_counter_ns()-start,
                      compiler=version.stdout + version.stderr,
                      stdout=result.stdout, stderr=result.stderr)
        os.link(target, library)
        try:
            with receipt.open('x') as stream:
                json.dump(record, stream, sort_keys=True, indent=2)
                stream.write('\n')
        except BaseException:
            # Never leave a newly published library with a missing/partial receipt.
            library.unlink(missing_ok=True)
            receipt.unlink(missing_ok=True)
            raise
    return library


@dataclass(frozen=True)
class ExactAnswer:
    status: str
    labels: bytes
    proof: bytes
    visited: int
    arcs_scanned: int


@dataclass(frozen=True)
class ExactBankAnswer:
    """One already-serialized, independently checked SCQ2 answer bank."""

    output: bytes
    query_count: int
    sat_count: int
    unsat_count: int
    proof_bytes: int
    visited: int
    arcs_scanned: int


@dataclass(frozen=True)
class ExactSessionPacketAnswer(ExactBankAnswer):
    """One-shot prepared-input result including native construction metadata."""

    info: dict


def _checked_query_rows(n: int, rows: Iterable[tuple[tuple[int, int], ...]]
                        ) -> tuple[tuple[tuple[int, int], ...], ...]:
    if type(n) is not int or not 0 <= n <= 1_000_000:
        raise ValueError('n must be an exact bounded nonnegative integer')
    checked = []
    for row in rows:
        if type(row) is not tuple:
            raise ValueError('frozen query rows must be exact tuples')
        copied = []
        for pin in row:
            if type(pin) is not tuple or len(pin) != 2:
                raise ValueError('frozen query pins must be exact pairs')
            vertex, allowed = pin
            if (type(vertex) is not int or not 0 <= vertex < n or
                    type(allowed) is not int or not 0 <= allowed < 1 << 64):
                raise ValueError('invalid frozen query pin')
            copied.append((vertex, allowed))
        checked.append(tuple(copied))
    return tuple(checked)


def pack_query_bank(problem: BinaryListProblem,
                    queries: Iterable[Iterable[Sequence[int]]]) -> bytes:
    """Freeze and serialize current requests without consulting any solver.

    ``SBQ1`` is problem-independent except for original-vertex bounds. The native
    solver and observer each parse and validate the immutable byte string
    independently. Packing is real endpoint work and must be charged when an
    ordinary Python request bank is the advertised input.
    """
    p = BinaryListProblem.create(problem.n, problem.palette,
                                 problem.edges, problem.masks)
    rows = tuple(p.freeze_query(query) for query in queries)
    return pack_frozen_query_bank(p.n, rows)


def pack_frozen_query_bank(n: int,
                           rows: Iterable[tuple[tuple[int, int], ...]]) -> bytes:
    """Serialize already-frozen rows while independently rechecking pin syntax.

    This avoids rebuilding and sorting the fixed graph after :class:`Case` has
    already admitted it. It does not trust query contents: exact tuple geometry,
    integer types, uint64 masks, and original-vertex bounds are checked again.
    """
    rows = _checked_query_rows(n, rows)
    if len(rows) > 1_000_000:
        raise ValueError('query bank exceeds one million rows')
    pins = sum(map(len, rows))
    if pins > 1_000_000:
        raise ValueError('query bank exceeds one million pins')
    size = QUERY_BANK_HEADER.size + 4 * (len(rows) + 1) + 12 * pins
    payload = bytearray(size)
    QUERY_BANK_HEADER.pack_into(payload, 0, QUERY_BANK_MAGIC, len(rows), pins, 0)
    offset_at = QUERY_BANK_HEADER.size
    vertex_at = offset_at + 4 * (len(rows) + 1)
    mask_at = vertex_at + 4 * pins
    cursor = 0
    struct.pack_into('<I', payload, offset_at, 0)
    for index, row in enumerate(rows):
        for vertex, allowed in row:
            struct.pack_into('<I', payload, vertex_at + 4 * cursor, vertex)
            struct.pack_into('<Q', payload, mask_at + 8 * cursor, allowed)
            cursor += 1
        struct.pack_into('<I', payload, offset_at + 4 * (index + 1), cursor)
    if cursor != pins:
        raise AssertionError('query bank pin accounting differs')
    return bytes(payload)


def pack_session_packet(problem: BinaryListProblem,
                        queries: Iterable[Iterable[Sequence[int]]]) -> bytes:
    """Validate ordinary inputs and create one immutable prepared session packet."""
    p = BinaryListProblem.create(problem.n, problem.palette,
                                 problem.edges, problem.masks)
    rows = tuple(p.freeze_query(query) for query in queries)
    return pack_validated_session_packet(p, rows)


def pack_validated_session_packet(problem: BinaryListProblem,
                                  rows: Iterable[tuple[tuple[int, int], ...]]) -> bytes:
    """Serialize a previously admitted problem and frozen request bank.

    The graph is still scanned for exact types, sorted uniqueness and bounds, but
    it is not copied and re-sorted. This function is intended for study-freeze or
    deployment preparation; its cost is outside a ready-packet endpoint and must
    be reported separately.
    """
    n, palette_value = problem.n, problem.palette
    if type(n) is not int or not 0 <= n <= 100_000:
        raise ValueError('session packet n outside native bound')
    if type(palette_value) is not int or not 1 <= palette_value <= 64:
        raise ValueError('session packet palette outside [1, 64]')
    if type(problem.masks) is not tuple or len(problem.masks) != n:
        raise ValueError('session packet masks differ')
    palette_mask = (1 << palette_value) - 1 if palette_value < 64 else (1 << 64) - 1
    for mask in problem.masks:
        if (type(mask) is not int or mask <= 0 or mask & ~palette_mask or
                mask.bit_count() not in (1, 2)):
            raise ValueError('session packet original domain differs')
    if type(problem.edges) is not tuple or len(problem.edges) > 2_000_000:
        raise ValueError('session packet edge bank differs')
    previous = None
    for edge in problem.edges:
        if (type(edge) is not tuple or len(edge) != 2 or
                any(type(v) is not int for v in edge) or
                not 0 <= edge[0] < edge[1] < n or
                (previous is not None and edge <= previous)):
            raise ValueError('session packet edges must be sorted unique pairs')
        previous = edge
    rows = _checked_query_rows(n, rows)
    pins = sum(map(len, rows))
    if len(rows) > 1_000_000 or pins > 1_000_000:
        raise ValueError('session packet query bank exceeds supported bound')
    edge_count = len(problem.edges)
    size = (SESSION_PACKET_HEADER.size + 8 * n + 8 * edge_count +
            4 * (len(rows) + 1) + 12 * pins)
    payload = bytearray(size)
    SESSION_PACKET_HEADER.pack_into(payload, 0, SESSION_PACKET_MAGIC, n,
                                    palette_value, edge_count, len(rows), pins, 0, 0)
    masks_at = SESSION_PACKET_HEADER.size
    edges_at = masks_at + 8 * n
    offsets_at = edges_at + 8 * edge_count
    vertices_at = offsets_at + 4 * (len(rows) + 1)
    allowed_at = vertices_at + 4 * pins
    for index, mask in enumerate(problem.masks):
        struct.pack_into('<Q', payload, masks_at + 8 * index, mask)
    for index, (left, right) in enumerate(problem.edges):
        struct.pack_into('<II', payload, edges_at + 8 * index, left, right)
    cursor = 0
    struct.pack_into('<I', payload, offsets_at, 0)
    for row_index, row in enumerate(rows):
        for vertex, allowed in row:
            struct.pack_into('<I', payload, vertices_at + 4 * cursor, vertex)
            struct.pack_into('<Q', payload, allowed_at + 8 * cursor, allowed)
            cursor += 1
        struct.pack_into('<I', payload, offsets_at + 4 * (row_index + 1), cursor)
    if cursor != pins:
        raise AssertionError('session packet pin accounting differs')
    return bytes(payload)


def unpack_session_packet(payload: bytes) -> tuple[BinaryListProblem,
                                                    tuple[tuple[tuple[int, int], ...], ...]]:
    """Strict, slow Python reference decoder for tests and artifact inspection."""
    if type(payload) is not bytes or len(payload) < SESSION_PACKET_HEADER.size + 4:
        raise ValueError('invalid session packet')
    magic, n, palette_value, edge_count, count, pins, reserved0, reserved1 = \
        SESSION_PACKET_HEADER.unpack_from(payload)
    expected = (SESSION_PACKET_HEADER.size + 8 * n + 8 * edge_count +
                4 * (count + 1) + 12 * pins)
    if (magic != SESSION_PACKET_MAGIC or reserved0 or reserved1 or expected != len(payload)
            or n > 100_000 or edge_count > 2_000_000 or count > 1_000_000 or pins > 1_000_000):
        raise ValueError('session packet header or geometry differs')
    masks_at = SESSION_PACKET_HEADER.size
    edges_at = masks_at + 8 * n
    offsets_at = edges_at + 8 * edge_count
    vertices_at = offsets_at + 4 * (count + 1)
    allowed_at = vertices_at + 4 * pins
    masks = struct.unpack_from('<' + 'Q' * n, payload, masks_at) if n else ()
    edges = tuple(struct.unpack_from('<II', payload, edges_at + 8 * i)
                  for i in range(edge_count))
    # Canonical edge order is part of SBC1, not an opportunity to repair bytes.
    if any(not 0 <= u < v < n for u, v in edges) or any(
            left >= right for left, right in zip(edges, edges[1:])):
        raise ValueError('session packet edges must be sorted unique pairs')
    problem = BinaryListProblem.create(n, palette_value, edges, masks)
    offsets = struct.unpack_from('<' + 'I' * (count + 1), payload, offsets_at)
    if offsets[0] != 0 or offsets[-1] != pins or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError('session packet offsets differ')
    rows = []
    for begin, end in zip(offsets, offsets[1:]):
        row = tuple((struct.unpack_from('<I', payload, vertices_at + 4 * i)[0],
                     struct.unpack_from('<Q', payload, allowed_at + 8 * i)[0])
                    for i in range(begin, end))
        rows.append(problem.freeze_query(row))
    return problem, tuple(rows)


def unpack_query_bank(payload: bytes, n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Strict Python reference decoder used by tests and artifact inspection."""
    if type(payload) is not bytes or type(n) is not int or n < 0:
        raise ValueError('query bank requires immutable bytes and nonnegative n')
    if len(payload) < QUERY_BANK_HEADER.size + 4:
        raise ValueError('truncated query bank')
    magic, count, pins, reserved = QUERY_BANK_HEADER.unpack_from(payload)
    expected = QUERY_BANK_HEADER.size + 4 * (count + 1) + 12 * pins
    if magic != QUERY_BANK_MAGIC or reserved != 0 or expected != len(payload):
        raise ValueError('query bank header or length differs')
    if count > 1_000_000 or pins > 1_000_000:
        raise ValueError('query bank count exceeds supported bound')
    offset_at = QUERY_BANK_HEADER.size
    vertex_at = offset_at + 4 * (count + 1)
    mask_at = vertex_at + 4 * pins
    offsets = struct.unpack_from('<' + 'I' * (count + 1), payload, offset_at)
    if offsets[0] != 0 or offsets[-1] != pins or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError('query bank offsets differ')
    rows = []
    for begin, end in zip(offsets, offsets[1:]):
        row = []
        for index in range(begin, end):
            vertex = struct.unpack_from('<I', payload, vertex_at + 4 * index)[0]
            allowed = struct.unpack_from('<Q', payload, mask_at + 8 * index)[0]
            if vertex >= n:
                raise ValueError('query bank vertex outside graph')
            row.append((vertex, allowed))
        rows.append(tuple(row))
    return tuple(rows)


class ExactRuntime:
    def __init__(self, library: Path):
        spec = importlib.util.spec_from_file_location('_spectra_exact_binary', Path(library).resolve())
        if spec is None or spec.loader is None:
            raise ValueError('invalid trusted native extension path')
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        if self._module.abi() != 3:
            raise ValueError('unsupported exact binary ABI')

    def prepare(self, problem: BinaryListProblem, *, mode: str = 'scc',
                max_bytes: int = DEFAULT_BYTES) -> 'ExactSession':
        return ExactSession(self._module, problem, mode=mode, max_bytes=max_bytes)

    def solve_session_packet(self, packet: bytes, *, mode: str = 'auto',
                             max_bytes: int = DEFAULT_BYTES) -> ExactSessionPacketAnswer:
        """Construct, solve, verify, serialize and dispose one prepared SBC1 session.

        The default ``auto`` mode uses structural policy v5. It keeps a
        base-infeasible graph on raw traversal. A request bank also stays raw
        when it averages at least eight pins per request and those pins cover at
        least one original vertex in 24, or when an uncompressed graph with at
        least four original edges per vertex has at most 128 requests. Feasible
        banks of at least 64 requests otherwise use SCC; banks from eight
        requests use SCC earlier when SCCs compress the literal graph by at least
        fourfold. Selection uses immutable graph/request geometry only, never
        answers or measured timings. ``info['selected_mode']`` and
        ``info['auto_policy_reason']`` expose the decision. Explicit modes keep
        their meaning; the automatic policy is a frozen structural dispatcher.
        """
        if type(packet) is not bytes:
            raise ValueError('session packet must be immutable exact bytes')
        if mode not in ('scc', 'raw', 'sparse', 'route', 'auto'):
            raise ValueError('mode must be scc, raw, sparse, route or auto')
        if type(max_bytes) is not int or not (1 << 20) <= max_bytes <= MAX_BYTES:
            raise ValueError('max_bytes must be an exact uint64 of at least 1 MiB')
        values = self._module.solve_session_packet(
            packet, {'raw': 0, 'scc': 1, 'sparse': 2, 'route': 3, 'auto': 4}[mode], max_bytes)
        output, count, sat, unsat, proof_bytes, visits, arcs, info = values
        info = {**info, 'backend': 'native-exact-binary', 'mode': mode,
                'prepared_input': True, 'input_schema': 'SBC1',
                'output_schema': 'SCQ2', 'portable_unsat_certificate': True,
                'query_answer_cache': False,
                'resource_limit_kind': 'conservative native array admission, not RSS'}
        return ExactSessionPacketAnswer(output, count, sat, unsat, proof_bytes,
                                        visits, arcs, info)


class ExactSession:
    def __init__(self, module, problem: BinaryListProblem, *, mode: str,
                 max_bytes: int):
        if mode not in ('scc', 'raw', 'sparse', 'route'):
            raise ValueError('mode must be scc, raw, sparse or route')
        if type(max_bytes) is not int or not (1 << 20) <= max_bytes <= MAX_BYTES:
            raise ValueError('max_bytes must be an exact uint64 of at least 1 MiB')
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
            if observer_bytes >= max_bytes:
                raise MemoryError('combined native owners exhaust the session cap')
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

    def solve_bank(self, packet: bytes) -> ExactBankAnswer:
        """Solve, independently verify, and serialize a complete SBQ1 bank."""
        if type(packet) is not bytes:
            raise ValueError('query bank must be immutable exact bytes')
        with self._lock:
            self._require_open()
            output, count, sat, unsat, proof_bytes, visits, arcs = self._module.solve_bank(
                self._solver, self._observer, packet)
            if type(output) is not bytes:
                raise AssertionError('native bank endpoint returned mutable output')
            return ExactBankAnswer(output, count, sat, unsat, proof_bytes, visits, arcs)

    def check_bank(self, packet: bytes, output: bytes) -> bool:
        """Reparse and check a serialized answer bank with the native observer."""
        if type(packet) is not bytes or type(output) is not bytes:
            return False
        with self._lock:
            self._require_open()
            return bool(self._module.check_bank(self._observer, packet, output))

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
