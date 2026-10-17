"""Matched complete-cost feasibility endpoints, separate from the frozen study.

Every SAT answer receives exactly one separately owned native original-input
check. UNKNOWN is never relabelled UNSAT. The binary output bank, its SHA-256,
backend diagnostics, and explicit disposal are inside the complete timer.
"""
from __future__ import annotations

from collections import Counter
import importlib
import struct
import time
from pathlib import Path

from .binary_list import BinaryListProblem, CompiledAnswer, CompiledListSession
from .cases import digest, load_case
from .exact import pack_frozen_query_bank

ARMS = ('scc', 'none', 'minicard', 'cadical195',
        'cudd-natural', 'cudd-bfs', 'cudd-occurrence',
        'cudd-frontier-natural', 'cudd-frontier-bfs', 'cudd-frontier-occurrence',
        'cudd-seeded-bfs', 'cudd-reduced-natural', 'cudd-reduced-bfs',
        'cudd-reduced-occurrence')
EXACT_SINGLE_ARMS = ('exact-scc', 'exact-raw', 'exact-sparse', 'exact-route')
EXACT_BANK_ARMS = ('exact-bank-scc', 'exact-bank-raw',
                   'exact-bank-sparse', 'exact-bank-route')
EXACT_PACKET_ARMS = ('exact-packet-scc', 'exact-packet-raw',
                     'exact-packet-sparse', 'exact-packet-route')
EXACT_ARMS = EXACT_SINGLE_ARMS + EXACT_BANK_ARMS + EXACT_PACKET_ARMS
OPTIONAL_ARMS = ('cadical3', 'kissat4-rebuild', 'ortools-cpsat')
ALL_ARMS = ARMS + OPTIONAL_ARMS + EXACT_ARMS
EXTERNAL_LIBRARY_ARMS = {'cadical3': 'cadical3', 'kissat4-rebuild': 'kissat4'}
STATUS = {'UNSAT': 0, 'SAT_VERIFIED': 1, 'UNKNOWN': 2}
HEADER = struct.Struct('<4sIQ')


class SATSession:
    def __init__(self, problem, runtime, name, max_bytes):
        from pysat.solvers import Solver
        self.problem = BinaryListProblem.create(problem.n, problem.palette,
                                               problem.edges, problem.masks)
        self._choices = self.problem.colors()
        self._checker = runtime.checker(problem.n, problem.palette, problem.edges,
                                        problem.masks, max_bytes)
        self._solver = None
        try:
            self._solver = Solver(name=name, bootstrap_with=self.problem.cnf())
        except BaseException:
            self._checker.close()
            raise

    def solve(self, restrictions):
        query = self.problem.freeze_query(restrictions)
        assumptions = self.problem.assumptions(query)
        if assumptions is None or not self._solver.solve(assumptions=list(assumptions)):
            return CompiledAnswer('UNSAT', b'')
        model = self._solver.get_model()
        if model is None:
            raise AssertionError('native SAT solver reported SAT without a model')
        values = [False] * self.problem.n
        for lit in model:
            if abs(lit) <= self.problem.n:
                values[abs(lit) - 1] = lit > 0
        labels = bytes(self._choices[v][int(value)] for v, value in enumerate(values))
        if not self._checker.check(labels, query):
            raise AssertionError('native SAT output fails original-input observer')
        return CompiledAnswer('SAT_VERIFIED', labels)

    @property
    def info(self):
        return dict(backend='pysat', statistics=self._solver.accum_stats(),
                    query_answer_cache=False, unsat_certificate=False)

    def close(self):
        try:
            if self._solver is not None:
                self._solver.delete()
                self._solver = None
        finally:
            self._checker.close()


class BDDSession:
    def __init__(self, problem, runtime, order, max_bytes, max_nodes):
        construction = ('reduced' if order.startswith('reduced-') else
                        'seeded' if order.startswith('seeded-') else
                        'frontier' if order.startswith('frontier-') else 'balanced64')
        order = order.removeprefix('frontier-').removeprefix('seeded-').removeprefix('reduced-')
        self._checker = runtime.checker(problem.n, problem.palette, problem.edges,
                                        problem.masks, max_bytes)
        self._session = None
        try:
            self._session = CompiledListSession(problem, backend='cudd', order=order,
                                                checker=self._checker.check,
                                                max_nodes=max_nodes, memory_bytes=max_bytes,
                                                construction=construction)
        except BaseException:
            self._checker.close()
            raise

    def solve(self, query):
        return self._session.solve(query)

    @property
    def info(self):
        return self._session.info

    def close(self):
        try:
            if self._session is not None:
                self._session.close()
                self._session = None
        finally:
            self._checker.close()


def preload(arm: str) -> None:
    if arm not in ALL_ARMS:
        raise ValueError('unknown arm')
    if arm.startswith('cudd-'):
        importlib.import_module('dd.cudd')
    elif arm in ('minicard', 'cadical195'):
        importlib.import_module('pysat.solvers')
    elif arm == 'ortools-cpsat':
        importlib.import_module('ortools.sat.python.cp_model')


def query_order(count: int, order: str, seed: int) -> tuple[int, ...]:
    import random
    indices = list(range(count))
    if order == 'reverse':
        indices.reverse()
    elif order == 'shuffle':
        random.Random(seed).shuffle(indices)
    elif order != 'forward':
        raise ValueError('unknown query order')
    return tuple(indices)


def run_session(case_path: Path, runtime, arm: str, *, count: int,
                order: str = 'forward', seed: int = 0, max_bytes: int = 512 << 20,
                max_nodes: int = 2_000_000, exact_runtime=None,
                external_libraries: dict[str, Path] | None = None) -> tuple[dict, bytes]:
    if arm not in ALL_ARMS:
        raise ValueError('unknown arm')
    if arm in EXACT_PACKET_ARMS:
        raise ValueError('prepared packet arms require run_prepared_session')
    start = time.perf_counter_ns()
    case, case_hash = load_case(case_path)
    if type(count) is not int or count != len(case.queries) or count < 1:
        raise ValueError('input must contain exactly the requested nonempty prefix')
    indices = query_order(count, order, seed)
    p = case.problem
    identity = (case.name, case.cluster, case.provenance)
    packet = None
    if arm in EXACT_BANK_ARMS:
        packet = pack_frozen_query_bank(p.n, (case.queries[index] for index in indices))
    input_end = time.perf_counter_ns()
    session = None
    try:
        if arm in EXACT_ARMS:
            if exact_runtime is None:
                raise ValueError('exact runtime must be supplied explicitly; no hidden build')
            mode = (arm.removeprefix('exact-bank-') if arm in EXACT_BANK_ARMS
                    else arm.removeprefix('exact-'))
            session = exact_runtime.prepare(p, mode=mode, max_bytes=max_bytes)
        elif arm in ('scc', 'none'):
            session = runtime.prepare(p.n, p.palette, p.edges, masks=p.masks,
                                      mode=arm, max_build_bytes=max_bytes)
        elif arm in ('minicard', 'cadical195'):
            session = SATSession(p, runtime, arm, max_bytes)
        elif arm in ('cadical3', 'kissat4-rebuild', 'ortools-cpsat'):
            from .champion_competitors import (CaDiCaL3Session,
                Kissat4RebuildSession, ORToolsCPSATSession)
            external_libraries = external_libraries or {}
            if arm == 'cadical3':
                library = external_libraries.get('cadical3')
                if library is None:
                    raise ValueError('cadical3 arm requires an explicit pinned shared library')
                session = CaDiCaL3Session(p, runtime, library, max_bytes)
            elif arm == 'kissat4-rebuild':
                library = external_libraries.get('kissat4')
                if library is None:
                    raise ValueError('kissat4 arm requires an explicit pinned shared library')
                session = Kissat4RebuildSession(p, runtime, library, max_bytes)
            else:
                session = ORToolsCPSATSession(p, runtime, max_bytes)
        else:
            session = BDDSession(p, runtime, arm.removeprefix('cudd-'), max_bytes, max_nodes)
        setup_end = time.perf_counter_ns()
        rows, per_query, statuses = [], [], []
        proof_bytes = 0
        timing_granularity = 'query'
        if arm in EXACT_BANK_ARMS:
            bank = session.solve_bank(packet)
            queries_end = time.perf_counter_ns()
            if bank.query_count != count:
                raise AssertionError('native bank count differs')
            output = bank.output
            counts = {key: value for key, value in
                      (('SAT_VERIFIED', bank.sat_count), ('UNSAT', bank.unsat_count)) if value}
            proof_bytes = bank.proof_bytes
            query_ns = queries_end - setup_end
            timing_granularity = 'bank'
        else:
            for index in indices:
                tick = time.perf_counter_ns()
                if arm in ('scc', 'none'):
                    answer = session.solve(case.queries[index], max_work=10_000_000,
                                           max_state_bytes=max_bytes, core_first=True)
                else:
                    answer = session.solve(case.queries[index])
                per_query.append(time.perf_counter_ns() - tick)
                status = 'UNSAT' if answer.status == 'UNSAT_VERIFIED' and arm in EXACT_ARMS else answer.status
                if status not in STATUS:
                    raise AssertionError('unsupported solver decision')
                if answer.status == 'SAT_VERIFIED' and len(answer.labels) != p.n:
                    raise AssertionError('partial original-vertex witness')
                if answer.status != 'SAT_VERIFIED' and answer.labels:
                    raise AssertionError('non-SAT result supplied a witness')
                row = bytes((STATUS[status],)) + (answer.labels if status == 'SAT_VERIFIED' else bytes(p.n))
                if arm in EXACT_ARMS:
                    row += struct.pack('<I', len(answer.proof)) + answer.proof
                    proof_bytes += len(answer.proof)
                rows.append(row)
                statuses.append(status)
            queries_end = time.perf_counter_ns()
            output = HEADER.pack(b'SCQ2' if arm in EXACT_ARMS else b'SCQ1', p.n, count) + b''.join(rows)
            counts = dict(Counter(statuses))
            query_ns = sum(per_query)
        output_hash = digest(output)
        output_end = time.perf_counter_ns()
        info = session.info
        diagnostics_end = time.perf_counter_ns()
    finally:
        if session is not None:
            session.close()
    # Release every non-output input/staging owner before stopping the timer.
    del session, rows, statuses, indices, p, case, packet
    end = time.perf_counter_ns()
    record = dict(schema='spectra.compiled_query.session.v1', arm=arm, case=identity[0],
                  cluster=identity[1], provenance=identity[2], case_sha256=case_hash,
                  query_count=count, order=order, order_seed=seed,
                  status='INCOMPLETE' if counts.get('UNKNOWN', 0) else 'COMPLETE',
                  counts=counts, complete_ns=end-start, input_ns=input_end-start,
                  setup_ns=setup_end-input_end, query_loop_ns=queries_end-setup_end,
                  query_ns=query_ns, per_query_ns=per_query,
                  timing_granularity=timing_granularity,
                  materialize_ns=output_end-queries_end,
                  diagnostics_ns=diagnostics_end-output_end, dispose_ns=end-diagnostics_end,
                  output_sha256=output_hash, output_bytes=len(output), solver_info=info,
                  unsat_certificate=arm in EXACT_ARMS, proof_bytes=proof_bytes,
                  decision_contract='checked-SAT-and-UNSAT' if arm in EXACT_ARMS else 'checked-SAT-only')
    return record, output


def run_prepared_session(packet_path: Path, exact_runtime, arm: str, *, count: int,
                         order: str, seed: int, case_name: str, cluster: str,
                         provenance: str, case_sha256: str, packet_sha256: str,
                         preparation_ns: int | None = None,
                         max_bytes: int = 512 << 20) -> tuple[dict, bytes]:
    """Run a ready SBC1 deployment packet with all native work charged.

    Packet construction belongs to study preparation and is deliberately not
    hidden inside this endpoint. File read, packet hashing, two independent native
    parses, solver/observer construction, every decision/proof check, SCQ2 output,
    output hashing, and native disposal are inside ``complete_ns``.
    """
    if arm not in EXACT_PACKET_ARMS:
        raise ValueError('arm is not a prepared exact packet endpoint')
    if exact_runtime is None:
        raise ValueError('exact runtime must be supplied explicitly; no hidden build')
    if preparation_ns is not None and (type(preparation_ns) is not int or preparation_ns < 0):
        raise ValueError('invalid packet preparation receipt')
    start = time.perf_counter_ns()
    packet = Path(packet_path).read_bytes()
    actual_packet_hash = digest(packet)
    if actual_packet_hash != packet_sha256:
        raise ValueError('prepared SBC1 packet changed')
    input_end = time.perf_counter_ns()
    mode = arm.removeprefix('exact-packet-')
    answer = exact_runtime.solve_session_packet(packet, mode=mode, max_bytes=max_bytes)
    queries_end = time.perf_counter_ns()
    if answer.query_count != count:
        raise AssertionError('prepared packet query count differs')
    output = answer.output
    counts = {key: value for key, value in
              (('SAT_VERIFIED', answer.sat_count), ('UNSAT', answer.unsat_count)) if value}
    proof_bytes = answer.proof_bytes
    packet_bytes = len(packet)
    output_hash = digest(output)
    output_end = time.perf_counter_ns()
    info = answer.info
    diagnostics_end = time.perf_counter_ns()
    del answer, packet
    end = time.perf_counter_ns()
    record = dict(schema='spectra.compiled_query.session.v1', arm=arm,
                  case=case_name, cluster=cluster, provenance=provenance,
                  case_sha256=case_sha256, query_count=count, order=order,
                  order_seed=seed, status='COMPLETE', counts=counts,
                  complete_ns=end-start, input_ns=input_end-start, setup_ns=0,
                  query_loop_ns=queries_end-input_end,
                  query_ns=queries_end-input_end, per_query_ns=[],
                  timing_granularity='session-packet',
                  materialize_ns=output_end-queries_end,
                  diagnostics_ns=diagnostics_end-output_end,
                  dispose_ns=end-diagnostics_end,
                  output_sha256=output_hash, output_bytes=len(output),
                  solver_info=info, unsat_certificate=True,
                  proof_bytes=proof_bytes,
                  decision_contract='checked-SAT-and-UNSAT',
                  prepared_input_sha256=actual_packet_hash,
                  prepared_input_bytes=packet_bytes,
                  packet_preparation_charged=False,
                  packet_preparation_ns=preparation_ns,
                  packet_preparation_stage='immutable study freeze')
    return record, output


def decode_output(payload: bytes, n: int, count: int) -> tuple[tuple[int, bytes], ...]:
    if (len(payload) != HEADER.size + count * (n + 1)
            or HEADER.unpack_from(payload) != (b'SCQ1', n, count)):
        raise ValueError('output header/geometry differs')
    rows = []
    for i in range(count):
        offset = HEADER.size + i * (n + 1)
        status, labels = payload[offset], payload[offset+1:offset+1+n]
        if status not in STATUS.values() or (status != 1 and any(labels)):
            raise ValueError('output decision/nonwitness bytes differ')
        rows.append((status, labels))
    return tuple(rows)


def decode_certified_output(payload: bytes, n: int, count: int) -> tuple[tuple[int, bytes, bytes], ...]:
    """Strict SCQ2 decoder; every UNSAT row must carry an SBP1 proof."""
    from .proof import decode_proof
    if type(n) is not int or n < 0 or type(count) is not int or count < 0:
        raise ValueError('invalid expected geometry')
    if type(payload) is not bytes or len(payload) < HEADER.size:
        raise ValueError('invalid certified output')
    if HEADER.unpack_from(payload) != (b'SCQ2', n, count):
        raise ValueError('certified output header differs')
    rows, offset = [], HEADER.size
    if count > (len(payload) - offset) // (n + 5):
        raise ValueError('truncated certified row bank')
    for _ in range(count):
        if offset + n + 5 > len(payload):
            raise ValueError('truncated certified row')
        code = payload[offset]
        labels = payload[offset+1:offset+1+n]
        length = struct.unpack_from('<I', payload, offset+1+n)[0]
        offset += n + 5
        if offset + length > len(payload):
            raise ValueError('truncated proof bank')
        proof = payload[offset:offset+length]
        offset += length
        if code == 1:
            if proof:
                raise ValueError('SAT row must not carry contradiction evidence')
        elif code == 0:
            if any(labels):
                raise ValueError('UNSAT padding must be zero')
            decode_proof(proof, n)
        else:
            raise ValueError('unknown certified decision')
        rows.append((code, labels, proof))
    if offset != len(payload):
        raise ValueError('unconsumed certified output bytes')
    return tuple(rows)
