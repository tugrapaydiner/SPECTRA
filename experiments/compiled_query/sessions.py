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

ARMS = ('scc', 'none', 'minicard', 'cadical195',
        'cudd-natural', 'cudd-bfs', 'cudd-occurrence',
        'cudd-frontier-natural', 'cudd-frontier-bfs', 'cudd-frontier-occurrence',
        'cudd-seeded-bfs', 'cudd-reduced-natural', 'cudd-reduced-bfs',
        'cudd-reduced-occurrence')
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
    if arm not in ARMS:
        raise ValueError('unknown arm')
    if arm.startswith('cudd-'):
        importlib.import_module('dd.cudd')
    elif arm in ('minicard', 'cadical195'):
        importlib.import_module('pysat.solvers')


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
                max_nodes: int = 2_000_000) -> tuple[dict, bytes]:
    if arm not in ARMS:
        raise ValueError('unknown arm')
    start = time.perf_counter_ns()
    case, case_hash = load_case(case_path)
    if type(count) is not int or count != len(case.queries) or count < 1:
        raise ValueError('input must contain exactly the requested nonempty prefix')
    indices = query_order(count, order, seed)
    p = case.problem
    identity = (case.name, case.cluster, case.provenance)
    input_end = time.perf_counter_ns()
    session = None
    try:
        if arm in ('scc', 'none'):
            session = runtime.prepare(p.n, p.palette, p.edges, masks=p.masks,
                                      mode=arm, max_build_bytes=max_bytes)
        elif arm in ('minicard', 'cadical195'):
            session = SATSession(p, runtime, arm, max_bytes)
        else:
            session = BDDSession(p, runtime, arm.removeprefix('cudd-'), max_bytes, max_nodes)
        setup_end = time.perf_counter_ns()
        rows, per_query, statuses = [], [], []
        for index in indices:
            tick = time.perf_counter_ns()
            if arm in ('scc', 'none'):
                answer = session.solve(case.queries[index], max_work=10_000_000,
                                       max_state_bytes=max_bytes, core_first=True)
            else:
                answer = session.solve(case.queries[index])
            per_query.append(time.perf_counter_ns() - tick)
            if answer.status not in STATUS:
                raise AssertionError('unsupported solver decision')
            if answer.status == 'SAT_VERIFIED' and len(answer.labels) != p.n:
                raise AssertionError('partial original-vertex witness')
            if answer.status != 'SAT_VERIFIED' and answer.labels:
                raise AssertionError('non-SAT result supplied a witness')
            rows.append(bytes((STATUS[answer.status],)) +
                        (answer.labels if answer.status == 'SAT_VERIFIED' else bytes(p.n)))
            statuses.append(answer.status)
        queries_end = time.perf_counter_ns()
        output = HEADER.pack(b'SCQ1', p.n, count) + b''.join(rows)
        output_hash = digest(output)
        output_end = time.perf_counter_ns()
        info = session.info
        counts = dict(Counter(statuses))
        diagnostics_end = time.perf_counter_ns()
    finally:
        if session is not None:
            session.close()
    # Release every non-output input/staging owner before stopping the timer.
    del session, rows, statuses, indices, answer, p, case
    end = time.perf_counter_ns()
    record = dict(schema='spectra.compiled_query.session.v1', arm=arm, case=identity[0],
                  cluster=identity[1], provenance=identity[2], case_sha256=case_hash,
                  query_count=count, order=order, order_seed=seed,
                  status='INCOMPLETE' if counts.get('UNKNOWN', 0) else 'COMPLETE',
                  counts=counts, complete_ns=end-start, input_ns=input_end-start,
                  setup_ns=setup_end-input_end, query_loop_ns=queries_end-setup_end,
                  query_ns=sum(per_query), per_query_ns=per_query,
                  materialize_ns=output_end-queries_end,
                  diagnostics_ns=diagnostics_end-output_end, dispose_ns=end-diagnostics_end,
                  output_sha256=output_hash, output_bytes=len(output), solver_info=info,
                  unsat_certificate=False)
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
