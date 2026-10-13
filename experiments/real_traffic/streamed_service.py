"""Opt-in checked migration service with an independent streamed observer.

All query iterables are snapshotted before native execution. The immutable bank is
reused by the solver and checker, but both independently validate all fields.
Every returned batch has passed original feasibility, minimum-change and proof
checking. No certificate or answer cache is supplied to either implementation.
"""
from __future__ import annotations
from pathlib import Path
import time
from experiments.real_traffic.compact_migration import (
    CompactMigrationRuntime, build_compact_migration, DEFAULT_BYTES,
)
from experiments.real_traffic.compact_service import CheckedBatch, CheckedSession
from experiments.real_traffic.streamed_observer import StreamedObserverRuntime, build_streamed_observer

class StreamedService:
    def __init__(self,solver_library,observer_library):
        self.solver=CompactMigrationRuntime(solver_library)
        self.observer=StreamedObserverRuntime(observer_library)
    @classmethod
    def build(cls,directory,*,compiler='g++',sanitize=False):
        root=Path(directory)
        return cls(build_compact_migration(root/'solver',compiler=compiler,sanitize=sanitize),
                   build_streamed_observer(root/'observer',compiler=compiler,sanitize=sanitize))
    def prepare(self,edges,plan_a,plan_b,*,max_bytes=DEFAULT_BYTES):
        edges=tuple(map(tuple,edges));a=tuple(plan_a);b=tuple(plan_b)
        solver=self.solver.prepare(edges,a,b,max_bytes=max_bytes)
        try:
            observer=self.observer.prepare(edges,a,b,max_bytes=max_bytes-solver.info['payload_bytes'])
        except BaseException:
            solver.close();raise
        return StreamedSession(solver,observer)

class StreamedSession(CheckedSession):
    def solve(self,queries,*,mode='auto',check_mode='auto',max_bytes=DEFAULT_BYTES):
        """Return only independently checked output; caps bound arrays, not RSS.

        The explicit `bfs` and `tiled` modes allow matched comparisons. `auto`
        currently chooses tiling at 64 requests, a heuristic, not an oracle.
        """
        begin=time.perf_counter_ns()
        with self._lock:
            if self._closed:raise RuntimeError('checked session is closed')
            bank=self.observer.freeze_queries(queries)
            result=self.solver.solve_frozen(bank,mode=mode,max_bytes=max_bytes)
            published=sum(map(len,(result.statuses,result.labels,result.proof_kinds,result.proof_vertices_le)))
            if published>=max_bytes:raise MemoryError('published buffers exhaust native payload allowance')
            audit=self.observer.verify_frozen(bank,result,minimal=True,mode=check_mode,max_bytes=max_bytes-published)
            return CheckedBatch(result,audit,time.perf_counter_ns()-begin)
