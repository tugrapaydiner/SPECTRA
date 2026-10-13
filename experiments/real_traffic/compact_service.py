"""Checked sparse migration service. Native code is built only by explicit request.

Every returned result has passed original-edge, current-request and minimum-change
checking, or has a verified contradiction path. Caps describe native array payload;
Python objects, allocator overhead and application-held inputs are not RSS-bounded.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from experiments.real_traffic.compact_migration import (CompactMigrationRuntime, CompactBatch,
                                                       build_compact_migration, DEFAULT_BYTES)
from experiments.real_traffic.migration_observer import (MigrationObserverRuntime, OriginalAudit,
                                                        build_migration_observer)

@dataclass(frozen=True)
class CheckedBatch:
    batch: CompactBatch
    audit: OriginalAudit
    elapsed_ns: int

class CompactService:
    def __init__(self, solver_library, observer_library):
        self.solver=CompactMigrationRuntime(solver_library)
        self.observer=MigrationObserverRuntime(observer_library)
    @classmethod
    def build(cls, directory, *, compiler='g++', sanitize=False):
        root=Path(directory)
        solver=build_compact_migration(root/'solver',compiler=compiler,sanitize=sanitize)
        observer=build_migration_observer(root/'observer',compiler=compiler,sanitize=sanitize)
        return cls(solver,observer)
    def prepare(self, edges, plan_a, plan_b, *, max_bytes=DEFAULT_BYTES):
        edges=tuple(map(tuple,edges));a=tuple(plan_a);b=tuple(plan_b)
        solver=self.solver.prepare(edges,a,b,max_bytes=max_bytes)
        try:
            remaining=max_bytes-solver.info['payload_bytes']
            observer=self.observer.prepare(edges,a,b,max_bytes=remaining)
        except BaseException:
            solver.close();raise
        return CheckedSession(solver,observer)

class CheckedSession:
    def __init__(self, solver, observer):
        self.solver=solver;self.observer=observer;self._closed=False;self._lock=threading.RLock()
    @property
    def info(self):
        with self._lock:
            if self._closed:raise RuntimeError('checked session is closed')
            solver=self.solver.info;observer=self.observer.info
            return {'solver':solver,'observer':observer,
                    'combined_index_payload_bytes':solver['payload_bytes']+observer['payload_bytes']}
    def solve(self, queries, *, mode='auto', max_bytes=DEFAULT_BYTES):
        begin=time.perf_counter_ns();bank=tuple(tuple(map(tuple,q)) for q in queries)
        with self._lock:
            if self._closed:raise RuntimeError('checked session is closed')
            result=self.solver.solve_raw(bank,mode=mode,max_bytes=max_bytes)
            published=sum(map(len,(result.statuses,result.labels,result.proof_kinds,result.proof_vertices_le)))
            if published>=max_bytes:raise MemoryError('published buffers exhaust native payload allowance')
            audit=self.observer.verify(bank,result,minimal=True,max_bytes=max_bytes-published)
            return CheckedBatch(result,audit,time.perf_counter_ns()-begin)
    def close(self):
        with self._lock:
            self.solver.close();self.observer.close();self._closed=True
    def __enter__(self):
        with self._lock:
            if self._closed:raise RuntimeError('checked session is closed')
        return self
    def __exit__(self,*args):self.close()
