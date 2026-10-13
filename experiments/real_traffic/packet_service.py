"""Checked original-address requests with immutable, portable input packets.

Two native parsers validate the same read-only bytes independently. Inputs contain
only offsets, original vertex IDs and raw uint64 masks: no model, quotient, or
cached answer enters the packet. The original constraints and least migration
closure are checked before an answer is returned.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
from experiments.real_traffic.compact_service import CheckedBatch, CheckedSession
from experiments.real_traffic.packet_migration import (
    PacketMigrationRuntime, build_packet_migration, DEFAULT_BYTES,
)
from experiments.real_traffic.packet_observer import (
    PacketObserverRuntime, build_packet_observer,
)

@dataclass(frozen=True)
class PacketCheckedBatch(CheckedBatch):
    input_packet_bytes: int
    packing_ns: int
    input_form: str

class PacketService:
    def __init__(self,solver_library,observer_library):
        self.solver=PacketMigrationRuntime(solver_library)
        self.observer=PacketObserverRuntime(observer_library)

    @classmethod
    def build(cls,directory,*,compiler='g++',sanitize=False):
        root=Path(directory)
        return cls(build_packet_migration(root/'solver',compiler=compiler,sanitize=sanitize),
                   build_packet_observer(root/'observer',compiler=compiler,sanitize=sanitize))

    def pack(self,queries,*,max_bytes=DEFAULT_BYTES):
        """Snapshot iterables and serialize every original-address mask exactly."""
        bank=self.observer.freeze_queries(queries)
        return self.solver.pack_frozen(bank,max_bytes=max_bytes)

    def prepare(self,edges,plan_a,plan_b,*,max_bytes=DEFAULT_BYTES):
        edges=tuple(map(tuple,edges));a=tuple(plan_a);b=tuple(plan_b)
        solver=self.solver.prepare(edges,a,b,max_bytes=max_bytes)
        try:
            observer=self.observer.prepare(edges,a,b,max_bytes=max_bytes-solver.info['payload_bytes'])
        except BaseException:
            solver.close();raise
        return PacketSession(solver,observer,self)

class PacketSession(CheckedSession):
    def __init__(self,solver,observer,service):
        super().__init__(solver,observer)
        self._service=service

    def solve(self,queries,*,mode='auto',check_mode='auto',max_bytes=DEFAULT_BYTES):
        """Include freezing and packing; return only checked immutable output.

        The array guard is conservative, not an RSS or execution-time deadline.
        `auto` is a fixed heuristic, not a promise of best performance.
        """
        begin=time.perf_counter_ns()
        with self._lock:
            if self._closed:raise RuntimeError('checked session is closed')
            packet=self._service.pack(queries,max_bytes=max_bytes)
            packing=time.perf_counter_ns()-begin
            result,audit=self._checked(packet,mode,check_mode,max_bytes)
        return PacketCheckedBatch(result,audit,time.perf_counter_ns()-begin,
                                  len(packet),packing,'iterable')

    def solve_packet(self,packet,*,mode='auto',check_mode='auto',max_bytes=DEFAULT_BYTES):
        """For a pre-existing SPQ1 byte bank; creation cost is explicitly excluded.

        Both components validate the entire packet, independently, on every call.
        """
        begin=time.perf_counter_ns()
        if type(packet) is not bytes:raise ValueError('packet must be exact immutable bytes')
        with self._lock:
            if self._closed:raise RuntimeError('checked session is closed')
            result,audit=self._checked(packet,mode,check_mode,max_bytes)
        return PacketCheckedBatch(result,audit,time.perf_counter_ns()-begin,
                                  len(packet),0,'packet')

    def _checked(self,packet,mode,check_mode,max_bytes):
        result=self.solver.solve_packet(packet,mode=mode,max_bytes=max_bytes)
        published=sum(map(len,(result.statuses,result.labels,result.proof_kinds,result.proof_vertices_le)))
        if published>=max_bytes:raise MemoryError('published buffers exhaust native payload allowance')
        audit=self.observer.verify_packet(packet,result,minimal=True,mode=check_mode,max_bytes=max_bytes-published)
        return result,audit
