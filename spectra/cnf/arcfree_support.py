"""Exact support-query table for an arc-free quotient relation.

This is an opt-in specialization of :mod:`spectra.cnf.quotient_query`.  It first
builds the ordinary SCC quotient, then admits the instance only when the compiled
relation contains no residual implication arcs.  Every quotient component is then
an independent finite-domain choice.  Queries can restrict any original vertex;
a complete immutable byte witness is lifted and checked against all original
masks, edges and current restrictions before SAT_VERIFIED is returned.

The specialization never reports UNSAT.  An empty restricted component returns
UNKNOWN/restriction_conflict.  It does not bypass quotient construction, certificate
export, full witness materialization, or the original checker.  Arc-free 2-SAT
relations and equivalence-class substitution are established techniques; this API
is a measured systems specialization, not an algorithmic novelty claim.
"""
from __future__ import annotations
from dataclasses import dataclass
import threading
import time

from spectra.cnf.quotient_query import DEFAULT_BYTES, PreparedQuotient, QuotientRuntime


class ResidualImplications(ValueError):
    """The exact quotient still has inter-component implications."""


def _exact_uint(value, maximum, label):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(label)
    return value


@dataclass(frozen=True)
class SupportTableResult:
    status: str
    labels: bytes
    reason: str
    restrictions: int
    touched_components: int
    changed_components: int
    setup_payload_bytes: int
    elapsed_ns: int

    def record(self):
        return {**self.__dict__, 'labels': list(self.labels),
                'schema': 'spectra.arc_free_support.v1', 'learned': False}


class ArcFreeSupportTable:
    """Prepared exact table with separately owned original-input checking.

    ``setup_ns`` includes SCC quotient construction, integer certificate export,
    table construction, full base-witness materialization and one complete original
    check.  The payload count covers retained Python integer/reference arrays and
    byte witnesses only approximately; it is not process RSS or allocator overhead.
    """
    def __init__(self, runtime: QuotientRuntime, n: int, k: int, edges: tuple, *,
                 masks: tuple=(), max_build_bytes: int=DEFAULT_BYTES):
        start = time.perf_counter_ns()
        self._lock = threading.RLock()
        self._prepared: PreparedQuotient | None = runtime.prepare(
            n, k, edges, masks=masks, mode='scc', max_build_bytes=max_build_bytes)
        self._masks = masks if masks else tuple(((1 << k) - 1) for _ in range(n))
        certificate = self._prepared.certificate()
        if certificate['impossible']:
            self._prepared.close(); self._prepared = None
            raise ValueError('base quotient is contradictory')
        if certificate['arcs'] or certificate['offsets'][-1] != 0:
            self._prepared.close(); self._prepared = None
            raise ResidualImplications('compiled quotient is not arc-free')
        self._certificate = certificate
        qn = len(certificate['palettes'])
        members = [[] for _ in range(qn)]
        for vertex, node in enumerate(certificate['node']):
            members[node].append(vertex)
        self._members = tuple(tuple(row) for row in members)
        defaults = []
        base = bytearray(n)
        for domain in certificate['initial']:
            if type(domain) is not int or domain <= 0:
                self.close(); raise AssertionError('invalid nonempty compiled domain')
            defaults.append((domain & -domain).bit_length() - 1)
        self._defaults = tuple(defaults)
        for vertex, node in enumerate(certificate['node']):
            side = self._defaults[node]
            if certificate['wide'][vertex]:
                colour = side
            else:
                colour = certificate['colour0'][vertex] if side == 0 else certificate['colour1'][vertex]
                if colour == 255:
                    self.close(); raise AssertionError('default side lacks an original colour')
            base[vertex] = colour
        self._base = bytes(base)
        if not self._prepared.check(self._base, ()):
            self.close(); raise AssertionError('compiled base witness fails original constraints')
        # Explicit, conservative retained logical payload estimate. Python object
        # headers/allocator capacities and native prepared-index storage are separate.
        self.setup_payload_bytes = (len(self._base) + 8 * len(self._defaults) +
            8 * sum(len(row) for row in self._members) + 8 * len(self._members))
        self.setup_ns = time.perf_counter_ns() - start

    @property
    def info(self):
        with self._lock:
            if self._prepared is None:
                raise RuntimeError('support table is closed')
            return {**self._prepared.info, 'arc_free': True,
                    'support_table_payload_bytes': self.setup_payload_bytes,
                    'support_table_setup_ns': self.setup_ns}

    def solve(self, restrictions: tuple=()) -> SupportTableResult:
        start = time.perf_counter_ns()
        if type(restrictions) is not tuple:
            raise ValueError('restrictions must be an exact tuple')
        with self._lock:
            if self._prepared is None:
                raise RuntimeError('support table is closed')
            certificate = self._certificate
            required = {}
            for pair in restrictions:
                if type(pair) is not tuple or len(pair) != 2:
                    raise ValueError('restriction must be an exact pair tuple')
                vertex = _exact_uint(pair[0], len(self._base) - 1, 'invalid restricted vertex')
                domain = _exact_uint(pair[1], (1 << certificate['k']) - 1, 'invalid restricted mask')
                node = certificate['node'][vertex]
                if certificate['wide'][vertex]:
                    translated = domain & self._masks[vertex]
                else:
                    translated = 0
                    c0, c1 = certificate['colour0'][vertex], certificate['colour1'][vertex]
                    if c0 != 255 and (domain >> c0) & 1: translated |= 1
                    if c1 != 255 and (domain >> c1) & 1: translated |= 2
                translated &= certificate['initial'][node]
                required[node] = translated if node not in required else required[node] & translated
                if not required[node]:
                    return SupportTableResult('UNKNOWN', b'', 'restriction_conflict',
                        len(restrictions), len(required), 0, self.setup_payload_bytes,
                        time.perf_counter_ns() - start)
            labels = bytearray(self._base)
            changed = 0
            for node, domain in required.items():
                side = self._defaults[node]
                if not (domain >> side) & 1:
                    side = (domain & -domain).bit_length() - 1
                    changed += 1
                    for vertex in self._members[node]:
                        if certificate['wide'][vertex]:
                            colour = side
                        else:
                            colour = certificate['colour0'][vertex] if side == 0 else certificate['colour1'][vertex]
                            if colour == 255:
                                raise AssertionError('selected side lacks an original colour')
                        labels[vertex] = colour
            answer = bytes(labels)
            if not self._prepared.check(answer, restrictions):
                raise AssertionError('support-table answer fails original constraints')
            return SupportTableResult('SAT_VERIFIED', answer, 'satisfied',
                len(restrictions), len(required), changed, self.setup_payload_bytes,
                time.perf_counter_ns() - start)

    def close(self):
        with self._lock:
            if self._prepared is not None:
                self._prepared.close()
                self._prepared = None

    def __enter__(self):
        with self._lock:
            if self._prepared is None:
                raise RuntimeError('support table is closed')
        return self

    def __exit__(self, *_):
        self.close()
