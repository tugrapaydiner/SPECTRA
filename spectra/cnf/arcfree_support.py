"""Exact support-query specialization for an arc-free quotient relation.

This is an opt-in specialization of :mod:`spectra.cnf.quotient_query`. It builds
an ordinary SCC quotient and admits the instance only when no residual implication
arcs remain. Every query is solved by the committed native runtime, which lifts a
complete immutable witness and then runs the independently owned original-input
checker before reporting ``SAT_VERIFIED``.

The specialization never claims a proof of UNSAT: an empty restricted component
is returned as ``UNKNOWN/restriction_conflict``. Arc-free 2-SAT relations and
literal-equivalence substitution are established techniques; this module exposes
a measured systems specialization rather than an algorithmic-novelty claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from spectra.cnf.quotient_query import DEFAULT_BYTES, PreparedQuotient, QuotientRuntime


class ResidualImplications(ValueError):
    """The exact quotient still has inter-component implications."""


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

    def record(self) -> dict:
        return {
            **self.__dict__,
            "labels": list(self.labels),
            "schema": "spectra.arc_free_support.v2",
            "learned": False,
        }


class ArcFreeSupportTable:
    """Prepared native support table with separately owned original checking.

    ``setup_ns`` includes SCC quotient construction, the independent checker,
    arc-freedom inspection, complete base-witness materialization, and one original
    check. ``setup_payload_bytes`` is the retained native quotient plus checker
    payload reported by the runtime; it is not process RSS.
    """

    def __init__(
        self,
        runtime: QuotientRuntime,
        n: int,
        k: int,
        edges: tuple,
        *,
        masks: tuple = (),
        max_build_bytes: int = DEFAULT_BYTES,
    ) -> None:
        start = time.perf_counter_ns()
        self._lock = threading.RLock()
        self._prepared: PreparedQuotient | None = runtime.prepare(
            n,
            k,
            edges,
            masks=masks,
            mode="scc",
            max_build_bytes=max_build_bytes,
        )
        info = self._prepared.info
        if info["quotient_arcs"]:
            self.close()
            raise ResidualImplications("compiled quotient is not arc-free")
        labels, reason, _, _ = self._prepared.solve_support(())
        if reason == 3:
            self.close()
            raise ValueError("base quotient is contradictory")
        if reason != 0:
            self.close()
            raise AssertionError("arc-free base support construction failed")
        self._base = labels
        self.setup_payload_bytes = info["total_owned_index_payload_bytes"]
        self.setup_ns = time.perf_counter_ns() - start

    @property
    def info(self) -> dict:
        with self._lock:
            if self._prepared is None:
                raise RuntimeError("support table is closed")
            return {
                **self._prepared.info,
                "arc_free": True,
                "support_table_payload_bytes": self.setup_payload_bytes,
                "support_table_setup_ns": self.setup_ns,
            }

    def solve(self, restrictions: tuple = ()) -> SupportTableResult:
        start = time.perf_counter_ns()
        if type(restrictions) is not tuple:
            raise ValueError("restrictions must be an exact tuple")
        with self._lock:
            if self._prepared is None:
                raise RuntimeError("support table is closed")
            labels, reason, touched, changed = self._prepared.solve_support(restrictions)
        if reason == 0:
            status, text = "SAT_VERIFIED", "satisfied"
        elif reason == 4:
            status, text = "UNKNOWN", "restriction_conflict"
        elif reason == 3:
            status, text = "UNKNOWN", "base_contradiction"
        else:
            raise AssertionError(f"unexpected native support reason: {reason}")
        return SupportTableResult(
            status,
            labels,
            text,
            len(restrictions),
            touched,
            changed,
            self.setup_payload_bytes,
            time.perf_counter_ns() - start,
        )

    def close(self) -> None:
        with self._lock:
            if self._prepared is not None:
                self._prepared.close()
                self._prepared = None

    def __enter__(self) -> "ArcFreeSupportTable":
        with self._lock:
            if self._prepared is None:
                raise RuntimeError("support table is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
