"""Online CPU accounting. Cooperative deadlines retain full callback overshoot."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Callable, TypeVar
from .identity import positive_int
T = TypeVar("T")
KINDS = ("transition", "decode", "check", "value", "policy")


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class Budget:
    transitions: int
    checks: int
    values: int
    decodes: int
    policies: int
    deadline_ms: float | None = None

    def __post_init__(self):
        for name in ("transitions", "checks", "values", "decodes", "policies"):
            positive_int(getattr(self, name), name, allow_zero=True)
        if self.deadline_ms is not None and (isinstance(self.deadline_ms, bool) or
            not math.isfinite(float(self.deadline_ms)) or self.deadline_ms < 0):
            raise ValueError("deadline_ms must be finite and non-negative")


class WorkLedger:
    def __init__(self, budget: Budget, *, clock: Callable[[], int] = time.perf_counter_ns,
                 cpu_clock: Callable[[], int] = time.process_time_ns):
        self.budget = budget; self.clock = clock; self.cpu_clock = cpu_clock
        self.start = clock(); self.cpu_start = cpu_clock()
        self.deadline = None if budget.deadline_ms is None else self.start + int(budget.deadline_ms * 1e6)
        self.counts = dict.fromkeys(KINDS, 0); self.wall_ns = dict.fromkeys(KINDS, 0)
        self.cpu_ns = dict.fromkeys(KINDS, 0); self.failures = dict.fromkeys(KINDS, 0); self._busy = False

    def call(self, kind: str, fn: Callable[..., T], *args, **kwargs) -> T:
        if kind not in KINDS: raise ValueError(f"unknown work kind {kind}")
        if self._busy: raise RuntimeError("nested accounting prohibited; charge non-overlapping boundaries")
        if self.deadline is not None and self.clock() >= self.deadline: raise BudgetExhausted("wall deadline reached")
        limit = getattr(self.budget, {"policy": "policies"}.get(kind, kind + "s"))
        if self.counts[kind] >= limit: raise BudgetExhausted(f"{kind} budget exhausted")
        self.counts[kind] += 1; self._busy = True
        start = self.clock(); cpu_start = self.cpu_clock()
        try:
            return fn(*args, **kwargs)
        except Exception:
            self.failures[kind] += 1
            raise
        finally:
            self.cpu_ns[kind] += max(0, self.cpu_clock() - cpu_start)
            self.wall_ns[kind] += max(0, self.clock() - start); self._busy = False

    def snapshot(self) -> dict:
        now = self.clock(); elapsed = max(0, now - self.start)
        return {"counts": dict(self.counts), "callback_failures": dict(self.failures),
            "operation_wall_ns": dict(self.wall_ns), "operation_cpu_ns": dict(self.cpu_ns),
            "elapsed_wall_ns": elapsed, "elapsed_process_cpu_ns": max(0, self.cpu_clock() - self.cpu_start),
            "control_wall_ns": max(0, elapsed - sum(self.wall_ns.values())), "deadline_ms": self.budget.deadline_ms,
            "deadline_overshoot_ns": 0 if self.deadline is None else max(0, now - self.deadline),
            "deadline_semantics": "cooperative; started callbacks not preempted; full overshoot retained",
            "physical_energy_joules": None, "physical_energy_status": "not_measured"}
