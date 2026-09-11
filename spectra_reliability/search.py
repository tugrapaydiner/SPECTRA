"""Budgeted best-first search with task-bound immutable verified incumbents.

This is not the historical MCTS. With a sound checker, a valid incumbent cannot
be replaced by an invalid proposal in this run. This is NOT a learned-value,
findability, hard-deadline, or cross-run budget-monotonicity guarantee.
"""
from __future__ import annotations
import heapq
import math
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, TypeVar
from .budget import Budget, BudgetExhausted, WorkLedger
from .identity import digest_parts, integral_tuple, positive_int, tokens_bytes
from .semantics import CheckedEvaluator
S = TypeVar("S"); A = TypeVar("A")


@dataclass(frozen=True)
class Task:
    name: str
    checker_id: str
    puzzle: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or not isinstance(self.checker_id, str) or not self.checker_id:
            raise ValueError("task and checker IDs required")
        object.__setattr__(self, "puzzle", integral_tuple(self.puzzle))

    @property
    def identity(self) -> str:
        return digest_parts("spectra.task.v1", self.name, self.checker_id, tokens_bytes(self.puzzle))


@dataclass(frozen=True)
class Candidate:
    answer: tuple[int, ...]
    path: tuple[int, ...]
    score: float
    valid: bool
    task_identity: str


class Incumbent:
    def __init__(self, task: Task, checker: Callable[[tuple[int, ...], tuple[int, ...]], bool]):
        self.task = task; self.checker = checker; self.best: Candidate | None = None
        self.offers = 0; self.valid_offers = 0; self.invalid_after_valid = 0

    def offer(self, answer: Iterable[int], path: tuple[int, ...], score: float,
              ledger: WorkLedger | None = None) -> Candidate:
        frozen = integral_tuple(answer)  # snapshot before certification
        if len(frozen) != len(self.task.puzzle): raise ValueError("answer geometry mismatch")
        if isinstance(score, bool) or not math.isfinite(float(score)): raise ValueError("score must be finite")
        if any(type(i) is not int or i < 0 for i in path): raise ValueError("invalid action ordinals")
        valid = self.checker(self.task.puzzle, frozen) if ledger is None else ledger.call("check", self.checker, self.task.puzzle, frozen)
        if type(valid) is not bool: raise TypeError("checker must return bool; scores/truthy objects are not certificates")
        candidate = Candidate(frozen, tuple(path), float(score), valid, self.task.identity)
        self.offers += 1; self.valid_offers += int(valid)
        if self.best is not None and self.best.valid and not valid: self.invalid_after_valid += 1
        if self.best is None or (valid and not self.best.valid) or (
            valid == self.best.valid and not valid and score > self.best.score):
            self.best = candidate
        return candidate


@dataclass
class SearchResult:
    answer: tuple[int, ...] | None
    valid: bool
    status: str
    task_identity: str
    path: tuple[int, ...] | None
    stats: dict[str, Any]
    candidates: list[dict[str, Any]]
    error: str | None = None


class ReliableSearch(Generic[S, A]):
    def __init__(self, *, task: Task, checker: Callable, evaluator: CheckedEvaluator[S],
                 actions: Callable[[S], Iterable[A]], transition: Callable[[S, A], S],
                 decode: Callable[[S], Iterable[int]], max_depth: int, max_actions: int = 64):
        if task.name != evaluator.contract.task or task.checker_id != evaluator.contract.checker:
            raise ValueError("task differs from evaluator task/checker contract")
        self.task = task; self.checker = checker; self.evaluator = evaluator
        self.actions = actions; self.transition = transition; self.decode = decode
        self.max_depth = positive_int(max_depth, "max_depth"); self.max_actions = positive_int(max_actions, "max_actions")

    def _actions(self, state: S) -> tuple[A, ...]:
        values = []
        for action in self.actions(state):
            values.append(action)
            if len(values) > self.max_actions: raise ValueError("action provider exceeded max_actions")
        return tuple(values)

    def run(self, root: S, budget: Budget, *, baseline_actions: Iterable[A] = (),
            stop_on_valid: bool = True, collect_candidates: bool = False) -> SearchResult:
        if type(stop_on_valid) is not bool or type(collect_candidates) is not bool:
            raise ValueError("stop_on_valid/collect_candidates must be boolean")
        ledger = WorkLedger(budget); incumbent = Incumbent(self.task, self.checker)
        records = []; sequence = 0; queue = []; max_reached = 0; baseline_steps = 0
        status = "tree_exhausted"; error = None

        def observe(state, path, *, value_required=True):
            answer = ledger.call("decode", self.decode, state)
            candidate = incumbent.offer(answer, path, 0.0, ledger)
            value = 1.0 if candidate.valid else (ledger.call("value", self.evaluator, state) if value_required else 0.0)
            if not candidate.valid and not incumbent.best.valid and value > incumbent.best.score:
                incumbent.best = Candidate(candidate.answer, candidate.path, value, False, candidate.task_identity)
            if collect_candidates:
                records.append({"path": list(path), "valid": candidate.valid, "value": value,
                                "answer": list(candidate.answer), "task_identity": candidate.task_identity})
            return candidate.valid, value

        try:
            state = root
            for ordinal, action in enumerate(baseline_actions):
                if ordinal >= self.max_depth: raise ValueError("baseline exceeds max_depth")
                state = ledger.call("transition", self.transition, state, action)
                baseline_steps += 1; max_reached = max(max_reached, ordinal + 1)
                valid, _ = observe(state, (0,) * (ordinal + 1), value_required=False)
                if valid and stop_on_valid:
                    status = "verified_baseline"; break
            else:
                # Repeating a prefix is explicitly charged, not presented as cache reuse.
                heapq.heappush(queue, (0.0, sequence, (), root)); sequence += 1
            if not (incumbent.best is not None and incumbent.best.valid and stop_on_valid):
                while queue:
                    _, _, path, state = heapq.heappop(queue)
                    if len(path) >= self.max_depth: continue
                    actions = ledger.call("policy", self._actions, state)
                    for ordinal, action in enumerate(actions):
                        child = ledger.call("transition", self.transition, state, action)
                        child_path = path + (ordinal,); max_reached = max(max_reached, len(child_path))
                        valid, value = observe(child, child_path)
                        if valid and stop_on_valid:
                            status = "verified_search"; queue.clear(); break
                        if len(child_path) < self.max_depth and not valid:
                            heapq.heappush(queue, (-value, sequence, child_path, child)); sequence += 1
                    if incumbent.best is not None and incumbent.best.valid and stop_on_valid: break
        except BudgetExhausted as exc:
            status = "budget_exhausted"; error = str(exc)
        except Exception as exc:
            status = "callback_error"; error = f"{type(exc).__name__}: {exc}"
        best = incumbent.best; stats = ledger.snapshot()
        stats.update({"baseline_steps": baseline_steps, "max_depth_reached": max_reached,
            "candidate_checks": incumbent.offers, "valid_candidates": incumbent.valid_offers,
            "invalid_proposals_after_valid": incumbent.invalid_after_valid, "algorithm": "budgeted_best_first_v1",
            "stop_on_valid": stop_on_valid, "baseline_cost_included": True,
            "baseline_invalid_states_evaluated_by_proxy": False})
        return SearchResult(None if best is None else best.answer, bool(best and best.valid), status,
            self.task.identity, None if best is None else best.path, stats, records, error)
