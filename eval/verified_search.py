"""Typed values and budgeted checked best-first search (not an MCTS claim).

All generated candidates are decoded and checked online. An exact checker is a
required task-specific dependency, not an inferred property of a learned score.
Legacy M08/M15 reproductions remain separate, explicitly unaligned experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from copy import deepcopy
import heapq
import math
import time
from typing import Any, Callable, Sequence

from data.ancestry import require_sha


class ValueTarget(str, Enum):
    IMPROVEMENT = "one_cycle_improvement"
    QUALITY = "current_structural_quality"
    TERMINAL = "current_terminal_probability"
    BUDGET = "budget_conditioned_solve_probability"


@dataclass(frozen=True)
class ValueContract:
    target: ValueTarget
    model_sha256: str
    evaluator_sha256: str
    transition_id: str
    state_schema: str = "search_state_xyz_v1"
    continuation_policy: str | None = None
    maximum_horizon: int | None = None

    def __post_init__(self):
        if not isinstance(self.target, ValueTarget):
            raise TypeError("target must be an explicit ValueTarget")
        require_sha(self.model_sha256)
        require_sha(self.evaluator_sha256)
        if any(not isinstance(v, str) or not v for v in (self.transition_id, self.state_schema)):
            raise ValueError("transition and state identities are required")
        if self.target is ValueTarget.BUDGET:
            if not self.continuation_policy or type(self.maximum_horizon) is not int or self.maximum_horizon < 0:
                raise ValueError("budget value requires a continuation policy and integer horizon")
        elif self.continuation_policy is not None or self.maximum_horizon is not None:
            raise ValueError("non-budget value must not silently carry budget semantics")

    def bind_selection(self, *, model_sha256: str, transition_id: str,
                       state_schema: str = "search_state_xyz_v1",
                       continuation_policy: str | None = None,
                       horizon: int | None = None) -> None:
        if (self.model_sha256, self.transition_id, self.state_schema) != (model_sha256, transition_id, state_schema):
            raise ValueError("evaluator/reasoner/transition/state contract mismatch")
        if self.target is ValueTarget.IMPROVEMENT:
            raise ValueError("one-cycle improvement is not an absolute candidate-selection value")
        if self.target is ValueTarget.BUDGET:
            if continuation_policy != self.continuation_policy or type(horizon) is not int or not 0 <= horizon <= self.maximum_horizon:
                raise ValueError("budget-conditioned value used under an incompatible policy or horizon")
        elif continuation_policy is not None or horizon is not None:
            raise ValueError("absolute non-budget selector does not accept a horizon")

    def check_value(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("evaluator produced a non-finite value")
        if not 0.0 <= value <= 1.0:
            raise ValueError("declared probability/normalized quality must lie in [0,1]")
        return value

    def metadata(self) -> dict:
        result = asdict(self)
        result["target"] = self.target.value
        return result


@dataclass(frozen=True)
class CheckedAnswer:
    answer: Any
    valid: bool
    score: float
    path: tuple[int, ...]


class CheckedIncumbent:
    """Retain a copied, checked answer; learned scores cannot demote validity."""

    def __init__(self, checker: Callable[[Any], bool]):
        self._checker = checker
        self._best: CheckedAnswer | None = None
        self.checks = 0

    def offer(self, answer: Any, score: float, path: tuple[int, ...]) -> bool:
        if not math.isfinite(float(score)):
            raise ValueError("candidate score must be finite")
        owned = deepcopy(answer)
        # The checker receives no alias to the retained answer.
        valid = self._checker(deepcopy(owned))
        self.checks += 1
        if type(valid) is not bool:
            raise TypeError("checker must return a bool, not a proxy score or tensor")
        item = CheckedAnswer(owned, valid, float(score), tuple(path))
        if self._best is None or (item.valid, item.score) > (self._best.valid, self._best.score):
            self._best = item
        return valid

    def snapshot(self) -> CheckedAnswer | None:
        return deepcopy(self._best)


@dataclass
class _Node:
    state: Any
    path: tuple[int, ...]
    value: float


@dataclass(frozen=True)
class SearchResult:
    answer: Any
    valid: bool
    path: tuple[int, ...]
    work: dict
    candidates: tuple[dict, ...]


class BudgetedVerifiedSearch:
    """Serial checked best-first search with a hard transition-work ceiling.

    Each transition creates exactly one child. No hidden multi-child expansion,
    pretrained baseline or fallback is outside the measured budget. The optional
    identity prefix is part of this same budget, not a free baseline.
    Wall deadlines are soft: an in-flight atomic call may finish after its deadline.
    They are not hard real-time guarantees. No batched API silently changes this.

    Nodes own defensive deep snapshots. In-place transitions and reused scratch
    outputs cannot corrupt siblings. Decoder/value callbacks also receive private
    snapshots. Copy overhead is inside solve timing and counted explicitly. The
    checker must truthfully check its argument without changing its meaning;
    arbitrary Python callbacks are not a security or purity boundary.
    """

    def __init__(self, *, initial: Callable[[], Any], actions: Sequence[int],
                 transition: Callable[[Any, int], Any], decode: Callable[[Any], Any],
                 checker: Callable[[Any], bool], value: Callable[[Any], float],
                 contract: ValueContract, model_sha256: str, transition_id: str,
                 state_schema: str = "search_state_xyz_v1"):
        contract.bind_selection(model_sha256=model_sha256, transition_id=transition_id, state_schema=state_schema)
        actions = tuple(actions)
        if not actions or any(type(a) is not int for a in actions) or len(set(actions)) != len(actions):
            raise ValueError("actions must be a nonempty sequence of distinct integers")
        self.initial, self.actions = initial, actions
        self.transition, self.decode = transition, decode
        self.checker, self.value, self.contract = checker, value, contract

    def solve(self, *, max_transitions: int, max_depth: int, identity_prefix: int = 0,
              identity_action: int = 0, deadline_ms: float | None = None) -> SearchResult:
        for name, v in [("max_transitions", max_transitions), ("max_depth", max_depth)]:
            if type(v) is not int or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(identity_prefix) is not int or not 0 <= identity_prefix <= max_depth:
            raise ValueError("identity_prefix must be an integer within max_depth")
        if identity_prefix and identity_action not in self.actions:
            raise ValueError("identity prefix action is unavailable")
        if deadline_ms is not None and (not math.isfinite(deadline_ms) or deadline_ms <= 0):
            raise ValueError("deadline_ms must be positive and finite")
        start = time.perf_counter_ns()
        work = {"transitions": 0, "decodes": 0, "checks": 0, "value_calls": 0,
                "initializations": 1, "state_snapshots": 0, "max_transitions": max_transitions,
                "deadline_ms": deadline_ms, "reference_target_used": False}
        def snapshot(state):
            work["state_snapshots"] += 1
            return deepcopy(state)

        root = _Node(snapshot(self.initial()), (), 0.0)
        queue: list[tuple[float, tuple[int, ...], _Node]] = [(0.0, (), root)]
        generated: dict[tuple[int, ...], _Node] = {}
        rows: list[dict] = []
        best: CheckedAnswer | None = None

        def stop():
            if work["transitions"] >= max_transitions:
                return "transition_budget"
            if deadline_ms is not None and (time.perf_counter_ns() - start) / 1e6 >= deadline_ms:
                return "soft_deadline"
            return None

        def step(parent: _Node, action: int) -> _Node:
            nonlocal best
            path = parent.path + (action,)
            state = snapshot(self.transition(snapshot(parent.state), action))
            work["transitions"] += 1
            answer = deepcopy(self.decode(snapshot(state)))
            work["decodes"] += 1
            valid = self.checker(deepcopy(answer))
            work["checks"] += 1
            if type(valid) is not bool:
                raise TypeError("checker must return bool")
            # A valid answer needs no learned score and is terminal.
            if valid:
                score = 1.0
            else:
                score = self.contract.check_value(self.value(snapshot(state)))
                work["value_calls"] += 1
            node = _Node(state, path, score)
            generated[path] = node
            if len(path) < max_depth and not valid:
                heapq.heappush(queue, (-score, path, node))
            item = CheckedAnswer(deepcopy(answer), valid, score, path)
            if best is None or (valid, score) > (best.valid, best.score):
                best = item
            rows.append({"path": list(path), "valid": valid, "value": score})
            return node

        node = root
        for _ in range(identity_prefix):
            if stop():
                break
            node = step(node, identity_action)
            if best is not None and best.valid:
                break
        while queue and not stop() and not (best is not None and best.valid):
            _, _, parent = heapq.heappop(queue)
            for action in self.actions:
                if stop() or (best is not None and best.valid):
                    break
                if parent.path + (action,) not in generated:
                    step(parent, action)
        reason = "valid_answer" if best is not None and best.valid else (stop() or "tree_exhausted")
        result_answer = None if best is None else deepcopy(best.answer)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        work.update(state_ownership="defensive_snapshots_v1", stop_reason=reason, elapsed_ms=elapsed, candidate_count=len(rows),
                    deadline_overrun_ms=0.0 if deadline_ms is None else max(0.0, elapsed-deadline_ms))
        return SearchResult(result_answer,
                            False if best is None else best.valid,
                            () if best is None else best.path, work, tuple(rows))


def pool_metrics(groups: Sequence[dict]) -> dict:
    """Each row: pool_id, candidate_validity (bools), selected_index.

    Counts are model-example pools, not necessarily independent task instances.
    Bootstrap or clustered inference must be performed separately.
    """
    seen = set()
    covered = returned = selection_failures = 0
    for group in groups:
        key = group["pool_id"]
        if key in seen:
            raise ValueError("duplicate pool_id would overcount observations")
        seen.add(key)
        flags = group["candidate_validity"]
        i = group["selected_index"]
        if not flags or any(type(v) is not bool for v in flags) or type(i) is not int or not 0 <= i < len(flags):
            raise ValueError("invalid fixed-pool selection record")
        has_valid = any(flags)
        covered += int(has_valid)
        returned += int(flags[i])
        selection_failures += int(has_valid and not flags[i])
    n = len(groups)
    return {"pools": n, "covered": covered, "returned_valid": returned,
            "selection_failures": selection_failures,
            "coverage": covered/n if n else None,
            "conditional_selection_reliability": returned/covered if covered else None,
            "success": returned/n if n else None}
