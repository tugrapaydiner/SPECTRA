"""Evaluator target/identity contracts, not calibration or OOD certificates."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Generic, TypeVar
from .identity import positive_int, require_digest
S = TypeVar("S")


class TargetKind(str, Enum):
    CURRENT_VALIDITY = "current_validity"
    STRUCTURAL_QUALITY = "absolute_structural_quality"
    ONE_STEP_IMPROVEMENT = "one_step_improvement"
    BUDGETED_SUCCESS = "budgeted_success"


class SemanticMismatch(ValueError):
    pass


@dataclass(frozen=True)
class EvaluatorContract:
    target: TargetKind
    task: str
    model_sha256: str
    decoder: str
    checker: str
    continuation_policy: str | None = None
    budget_unit: str | None = None
    horizon: int | None = None
    version: int = 1

    def __post_init__(self):
        if not isinstance(self.target, TargetKind): raise SemanticMismatch("explicit TargetKind required")
        require_digest(self.model_sha256, "reasoner identity")
        if any(not isinstance(v, str) or not v for v in (self.task, self.decoder, self.checker)):
            raise SemanticMismatch("task/decoder/checker identities required")
        if type(self.version) is not int or self.version != 1: raise SemanticMismatch("unknown contract version")
        if self.target in (TargetKind.BUDGETED_SUCCESS, TargetKind.ONE_STEP_IMPROVEMENT):
            if not isinstance(self.continuation_policy, str) or not self.continuation_policy:
                raise SemanticMismatch("continuation targets must name their policy")
            if self.budget_unit != "recursive_cycles": raise SemanticMismatch("unsupported horizon unit")
            positive_int(self.horizon, "horizon")
            if self.target is TargetKind.ONE_STEP_IMPROVEMENT and self.horizon != 1:
                raise SemanticMismatch("one-step improvement requires horizon=1")
        elif any(v is not None for v in (self.continuation_policy, self.budget_unit, self.horizon)):
            raise SemanticMismatch("state-only targets must not carry continuation semantics")

    def require(self, expected: EvaluatorContract) -> None:
        if self != expected:
            differences = [k for k, v in asdict(self).items() if v != asdict(expected)[k]]
            raise SemanticMismatch("incompatible evaluator fields: " + ", ".join(differences))


class CheckedEvaluator(Generic[S]):
    def __init__(self, contract: EvaluatorContract, expected: EvaluatorContract, predict: Callable[[S], float]):
        contract.require(expected)
        if contract.target is TargetKind.ONE_STEP_IMPROVEMENT:
            raise SemanticMismatch("one-step improvement is not an absolute search value; use the explicit diagnostic")
        if not callable(predict): raise TypeError("predict must be callable")
        self.contract = contract; self._predict = predict

    def __call__(self, state: S) -> float:
        value = self._predict(state)
        if isinstance(value, bool): raise SemanticMismatch("score cannot be a checker boolean")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise SemanticMismatch("score must be finite and in [0,1]")
        return value


@dataclass(frozen=True)
class FirstHitFamilyContract:
    """A family of horizon-specific targets, not one silently changing target."""
    task: str
    model_sha256: str
    decoder: str
    checker: str
    continuation_policy: str
    max_horizon: int
    version: int = 1

    def __post_init__(self):
        require_digest(self.model_sha256, 'reasoner identity')
        positive_int(self.max_horizon, 'max_horizon')
        if self.version != 1 or any(not isinstance(v,str) or not v for v in (
            self.task,self.decoder,self.checker,self.continuation_policy)):
            raise SemanticMismatch('invalid first-hit family contract')

    def at(self, horizon: int) -> EvaluatorContract:
        horizon=positive_int(horizon,'horizon',allow_zero=True)
        if horizon>self.max_horizon:
            raise SemanticMismatch('horizon outside evaluator family support')
        if horizon==0:
            return EvaluatorContract(TargetKind.CURRENT_VALIDITY,self.task,self.model_sha256,self.decoder,self.checker)
        return EvaluatorContract(TargetKind.BUDGETED_SUCCESS,self.task,self.model_sha256,self.decoder,self.checker,
                                 self.continuation_policy,'recursive_cycles',horizon)


class CheckedHorizonEvaluator(Generic[S]):
    """Bind EACH query to its actual horizon before evaluating a first-hit CDF."""
    def __init__(self, contract: FirstHitFamilyContract, expected: FirstHitFamilyContract,
                 horizon: Callable[[S], int], predict: Callable[[S, EvaluatorContract], float]):
        if contract != expected:
            raise SemanticMismatch('first-hit family mismatch')
        self.contract=contract; self.horizon=horizon; self.predict=predict
        self.last_query_contract: EvaluatorContract | None=None
        self.query_counts: dict[int,int]={}

    def __call__(self,state:S) -> float:
        h=self.horizon(state); query=self.contract.at(h)
        value=self.predict(state,query)
        if isinstance(value,bool):
            raise SemanticMismatch('score must be numeric, not checker boolean')
        value=float(value)
        if not math.isfinite(value) or not 0<=value<=1:
            raise SemanticMismatch('invalid horizon probability')
        self.last_query_contract=query
        self.query_counts[h]=self.query_counts.get(h,0)+1
        return value
