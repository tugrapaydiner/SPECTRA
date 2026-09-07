"""Synthetic smart-home state resolution task.

The state space is finite (six binary features = 64 exact states).  Milestone 03
exposes stable state codes so grouped train/validation/test splits can partition
that finite universe instead of leaking identical states across splits.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

PAD = 0
FEATURES = ["door", "lock", "motion", "user", "time", "battery"]
ACTION_BASE = 1 + 2 * len(FEATURES)
IGNORE, ALERT, LOCK_ALERT, CONSERVE = ACTION_BASE, ACTION_BASE + 1, ACTION_BASE + 2, ACTION_BASE + 3
NUM_TOKENS = ACTION_BASE + 4
SEQ_LEN = 16
_CRITICAL = {ALERT, LOCK_ALERT}
NUM_STATES = 1 << len(FEATURES)


@dataclass
class SmartHomeTask:
    state: dict[str, int]
    sensors: np.ndarray
    action: int


def safe_action(state: dict[str, int]) -> int:
    door, lock, motion, user = state["door"], state["lock"], state["motion"], state["user"]
    battery = state["battery"]
    if user == 1 and door == 1 and lock == 1:
        return ALERT
    if user == 1 and door == 1 and lock == 0:
        return LOCK_ALERT
    if user == 1 and motion == 1:
        return ALERT
    if battery == 1:
        return CONSERVE
    return IGNORE


def state_code(state: dict[str, int]) -> int:
    """Stable integer code in [0,63] for an exact six-feature state."""
    code = 0
    for i, feat in enumerate(FEATURES):
        value = int(state[feat])
        if value not in (0, 1):
            raise ValueError(f"{feat} must be binary")
        code |= value << i
    return code


def state_from_code(code: int) -> dict[str, int]:
    if not isinstance(code, (int, np.integer)) or not 0 <= int(code) < NUM_STATES:
        raise ValueError(f"state code must be in [0,{NUM_STATES})")
    return {feat: (int(code) >> i) & 1 for i, feat in enumerate(FEATURES)}


def encode_state(state: dict[str, int]) -> np.ndarray:
    seq = np.full(SEQ_LEN, PAD, dtype=np.int64)
    for i, feat in enumerate(FEATURES):
        value = int(state[feat])
        if value not in (0, 1):
            raise ValueError(f"{feat} must be binary")
        seq[i] = 1 + 2 * i + value
    return seq


def task_from_state(state: dict[str, int]) -> SmartHomeTask:
    return SmartHomeTask(state=dict(state), sensors=encode_state(state), action=safe_action(state))


def pair_from_state(state: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    task = task_from_state(state)
    return task.sensors, np.full(SEQ_LEN, task.action, dtype=np.int64)


def generate_task(
    rng: np.random.Generator,
    p_risky: float = 0.5,
    allowed_codes: np.ndarray | list[int] | None = None,
) -> SmartHomeTask:
    if allowed_codes is None:
        state = {f: int(rng.random() < p_risky) for f in FEATURES}
        return task_from_state(state)
    codes = np.asarray(allowed_codes, dtype=np.int64)
    if codes.ndim != 1 or codes.size == 0 or int(codes.min()) < 0 or int(codes.max()) >= NUM_STATES:
        raise ValueError("allowed_codes must be a non-empty 1D subset of valid state codes")
    # Preserve the original independent-Bernoulli distribution conditional on
    # membership in the split's allowed state set.
    risks = np.array([int(c).bit_count() for c in codes], dtype=np.int64)
    weights = (p_risky ** risks) * ((1.0 - p_risky) ** (len(FEATURES) - risks))
    if float(weights.sum()) <= 0:
        weights = np.ones_like(weights, dtype=np.float64)
    weights = weights / weights.sum()
    code = int(rng.choice(codes, p=weights))
    return task_from_state(state_from_code(code))


def generate_pair(
    rng: np.random.Generator,
    allowed_codes: np.ndarray | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    task = generate_task(rng, allowed_codes=allowed_codes)
    return task.sensors, np.full(SEQ_LEN, task.action, dtype=np.int64)


def is_safety_critical_action(action: int) -> bool:
    return action in _CRITICAL


def safety_false_negative(true_action: int, pred_action: int) -> bool:
    return is_safety_critical_action(true_action) and not is_safety_critical_action(pred_action)
