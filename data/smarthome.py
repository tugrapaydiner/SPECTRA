"""Smart-home state resolution task (BLUEPRINT section 24.5).

A privacy-preserving edge decision task: given a snapshot of sensor/context state,
choose the correct action. We encode the state as a fixed-length token sequence
and the target as the action (broadcast over the sequence so the existing
cell/board metrics apply). A deterministic safety *policy* provides the ground
truth and the "known safe action" of the section 10.2 generator interface, with a
validator that flags safety-critical false negatives (missing a real alert).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Token vocabulary (0 = PAD). Six binary features -> 12 value tokens; 4 actions.
PAD = 0
# Feature value tokens: (feature_index, value) -> 1 + 2*feature + value.
_FEATURES = ["door", "lock", "motion", "user", "time", "battery"]
# Human-readable value meaning per feature: index 0 = "safe" state, 1 = "risky".
#   door:   0 closed / 1 open
#   lock:   0 locked / 1 unlocked
#   motion: 0 none   / 1 detected
#   user:   0 home   / 1 away
#   time:   0 day    / 1 night
#   battery:0 ok     / 1 low
ACTION_BASE = 1 + 2 * len(_FEATURES)  # 13
IGNORE, ALERT, LOCK_ALERT, CONSERVE = ACTION_BASE, ACTION_BASE + 1, ACTION_BASE + 2, ACTION_BASE + 3
NUM_TOKENS = ACTION_BASE + 4  # 17
SEQ_LEN = 16

# Actions considered safety-critical (a false negative here is dangerous).
_CRITICAL = {ALERT, LOCK_ALERT}


@dataclass
class SmartHomeTask:
    state: dict[str, int]
    sensors: np.ndarray  # input token sequence [SEQ_LEN]
    action: int          # target action token


def safe_action(state: dict[str, int]) -> int:
    """Deterministic safety policy mapping a state to the correct action."""
    door, lock, motion, user = state["door"], state["lock"], state["motion"], state["user"]
    battery = state["battery"]
    if user == 1 and door == 1 and lock == 1:  # away, door open, but locked -> alert
        return ALERT
    if user == 1 and door == 1 and lock == 0:  # away, door open, unlocked -> lock + alert
        return LOCK_ALERT
    if user == 1 and motion == 1:              # away but motion detected -> alert
        return ALERT
    if battery == 1:                            # low battery -> conserve
        return CONSERVE
    return IGNORE


def encode_state(state: dict[str, int]) -> np.ndarray:
    """Encode a state dict to a token sequence ``[SEQ_LEN]`` (PAD-filled tail)."""
    seq = np.full(SEQ_LEN, PAD, dtype=np.int64)
    for i, feat in enumerate(_FEATURES):
        seq[i] = 1 + 2 * i + state[feat]  # unique token per (feature, value)
    return seq


def generate_task(rng: np.random.Generator, p_risky: float = 0.5) -> SmartHomeTask:
    """Sample a state (biased toward risky cases) and its safe action."""
    state = {f: int(rng.random() < p_risky) for f in _FEATURES}
    return SmartHomeTask(state=state, sensors=encode_state(state), action=safe_action(state))


def generate_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(input [SEQ_LEN], target [SEQ_LEN])`` with the action broadcast."""
    task = generate_task(rng)
    target = np.full(SEQ_LEN, task.action, dtype=np.int64)
    return task.sensors, target


def is_safety_critical_action(action: int) -> bool:
    return action in _CRITICAL


def safety_false_negative(true_action: int, pred_action: int) -> bool:
    """True if a safety-critical action was required but not predicted (section 24.5)."""
    return is_safety_critical_action(true_action) and not is_safety_critical_action(pred_action)
