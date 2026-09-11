"""Reject incomplete, unpaired or non-finite repeated-solve measurements.

Round repeats measure timing noise, not independent correctness observations.
These checks validate the record inventory; they cannot prove a timer was honest
or that the source data were independent. Ancestry and source audits are separate.
"""
from __future__ import annotations

from collections import defaultdict
import math
from numbers import Real
from typing import Mapping, Sequence


def validate_timing_rows(rows: Sequence[Mapping], *, seed_key: str,
                         example_key: str, required_arms: Sequence[str],
                         rounds: int = 3) -> None:
    if not rows or type(rounds) is not int or rounds < 1:
        raise ValueError("nonempty timing rows and a positive round count are required")
    expected_rounds = set(range(rounds))
    seen = set()
    groups = defaultdict(list)
    arms = defaultdict(set)
    by_seed = defaultdict(set)
    for row in rows:
        try:
            seed, example, arm = row[seed_key], row[example_key], row["arm"]
            repeat, latency, valid = row["round"], row["latency_ms"], row["valid"]
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete timing record") from exc
        if type(seed) is not int or type(example) not in (int, str) or example == "":
            raise ValueError("invalid model/example identity")
        if not isinstance(arm, str) or not arm or type(repeat) is not int or repeat not in expected_rounds:
            raise ValueError("invalid arm or timing round")
        if type(valid) is not bool:
            raise ValueError("timing correctness must be an exact boolean")
        if isinstance(latency, bool) or not isinstance(latency, Real) or not math.isfinite(latency) or latency <= 0:
            raise ValueError("timing latency must be finite and positive")
        key = (seed, example, arm, repeat)
        if key in seen:
            raise ValueError("duplicate timing round would overcount observations")
        seen.add(key)
        groups[(seed, example, arm)].append(row)
        arms[arm].add((seed, example))
        by_seed[seed].add(example)
    if not required_arms or not set(required_arms) <= set(arms):
        raise ValueError("required timing arms are missing")
    reference_pairs = arms[required_arms[0]]
    if any(pairs != reference_pairs for pairs in arms.values()):
        raise ValueError("timing arms are not paired on the same model-examples")
    examples = next(iter(by_seed.values()))
    if any(ids != examples for ids in by_seed.values()):
        raise ValueError("crossed timing inference requires common examples across seeds")
    for group in groups.values():
        if len(group) != rounds or {r["round"] for r in group} != expected_rounds:
            raise ValueError("timing group must have exactly three distinct rounds" if rounds == 3
                             else "timing group has incomplete rounds")
        if any(r["valid"] != group[0]["valid"] for r in group[1:]):
            raise ValueError("deterministic correctness changed across timing rounds")
