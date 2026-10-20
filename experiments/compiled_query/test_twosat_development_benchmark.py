"""Structural contracts for the exposed native 2-SAT development benchmark."""
from __future__ import annotations

from collections import Counter

from experiments.compiled_query.twosat_development_benchmark import (
    ARMS,
    PREFIXES,
    _make_cases,
    _position_balanced_orders,
)


def test_position_balanced_schedule_uses_every_arm_in_every_slot():
    repeats = 24
    orders = _position_balanced_orders(repeats=repeats, seed=20261019)
    assert len(orders) == repeats
    assert all(tuple(sorted(order)) == tuple(sorted(ARMS)) for order in orders)
    for arm in ARMS:
        positions = Counter(
            position
            for order in orders
            for position, candidate in enumerate(order)
            if candidate == arm
        )
        assert positions == {position: repeats // len(ARMS) for position in range(len(ARMS))}


def test_development_panel_covers_the_frozen_dimensions():
    cases = _make_cases()
    assert len(cases) == 7
    assert all(len(case.rows) == max(PREFIXES) for case in cases)
    tags = {tag for case in cases for tag in case.tags}
    assert {
        "weak-compression",
        "strong-compression",
        "moderate-compression",
        "sparse",
        "dense",
        "low-pin",
        "high-pin",
        "palette64",
        "single-query-unsat",
        "base-unsat",
        "mixed-result",
    }.issubset(tags)
    assert PREFIXES == (1, 8, 64, 1024)
