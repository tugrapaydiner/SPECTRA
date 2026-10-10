from __future__ import annotations

import pytest

from experiments.real_traffic.series import combine_distinct_weeks


def test_series_keeps_first_distinct_trace_request() -> None:
    q1 = ((0, 1),)
    q2 = ((1, 2),)
    q3 = ((2, 4),)
    result = combine_distinct_weeks(
        ("X01", "X02"),
        ((q1, q2), (q2, q3)),
        ((1, 2), (3, 4)),
        (("SAT", "UNSAT"), ("UNSAT", "SAT")),
        (("a", None), (None, "b")),
        ((None, {"proof": 1}), ({"proof": 1}, None)),
    )
    queries, timestamps, statuses, witnesses, proofs, duplicates = result
    assert queries == (q1, q2, q3)
    assert timestamps == (1, 2, 2016 + 4)
    assert statuses == ("SAT", "UNSAT", "SAT")
    assert witnesses == ("a", None, "b")
    assert proofs == (None, {"proof": 1}, None)
    assert duplicates == 1


def test_series_refuses_ragged_or_repeated_week_inventory() -> None:
    with pytest.raises(ValueError, match="nonempty and distinct"):
        combine_distinct_weeks((), (), (), (), (), ())
    with pytest.raises(ValueError, match="inventories differ"):
        combine_distinct_weeks(("X01",), (), (), (), (), ())
    with pytest.raises(ValueError, match="ragged"):
        combine_distinct_weeks(
            ("X01",), (((0, 1),),), ((1,),), (("SAT",),), ((),), ((),))
