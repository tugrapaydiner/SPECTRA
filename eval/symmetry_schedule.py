"""Exact stopping-time counterfactuals for deterministic, freshly reset views.

These are conditional statements about previously measured trajectories. They
predict validity/transition counts, not wall-clock latency or novel examples.
"""
from __future__ import annotations


def predict_schedule(identity_time: int | None, dihedral_time: int | None, *,
                     identity_cycles: int, view_limit: int, budget: int) -> dict:
    """Compose identity-32 and fixed-order 8x4-view first-success observations.

    A None observation certifies failure through all 32 calls of its parent arm.
    The first four identity calls are shared by both parents. All later views
    start from zero state and share fixed parameters/input transforms. This
    function cannot support adaptive transforms or trajectories sharing state.
    """
    for t in (identity_time, dihedral_time):
        if t is not None and (type(t) is not int or not 1 <= t <= 32):
            raise ValueError('parent success time must be None or integer in [1,32]')
    if (type(identity_cycles) is not int or type(view_limit) is not int or type(budget) is not int
        or not 4 <= identity_cycles <= budget <= 32 or not 1 <= view_limit <= 8
        or budget > identity_cycles+4*(view_limit-1)):
        raise ValueError('unsupported bounded prefix/view schedule')
    a = identity_time if identity_time is not None else 33
    d = dihedral_time if dihedral_time is not None else 33
    if (a <= 4 or d <= 4) and a != d:
        raise ValueError('parent trajectories disagree on the shared identity prefix')
    if a <= identity_cycles:
        return {'valid': True, 'transitions': a, 'answer_source': 'identity_32'}
    transformed_calls = d-4
    if dihedral_time is not None and identity_cycles+transformed_calls <= budget:
        return {'valid': True, 'transitions': identity_cycles+transformed_calls, 'answer_source':'dihedral_32'}
    return {'valid':False,'transitions':budget,'answer_source':None}
