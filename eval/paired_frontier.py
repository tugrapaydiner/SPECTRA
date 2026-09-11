"""Strict paired quality/latency analysis, with explicit sampling scope.

Timing rounds measure noise; they are never additional task/model samples.
Crossed resampling preserves each sampled model's shared example inventory and
uses the same draws for both arms and all endpoints. Descriptive development
intervals do not undo adaptive policy selection or establish confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np


@dataclass(frozen=True)
class FrontierGate:
    min_success_gain: float = .05
    max_mean_latency_ratio: float = 1.
    max_p95_latency_ratio: float = 1.

    def __post_init__(self):
        for value in (self.min_success_gain, self.max_mean_latency_ratio,
                      self.max_p95_latency_ratio):
            if (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError('gate thresholds must be finite numbers')
        if not 0 <= self.min_success_gain <= 1:
            raise ValueError('success gain must lie in [0,1]')
        if min(self.max_mean_latency_ratio, self.max_p95_latency_ratio) <= 0:
            raise ValueError('latency ratios must be positive')


def quality_headroom(baseline_valid: int, pairs: int, min_gain: float) -> dict:
    """Exact finite-inventory ceiling; not a population or power calculation."""
    if (type(pairs) is not int or pairs < 1 or type(baseline_valid) is not int
            or not 0 <= baseline_valid <= pairs):
        raise ValueError('invalid baseline count or inventory size')
    FrontierGate(min_success_gain=min_gain)
    required_net = math.ceil(Fraction(str(min_gain)) * pairs)
    available = pairs - baseline_valid
    return {'baseline_valid': baseline_valid, 'model_example_pairs': pairs,
            'available_additional_solves': available,
            'required_net_additional_solves': required_net,
            'maximum_possible_success_gain': available / pairs,
            'point_gain_attainable_on_this_inventory': required_net <= available}


def _inventory(values, *, integer: bool):
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError('an explicit nonempty model/example inventory is required')
    if any((type(v) is not int or v < 0) if integer else
           (not isinstance(v, str) or not v) for v in values):
        raise ValueError('invalid inventory identity')
    if len(set(values)) != len(values):
        raise ValueError('duplicate inventory identity')
    return tuple(values)


def paired_frontier(rows: list[dict], *, family: str, candidate: str, comparator: str,
                    model_seeds: tuple[int, ...], example_ids: tuple[str, ...],
                    rounds: int = 3, replicates: int = 2000, seed: int = 2026091110,
                    gate: FrontierGate = FrontierGate(),
                    example_groups: tuple[str, ...] | None = None) -> dict:
    """Analyze exactly the supplied two-arm inventory, refusing silent omissions.

    All rows must belong to this family and pair of arms. Each model must have
    every declared example and timing round. Within a model/example/arm, answer,
    validity and work must be invariant across rounds; only latency may change.
    The caller must bind the expected inventories to independent source manifests.
    This routine analyzes records; it does not rerun models or verify answers.
    """
    models = _inventory(model_seeds, integer=True)
    examples = _inventory(example_ids, integer=False)
    group_members = None
    if example_groups is not None:
        if (not isinstance(example_groups, (tuple, list)) or len(example_groups) != len(examples)
                or any(not isinstance(g, str) or not g for g in example_groups)):
            raise ValueError('one nonempty group key is required for each example')
        group_keys = sorted(set(example_groups))
        if len(group_keys) < 2:
            raise ValueError('grouped resampling requires at least two groups')
        group_members = [np.array([i for i, g in enumerate(example_groups) if g == group]) for group in group_keys]
    if len(models) < 2 or len(examples) < 2:
        raise ValueError('crossed inference needs at least two models and examples')
    if (any(not isinstance(v, str) or not v for v in (family, candidate, comparator))
            or candidate == comparator):
        raise ValueError('distinct named arms and family are required')
    if type(rounds) is not int or rounds < 1 or type(replicates) is not int or replicates < 100:
        raise ValueError('positive rounds and at least 100 bootstrap replicates required')
    if type(seed) is not int or seed < 0 or not isinstance(gate, FrontierGate):
        raise ValueError('invalid bootstrap seed or gate')
    arms = (candidate, comparator)
    mi, ei = {m: i for i, m in enumerate(models)}, {e: i for i, e in enumerate(examples)}
    seen, groups = set(), {}
    times = np.empty((2, len(models), len(examples), rounds), dtype=np.float64)
    valid = np.empty((2, len(models), len(examples)), dtype=np.int64)
    for row in rows:
        m, e, a, r = (row[k] for k in ('core_seed', 'example_id', 'arm', 'round'))
        if (row['family'] != family or type(m) is not int or m not in mi or e not in ei
                or a not in arms or type(r) is not int or not 0 <= r < rounds):
            raise ValueError('row falls outside declared paired inventory')
        key = (m, e, a, r)
        if key in seen:
            raise ValueError('duplicate timing record')
        seen.add(key)
        latency = row['latency_ms']
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0:
            raise ValueError('latency must be positive and finite')
        work = row['work']
        if (type(row['valid']) is not bool or not isinstance(work, dict)
                or type(work.get('valid')) is not bool or work['valid'] != row['valid']
                or (row['valid'] and row['answer'] is None)):
            raise ValueError('invalid semantic/work record')
        outcome = (row['valid'], row['answer'], work)
        group = (m, e, a)
        if group in groups and outcome != groups[group]:
            raise ValueError('answer, validity or work changed across timing rounds')
        groups[group] = outcome
        ai = arms.index(a)
        times[ai, mi[m], ei[e], r] = latency
        valid[ai, mi[m], ei[e]] = row['valid']
    expected = 2 * len(models) * len(examples) * rounds
    if len(seen) != expected:
        raise ValueError('missing declared model/example/arm/round records')
    # Seed x example cells remain distinct after collapsing timing noise.
    times = np.median(times, axis=-1)
    delta = valid[0] - valid[1]

    def endpoints(v, t):
        result = np.array([v.mean(), t[0].mean() / t[1].mean(),
                           np.quantile(t[0], .95) / np.quantile(t[1], .95)])
        if not np.isfinite(result).all() or not (result[1:] > 0).all():
            raise ValueError('timing arithmetic overflowed or underflowed')
        return result

    point = endpoints(delta, times)
    rng = np.random.default_rng(seed)
    sampled = np.empty((replicates, 3))
    for k in range(replicates):
        mm = rng.integers(len(models), size=len(models))
        ee = (rng.integers(len(examples), size=len(examples)) if group_members is None else
              np.concatenate([group_members[g] for g in rng.integers(len(group_keys), size=len(group_keys))]))
        sampled[k] = endpoints(delta[mm[:, None], ee], times[:, mm[:, None], ee])
    intervals = np.quantile(sampled, [.025, .975], axis=0).T
    headroom = quality_headroom(int(valid[1].sum()), int(delta.size), gate.min_success_gain)
    conditions = {
        'minimum_point_gain': int(delta.sum()) >= headroom['required_net_additional_solves'],
        'positive_quality_interval_lower': bool(intervals[0, 0] > 0),
        'mean_latency_point_ratio': bool(point[1] <= gate.max_mean_latency_ratio),
        'p95_latency_point_ratio': bool(point[2] <= gate.max_p95_latency_ratio),
    }
    names = ('success_gain', 'mean_latency_ratio', 'p95_latency_ratio')
    return {'family': family, 'candidate': candidate, 'comparator': comparator,
            'model_seeds': list(models), 'unique_examples': len(examples),
            'model_example_pairs': int(delta.size), 'timing_rounds': rounds,
            'candidate_valid': int(valid[0].sum()), 'comparator_valid': int(valid[1].sum()),
            'new_solves': int((delta == 1).sum()), 'regressions': int((delta == -1).sum()),
            'endpoints': {name: {'point': float(point[i]), 'crossed_percentile_95': intervals[i].tolist()}
                          for i, name in enumerate(names)},
            'by_model_seed': {str(m): {'candidate_valid': int(valid[0, i].sum()),
                'comparator_valid': int(valid[1, i].sum()), 'examples': len(examples),
                'new_solves': int((delta[i] == 1).sum()), 'regressions': int((delta[i] == -1).sum())}
                for i, m in enumerate(models)},
            'headroom': headroom, 'descriptive_gate_conditions': conditions,
            'descriptive_gate_pass': all(conditions.values()),
            'bootstrap': {'replicates': replicates, 'seed': seed,
                'units': ('independently resampled model seeds and shared example IDs' if group_members is None
                          else 'independently resampled model seeds and shared complete example groups'),
                **({} if group_members is None else {'unique_example_groups': len(group_keys),
                    'conditional_on_fitted_policies': True, 'policies_refitted_inside_bootstrap': False}),
                'timing_rounds_resampled_as_examples': False,
                'multiple_comparison_adjustment': False},
            'scope': 'Descriptive paired analysis; not independent confirmation, a selection-adjusted interval, '
                     'a power calculation, or an equal-wall-clock-budget comparison. '
                     'Latency gate uses point estimates, not interval upper bounds.'}
