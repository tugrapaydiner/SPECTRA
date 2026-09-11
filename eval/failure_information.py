"""Bounded allocation using only inputs and already observed failed answers.

This is a diagnostic linear selector, not a new algorithm-selection method.
Candidate trajectories reset independently. A selector never receives untried
answers, reference targets, example IDs, model IDs, or the counterfactual table.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
import hashlib

import numpy as np

MODES = ('input_only', 'prefix_only', 'failure_aware')
ALTERNATIVES = 7


def _symbols(values, cells):
    if (not isinstance(values, tuple) or len(values) != cells
            or any(type(v) is not int or not 0 <= v < 5 for v in values)):
        raise ValueError('expected immutable integer symbols in [0,4]')


@dataclass(frozen=True)
class Observation:
    """Only information available after an unsuccessful prefix/attempt."""
    input: tuple[int, ...]
    prefix_answer: tuple[int, ...]
    failures: tuple[tuple[int, tuple[int, ...]], ...] = ()

    def __post_init__(self):
        cells = len(self.input)
        if cells not in (16, 121):
            raise ValueError('this pilot supports Sudoku4 and maze11 only')
        _symbols(self.input, cells)
        _symbols(self.prefix_answer, cells)
        if not isinstance(self.failures, tuple) or len(self.failures) > 3:
            raise ValueError('at most three immutable observed failures')
        seen = set()
        for action, answer in self.failures:
            if type(action) is not int or not 0 <= action < ALTERNATIVES or action in seen:
                raise ValueError('invalid or repeated observed action')
            seen.add(action)
            _symbols(answer, cells)

    @property
    def available(self):
        return tuple(a for a in range(ALTERNATIVES) if a not in {a for a, _ in self.failures})


def pooled_symbols(values: tuple[int, ...]) -> np.ndarray:
    """Fixed 4x4 spatial bins of five-symbol frequencies; no fitted preprocessing."""
    side = int(len(values)**.5)
    a = np.asarray(values).reshape(side, side)
    out = []
    for r in range(4):
        for c in range(4):
            block = a[r*side//4:(r+1)*side//4, c*side//4:(c+1)*side//4]
            out.extend(np.bincount(block.ravel(), minlength=5) / block.size)
    return np.asarray(out, dtype=np.float64)


def features(observation: Observation, mode: str) -> np.ndarray:
    if not isinstance(observation, Observation) or mode not in MODES:
        raise ValueError('declared observation and selector mode required')
    mask = np.zeros(ALTERNATIVES)
    for action, _ in observation.failures:
        mask[action] = 1
    parts = [np.ones(1), pooled_symbols(observation.input), mask,
             np.array([(3-len(observation.failures))/3])]
    if mode != 'input_only':
        prefix = pooled_symbols(observation.prefix_answer)
        parts.append(prefix)
        if mode == 'failure_aware':
            # Zero means no additional feedback. Each observed answer is restored
            # to the original coordinates before it reaches this function.
            parts.append(np.mean([pooled_symbols(ans) for _, ans in observation.failures], axis=0)-prefix
                         if observation.failures else np.zeros_like(prefix))
    return np.concatenate(parts)


@dataclass(frozen=True)
class LinearSelector:
    mode: str
    weights: np.ndarray

    def __post_init__(self):
        expected = {'input_only': 89, 'prefix_only': 169, 'failure_aware': 249}
        if (self.mode not in expected or not isinstance(self.weights, np.ndarray)
                or self.weights.shape != (7, expected[self.mode])
                or not np.isfinite(self.weights).all()):
            raise ValueError('invalid selector coefficients')
        copy = self.weights.astype(np.float64, copy=True)
        copy.setflags(write=False)
        object.__setattr__(self, 'weights', copy)

    def choose(self, observation: Observation) -> int:
        if len(observation.failures) >= 3:
            raise ValueError('selector budget exhausted')
        scores = self.weights @ features(observation, self.mode)
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite selector score')
        # Stable tie order is the original declared D4 order.
        return max(observation.available, key=lambda a: (scores[a], -a))


def first_success(trajectory: list[dict], cap: int) -> int | None:
    return next((i+1 for i, step in enumerate(trajectory[:cap]) if step['valid']), None)


def simulate(case: dict, choose, budget: int = 20) -> dict:
    """Counterfactual decisions/work only. Never an elapsed-time estimate."""
    if type(budget) is not int or budget not in (12, 16, 20):
        raise ValueError('pilot budgets are 12, 16, or 20 cycles')
    t = first_success(case['identity'], 8)
    if t is not None:
        return {'valid': True, 'transitions': t, 'actions': [],
                'answer': case['identity'][t-1]['answer']}
    observation = Observation(tuple(case['input']), tuple(case['identity'][7]['answer']))
    actions = []
    for _ in range((budget-8)//4):
        action = choose(observation)
        if type(action) is not int or action not in observation.available:
            raise ValueError('selector returned a repeated or invalid action')
        actions.append(action)
        trajectory = case['views'][action]
        t = first_success(trajectory, 4)
        if t is not None:
            return {'valid': True, 'transitions': 8+4*(len(actions)-1)+t,
                    'actions': actions, 'answer': trajectory[t-1]['answer']}
        observation = Observation(observation.input, observation.prefix_answer,
            observation.failures + ((action, tuple(trajectory[3]['answer'])),))
    return {'valid': False, 'transitions': budget, 'actions': actions,
            'answer': list(observation.failures[-1][1])}


def fixed_order(order):
    order = tuple(order)
    if len(order) != len(set(order)) or not order or any(type(a) is not int or not 0 <= a < 7 for a in order):
        raise ValueError('distinct declared actions required')
    def choose(observation):
        return next(a for a in order if a in observation.available)
    return choose


def input_random_order(inp):
    seed = int.from_bytes(hashlib.sha256(b'spectra.failure.random.v1\0'+bytes(inp)).digest()[:8], 'little')
    return tuple(map(int, np.random.default_rng(seed).permutation(7)))


def best_static_order(cases: list[dict]) -> tuple[int, ...]:
    """Exhaustive 210 three-view orders, selected on training folds only."""
    if not cases:
        raise ValueError('static selection requires training cases')
    def score(order):
        outcomes = [simulate(c, fixed_order(order)) for c in cases]
        return (-sum(r['valid'] for r in outcomes), sum(r['transitions'] for r in outcomes), order)
    return min(permutations(range(7), 3), key=score)


def fit_selector(cases: list[dict], mode: str, ridge: float = 10.) -> LinearSelector:
    """Case-balanced ridge scores, trained only on reachable failure histories.

    Each action sees subsets of up to two other failed attempts. Each eligible
    model/example contributes total weight one per action, regardless of the
    number of reachable histories. These scores are not calibrated probabilities.
    """
    if mode not in MODES or type(ridge) not in (int, float) or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('declared mode and positive ridge penalty required')
    if not cases:
        raise ValueError('nonempty training inventory required')
    weights = []
    dimension = {'input_only': 89, 'prefix_only': 169, 'failure_aware': 249}[mode]
    for action in range(7):
        xs, ys, ws = [], [], []
        for case in cases:
            if first_success(case['identity'], 8) is not None:
                continue
            failures = [a for a in range(7) if a != action and first_success(case['views'][a], 4) is None]
            subsets = [s for n in range(3) for s in combinations(failures, n)]
            for subset in subsets:
                obs = Observation(tuple(case['input']), tuple(case['identity'][7]['answer']),
                    tuple((a, tuple(case['views'][a][3]['answer'])) for a in subset))
                xs.append(features(obs, mode))
                ys.append(float(first_success(case['views'][action], 4) is not None))
                ws.append(1/len(subsets))
        if not xs:
            weights.append(np.zeros(dimension))
            continue
        x, y, w = np.asarray(xs), np.asarray(ys), np.asarray(ws)
        gram = x.T @ (w[:, None]*x)
        penalty = np.eye(dimension)*ridge
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(gram+penalty, x.T @ (w*y))
        weights.append(coefficients)
    return LinearSelector(mode, np.stack(weights))


def coverage(cases: list[dict]) -> dict:
    """An optimistic inventory ceiling, with free knowledge of candidate validity."""
    if not cases:
        raise ValueError('nonempty inventory required')
    prefix = np.array([first_success(c['identity'], 8) is not None for c in cases])
    views = np.array([[first_success(v, 4) is not None for v in c['views']] for c in cases])
    identity32 = np.array([first_success(c['identity'], 32) is not None for c in cases])
    union = prefix | views.any(axis=1)
    original = prefix | views[:, :3].any(axis=1)
    return {'pairs': len(cases), 'identity8_valid': int(prefix.sum()),
        'identity32_valid': int(identity32.sum()), 'original_order_valid': int(original.sum()),
        'prefix_plus_all_views_oracle_valid': int(union.sum()),
        'all_trajectories_union_valid': int((union | identity32).sum()),
        'oracle_new_vs_original': int((union & ~original).sum()),
        'oracle_new_vs_identity32': int((union & ~identity32).sum()),
        'oracle_regressions_vs_identity32': int((~union & identity32).sum()),
        'unsolved_by_every_trajectory': int((~union & ~identity32).sum()),
        'valid_views_among_prefix_failures_histogram': np.bincount(views[~prefix].sum(axis=1), minlength=8).tolist(),
        'view_additional_solves_after_prefix': (views & ~prefix[:, None]).sum(axis=0).tolist(),
        'scope': 'Oracle knows untried outcomes for free; upper bound for this fixed inventory, not a deployable policy.'}


def solve_with_selector(core, x, spec, choose, budget=20):
    """Actual B=1 execution, including online features, decisions and checks."""
    import torch
    from eval.checkable_tasks import require_core
    from eval.symmetry_search import grid_views
    if type(budget) is not int or budget not in (12, 16, 20):
        raise ValueError('pilot budgets are 12, 16, or 20 cycles')
    require_core(core, spec)
    spec.validate_input(x, single=True)
    with torch.inference_mode():
        original = spec.native_problem(x)
        views = grid_views(spec)
        work = {'valid': False, 'transitions': 0, 'decodes': 0, 'checks': 0,
            'input_embeddings': 0, 'checker_constructions': 1, 'input_transforms': 0,
            'answer_inverse_transforms': 0, 'restoration_checks': 0,
            'selector_calls': 0, 'actions': [], 'target_used': False, 'requested_cycles': budget}
        observation = None
        for stage in range(1+(budget-8)//4):
            if stage == 0:
                transformed, problem, view = x, original, views[0]
                cycles = 8
            else:
                action = choose(observation)
                if type(action) is not int or action not in observation.available:
                    raise ValueError('selector returned a repeated or invalid action')
                work['selector_calls'] += 1
                work['actions'].append(action)
                view = views[action+1]
                transformed = view.apply(x)
                problem = spec.native_problem(transformed)
                work['input_transforms'] += 1
                work['checker_constructions'] += 1
                cycles = 4
            embedded = core.token_embed(transformed)+core.encode_positions(transformed, spec.height, spec.width)
            y, z = torch.zeros_like(embedded), torch.zeros_like(embedded)
            work['input_embeddings'] += 1
            for _ in range(cycles):
                y, z = core.recursive_cycle(embedded, y, z)
                answer, valid = problem.decode(core.out_head(y).contiguous())
                work['transitions'] += 1
                work['decodes'] += 1
                work['checks'] += 1
                if stage:
                    answer = view.restore(answer)
                    work['answer_inverse_transforms'] += 1
                if valid:
                    if stage:
                        work['checks'] += 1
                        work['restoration_checks'] += 1
                        if not bool(original.check(answer)):
                            raise RuntimeError('restored answer failed original-input verification')
                    work['valid'] = True
                    return answer.clone(), work
            if stage == 0:
                observation = Observation(tuple(x.flatten().tolist()), tuple(answer.flatten().tolist()))
            else:
                observation = Observation(observation.input, observation.prefix_answer,
                    observation.failures+((action, tuple(answer.flatten().tolist())),))
        return answer.clone(), work
