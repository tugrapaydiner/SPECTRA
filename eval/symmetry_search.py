"""Bounded CPU test-time grid symmetries with exact answer verification.

This is a diagnostic proposal policy, not a claim of novel search or superiority.
The identity view is first. A shared identity prefix retains its valid incumbent.
Transforms are exact task automorphisms; learned inference need not be equivariant.
No reference answer, quality predictor, parameter update or GPU enters the policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import torch

from eval.checkable_tasks import TaskSpec, require_core

Policy = Literal['identity', 'repeat_identity', 'dihedral']
POLICY_ID = 'square_d4_endpoint_swap_budgeted_v2'


@lru_cache(maxsize=16)
def _position_maps(side: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if type(side) is not int or side < 2:
        raise ValueError('square side must be an integer >= 2')
    a = np.arange(side*side).reshape(side, side)
    # Freeze this order before inspecting the experimental outcomes. The first
    # four preserve the two diagonal corners as a set. Maze endpoint swaps in
    # views 2 and 3 preserve the generator's START/GOAL convention as well.
    arrays = [('identity', a), ('transpose', a.T), ('rotate180', a[::-1, ::-1]),
              ('anti_transpose', a.T[::-1, ::-1]), ('rotate90', np.rot90(a)),
              ('rotate270', np.rot90(a, 3)), ('flip_horizontal', a[:, ::-1]),
              ('flip_vertical', a[::-1, :])]
    return tuple((name, tuple(map(int, p.ravel()))) for name, p in arrays)


@dataclass(frozen=True)
class GridView:
    """An immutable, invertible spatial/symbol map with a task-specific contract."""
    spec: TaskSpec
    name: str
    positions: tuple[int, ...]  # transformed position -> original position
    labels: tuple[int, ...]     # original symbol -> transformed symbol

    def __post_init__(self):
        if not isinstance(self.spec, TaskSpec) or self.spec.height != self.spec.width:
            raise ValueError('square TaskSpec required')
        allowed = dict(_position_maps(self.spec.width))
        if self.name not in allowed or self.positions != allowed[self.name]:
            raise ValueError('view must be an explicitly declared D4 spatial map')
        if type(self.positions) is not tuple or type(self.labels) is not tuple:
            raise TypeError('view positions and labels must be immutable tuples')
        if any(type(v) is not int for v in (*self.positions, *self.labels)):
            raise ValueError('view entries must be integers, not booleans')
        identity = tuple(range(self.spec.num_tokens))
        allowed_labels = (identity,) if self.spec.task == 'sudoku' else (identity, (0, 1, 3, 2, 4))
        if self.labels not in allowed_labels:
            raise ValueError('only identity digits or an exact maze endpoint swap is supported')

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        self.spec.validate_input(x)
        if not bool(((x >= 0) & (x < self.spec.num_tokens)).all()):
            raise ValueError('input symbols outside vocabulary')
        p = torch.tensor(self.positions, dtype=torch.int64)
        labels = torch.tensor(self.labels, dtype=torch.int64)
        return labels[x.index_select(1, p)].contiguous()

    def restore(self, answer: torch.Tensor) -> torch.Tensor:
        self.spec.validate_input(answer)
        if not bool(((answer >= 0) & (answer < self.spec.num_tokens)).all()):
            raise ValueError('answer symbols outside vocabulary')
        p = torch.argsort(torch.tensor(self.positions, dtype=torch.int64))
        labels = torch.argsort(torch.tensor(self.labels, dtype=torch.int64))
        return labels[answer.index_select(1, p)].contiguous()

    def metadata(self) -> dict:
        return {'name': self.name, 'positions': list(self.positions), 'labels': list(self.labels)}


def grid_views(spec: TaskSpec) -> tuple[GridView, ...]:
    if spec.height != spec.width:
        raise ValueError('the D4 policy requires a square task')
    return tuple(GridView(spec, name, pos,
        (0, 1, 3, 2, 4) if spec.task == 'maze' and name in ('rotate180', 'anti_transpose')
        else tuple(range(spec.num_tokens))) for name, pos in _position_maps(spec.width))


@torch.inference_mode()
def symmetry_solve(core, x: torch.Tensor, spec: TaskSpec, *, policy: Policy = 'dihedral',
                   transition_budget: int = 32, cycles_per_view: int = 4,
                   identity_cycles: int | None = None, view_limit: int = 8):
    """Complete sequential B=1 solve. All transformation/checker work is online.

    The cap counts recursive_cycle calls, not elapsed time or FLOPs. Different
    views incur real transformation, embedding and checker construction costs.
    Do not call equal transition limits equal wall-clock budgets. The algorithm
    stores only the current trajectory and candidate, never all 32 states.
    """
    require_core(core, spec)
    spec.validate_input(x, single=True)
    if policy not in ('identity', 'repeat_identity', 'dihedral'):
        raise ValueError('unknown proposal policy')
    if type(transition_budget) is not int or not 1 <= transition_budget <= 256:
        raise ValueError('transition_budget must be an integer in [1,256]')
    if type(cycles_per_view) is not int or not 1 <= cycles_per_view <= 256:
        raise ValueError('cycles_per_view must be an integer in [1,256]')
    if identity_cycles is not None and (type(identity_cycles) is not int or not 1 <= identity_cycles <= 256):
        raise ValueError('identity_cycles must be None or an integer in [1,256]')
    if type(view_limit) is not int or not 1 <= view_limit <= 8:
        raise ValueError('view_limit must be an integer in [1,8]')
    if policy != 'dihedral' and (identity_cycles is not None or view_limit != 8):
        raise ValueError('view scheduling arguments require the dihedral policy')
    prefix_cycles = cycles_per_view if identity_cycles is None else identity_cycles
    if policy == 'dihedral' and transition_budget > prefix_cycles+(view_limit-1)*cycles_per_view:
        raise ValueError('requested budget exceeds the declared grid views (at most eight)')
    views = grid_views(spec)[:view_limit]
    original_problem = spec.native_problem(x)
    count = 0
    work = {'valid': False, 'transitions': 0, 'decodes': 0, 'checks': 0,
            'restoration_checks': 0, 'input_embeddings': 0, 'checker_constructions': 1,
            'input_transforms': 0, 'answer_inverse_transforms': 0, 'value_calls': 0,
            'policy': policy, 'policy_id': POLICY_ID, 'requested_cycles': transition_budget,
            'cycles_per_view': cycles_per_view, 'identity_cycles': prefix_cycles,
            'view_limit': view_limit, 'views_started': [], 'target_used': False}
    last = None
    while count < transition_budget:
        view_i = len(work['views_started']) if policy == 'dihedral' else 0
        view = views[view_i]
        work['views_started'].append(view.name)
        if view_i == 0:
            transformed = x
            problem = original_problem
        else:
            transformed = view.apply(x)
            work['input_transforms'] += 1
            problem = spec.native_problem(transformed)
            work['checker_constructions'] += 1
        embedded = core.token_embed(transformed) + core.encode_positions(transformed, spec.height, spec.width)
        work['input_embeddings'] += 1
        y, z = torch.zeros_like(embedded), torch.zeros_like(embedded)
        steps = (transition_budget if policy == 'identity' else
                 prefix_cycles if policy == 'dihedral' and view_i == 0 else cycles_per_view)
        for _ in range(min(steps, transition_budget-count)):
            y, z = core.recursive_cycle(embedded, y, z)
            answer, valid = problem.decode(core.out_head(y).contiguous())
            count += 1
            work['transitions'] = work['decodes'] = work['checks'] = count
            if view_i:
                answer = view.restore(answer)
                work['answer_inverse_transforms'] += 1
            last = answer
            if valid:
                # Do not rely only on the transform proof in a deployed call.
                # Check the returned answer against the original input as well.
                if view_i:
                    work['restoration_checks'] += 1
                    work['checks'] += 1
                    if not bool(original_problem.check(answer)):
                        raise RuntimeError('transformed valid solution failed original-input verification')
                work.update(valid=True, stop_reason='valid_answer',
                            block_applications=2*len(core.blocks)*count)
                return answer.clone(), work
    work.update(stop_reason='transition_budget', block_applications=2*len(core.blocks)*count)
    return last.clone(), work
