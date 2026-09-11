from dataclasses import replace

import numpy as np
import pytest
import torch

from data import maze, sudoku
from eval.checkable_tasks import SUDOKU_SHIFT, MAZE11, semantic_exit
from eval.symmetry_search import GridView, grid_views, symmetry_solve
from model.trm import TRM
from scripts.m17_models import core_architecture, freeze

SOLUTION = torch.tensor([[1,2,3,4,3,4,1,2,2,1,4,3,4,3,2,1]])
PUZZLE = SOLUTION.clone()
PUZZLE[0,[0,2,5,6,9,11,12,14]] = 0


@pytest.mark.parametrize('spec', [SUDOKU_SHIFT, MAZE11])
def test_maps_are_eight_distinct_bijections_and_roundtrip(spec):
    views = grid_views(spec)
    assert len(views) == len({v.positions for v in views}) == 8
    x = torch.arange(spec.height*spec.width).remainder(spec.num_tokens).reshape(1,-1)
    for v in views:
        assert sorted(v.positions) == list(range(x.numel()))
        assert torch.equal(v.restore(v.apply(x)), x)
        changed = v.apply(x); changed.zero_()
        assert x.count_nonzero() > 0
    assert views[0].name == 'identity'


def test_every_sudoku_view_preserves_exact_validity_and_clues():
    for v in grid_views(SUDOKU_SHIFT):
        p, y = v.apply(PUZZLE), v.apply(SOLUTION)
        assert SUDOKU_SHIFT.independent_correct(p, y)
        assert SUDOKU_SHIFT.independent_correct(PUZZLE, v.restore(y))
        y[0,0] = 0
        assert not SUDOKU_SHIFT.independent_correct(p, y)


@pytest.mark.parametrize('seed', [0, 1, 73])
def test_every_maze_view_preserves_shortest_path_in_both_directions(seed):
    grid,s,g = maze.generate_maze(11,11,np.random.default_rng(seed))
    x = torch.from_numpy(maze.make_input(grid,s,g).reshape(1,-1))
    y = torch.from_numpy(maze.make_target(grid,s,g,maze.shortest_path(grid,s,g)).reshape(1,-1))
    for i,v in enumerate(grid_views(MAZE11)):
        tx,ty = v.apply(x), v.apply(y)
        assert MAZE11.independent_correct(tx,ty)
        assert MAZE11.independent_correct(x,v.restore(ty))
        if i < 4:
            assert torch.equal(tx == maze.START, x == maze.START)
            assert torch.equal(tx == maze.GOAL, x == maze.GOAL)


def test_invalid_maps_and_symbols_are_rejected():
    v = grid_views(SUDOKU_SHIFT)[0]
    with pytest.raises(ValueError, match='D4'):
        replace(v, positions=tuple(reversed(range(15)))+(15,))
    with pytest.raises(ValueError, match='digits'):
        replace(v, labels=(1,0,2,3,4))
    with pytest.raises(ValueError, match='integers'):
        replace(v, labels=(False,1,2,3,4))
    with pytest.raises(ValueError, match='vocabulary'):
        v.apply(torch.full((1,16),5))
    with pytest.raises(ValueError, match='vocabulary'):
        v.restore(torch.full((1,16),-1))
    with pytest.raises(ValueError, match='square'):
        grid_views(replace(MAZE11,width=13))


@pytest.fixture
def core():
    torch.manual_seed(481)
    return freeze(TRM(**core_architecture(SUDOKU_SHIFT)))


@pytest.mark.parametrize('policy', ['identity','repeat_identity','dihedral'])
def test_identity_prefix_matches_unchanged_runtime(core, policy):
    before = {k:v.clone() for k,v in core.state_dict().items()}
    answer,work = symmetry_solve(core,PUZZLE,SUDOKU_SHIFT,policy=policy,transition_budget=4)
    reference,rw = semantic_exit(core,PUZZLE,SUDOKU_SHIFT,4)
    assert torch.equal(answer,reference)
    assert work['valid'] == rw['valid']
    assert work['transitions'] == rw['transitions'] <= 4
    assert work['input_embeddings'] == 1
    assert all(torch.equal(v,core.state_dict()[k]) for k,v in before.items())
    assert not work['target_used'] and work['value_calls'] == 0


@pytest.mark.parametrize('budget', [0, -1, 257, True, 1.0])
def test_bad_budget_rejected_before_native_work(core,budget):
    with pytest.raises(ValueError,match='transition_budget'):
        symmetry_solve(core,PUZZLE,SUDOKU_SHIFT,transition_budget=budget)


def test_repeated_identity_cannot_invent_new_solution(core):
    a,w = symmetry_solve(core,PUZZLE,SUDOKU_SHIFT,policy='identity',transition_budget=4)
    b,v = symmetry_solve(core,PUZZLE,SUDOKU_SHIFT,policy='repeat_identity',transition_budget=32)
    assert torch.equal(a,b) and w['valid'] == v['valid']
    assert v['transitions'] == (w['transitions'] if w['valid'] else 32)
    if not v['valid']:
        assert len(v['views_started']) == 8
        assert v['checker_constructions'] == 1


def test_valid_prefix_stops_before_any_later_views(core):
    a,w = symmetry_solve(core,SOLUTION,SUDOKU_SHIFT,transition_budget=32)
    assert torch.equal(a,SOLUTION) and w['valid'] and w['transitions'] == 1
    assert w['views_started'] == ['identity'] and w['input_transforms'] == 0
    with torch.inference_mode():
        a.zero_()
    assert SOLUTION.count_nonzero() == 16


def test_no_silent_view_exhaustion_and_invalid_policy(core):
    with pytest.raises(ValueError,match='eight'):
        symmetry_solve(core,PUZZLE,SUDOKU_SHIFT,transition_budget=33)
    with pytest.raises(ValueError,match='unknown'):
        symmetry_solve(core,PUZZLE,SUDOKU_SHIFT,policy='learned')
    with pytest.raises(ValueError,match='cycles_per_view'):
        symmetry_solve(core,PUZZLE,SUDOKU_SHIFT,cycles_per_view=0)


def test_extended_identity_prefix_matches_unchanged_eight_cycle_reference(core):
    a,w=symmetry_solve(core,PUZZLE,SUDOKU_SHIFT,transition_budget=8,identity_cycles=8,view_limit=1)
    b,r=semantic_exit(core,PUZZLE,SUDOKU_SHIFT,8)
    assert torch.equal(a,b) and w['valid']==r['valid'] and w['transitions']==r['transitions']
    assert w['views_started']==['identity']


def test_refined_policy_never_exceeds_its_twenty_cycle_or_four_view_cap(core):
    _,w=symmetry_solve(core,PUZZLE,SUDOKU_SHIFT,transition_budget=20,identity_cycles=8,view_limit=4)
    assert w['transitions']<=20 and len(w['views_started'])<=4
    assert w['views_started']==[v.name for v in grid_views(SUDOKU_SHIFT)[:len(w['views_started'])]]


@pytest.mark.parametrize('kwargs',[{'identity_cycles':True},{'identity_cycles':0},{'view_limit':0},
    {'view_limit':9},{'view_limit':True},{'policy':'identity','identity_cycles':8}])
def test_invalid_new_schedule_options_rejected(core,kwargs):
    with pytest.raises(ValueError):symmetry_solve(core,PUZZLE,SUDOKU_SHIFT,**kwargs)
