"""Independent truth tables, backtracking and strict branch-budget contracts."""
import itertools
import random

import pytest

from data.cnf import CNF
from spectra.cnf.dpll import solve_dpll
from experiments.structured_search.task import check_grid, check_cnf, encode, decode


def satisfies(problem, witness):
    return all(any(witness[abs(lit)-1] == (lit > 0) for lit in clause) for clause in problem.clauses)


def test_exhaustive_two_variable_formulas_agree_with_truth_table():
    clauses = [(), (1,), (-1,), (2,), (-2,), (1, 2), (1, -2), (-1, 2), (-1, -2)]
    for mask in range(512):
        p = CNF(2, tuple(c for i, c in enumerate(clauses) if mask & (1 << i)))
        expected = any(satisfies(p, w) for w in itertools.product((False, True), repeat=2))
        r = solve_dpll(p, max_decisions=32)
        assert (r.status == 'SAT_VERIFIED') == expected
        assert satisfies(p, r.witness) == expected
        assert r.decisions <= 32


def test_random_noncanonical_formulas_against_brute_force():
    rng = random.Random(202610056201)
    for _ in range(800):
        p = CNF(5, tuple(tuple(rng.choice((-5,-4,-3,-2,-1,1,2,3,4,5))
                              for _ in range(rng.randrange(1, 6))) for _ in range(rng.randrange(1, 25))))
        expected = any(satisfies(p, w) for w in itertools.product((False, True), repeat=5))
        r = solve_dpll(p, max_decisions=128)
        assert (r.status == 'SAT_VERIFIED') == expected
        assert r.unsatisfied == tuple(i for i,c in enumerate(p.clauses) if not any(r.witness[abs(l)-1] == (l>0) for l in c))


@pytest.mark.parametrize('limit', [0, 1, 2, 3, 7])
def test_both_branch_polarities_charge_budget_and_repeat_deterministically(limit):
    p = CNF(3, ((1,2),(-1,2),(1,-2),(-1,-2),(3,)))
    a, b = solve_dpll(p, max_decisions=limit), solve_dpll(p, max_decisions=limit)
    assert 0 <= a.decisions <= limit
    assert a.status == 'UNKNOWN'
    assert a.reason in ('budget', 'exhausted')
    assert {k:v for k,v in a.record().items() if k != 'elapsed_ns'} == {k:v for k,v in b.record().items() if k != 'elapsed_ns'}
    assert a.record()['learned'] is False


def test_zero_decisions_still_propagates_and_handles_empty_and_tautological_clauses():
    for p, status in [(CNF(0,()),'SAT_VERIFIED'), (CNF(0,((),)),'UNKNOWN'),
                      (CNF(2,((1,-1),(2,2))), 'SAT_VERIFIED')]:
        assert solve_dpll(p,max_decisions=0).status == status
    chain=CNF(1500, ((1,),)+tuple((-i,i+1) for i in range(1,1500)))
    r=solve_dpll(chain,max_decisions=0)
    assert r.status=='SAT_VERIFIED' and all(r.witness) and r.propagations==1500


def test_decision_depth_does_not_use_python_recursion():
    p=CNF(2200,tuple((2*i+1,2*i+2) for i in range(1100)))
    r=solve_dpll(p,max_decisions=1100)
    assert r.status=='SAT_VERIFIED' and r.decisions==1100


@pytest.mark.parametrize('budget',[True,-1,1.5,None])
def test_invalid_budgets(budget):
    with pytest.raises(ValueError): solve_dpll(CNF(1,((1,),)),max_decisions=budget)


def test_non_cnf_rejected():
    with pytest.raises(TypeError): solve_dpll([[1]])


def test_task_encoding_and_checker_accept_exact_solution_and_reject_wrong_cells():
    grid=''.join(str((3*r+r//3+c)%9+1) for r in range(9) for c in range(9))
    puzzle=''.join(v if i%3 else '.' for i,v in enumerate(grid))
    p=encode(puzzle)
    witness=tuple(int(grid[i//9])==i%9+1 for i in range(729))
    assert check_grid(puzzle,grid) and check_cnf(p,witness) and decode(witness)==grid
    result=solve_dpll(encode(grid),max_decisions=0)
    assert result.status=='SAT_VERIFIED' and decode(result.witness)==grid
    for i in range(81):
        wrong=grid[:i]+str(int(grid[i])%9+1)+grid[i+1:]
        bad=tuple(int(wrong[j//9])==j%9+1 for j in range(729))
        assert not check_grid(puzzle,wrong) and not check_cnf(p,bad)
    with pytest.raises(ValueError): decode([False]*729)
    with pytest.raises(ValueError): encode('0'*80)
