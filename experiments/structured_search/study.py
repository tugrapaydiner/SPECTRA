"""Task adapters and immutable dataset split for the structured-search study."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import time

from spectra.cnf import solve_focused, solve_deductive
from spectra.cnf.dpll import solve_dpll
from .task import check_grid, check_cnf, encode, decode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ('focused', 'deductive', 'dpll', 'glucose4', 'norvig')
ROUNDS = 3
MAX_DECISIONS = 2048
SOURCE_PATHS = (
    'data/__init__.py', 'data/cnf.py', 'eval/__init__.py', 'eval/cnf_cached.py',
    'spectra/__init__.py', 'spectra/cnf/__init__.py', 'spectra/cnf/dimacs.py',
    'spectra/cnf/state.py', 'spectra/cnf/search.py', 'spectra/cnf/ranked.py',
    'spectra/cnf/indexed.py', 'spectra/cnf/deductive.py', 'spectra/cnf/focused.py',
    'spectra/cnf/dpll.py',
    *('experiments/structured_search/'+name for name in (
        'study.py', 'task.py', 'run.py', 'analyse.py', 'PROTOCOL.md',
        'DEVELOPMENT_PLAN.md', 'UPSTREAM.json', 'upstream/sudoku.py',
        'upstream/sudoku-top95.txt', 'upstream/LICENSE')),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_upstream():
    manifest=json.loads((HERE/'UPSTREAM.json').read_text())
    for name, expected in manifest['files'].items():
        data=(HERE/'upstream'/name).read_bytes()
        require(len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'], 'upstream identity mismatch')
    return manifest


def cases(split):
    require(split in ('development', 'evaluation'), 'unknown split')
    load_upstream()
    puzzles=(HERE/'upstream/sudoku-top95.txt').read_text().splitlines()
    require(len(puzzles)==95 and len(set(puzzles))==95, 'unexpected public corpus')
    start, stop = (0,10) if split=='development' else (10,95)
    return [{'id':f'top95:{i+1:02}', 'row':i+1, 'puzzle':puzzles[i],
             'sha256':hashlib.sha256(puzzles[i].encode('ascii')).hexdigest()} for i in range(start,stop)]


def native_and_domain():
    import pysat
    from pysat.solvers import Solver
    require(pysat.__version__=='1.9.dev15', 'unexpected python-sat version')
    return Solver,domain_solver()


def domain_solver():
    spec=importlib.util.spec_from_file_location('norvig_original',HERE/'upstream/sudoku.py')
    norvig=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(norvig)
    return norvig


def call(case, arm, native, norvig):
    require(arm in ARMS, 'unknown arm')
    puzzle=case['puzzle']; start=time.perf_counter_ns()
    if arm=='norvig':
        values=norvig.solve(puzzle)
        grid=''.join(values[s] for s in norvig.squares) if values else None
        result={'status':'SAT_VERIFIED' if grid else 'UNKNOWN'}
        encode_ns=0
        solve_ns=time.perf_counter_ns()-start
    else:
        problem=encode(puzzle)
        encode_ns=time.perf_counter_ns()-start
        search_start=time.perf_counter_ns()
        if arm=='glucose4':
            with native(name='glucose4',bootstrap_with=[list(c) for c in problem.clauses]) as solver:
                solver.conf_budget(2000)
                answer=solver.solve_limited()
                model=solver.get_model() if answer is True else None
                stats=solver.accum_stats()
            positive={lit for lit in model if lit>0} if model is not None else set()
            witness=tuple(i+1 in positive for i in range(problem.nvars))
            result={'status':'SAT_VERIFIED' if answer is True else 'UNSAT_REPORTED' if answer is False else 'UNKNOWN',
                    'stats':stats,'conflict_budget_requested':2000}
        else:
            r=(solve_dpll(problem,max_decisions=MAX_DECISIONS) if arm=='dpll' else
               (solve_focused if arm=='focused' else solve_deductive)(problem,seed=17,max_flips=2048))
            result=r.record();witness=r.witness
        solve_ns=time.perf_counter_ns()-search_start
        grid=None
        if result['status']=='SAT_VERIFIED':
            require(check_cnf(problem,witness),'invalid original-CNF witness')
            grid=decode(witness)
        # Every raw record retains the independent task answer, not only a SAT label.
        del problem
    if grid is not None:
        require(check_grid(puzzle,grid),'invalid original-grid solution')
    result['grid']=grid
    return {'result':result,'encoding_ns':encode_ns,'solve_ns':solve_ns}


def semantic(record):
    return {k:v for k,v in record.items() if k!='elapsed_ns'}


def check_freeze(freeze):
    require(set(freeze['sources'])==set(SOURCE_PATHS), 'incomplete frozen source inventory')
    for path,digest in freeze['sources'].items():
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest, 'frozen source mismatch: '+path)


def append(stream, row):
    import os
    stream.write(json.dumps(row, separators=(',', ':'))+'\n')
    stream.flush()
    os.fsync(stream.fileno())
