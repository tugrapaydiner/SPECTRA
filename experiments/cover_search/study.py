"""Prospective matrix, solver adapters, and source-bound complete-job measurements."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from data.cnf import SplitMix64
from spectra.cnf.cover import solve_cover
from spectra.cnf.dpll import solve_dpll
from .tasks import build,cnf,check,colouring

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE_COMMIT='eabcc9c5a8ebe0beee364da3ecd1a68bedeb7253'
ARMS=('cover','dense','cover_cnf','dpll','minicard','gluecard4','glucose42','cadical300','kissat404')
ROUNDS=3
MAX_NODES=1000000
MAX_STATE_BYTES=64*1024*1024
MAX_INDEX_BYTES=64*1024*1024
DEADLINE_SECONDS=1.0
MASTER_SEED=0x9AF490C711DE7301
SOURCE_PATHS=('experiments/cover_search/BASELINE_SHA256.json','spectra/cnf/cover.py','spectra/_native/cover_search.cpp',
              'experiments/cover_search/tasks.py','experiments/cover_search/study.py',
              'experiments/cover_search/run.py','experiments/cover_search/analyse.py',
              'experiments/cover_search/PROTOCOL.md','tests/public/test_cover_search.py','tests/public/test_cover_study.py',
              'data/cnf.py','spectra/cnf/dpll.py',
              'experiments/structured_search/task.py','experiments/structured_search/study.py',
              'experiments/structured_search/UPSTREAM.json',
              'experiments/structured_search/upstream/sudoku.py',
              'experiments/structured_search/upstream/sudoku-top95.txt')


def require(test,message):
    if not test:raise ValueError(message)


def digest(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def graph_signature(case):
    """Unlabelled colour-refinement invariant, not a complete isomorphism test.

    Distinct signatures exclude isomorphism (subject to SHA256 collision
    resistance). Equal signatures are conservatively one dependent group.
    """
    n=case['n'];adj=[[] for _ in range(n)]
    for a,b in case['edges']:adj[a].append(b);adj[b].append(a)
    colours=[str(len(row)) for row in adj];history=[]
    for _ in range(n):
        colours=[digest([colours[i],sorted(colours[j] for j in adj[i])]) for i in range(n)]
        history.append(sorted(colours))
    return digest([n,case['colours'],history])


def make_cases():
    # Existing public rows are a compatibility panel, never new confirmation.
    from experiments.structured_search.study import cases as old_cases
    result=[{'id':c['id'],'kind':'sudoku','puzzle':c['puzzle'],
             'stratum':'sudoku_public','role':'exposed_compatibility'} for c in old_cases('evaluation')]
    rng=SplitMix64(MASTER_SEED)
    for n in (18,36,72):
        for planted in (False,True):
            for index in range(24):
                seed=rng.below(2**64)
                case=colouring(n,3,seed,planted=planted)
                case.update(id=f'graph:{n}:{int(planted)}:{index:02}',
                            stratum=f'graph:{n}:{int(planted)}',role='prospective_synthetic')
                case['group']=graph_signature(case)
                result.append(case)
    for case in result:
        case.setdefault('group',case['id'])
        case['sha256']=digest(case)
    # Reject overlap with actual development graphs; do not silently replace it.
    dev={graph_signature(colouring(n,3,100+n+j,planted=p))
         for n in (18,36,72) for p in (False,True) for j in range(2)}
    require(not (dev & {c['group'] for c in result if c['kind']=='colouring'}),'development structural overlap')
    require(len({c['group'] for c in result})==len(result),'dependent prospective graph groups; protocol requires review')
    return result


def arms(case):return ARMS+(('norvig',) if case['kind']=='sudoku' else ())


def schedule(cases):
    jobs=[(case['id'],arm,r) for r in range(ROUNDS) for case in cases for arm in arms(case)]
    rng=SplitMix64(0x81A397B8)
    for i in range(len(jobs)-1,0,-1):
        j=rng.below(i+1);jobs[i],jobs[j]=jobs[j],jobs[i]
    return jobs


def check_freeze():
    freeze=json.loads((HERE/'FREEZE.json').read_text())
    require(set(freeze['sources'])==set(SOURCE_PATHS),'incomplete source freeze')
    for name,sha in freeze['sources'].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,'frozen source changed: '+name)
    baseline=json.loads((HERE/'BASELINE_SHA256.json').read_text())
    for name,sha in baseline.items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,'baseline modified: '+name)
    return freeze


def load_controls():
    import pysat
    require(pysat.__version__=='1.9.dev15','unexpected python-sat version')
    from pysat.solvers import Solver
    from experiments.structured_search.study import domain_solver
    return Solver,domain_solver()


def attempt(case,arm,library,controls=None):
    require(arm in arms(case),'unknown arm')
    Solver,norvig=controls if controls else load_controls()
    start=time.perf_counter_ns();cpu=time.process_time_ns()
    details={};witness=None;status='UNKNOWN'
    if arm=='norvig':
        values=norvig.solve(case['puzzle'])
        if values:
            grid=''.join(values[s] for s in norvig.squares)
            witness=tuple(grid[i]==str(d+1) for i in range(81) for d in range(9))
            status='SAT_VERIFIED'
        encoding_ns=0
    else:
        p=build(case)
        f=cnf(p) if arm in ('cover_cnf','dpll','glucose42','cadical300','kissat404') else None
        encoding_ns=time.perf_counter_ns()-start
        if arm in ('cover','dense','cover_cnf'):
            r=solve_cover(f if f is not None else p,library,max_nodes=MAX_NODES,
                          max_index_bytes=MAX_INDEX_BYTES,max_state_bytes=MAX_STATE_BYTES,
                          incremental=arm!='dense')
            witness=r.witness;status=r.status;details=r.record();details.pop('witness');details.pop('unsatisfied')
        elif arm=='dpll':
            r=solve_dpll(f,max_decisions=MAX_NODES)
            witness=r.witness;status=r.status;details=r.record();details.pop('witness');details.pop('unsatisfied',None)
        else:
            with Solver(name=arm,bootstrap_with=p.covers if arm in ('minicard','gluecard4') else f.clauses) as solver:
                if arm in ('minicard','gluecard4'):
                    for group in p.exclusive:solver.add_atmost(group,1)
                answer=solver.solve()
                model=solver.get_model() if answer is True else None
                details={'stats':None if arm=='kissat404' else solver.accum_stats(),
                         'stats_available':arm!='kissat404','unsat_proof_checked':False}
            status='SAT_VERIFIED' if answer is True else 'UNSAT_REPORTED' if answer is False else 'UNKNOWN'
            if model is not None:
                positive={lit for lit in model if lit>0}
                witness=tuple(i+1 in positive for i in range(p.nvars))
                require(not (p if arm in ('minicard','gluecard4') else f).violated(witness),'invalid original-constraint witness')
        del p,f
    if status=='SAT_VERIFIED':require(check(case,witness),'invalid original-task witness')
    # Includes creating fresh answer/diagnostic structures. Serialization/I/O outside.
    result={'status':status,'witness':list(witness) if witness is not None else None,
            'details':details,'encoding_ns':encoding_ns}
    result['cpu_ns']=time.process_time_ns()-cpu
    result['elapsed_ns']=time.perf_counter_ns()-start
    return result
