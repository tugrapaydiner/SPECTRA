"""Synthetic clock/counter records exercise contracts, not performance evidence."""
import copy
import pytest
from data.cnf import CNF
from .external import SOLVERS,ROUNDS,CONFLICTS,WATCHDOG,VERSION,summarize


def fixture():
    p=CNF(1,((1,),))
    cfg={'families':['unit'],'eval_sizes':[1],'wall_budget_ms':[2,10]}
    case={'id':'fixture','formula':p.record(),'family':'unit','nvars':1,'ordered_sha256':p.sha256()}
    rows=[]
    for s in SOLVERS:
        for r in range(ROUNDS):
            rows.append({'case_id':case['id'],'ordered_sha256':p.sha256(),'solver':s,'round':r,
              'status':'SAT_VERIFIED','witness':[True],'conflict_budget_requested':CONFLICTS,'watchdog_seconds':WATCHDOG,
              'python_sat_version':VERSION,'supervised_ns':1000,'complete_ns':40,
              'setup_ns':10,'solver_call_ns':10,'extract_count_delete_ns':10,'independent_check_ns':10,
              'counters':{'conflicts':3000,'decisions':10,'propagations':20,'restarts':1}})
    return cfg,[case],rows


def test_explicit_overshoot_is_not_censored():
    cfg,items,rows=fixture();r=summarize(cfg,items,rows,[])
    assert r['statuses']=={'SAT_VERIFIED':4}
    assert all(s['budget_overshoot_rows']==2 and s['max_actual_conflicts']==3000 for s in r['strata'])

@pytest.mark.parametrize('damage',['duplicate','missing','witness','status','unsat_contradiction','unknown_witness','counter','time','budget','hash','nonboolean_witness'])
def test_external_evidence_fails_closed(damage):
    cfg,items,rows=fixture();rows=copy.deepcopy(rows);hybrid=[]
    if damage=='duplicate':rows.append(rows[0])
    elif damage=='missing':rows.pop()
    elif damage=='witness':rows[0]['witness']=[False]
    elif damage=='status':rows[0]['status']='ERROR'
    elif damage=='unsat_contradiction':
        rows[0]['status']='UNSAT_REPORTED';rows[0]['witness']=None
    elif damage=='unknown_witness':rows[0]['status']='UNKNOWN_BUDGET'
    elif damage=='counter':rows[0]['counters']['conflicts']=-1
    elif damage=='time':rows[0]['complete_ns']=1001
    elif damage=='budget':rows[0]['conflict_budget_requested']=999
    elif damage=='hash':rows[0]['ordered_sha256']='0'*64
    else:rows[0]['witness']=[1]
    with pytest.raises(ValueError):summarize(cfg,items,rows,hybrid)


def test_unknown_and_timeout_receive_no_invented_witness_credit():
    cfg,items,rows=fixture()
    for r in rows:r['status']='UNKNOWN_BUDGET';r['witness']=None
    rows[0].update(status='TIMEOUT',complete_ns=None)
    out=summarize(cfg,items,rows,[])
    assert out['statuses']=={'TIMEOUT':1,'UNKNOWN_BUDGET':3}
    assert all(v==0 for budgets in out['all_within_cutoff_rates'].values() for v in budgets.values())


@pytest.mark.parametrize('native_result', [True, False, None])
def test_worker_live_payload_matches_json_representation(monkeypatch, native_result):
    # Exercise the live multiprocessing payload, not only JSON fixtures. The
    # first CI attempt exposed a tuple/list mismatch after all 384 real runs.
    import json
    import sys
    import types
    from . import external

    class SolverDouble:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def conf_budget(self, budget): assert budget == CONFLICTS
        def solve_limited(self): return native_result
        def get_model(self): return [1]
        def accum_stats(self):
            return {'conflicts': 0, 'decisions': 1, 'propagations': 1, 'restarts': 0}

    class Pipe:
        def __init__(self): self.value = None; self.closed = False
        def send(self, value): self.value = value
        def close(self): self.closed = True

    package = types.ModuleType('pysat')
    module = types.ModuleType('pysat.solvers')
    module.Solver = SolverDouble
    package.solvers = module
    monkeypatch.setitem(sys.modules, 'pysat', package)
    monkeypatch.setitem(sys.modules, 'pysat.solvers', module)
    monkeypatch.setattr(external.importlib.metadata, 'version', lambda name: VERSION)
    pipe = Pipe()
    problem = CNF(1, ((1,),))
    external.worker(pipe, problem.record(), SOLVERS[0])
    expected = 'SAT_VERIFIED' if native_result is True else 'UNSAT_REPORTED' if native_result is False else 'UNKNOWN_BUDGET'
    assert pipe.closed and pipe.value['status'] == expected
    assert pipe.value == json.loads(json.dumps(pipe.value, allow_nan=False))
    assert pipe.value['witness'] == ([True] if native_result is True else None)
    if native_result is True:
        assert type(pipe.value['witness']) is list
