"""Pinned default CDCL context on the SAME declared development input inventory.

Native solver budgets are requests, not equal work. The 2/10ms figures are
retrospective complete-witness delivery cutoffs, not enforced CDCL deadlines.
A separate three-second subprocess watchdog contains long native calls. Solver
construction, clause conversion/insertion, solve, extraction, counters, deletion
and independent witness validation are timed; worker import/startup is separate.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
from data.cnf import CNF
from eval.sat_workload import model_to_witness,independent_check
from .data import cases,write_json
from .experiment import sha,seal,environment

VERSION='1.9.dev15'
SOLVERS=('cadical195','glucose4')
ROUNDS=2
CONFLICTS=2000
WATCHDOG=3.0


def source_hashes():
    here=Path(__file__).resolve()
    return {'external.py':sha(here.read_bytes()),'legacy/eval/sat_workload.py':sha((here.parents[2]/'eval/sat_workload.py').read_bytes()),
            'legacy/data/cnf.py':sha((here.parents[2]/'data/cnf.py').read_bytes())}


def worker(pipe,record,solver):
    try:
        version=importlib.metadata.version('python-sat')
        if version!=VERSION:raise ValueError('wrong python-sat version: '+version)
        from pysat.solvers import Solver
        formula=CNF.from_record(record)
        start=time.perf_counter_ns()
        with Solver(name=solver,bootstrap_with=[list(c) for c in formula.clauses]) as s:
            s.conf_budget(CONFLICTS)
            setup_end=time.perf_counter_ns()
            result=s.solve_limited()
            solve_end=time.perf_counter_ns()
            if result is not None and type(result) is not bool:raise ValueError('unexpected native status')
            witness=model_to_witness(formula,s.get_model()) if result is True else None
            counters=s.accum_stats()
        deletion_end=time.perf_counter_ns()
        if witness is not None and not formula.satisfied(witness):raise ValueError('false SAT witness')
        complete_end=time.perf_counter_ns()
        pipe.send({'status':'SAT_VERIFIED' if result is True else 'UNSAT_REPORTED' if result is False else 'UNKNOWN_BUDGET',
          'witness':witness,'setup_ns':setup_end-start,'solver_call_ns':solve_end-setup_end,
          'extract_count_delete_ns':deletion_end-solve_end,'independent_check_ns':complete_end-deletion_end,
          'complete_ns':complete_end-start,'counters':counters,'python_sat_version':version})
    except Exception:
        pipe.send({'status':'ERROR','error':traceback.format_exc(),'witness':None,'complete_ns':None})
    finally:pipe.close()


def run_one(case,solver,round_id):
    ctx=mp.get_context('spawn');read,write=ctx.Pipe(duplex=False)
    process=ctx.Process(target=worker,args=(write,case['formula'],solver))
    start=time.perf_counter_ns()
    try:
        process.start();write.close()
        remaining=max(0.,WATCHDOG-(time.perf_counter_ns()-start)/1e9)
        if read.poll(remaining):
            try:r=read.recv()
            except EOFError:r={'status':'ERROR','error':'worker EOF','witness':None,'complete_ns':None}
        else:r={'status':'TIMEOUT','witness':None,'complete_ns':None}
    finally:
        write.close();read.close()
        if process.pid is not None:
            process.join(timeout=.1)
            if process.is_alive():process.terminate();process.join(timeout=1)
            if process.is_alive():process.kill();process.join()
            process.close()
    r.update(case_id=case['id'],ordered_sha256=case['ordered_sha256'],solver=solver,round=round_id,
       conflict_budget_requested=CONFLICTS,watchdog_seconds=WATCHDOG,supervised_ns=time.perf_counter_ns()-start)
    return r


def summarize(cfg,items,rows,hybrid):
    lookup={c['id']:c for c in items}
    expected={(c['id'],s,r) for c in items for s in SOLVERS for r in range(ROUNDS)}
    found=set()
    for row in rows:
        key=row['case_id'],row['solver'],row['round']
        if key in found or key not in expected or type(row['round']) is not int:raise ValueError('external inventory mismatch')
        found.add(key);c=lookup[row['case_id']];p=CNF.from_record(c['formula']);status=row['status']
        if row['ordered_sha256']!=p.sha256() or row['conflict_budget_requested']!=CONFLICTS or row['watchdog_seconds']!=WATCHDOG:raise ValueError('external input/configuration mismatch')
        if type(row['supervised_ns']) is not int or row['supervised_ns']<=0:raise ValueError('bad supervising duration')
        if status not in ('SAT_VERIFIED','UNSAT_REPORTED','UNKNOWN_BUDGET','TIMEOUT'):raise ValueError('execution error is failed evidence')
        if status=='TIMEOUT':
            if row['complete_ns'] is not None or row['witness'] is not None:raise ValueError('fabricated timeout measurement')
        else:
            if row['python_sat_version']!=VERSION:raise ValueError('wrong native dependency')
            names=('setup_ns','solver_call_ns','extract_count_delete_ns','independent_check_ns')
            if any(type(row[k]) is not int or row[k]<0 for k in names) or type(row['complete_ns']) is not int or not 0<row['complete_ns']<=row['supervised_ns'] or sum(row[k] for k in names)!=row['complete_ns']:raise ValueError('inconsistent measured cost')
            stats=row['counters']
            if type(stats) is not dict or not {'conflicts','decisions','propagations','restarts'}<=stats.keys() or any(type(v) is not int or v<0 for v in stats.values()):raise ValueError('bad native counters')
            if status=='SAT_VERIFIED':
                if type(row['witness']) is not list or not independent_check(p,tuple(row['witness'])):raise ValueError('false external SAT witness')
            elif row['witness'] is not None:raise ValueError('fabricated non-SAT witness')
    if found!=expected:raise ValueError('missing external rows')
    hybrid_witnesses={(r['case_id'],tuple(r['witness'])) for r in hybrid if r['valid']}
    for case_id,witness in hybrid_witnesses:
        if case_id not in lookup or not independent_check(CNF.from_record(lookup[case_id]['formula']),witness):
            raise ValueError('false hybrid witness in external comparison')
    hybrid_valid={case_id for case_id,_ in hybrid_witnesses}
    for c in items:
        reports=[r for r in rows if r['case_id']==c['id']]
        if any(r['status']=='UNSAT_REPORTED' for r in reports) and (c['id'] in hybrid_valid or any(r['status']=='SAT_VERIFIED' for r in reports)):raise ValueError('UNSAT report contradicts a valid observed witness')
    strata=[]
    for family in cfg['families']:
        for n in cfg['eval_sizes']:
            ids={c['id'] for c in items if c['family']==family and c['nvars']==n}
            for solver in SOLVERS:
                rs=[r for r in rows if r['case_id'] in ids and r['solver']==solver]
                stats=[r['counters']['conflicts'] for r in rs if r['status']!='TIMEOUT']
                for budget in cfg['wall_budget_ms']:
                    delivered=sum(r['status']=='SAT_VERIFIED' and r['complete_ns']<=budget*1000000 for r in rs)
                    strata.append({'family':family,'nvars':n,'solver':solver,'cutoff_ms':budget,'observations':len(rs),
                        'verified_witness_within_cutoff':delivered,'delivery_rate':delivered/len(rs),
                        'statuses':dict(sorted(Counter(r['status'] for r in rs).items())),
                        'max_actual_conflicts':max(stats,default=None),'budget_overshoot_rows':sum(v>CONFLICTS for v in stats)})
    return {'schema':'spectra.residual_patch.external.v1','cases':len(items),'observations':len(rows),
      'statuses':dict(sorted(Counter(r['status'] for r in rows).items())), 'strata':strata,
      'all_within_cutoff_rates':{s:{str(b):sum(r['solver']==s and r['status']=='SAT_VERIFIED' and r['complete_ns']<=b*1000000 for r in rows)/(len(items)*ROUNDS) for b in cfg['wall_budget_ms']} for s in SOLVERS},
      'scope':'two pinned default CDCL configurations; warm complete in-worker cost, separate startup; retrospective witness-delivery cutoffs; no enforced 2/10ms CDCL deadline or certified UNSAT',
      'same_input_hybrid_rows_considered':len(hybrid),'distinct_hybrid_witnesses_checked':len(hybrid_witnesses),'independent_confirmation':False,'sota_claim':False}


def execute(cfg,inputs,out):
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',cfg);write_json(out/'cases.json',inputs)
    write_json(out/'environment.json',environment())
    write_json(out/'protocol.json',{'version':VERSION,'solvers':SOLVERS,'rounds':ROUNDS,'conflicts_requested':CONFLICTS,
       'watchdog_seconds':WATCHDOG,'seed':1930911,'phase':'post-development classical context, no training/selection change',
       'sources':source_hashes()})
    events=[(i,s,r) for r in range(ROUNDS) for i in range(len(inputs)) for s in SOLVERS]
    np.random.default_rng(1930911).shuffle(events);write_json(out/'schedule.json',events)
    rows=[]
    with gzip.open(out/'rows.jsonl.gz','wt') as f:
        for index,(i,s,r) in enumerate(events):
            row=run_one(inputs[i],s,r);f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush();rows.append(row)
            if row['status']=='ERROR':raise RuntimeError(row['error'])
            if index%64==0:print('external',index,len(events),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','verify'])
    p.add_argument('--followup',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cfg=json.loads((a.followup/'config.json').read_text());inputs=cases(cfg,'development')
    if inputs!=json.loads((a.followup/'development/cases.json').read_text()):raise ValueError('not the same development inventory')
    with gzip.open(a.followup/'development/rows.jsonl.gz','rt') as f:hybrid=[json.loads(line) for line in f]
    if a.command=='run':
        try:
            rows=execute(cfg,inputs,a.out);report=summarize(cfg,inputs,rows,hybrid)
            write_json(a.out/'summary.json',report);seal(a.out)
        except Exception:
            if a.out.exists():(a.out/'failure.txt').write_text(traceback.format_exc());seal(a.out)
            raise
    else:
        from .audit import hash_inventory,records
        hash_inventory(a.out)
        protocol=json.loads((a.out/'protocol.json').read_text())
        if protocol['sources']!=source_hashes() or protocol['version']!=VERSION or protocol['solvers']!=list(SOLVERS) or protocol['rounds']!=ROUNDS or protocol['conflicts_requested']!=CONFLICTS or protocol['watchdog_seconds']!=WATCHDOG:
            raise ValueError('external source/protocol changed')
        if inputs!=json.loads((a.out/'cases.json').read_text()) or cfg!=json.loads((a.out/'config.json').read_text()):raise ValueError('altered inputs/config')
        rows=list(records(a.out/'rows.jsonl.gz'));report=summarize(cfg,inputs,rows,hybrid)
        if report!=json.loads((a.out/'summary.json').read_text()):raise ValueError('external summary differs')
        schedule=json.loads((a.out/'schedule.json').read_text())
        observed=[(next(i for i,c in enumerate(inputs) if c['id']==r['case_id']),r['solver'],r['round']) for r in rows]
        if observed!=list(map(tuple,schedule)):raise ValueError('external order mismatch')
    print(json.dumps(report,sort_keys=True,indent=2,allow_nan=False))

if __name__=='__main__':main()
