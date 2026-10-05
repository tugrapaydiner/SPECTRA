"""Dependency-free retained-evidence verification and optional Python replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.structured_search.study import (
    ARMS, ROUNDS, call, cases, check_freeze, domain_solver, require, semantic,
)
from experiments.structured_search.task import check_grid, check_cnf, decode, encode

RAW_FILES={'LOCK.json','cases.jsonl','rows.jsonl','memory.jsonl'}


def load(path):
    if path.is_dir():
        names={p.name for p in path.iterdir() if p.is_file()}
        read=lambda name:(path/name).read_bytes()
    else:
        archive=zipfile.ZipFile(path)
        names=set(archive.namelist())
        require(len(names)==len(archive.namelist()),'duplicate archive member')
        read=archive.read
    require(names==RAW_FILES|{'MANIFEST.json'},'unexpected artifact inventory')
    manifest=json.loads(read('MANIFEST.json'))
    require(manifest['status']=='COMPLETE' and set(manifest['files'])==RAW_FILES,
            'incomplete artifact bindings')
    for name,expected in manifest['files'].items():
        data=read(name)
        require(len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'],
                'artifact mismatch: '+name)
    return (json.loads(read('LOCK.json')),
            *([json.loads(line) for line in read(name).splitlines()]
              for name in ('cases.jsonl','rows.jsonl','memory.jsonl')))


def verify(lock, inputs, rows, memory, replay=False):
    check_freeze(lock['freeze'])
    require(lock['status']=='RUN_STARTED' and lock['python_sat']=='1.9.dev15','unexpected environment')
    require(len(lock['affinity'])==1 and len(lock['native_sources'])==3,'missing environment identities')
    require(inputs==cases('evaluation'),'case inventory differs')
    expected={(c['id'],arm,r) for c in inputs for arm in ARMS for r in range(ROUNDS)}
    keyed={(r['id'],r['arm'],r['round']):r for r in rows}
    require(len(rows)==len(keyed) and set(keyed)==expected,'incomplete timing inventory')
    by_case={c['id']:c for c in inputs}
    reference={}
    problems={c['id']:encode(c['puzzle']) for c in inputs}
    for row in rows:
        case=by_case[row['id']]; result=row['result']; arm=row['arm']
        require(row['sha256']==case['sha256'],'case identity differs')
        for metric in ('wall_ns','cpu_ns','encoding_ns','solve_ns'):
            require(type(row[metric]) is int and row[metric]>=0,'invalid timing')
        require(row['wall_ns']>=row['encoding_ns']+row['solve_ns']>0,'phase timing exceeds full cost')
        require(result['status'] in ('SAT_VERIFIED','UNKNOWN','UNSAT_REPORTED'),'unexpected outcome')
        grid=result['grid']
        require((grid is not None)==(result['status']=='SAT_VERIFIED'),'grid/status mismatch')
        if grid is not None:
            require(check_grid(case['puzzle'],grid),'invalid original-grid answer')
            witness=tuple(int(grid[i//9])==i%9+1 for i in range(729))
            require(check_cnf(problems[row['id']],witness),'invalid grid-derived CNF witness')
        if arm in ('focused','deductive','dpll'):
            witness=tuple(result['witness'])
            require(len(witness)==729 and all(type(v) is bool for v in witness),'invalid witness')
            problem=problems[row['id']]
            bad=[i for i,c in enumerate(problem.clauses)
                 if not any(witness[abs(l)-1]==(l>0) for l in c)]
            require(result['unsatisfied']==bad and (not bad)==(result['status']=='SAT_VERIFIED'),
                    'original-clause/status mismatch')
            if grid is not None:
                require(decode(witness)==grid,'witness/grid mismatch')
            if arm=='dpll':
                require(result['max_decisions']==2048 and 0<=result['decisions']<=2048,'branch budget mismatch')
                require(result['reason'] in ('satisfied','budget','exhausted'),'unexpected termination')
            else:
                require(result['seed']==17 and result['max_flips']==2048 and 0<=result['flips']<=2048,
                        'flip budget mismatch')
        if arm=='glucose4':
            require(result['conflict_budget_requested']==2000,'native budget differs')
            require(all(type(v) is int and v>=0 for v in result['stats'].values()),'invalid native stats')
        key=row['id'],arm
        current=semantic(result)
        if key in reference:
            require(current==reference[key],'round trajectory differs')
        reference[key]=current
    memory_keys={(m['id'],m['arm']) for m in memory}
    require(len(memory)==len(memory_keys) and memory_keys=={(c['id'],a) for c in inputs[:5] for a in ARMS},
            'incomplete memory inventory')
    for m in memory:
        require(type(m['baseline_rss_kib']) is int and type(m['peak_rss_kib']) is int
                and 0<m['baseline_rss_kib']<=m['peak_rss_kib'],'invalid RSS')
        require((m['python_peak_bytes'] is None if m['arm']=='glucose4' else
                 type(m['python_peak_bytes']) is int and m['python_peak_bytes']>0),'invalid allocation peak')
        require(m['status']==reference[m['id'],m['arm']]['status'],'memory outcome differs')
    replayed=0
    if replay:
        norvig=domain_solver()
        for case in inputs:
            for arm in ARMS:
                if arm=='glucose4':
                    continue
                current=call(case,arm,None,norvig)['result']
                require(semantic(current)==reference[case['id'],arm],'Python replay differs')
                replayed+=1
    return keyed,replayed


def summarize(lock, inputs, rows, memory, keyed, replayed):
    arms={}
    for arm in ARMS:
        selected=[r for r in rows if r['arm']==arm]
        times=[statistics.mean(keyed[c['id'],arm,r]['wall_ns'] for r in range(ROUNDS))/1e6 for c in inputs]
        mem=[m for m in memory if m['arm']==arm]
        arms[arm]={
            'verified_tasks':sum(keyed[c['id'],arm,0]['result']['status']=='SAT_VERIFIED' for c in inputs),
            'tasks':len(inputs),'mean_wall_ms':statistics.mean(times),'median_wall_ms':statistics.median(times),
            'p95_wall_ms':sorted(times)[math.ceil(.95*len(times))-1],'max_wall_ms':max(times),
            'mean_cpu_ms':statistics.mean(r['cpu_ns'] for r in selected)/1e6,
            'mean_encoding_ms':statistics.mean(r['encoding_ns'] for r in selected)/1e6,
            'mean_solve_ms':statistics.mean(r['solve_ns'] for r in selected)/1e6,
            'median_rss_mib':statistics.median(m['peak_rss_kib'] for m in mem)/1024,
            'median_baseline_rss_mib':statistics.median(m['baseline_rss_kib'] for m in mem)/1024,
            'max_rss_mib':max(m['peak_rss_kib'] for m in mem)/1024,
            'median_incremental_rss_mib':statistics.median(m['peak_rss_kib']-m['baseline_rss_kib'] for m in mem)/1024,
            'median_python_peak_bytes':None if arm=='glucose4' else statistics.median(m['python_peak_bytes'] for m in mem),
            'unknown_ids':[c['id'] for c in inputs if keyed[c['id'],arm,0]['result']['status']!='SAT_VERIFIED'],
        }
    candidate=arms['dpll']
    capability=(candidate['verified_tasks']>=80 and all(candidate['verified_tasks']>=arms[a]['verified_tasks']+20
                                                       for a in ('focused','deductive')))
    efficiency=candidate['mean_wall_ms']<=arms['focused']['mean_wall_ms']
    return {'status':'PASS','scope':'retained bytes, complete inventories and independent solutions; timing is not replayed',
            'freeze_commit':lock['freeze']['commit'],'freeze_tree':lock['freeze']['tree'],
            'timing_rows':len(rows),'memory_rows':len(memory),'python_replays':replayed,'arms':arms,
            'capability_gate':'PASS' if capability else 'FAIL',
            'efficiency_gate':'PASS' if efficiency else 'FAIL',
            'dpll_over_focused_mean_wall':candidate['mean_wall_ms']/arms['focused']['mean_wall_ms'],
            'dpll_over_glucose_mean_wall':candidate['mean_wall_ms']/arms['glucose4']['mean_wall_ms'],
            'dpll_over_norvig_mean_wall':candidate['mean_wall_ms']/arms['norvig']['mean_wall_ms'],
            'native_max_conflicts':max(r['result']['stats']['conflicts'] for r in rows if r['arm']=='glucose4')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    lock,inputs,rows,memory=load(args.evidence)
    keyed,replayed=verify(lock,inputs,rows,memory,args.replay)
    result=summarize(lock,inputs,rows,memory,keyed,replayed)
    encoded=json.dumps(result,indent=2)+'\n'
    if args.out:
        with args.out.open('x') as stream:
            stream.write(encoded)
    print(encoded,end='')


if __name__=='__main__':
    main()
