"""Strict retained-evidence validation and case-clustered descriptive intervals.

No runtime fitting, native-library execution, deserialization of model pickles,
or numerical-framework dependency is needed. Integrity is not clock authentication.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.cover_search import study
from experiments.cover_search.tasks import check


def quantile(values,p):
    if not values:raise ValueError('empty quantile')
    x=sorted(values);at=(len(x)-1)*p;lo=int(at);hi=min(lo+1,len(x)-1)
    return x[lo]+(x[hi]-x[lo])*(at-lo)


def valid(row):return row['status']=='SAT_VERIFIED'


def comparison(selected,rows,candidate,baseline,resamples=2000):
    indexed={(r['case'],r['arm'],r['round']):r for r in rows}
    selected=[c for c in selected if baseline in study.arms(c)]
    strata=defaultdict(list)
    for c in selected:strata[c['stratum']].append(c['id'])
    def calculate(ids):
        a=[indexed[(i,candidate,j)] for i in ids for j in range(study.ROUNDS)]
        b=[indexed[(i,baseline,j)] for i in ids for j in range(study.ROUNDS)]
        av=[r['elapsed_ns'] for r in a];bv=[r['elapsed_ns'] for r in b]
        return (statistics.fmean(av)/statistics.fmean(bv),quantile(av,.95)/quantile(bv,.95),
                (sum(map(valid,a))-sum(map(valid,b)))/len(a))
    ids=[c['id'] for c in selected];point=calculate(ids)
    rng=random.Random(519018);draws=[]
    for _ in range(resamples):
        sample=[rng.choice(group) for group in strata.values() for __ in group]
        draws.append(calculate(sample))
    intervals=[[quantile([d[k] for d in draws],.025),quantile([d[k] for d in draws],.975)] for k in range(3)]
    a=[r for r in rows if r['case'] in set(ids) and r['arm']==candidate]
    b=[r for r in rows if r['case'] in set(ids) and r['arm']==baseline]
    return {'problems':len(ids),'candidate':candidate,'baseline':baseline,
            'mean_ratio':point[0],'p95_ratio':point[1],'sat_delta':point[2],
            'descriptive_95_intervals':dict(zip(('mean_ratio','p95_ratio','sat_delta'),intervals)),
            'all_candidate_sat':all(map(valid,a)),'all_baseline_sat':all(map(valid,b)),
            'speed_gate':all(map(valid,a)) and all(map(valid,b)) and intervals[0][1]<=.5 and intervals[1][1]<=1.1}


def validate(folder):
    freeze=study.check_freeze()
    manifest=json.loads((folder/'MANIFEST.json').read_text())
    required={'controls.json','cases.json','freeze.json','environment.json','schedule.json','timings.jsonl','resources.jsonl',
              'native/build.json','native/spectra_cover.so'}
    study.require(required<=set(manifest),'missing required evidence roles')
    for name,sha in manifest.items():
        path=(folder/name).resolve()
        study.require(path.is_relative_to(folder.resolve()),'manifest path escapes evidence')
        study.require(hashlib.sha256(path.read_bytes()).hexdigest()==sha,'evidence hash mismatch: '+name)
    study.require(json.loads((folder/'freeze.json').read_text())==freeze,'frozen source identity mismatch')
    build=json.loads((folder/'native/build.json').read_text())
    study.require(build['source_sha256']==freeze['sources']['spectra/_native/cover_search.cpp'],'compiled source differs')
    study.require(build['library_sha256']==manifest['native/spectra_cover.so'],'library identity mismatch')
    cases=json.loads((folder/'cases.json').read_text())
    # Reconstruct complete prospective/public input inventory, not hashes alone.
    study.require(cases==study.make_cases(),'incomplete or altered task inventory')
    jobs=study.schedule(cases)
    study.require(json.loads((folder/'schedule.json').read_text())==[list(j) for j in jobs],'schedule mismatch')
    rows=[json.loads(line) for line in (folder/'timings.jsonl').read_text().splitlines()]
    study.require(len(rows)==len(jobs),'incomplete timing matrix')
    lookup={c['id']:c for c in cases}
    for i,(row,job) in enumerate(zip(rows,jobs)):
        study.require(type(row['job']) is int and row['job']==i,'invalid job index')
        study.require(type(row['round']) is int and (row['case'],row['arm'],row['round'])==job,'row order/identity mismatch')
        study.require(type(row['elapsed_ns']) is int and row['elapsed_ns']>0,'invalid timing')
        study.require(row['status'] in ('SAT_VERIFIED','UNKNOWN','UNSAT_REPORTED','TIMEOUT'),'invalid status')
        study.require(type(row['deadline_overrun']) is bool and row['deadline_overrun']==(row['elapsed_ns']>study.DEADLINE_SECONDS*1e9),'deadline mismatch')
        if row['status']=='UNSAT_REPORTED':study.require(row['arm'] in ('minicard','gluecard4','glucose42','cadical300','kissat404'),'unsupported UNSAT claim')
        if valid(row):
            study.require(type(row['witness']) is list and check(lookup[row['case']],tuple(row['witness'])),'invalid task witness')
        if row['arm'] in ('cover','dense','cover_cnf') and row['status']!='TIMEOUT':
            d=row['details']
            for field in ('nodes','max_nodes','max_state_bytes','state_word_bytes_peak','index_payload_bytes'):
                study.require(type(d[field]) is int and d[field]>=0,'invalid native work/resource counter')
            study.require(d['nodes']<=d['max_nodes']==study.MAX_NODES,'invalid node budget')
            study.require(d['state_word_bytes_peak']<=d['max_state_bytes']==study.MAX_STATE_BYTES,'invalid state budget')
            study.require(d['index_payload_bytes']<=study.MAX_INDEX_BYTES,'invalid index payload')
    resources=[json.loads(line) for line in (folder/'resources.jsonl').read_text().splitlines()]
    expected=[];per=defaultdict(int)
    for c in cases:
        limit=5 if c['kind']=='sudoku' else 2
        if per[c['stratum']]<limit:expected.extend((c['id'],arm) for arm in study.arms(c));per[c['stratum']]+=1
    study.require([(r['case'],r['arm']) for r in resources]==expected,'incomplete resource matrix')
    for r in resources:
        study.require(type(r['cold_process_wall_ns']) is int and r['cold_process_wall_ns']>0,'invalid cold time')
        if r.get('resource_error'):continue
        study.require(type(r['VmHWM_KiB']) is int and r['VmHWM_KiB']>0,'invalid process memory')
        study.require(not (set(r['numerical_frameworks_loaded']) & {'numpy','torch','scipy','sklearn'}),'contaminated worker')
        if r['result']['status']=='SAT_VERIFIED':study.require(check(lookup[r['case']],tuple(r['result']['witness'])),'invalid resource witness')
    return cases,rows,resources


def analyse(folder):
    cases,rows,resources=validate(folder)
    panels={'sudoku_public':[c for c in cases if c['kind']=='sudoku'],
            'graph_planted':[c for c in cases if c['kind']=='colouring' and c['planted']],
            'graph_uniform':[c for c in cases if c['kind']=='colouring' and not c['planted']]}
    summary={'schema':'spectra.cover.study.v1','cases':len(cases),'timing_rows':len(rows),
             'resource_rows':len(resources),'panels':{},'comparisons':{},'scaling':{},
             'resource_errors':[r for r in resources if r.get('resource_error')]}
    for name,selected in panels.items():
        ids={c['id'] for c in selected};table={}
        for arm in study.arms(selected[0]):
            obs=[r for r in rows if r['case'] in ids and r['arm']==arm]
            times=[r['elapsed_ns']/1e6 for r in obs]
            percase=defaultdict(set)
            for r in obs:percase[r['case']].add(r['status'])
            table[arm]={'verified_sat':sum(v=={'SAT_VERIFIED'} for v in percase.values()),
                       'problems':len(selected),'mean_ms':statistics.fmean(times),
                       'median_ms':statistics.median(times),'p95_ms':quantile(times,.95),
                       'max_ms':max(times),'deadline_overruns':sum(r['deadline_overrun'] for r in obs),
                       'statuses':{s:sum(r['status']==s for r in obs) for s in sorted({r['status'] for r in obs})},
                       'nondeterministic_status_cases':[k for k,v in percase.items() if len(v)>1]}
        summary['panels'][name]=table
        if name!='graph_uniform':
            summary['comparisons'][name]={arm:comparison(selected,rows,'cover',arm)
                                         for arm in ('dense','cover_cnf','dpll','minicard','gluecard4','glucose42','cadical300','kissat404')}
    for stratum in sorted({c['stratum'] for c in cases}):
        ids={c['id'] for c in cases if c['stratum']==stratum}
        summary['scaling'][stratum]={arm:{'mean_ms':statistics.fmean(r['elapsed_ns']/1e6 for r in rows if r['case'] in ids and r['arm']==arm),
             'sat_calls':sum(valid(r) for r in rows if r['case'] in ids and r['arm']==arm)} for arm in study.ARMS}
    summary['primary_external_gate']=summary['comparisons']['graph_planted']['minicard']['speed_gate']
    summary['internal_compatibility_gate']=summary['comparisons']['sudoku_public']['dpll']['speed_gate']
    summary['failure_taxonomy']={}
    for arm in study.ARMS:
        counts=defaultdict(int)
        for r in rows:
            if r['arm']!=arm or valid(r):continue
            key=('deadline' if r['status']=='TIMEOUT' else 'unverified_unsat' if r['status']=='UNSAT_REPORTED'
                 else r['details'].get('reason','unknown'))
            counts[key]+=1
        summary['failure_taxonomy'][arm]=dict(counts)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();result=analyse(a.folder)
    text=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if a.out:a.out.write_text(text)
    else:print(text,end='')
if __name__=='__main__':main()
