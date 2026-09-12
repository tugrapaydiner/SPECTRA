"""Paired fixed-policy constructor-memory/full-execution comparison; no ML claim."""
from __future__ import annotations
import argparse
import gc
import gzip
import hashlib
import json
import platform
from pathlib import Path
import sys
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.cnf import CNF, SplitMix64, random_3sat
from eval.cnf_cached import CachedCNFRepairState
from spectra.cnf.state import CompactCNFRepairState
from spectra.cnf.search import _execute

CONFIG = dict(sizes=[512, 4096, 16384], families=['uniform','planted'], per_cell=2,
              case_seed=91226000, search_seeds=[17001,27002], rounds=3, moves=128,
              bootstrap_repeats=4000, bootstrap_seed=91226,
              maximum_cell_mean_ratio=1.10, maximum_large_memory_ratio=0.70)
ENGINES = {'bitset':CachedCNFRepairState, 'compact':CompactCNFRepairState}
SOURCES = ['data/cnf.py','eval/cnf_cached.py','spectra/cnf/state.py',
           'spectra/cnf/search.py','scripts/bench_compact_cnf.py']


def write(path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+'\n')


def generate():
    cases=[]
    for n in CONFIG['sizes']:
        for family in CONFIG['families']:
            for i in range(CONFIG['per_cell']):
                seed=CONFIG['case_seed']+len(cases)
                p,_=random_3sat(n, n*21//5, seed, planted=family=='planted')
                cases.append(dict(id=f'{n}:{family}:{i}', nvars=n, family=family, seed=seed,
                                  sha256=p.sha256(), formula=p.record()))
    return cases


def schedule(cases):
    for r in range(CONFIG['rounds']):
        for i,c in enumerate(cases):
            for j,s in enumerate(CONFIG['search_seeds']):
                engines=['bitset','compact'] if (r+i+j)%2==0 else ['compact','bitset']
                for e in engines:
                    yield c,s,r,e


def memory_record(p, seed, engine):
    rng=SplitMix64(seed)
    witness=tuple(bool(rng.below(2)) for _ in range(p.nvars))
    gc.collect()
    tracemalloc.start()
    state=ENGINES[engine](p,witness)
    current,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert state.unsatisfied==p.violated(witness)
    del state
    return dict(current_python_bytes=current, peak_python_bytes=peak)


def analyze(cases, rows, memory):
    import numpy as np
    expected=[(c['id'],s,r,e) for c,s,r,e in schedule(cases)]
    if [(r['case'],r['search_seed'],r['round'],r['engine']) for r in rows]!=expected:
        raise ValueError('timing inventory/order mismatch')
    if [(r['case'],r['engine']) for r in memory] != [(c['id'],e) for c in cases for e in ENGINES]:
        raise ValueError('memory inventory/order mismatch')
    by_case={c['id']:c for c in cases}
    same={};times={};peaks={}
    for row in rows:
        c=by_case[row['case']];result=row['result']; p=CNF.from_record(c['formula'])
        if set(row)!={'case','engine','search_seed','round','result'}:
            raise ValueError('unexpected timing fields')
        for k in ('elapsed_ns','flips','queries','seed','max_flips'):
            if type(result[k]) is not int or result[k]< (1 if k=='elapsed_ns' else 0):
                raise ValueError('invalid integer work/time')
        if result['seed'] != row['search_seed'] or result['max_flips'] != CONFIG['moves']:
            raise ValueError('execution settings mismatch')
        if result['flips']>CONFIG['moves'] or result['queries']>3*CONFIG['moves']:
            raise ValueError('impossible work')
        if type(result['witness']) is not list or type(result['unsatisfied']) is not list:
            raise ValueError('invalid answer arrays')
        violated=p.violated(tuple(result['witness']))
        if list(violated)!=result['unsatisfied'] or result['status']!=('UNKNOWN' if violated else 'SAT_VERIFIED'):
            raise ValueError('original-formula answer mismatch')
        semantic={k:v for k,v in result.items() if k!='elapsed_ns'}
        key=(row['case'],row['search_seed'])
        if key in same and same[key]!=semantic:
            raise ValueError('backend/round trajectory mismatch')
        same[key]=semantic
        times.setdefault((row['case'],row['engine']),[]).append(result['elapsed_ns'])
    for m in memory:
        if set(m)!={'case','engine','current_python_bytes','peak_python_bytes'}:
            raise ValueError('unexpected memory fields')
        for field in ('current_python_bytes','peak_python_bytes'):
            if type(m[field]) is not int or m[field]<=0:
                raise ValueError('invalid memory count')
        if m['current_python_bytes']>m['peak_python_bytes']:raise ValueError('invalid memory peak')
        peaks[(m['case'],m['engine'])]=m['peak_python_bytes']
    rng=np.random.default_rng(CONFIG['bootstrap_seed'])
    cells={}
    for n in CONFIG['sizes']:
        for f in CONFIG['families']:
            subset=[c for c in cases if c['nvars']==n and c['family']==f]
            a=np.array([[np.mean(times[(c['id'],e)]) for e in ENGINES] for c in subset])
            draws=rng.integers(0,len(a),size=(CONFIG['bootstrap_repeats'],len(a)))
            b=a[draws].mean(axis=1)
            mem=np.array([[peaks[(c['id'],e)] for e in ENGINES] for c in subset])
            cells[f'{n}:{f}']=dict(mean_ms=(a.mean(axis=0)/1e6).tolist(),
                mean_ratio=float(a[:,1].mean()/a[:,0].mean()),
                mean_ratio_ci95=np.percentile(b[:,1]/b[:,0],[2.5,97.5]).tolist(),
                peak_python_bytes=mem.mean(axis=0).tolist(),
                peak_ratio=float(mem[:,1].mean()/mem[:,0].mean()))
    large=[c for c in cases if c['nvars']>=4096]
    large_ratio=sum(peaks[(c['id'],'compact')] for c in large)/sum(peaks[(c['id'],'bitset')] for c in large)
    gates=dict(same_answers=True,
               all_cell_mean=all(c['mean_ratio']<=CONFIG['maximum_cell_mean_ratio'] for c in cells.values()),
               large_python_peak=large_ratio<=CONFIG['maximum_large_memory_ratio'])
    return dict(cells=cells, memory_scope='tracemalloc Python construction allocations, not RSS',
                large_peak_ratio=large_ratio, gates=gates, gate='PASS' if all(gates.values()) else 'FAIL',
                observations=len(rows),paired_paths=len(same),memory_observations=len(memory),
                learned_capability=False, confirmation=False)


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    write(out/'config.json',CONFIG)
    write(out/'source_sha256.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES})
    write(out/'environment.json',dict(python=sys.version,platform=platform.platform()))
    cases=generate();write(out/'cases.json',cases)
    problems={c['id']:CNF.from_record(c['formula']) for c in cases}
    rows=[]
    with gzip.open(out/'timings.jsonl.gz','wt') as stream:
        for c,seed,r,engine in schedule(cases):
            result=_execute(problems[c['id']],seed=seed,max_flips=CONFIG['moves'],state_class=ENGINES[engine])
            row=dict(case=c['id'],search_seed=seed,round=r,engine=engine,result=result.record())
            rows.append(row);stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush()
    memory=[]
    with (out/'memory.jsonl').open('w') as stream:
        for c in cases:
            for engine in ENGINES:
                row=dict(case=c['id'],engine=engine,**memory_record(problems[c['id']],17001,engine))
                memory.append(row);stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush()
                print('memory',c['id'],engine,row['peak_python_bytes'],flush=True)
    report=analyze(cases,rows,memory);write(out/'summary.json',report)
    write(out/'SHA256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.name!='SHA256.json'})
    return report


def verify(out, replay=False):
    inventory=json.loads((out/'SHA256.json').read_text())
    if set(inventory)!={p.name for p in out.iterdir() if p.name!='SHA256.json'}:raise ValueError('inventory mismatch')
    for p,h in inventory.items():
        if hashlib.sha256((out/p).read_bytes()).hexdigest()!=h:raise ValueError('archive hash mismatch')
    if json.loads((out/'config.json').read_text())!=CONFIG:raise ValueError('config mismatch')
    if json.loads((out/'source_sha256.json').read_text())!={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}:raise ValueError('source mismatch')
    cases=json.loads((out/'cases.json').read_text())
    if cases!=generate():raise ValueError('frozen input inventory mismatch')
    with gzip.open(out/'timings.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
    memory=[json.loads(x) for x in (out/'memory.jsonl').read_text().splitlines()]
    summary=analyze(cases,rows,memory)
    if summary!=json.loads((out/'summary.json').read_text()):raise ValueError('statistics mismatch')
    count=0
    if replay:
        problems={c['id']:CNF.from_record(c['formula']) for c in cases}
        for row in rows:
            if row['round']!=0:continue
            actual=_execute(problems[row['case']],seed=row['search_seed'],max_flips=CONFIG['moves'],state_class=ENGINES[row['engine']]).record()
            if {k:v for k,v in actual.items() if k!='elapsed_ns'}!={k:v for k,v in row['result'].items() if k!='elapsed_ns'}:raise ValueError('path replay mismatch')
            count+=1
    return dict(integrity='PASS',answers=len(rows),path_replays=count,systems_gate=summary['gate'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','verify']);p.add_argument('--out',type=Path,required=True);p.add_argument('--replay',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.out) if a.command=='run' else verify(a.out,a.replay),indent=2))
