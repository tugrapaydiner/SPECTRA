"""Execute and audit the isolated residual-patch development experiment.

Run as ``python -m experiments.residual_patch.experiment ...`` from repo root.
No command updates legacy files or opens confirmation. Existing outputs are refused.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
import zipfile
import numpy as np
from data.cnf import CNF
from .data import assignment, cases, collect, write_json
from .runtime import State, build_info, parameters

HERE=Path(__file__).resolve().parent
CODE=('kernel.cpp','runtime.py','data.py','learn.py','experiment.py','config.json')

def sha(raw):return hashlib.sha256(raw).hexdigest()

def sources():return {**{name:sha((HERE/name).read_bytes()) for name in CODE},
    'legacy/data/cnf.py':sha((HERE.parents[1]/'data/cnf.py').read_bytes())}

def environment():
    return {'python':platform.python_version(),'numpy':np.__version__, 'platform':platform.platform(),
      'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
      'cpu':next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),None),
      'native_build':build_info(),'physical_energy_joules':None}

def seal(out):
    out=Path(out)
    inventory={str(p.relative_to(out)):sha(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256.json'}
    write_json(out/'SHA256.json',inventory)

def freeze(root,config):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'config.json',config);write_json(root/'source.json',sources());write_json(root/'environment.json',environment())
    with zipfile.ZipFile(root/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in CODE:z.write(HERE/name,name)
        z.write(HERE.parents[1]/'data/cnf.py','legacy/data/cnf.py')
    write_json(root/'design_boundary.json',{'base_main':'e73dc05379b38ab2689e0e3054f6eee74e0290c0',
      'scope':'locally frozen exact executable/protocol before data collection; branch-level protocol committed remotely first; not external preregistration',
      'confirmation_opened':False,'legacy_code_modified':False})

def assert_source(root):
    if sources()!=json.loads((Path(root)/'source.json').read_text()):raise ValueError('experiment source changed since freeze')

def model_bank(root,config):
    root=Path(root);bank={}
    label_sha=sha((root/'training/labels.npz').read_bytes())
    for mode in ('coarse','residual'):
        for seed in config['model_seeds']:
            path=root/f'models/{mode}_{seed}.json';p=json.loads(path.read_text())
            if p['mode']!=mode or p['seed']!=seed or p['training_labels_sha256']!=label_sha:raise ValueError('wrong model/data binding')
            bank[(mode,seed)]=parameters(p['weights'])
    return bank

def solve_one(case,arm,model_seed,search_seed,budget_ms,config,bank,*,work_only=False):
    p=CNF.from_record(case['formula']) # input parsing is outside the warm problem API
    digest=hashlib.sha256(f"{case['ordered_sha256']}:{search_seed}".encode()).digest()
    initial_seed=int.from_bytes(digest[:8],'little')
    weights=None
    if arm=='learned_patch':weights=bank[('residual',model_seed)]
    elif arm=='coarse_patch':weights=bank[('coarse',model_seed)]
    mode='learned_patch' if arm in ('learned_patch','coarse_patch') else arm
    budget_ns=int(budget_ms*1e6)
    start=time.perf_counter_ns()
    initial=assignment(p.nvars,initial_seed)
    with State(p,initial) as s:
        setup_end=time.perf_counter_ns()
        remaining=max(1,budget_ns-(setup_end-start)) if not work_only else 0
        moves=config['work_diagnostic_moves'] if work_only else 1000000
        # If setup consumes the deadline, do not perform uncredited extra search.
        if not work_only and setup_end-start>=budget_ns:moves=0
        result=s.run(initial_seed^0xa4b173f3,mode=mode,moves=moves,time_ns=remaining,
                    interval=config['patch_interval'],restart=config['restart_moves'],weights=weights,cb=config['break_exponent'])
        inference_end=time.perf_counter_ns()
        witness=result['witness']
        actual=p.satisfied(witness)
        if actual!=(result['native_status']==1):raise AssertionError('native result contradicts independent original-formula checker')
        check_end=time.perf_counter_ns()
    full=time.perf_counter_ns()-start
    return {'case_id':case['id'],'ordered_sha256':case['ordered_sha256'],'arm':arm,'model_seed':model_seed,
      'search_seed':search_seed,'budget_ms':budget_ms,'work_only':work_only,'initial_seed':initial_seed,
      'witness':witness,'valid':bool(actual),'within_budget':bool(actual and (work_only or full<=budget_ns)),
      'setup_ns':setup_end-start,'native_and_bridge_ns':inference_end-setup_end,'independent_check_ns':check_end-inference_end,
      'complete_ns':full,'overrun_ns':0 if work_only else max(0,full-budget_ns),
      'work':{k:v for k,v in result.items() if k!='witness'}}

def evaluate(root,split):
    root=Path(root);assert_source(root);config=json.loads((root/'config.json').read_text())
    if split not in ('validation','development'):raise ValueError('no confirmation evaluation is implemented')
    out=root/split;out.mkdir(exist_ok=False)
    bank=model_bank(root,config);items=cases(config,split);write_json(out/'cases.json',items)
    old=cases(config,'train')+(cases(config,'validation') if split=='development' else [])
    hashes=[c['normalized_sha256'] for c in old+items]
    if len(set(hashes))!=len(hashes):raise ValueError('order-normalized duplicate across declared input inventories')
    if split=='development':
        selection=json.loads((root/'validation/selection.json').read_text())
        model_hashes={p.name:sha(p.read_bytes()) for p in sorted((root/'models').glob('*.json'))}
        write_json(out/'pre_execution_lock.json',{'selection':selection,'model_hashes':model_hashes,'source':sources(),
                    'scope':'frozen before development reads; validation and all development results remain a pilot'})
    schedule=[]
    for ri in range(config['timing_rounds']):
        for ci in range(len(items)):
            for ms in config['model_seeds']:
                for ss in config['search_seeds']:
                    for budget in config['wall_budget_ms']:
                        for arm in config['arms']:schedule.append((ri,ci,ms,ss,budget,arm))
    rng=np.random.default_rng(92384 if split=='validation' else 29385);rng.shuffle(schedule)
    write_json(out/'schedule.json',schedule)
    rows=[];start=time.perf_counter_ns()
    with gzip.open(out/'rows.jsonl.gz','wt') as f:
        for i,(ri,ci,ms,ss,budget,arm) in enumerate(schedule):
            row=solve_one(items[ci],arm,ms,ss,budget,config,bank);row['round']=ri
            f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');rows.append(row)
            if i%1000==0:f.flush();print(split,'wall rows',i,len(schedule),flush=True)
    # Fixed-move runs bind actual native model behavior separately from wall timing.
    with gzip.open(out/'work_rows.jsonl.gz','wt') as f:
        for ci,case in enumerate(items):
            for ms in config['model_seeds']:
                for ss in config['search_seeds']:
                    for arm in config['arms']:
                        row=solve_one(case,arm,ms,ss,0,config,bank,work_only=True);row['round']=0
                        f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    summary=aggregate(config,items,rows)
    summary['evaluation_ns']=time.perf_counter_ns()-start
    write_json(out/'summary.json',summary)
    if split=='validation':
        selection={}
        for family in config['families']:
            for n in config['eval_sizes']:
                eligible=[r for r in summary['strata'] if r['family']==family and r['nvars']==n and
                    r['budget_ms']==config['primary_wall_budget_ms'] and r['arm'] in config['arms'][:4]]
                best=max(eligible,key=lambda r:(r['within_budget_rate'],-r['mean_complete_ns'],-config['arms'].index(r['arm'])))
                selection[f'{family}:{n}']=best['arm']
        write_json(out/'selection.json',selection)
    else:
        write_json(out/'paired_analysis.json',paired(config,items,rows,selection))
    assert_source(root);seal(root);return summary

def aggregate(config,items,rows):
    lookup={c['id']:c for c in items};strata=[]
    for family in config['families']:
        for n in config['eval_sizes']:
            for budget in config['wall_budget_ms']:
                for arm in config['arms']:
                    r=[x for x in rows if lookup[x['case_id']]['family']==family and lookup[x['case_id']]['nvars']==n and x['budget_ms']==budget and x['arm']==arm]
                    costs=[x['complete_ns'] for x in r]
                    strata.append({'family':family,'nvars':n,'budget_ms':budget,'arm':arm,'rows':len(r),
                        'within_budget_successes':sum(x['within_budget'] for x in r),'within_budget_rate':float(np.mean([x['within_budget'] for x in r])),
                        'valid_witnesses':sum(x['valid'] for x in r),'late_valid_witnesses':sum(x['valid'] and not x['within_budget'] for x in r),
                        'mean_complete_ns':float(np.mean(costs)),'p95_complete_ns':float(np.quantile(costs,.95)),
                        'mean_moves':float(np.mean([x['work']['moves'] for x in r])),
                        'mean_model_calls':float(np.mean([x['work']['model_calls'] for x in r]))})
    return {'schema':'spectra.residual_patch.evaluation.v1','cases':len(items),'observations':len(rows),
      'strata':strata,'independent_confirmation':False,'physical_energy_joules':None,
      'scope':'full warm solve from parsed CNF through independent answer check and native cleanup; build/model load outside, initialization inside'}

def paired(config,items,rows,selection):
    ms=config['model_seeds'];primary=config['primary_wall_budget_ms'];results={}
    groups=[[i for i,c in enumerate(items) if c['family']==family and c['nvars']==n] for family in config['families'] for n in config['eval_sizes']]
    lookup={(c['id'],s,a):[] for c in items for s in ms for a in config['arms']}
    for r in rows:
        if r['budget_ms']==primary:lookup[(r['case_id'],r['model_seed'],r['arm'])].append(float(r['within_budget']))
    arrays={a:np.array([[np.mean(lookup[(c['id'],s,a)]) for s in ms] for c in items]) for a in config['arms']}
    baseline=np.array([arrays[selection[f"{c['family']}:{c['nvars']}"]][i] for i,c in enumerate(items)])
    rng=np.random.default_rng(28177)
    for arm in ('coarse_patch','learned_patch'):
        delta=arrays[arm]-baseline;boots=[]
        for _ in range(config['bootstrap_repeats']):
            ci=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);si=rng.integers(0,len(ms),len(ms))
            boots.append(delta[np.ix_(ci,si)].mean())
        lo,hi=np.quantile(boots,[.025,.975]);point=float(delta.mean())
        results[arm]={'candidate_rate':float(arrays[arm].mean()),'selected_control_rate':float(baseline.mean()),
            'gain':point,'crossed_group_95_interval':[float(lo),float(hi)],
            'per_model_seed_gain':dict(zip(map(str,ms),map(float,delta.mean(0)))),
            'development_quality_gate':bool(point>=config['minimum_quality_gain'] and lo>0)}
    feedback=arrays['learned_patch']-arrays['coarse_patch']
    return {'primary_budget_ms':primary,'validation_selected_controls':selection,'results':results,
        'exact_feedback_minus_coarse_gain':float(feedback.mean()),
        'qualification':'stratified formula resampling crossed with three model seeds; search seeds/rounds collapsed; descriptive development intervals, no multiplicity correction or external confirmation',
        'model_seeds':ms,'distinct_formulas':len(items),'groundbreaking_established':False}

def verify(root,replay=False):
    root=Path(root);assert_source(root);config=json.loads((root/'config.json').read_text())
    hashes=json.loads((root/'SHA256.json').read_text())
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='SHA256.json'}
    if actual!=set(hashes):raise ValueError('evidence file inventory mismatch')
    for name,digest in hashes.items():
        if sha((root/name).read_bytes())!=digest:raise ValueError('evidence hash mismatch: '+name)
    bank=model_bank(root,config);checks=0;replays=0
    for split in ('validation','development'):
        out=root/split
        if not out.exists():continue
        items=json.loads((out/'cases.json').read_text())
        if items!=cases(config,split):raise ValueError('data regeneration mismatch')
        lookup={c['id']:c for c in items};rows=[];seen=set()
        with gzip.open(out/'rows.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line);key=(r['case_id'],r['arm'],r['model_seed'],r['search_seed'],r['budget_ms'],r['round'])
                if key in seen:raise ValueError('duplicate timing observation')
                seen.add(key);c=lookup[r['case_id']];p=CNF.from_record(c['formula'])
                valid=p.satisfied(tuple(r['witness']))
                if valid!=r['valid'] or valid!=(r['work']['native_status']==1) or r['within_budget']!=(valid and r['complete_ns']<=r['budget_ms']*1e6):raise ValueError('false witness/budget outcome')
                if r['ordered_sha256']!=c['ordered_sha256'] or type(r['complete_ns']) is not int or r['complete_ns']<=0 or r['work']['native_ns']>r['complete_ns']:raise ValueError('invalid input/time record')
                if r['setup_ns']+r['native_and_bridge_ns']+r['independent_check_ns']>r['complete_ns']:raise ValueError('time scope exceeds total')
                checks+=1;rows.append(r)
        expected={(c['id'],a,ms,ss,b,ri) for c in items for a in config['arms'] for ms in config['model_seeds'] for ss in config['search_seeds'] for b in config['wall_budget_ms'] for ri in range(config['timing_rounds'])}
        if seen!=expected:raise ValueError('incomplete/extra timing inventory')
        report=aggregate(config,items,rows);stored=json.loads((out/'summary.json').read_text());stored.pop('evaluation_ns')
        if report!=stored:raise ValueError('summary mismatch')
        if split=='development':
            lock=json.loads((out/'pre_execution_lock.json').read_text())
            if lock['source']!=sources() or lock['model_hashes']!={p.name:sha(p.read_bytes()) for p in sorted((root/'models').glob('*.json'))}:raise ValueError('model/source changed after development lock')
            if paired(config,items,rows,lock['selection'])!=json.loads((out/'paired_analysis.json').read_text()):raise ValueError('paired report mismatch')
        with gzip.open(out/'work_rows.jsonl.gz','rt') as f:
            work_rows=[json.loads(line) for line in f]
        expected_work={(c['id'],a,ms,ss) for c in items for a in config['arms'] for ms in config['model_seeds'] for ss in config['search_seeds']}
        found_work=[(r['case_id'],r['arm'],r['model_seed'],r['search_seed']) for r in work_rows]
        if len(set(found_work))!=len(found_work) or set(found_work)!=expected_work:raise ValueError('work replay inventory mismatch')
        for r in work_rows:
            p=CNF.from_record(lookup[r['case_id']]['formula']);checks+=1
            if p.satisfied(tuple(r['witness']))!=r['valid']:raise ValueError('false fixed-work witness')
            if replay:
                actual=solve_one(lookup[r['case_id']],r['arm'],r['model_seed'],r['search_seed'],0,config,bank,work_only=True)
                left={k:v for k,v in r['work'].items() if k!='native_ns'};right={k:v for k,v in actual['work'].items() if k!='native_ns'}
                if tuple(r['witness'])!=actual['witness'] or left!=right:raise ValueError('native model/search replay mismatch')
                replays+=1
    return {'integrity':'PASS','independent_answer_checks':checks,'exact_native_work_replays':replays,
            'timings_remeasured':False,'scientific_gate_preserved':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','collect','train','validation','development','verify'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();root=args.out
    config=json.loads((HERE/'config.json').read_text())
    if args.command=='freeze':freeze(root,config);return
    assert_source(root)
    try:
        if args.command=='collect':result=collect(config,root/'training');seal(root)
        elif args.command=='train':
            from .learn import train
            result=train(config,root/'training',root/'models');seal(root)
        elif args.command in ('validation','development'):result=evaluate(root,args.command)
        else:result=verify(root,args.replay)
        print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    except Exception:
        if args.command!='verify':
            failure=root/f'failure_{args.command}_{time.time_ns()}.txt';failure.write_text(traceback.format_exc());seal(root)
        raise

if __name__=='__main__':main()
