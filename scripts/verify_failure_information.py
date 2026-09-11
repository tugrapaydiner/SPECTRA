#!/usr/bin/env python3
"""Verify the retained development pilot, optionally refitting and replaying it."""
from __future__ import annotations
import argparse
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time

if __name__ == '__main__':
    os.environ.update(ATEN_CPU_CAPABILITY='avx2', MKL_ENABLE_INSTRUCTIONS='AVX2',
        ONEDNN_MAX_CPU_ISA='AVX2', DNNL_MAX_CPU_ISA='AVX2',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from eval.failure_information import (Observation, MODES, coverage, simulate, first_success, fit_selector,
    best_static_order, solve_with_selector)
from eval.symmetry_search import symmetry_solve, grid_views
from scripts.failure_information_pilot import (ARMS, setup, sha, write_json, load_cases,
    load_policies, selector_for, crossfit_partition, timed_analysis, sources,
    FAMILY_SPECS, load_sources, tensor_digest, trajectory, source_identity)
from scripts.transpose_continuation_pilot import ARMS as EXTENDED_ARMS
from scripts.transpose_continuation_pilot import predict, execute, analyze as extended_analysis
from scripts.m17_cross_task import symbolic_solve

EVIDENCE = ROOT/'results/failure_information'


def read_bound_rows(path, expected_sha, expected_count):
    raw = gzip.decompress(path.read_bytes())
    if sha(raw) != expected_sha:
        raise ValueError('raw evidence digest mismatch')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != expected_count:
        raise ValueError('raw evidence count mismatch')
    return rows


def verify_source(directory):
    receipt = json.loads((directory/'pre_execution_source.json').read_text())
    raw = (directory/'source.tar.gz').read_bytes()
    if sha(raw) != receipt['archive_sha256']:
        raise ValueError('archived executable source digest mismatch')
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(receipt['files']) or {m.name for m in members} != set(receipt['files']):
            raise ValueError('archived executable source inventory mismatch')
        for member in members:
            if not member.isfile() or sha(archive.extractfile(member).read()) != receipt['files'][member.name]:
                raise ValueError('archived executable source member mismatch')


def verify_inventory():
    inventory = json.loads((EVIDENCE/'inventory.json').read_text())
    stages = ('collection','fit','benchmark_attempt1','benchmark','continuation')
    expected = {str(p.relative_to(EVIDENCE)) for stage in stages
                for p in (EVIDENCE/stage).rglob('*') if p.is_file()}
    if set(inventory['files']) != expected:
        raise ValueError('retained file inventory is incomplete or contains extra members')
    for name, digest in inventory['files'].items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or sha((EVIDENCE/p).read_bytes()) != digest:
            raise ValueError('retained evidence file hash mismatch')
    for stage in stages:
        verify_source(EVIDENCE/stage)
    return len(inventory['files'])


def same_answer(actual, expected):
    answer, work = actual
    if (answer.flatten().tolist() != expected['answer'] or work != expected['work']
            or work['valid'] != expected['valid']):
        raise AssertionError('actual frozen solve differs from retained answer/work')


def verify(out, replay=False, refit=False):
    begin = time.perf_counter()
    before_source = source_identity()
    files = verify_inventory()
    cases, collection = load_cases(EVIDENCE/'collection')
    policies, fit = load_policies(EVIDENCE/'fit', cases, collection)
    case_map = {(c['family'],c['core_seed'],c['example_id']):c for c in cases}
    if collection['coverage'] != {f:coverage([c for c in cases if c['family']==f]) for f in FAMILY_SPECS}:
        raise ValueError('coverage summary differs from full trajectory union')
    predictions = read_bound_rows(EVIDENCE/'fit/predictions.jsonl.gz', fit['predictions_sha256'], 512*6*3)
    seen = set()
    for r in predictions:
        key = (r['family'],r['core_seed'],r['example_id'])
        case = case_map[key]
        full_key = (*key,r['arm'],r['budget'])
        if full_key in seen or r['arm'] not in ARMS[:6] or r['budget'] not in (12,16,20):
            raise ValueError('invalid or repeated prediction inventory')
        seen.add(full_key)
        if r['group'] != case['group'] or r['fold'] != case['fold']:
            raise ValueError('prediction differs from excluded fold')
        expected = simulate(case, selector_for(r['arm'],case,policies,fit),r['budget'])
        if any(r[k] != v for k,v in expected.items()):
            raise ValueError('stored policy prediction differs from observed-history-only replay')
    curves = {family: {arm: {str(budget): {
        'valid':sum(r['valid'] for r in predictions if r['family']==family and r['arm']==arm and r['budget']==budget),
        'transitions':sum(r['transitions'] for r in predictions if r['family']==family and r['arm']==arm and r['budget']==budget)}
        for budget in (12,16,20)} for arm in ARMS[:6]} for family in FAMILY_SPECS}
    if curves != fit['curves']:
        raise ValueError('cross-fit curve summary differs from complete predictions')
    fitted = 0
    max_coefficient_error = 0.
    if refit:
        for family in FAMILY_SPECS:
            for fold in range(5):
                train, _ = crossfit_partition([c for c in cases if c['family']==family],fold)
                key = f'{family}_{fold}'
                if list(best_static_order(train)) != fit['folds'][key]['order']:
                    raise AssertionError('training-fold static optimum changed')
                for mode in MODES:
                    actual = fit_selector(train,mode)
                    error = float(np.max(np.abs(actual.weights-policies[key][mode].weights)))
                    max_coefficient_error = max(max_coefficient_error,error)
                    if error > 1e-10:
                        raise AssertionError('refitted controller coefficients differ beyond declared tolerance')
                    fitted += 1
                print(f'REFIT {key}',flush=True)
    benchmark = json.loads((EVIDENCE/'benchmark/summary.json').read_text())
    if (benchmark['status']!='COMPLETE' or benchmark['collection_sha256']!=collection['cases_sha256']
            or benchmark['policies_sha256']!=fit['policies_sha256']):
        raise ValueError('benchmark is not bound to the accepted collection and policies')
    rows = read_bound_rows(EVIDENCE/'benchmark/rows.jsonl.gz',benchmark['rows_sha256'],512*9*3)
    if benchmark['comparisons'] != timed_analysis(rows,cases,collection) or benchmark['primary_pilot_gate_pass']:
        raise ValueError('primary statistical report/failed gate differs')
    continuation = json.loads((EVIDENCE/'continuation/summary.json').read_text())
    if continuation['status']!='COMPLETE' or continuation['collection_sha256']!=collection['cases_sha256']:
        raise ValueError('continuation is not bound to the accepted collection')
    extra = read_bound_rows(EVIDENCE/'continuation/extended.jsonl.gz',continuation['extended_sha256'],512)
    extended = {(r['family'],r['core_seed'],r['example_id']):r['steps'] for r in extra}
    if len(extended) != len(extra) or set(extended) != set(case_map):
        raise ValueError('incomplete or repeated extended trajectory inventory')
    for key, steps in extended.items():
        if len(steps) != 32 or steps[:4] != case_map[key]['views'][0]:
            raise ValueError('extended trajectory changed its original prefix')
    extended_rows = read_bound_rows(EVIDENCE/'continuation/rows.jsonl.gz',continuation['rows_sha256'],512*7*3)
    if continuation['comparisons'] != extended_analysis(extended_rows,collection):
        raise ValueError('continuation statistical report differs')
    # The failed aggregation attempt remains a full, separate timing inventory.
    first = json.loads((EVIDENCE/'benchmark_attempt1/failure.json').read_text())
    if first['type'] != 'IndexError':
        raise ValueError('unexpected initial failure record')
    first_rows = [json.loads(line) for line in gzip.decompress((EVIDENCE/'benchmark_attempt1/rows.jsonl.gz').read_bytes()).splitlines()]
    first_report = timed_analysis(first_rows,cases,collection)
    if any(r['prefix_only']['descriptive_gate_pass'] for r in first_report.values()):
        raise ValueError('initial attempt unexpectedly passed the quality gate')
    independent = 0
    for key, case in case_map.items():
        spec = FAMILY_SPECS[case['family']][0]
        inp = torch.tensor([case['input']],dtype=torch.int64)
        all_steps = case['identity']+[s for v in case['views'] for s in v]+extended[key]
        for step in all_steps:
            Observation(tuple(case['input']),tuple(step['answer']))
            if type(step['valid']) is not bool:
                raise ValueError('nonboolean extended validity')
            answer = torch.tensor([step['answer']],dtype=torch.int64)
            if bool(spec.independent_correct(inp,answer)) != step['valid']:
                raise AssertionError('stored validity differs from independent original-input check')
            independent += 1
    # Check all rounds against the observed-history simulator, including outputs
    # from the failed first timing attempt, rather than trusting summary counts.
    for r in rows+first_rows:
        key = (r['family'],r['core_seed'],r['example_id'])
        case = case_map[key]
        if r['arm'] in ARMS[:6]:
            expected = simulate(case,selector_for(r['arm'],case,policies,fit))
        elif r['arm'] != 'symbolic':
            expected = predict(case,extended[key],r['arm'])
        else:
            spec = FAMILY_SPECS[r['family']][0]
            if not r['valid'] or not spec.independent_correct(torch.tensor([case['input']],dtype=torch.int64),
                    torch.tensor([r['answer']],dtype=torch.int64)):
                raise ValueError('classical comparison changed')
            continue
        if any([r['valid'] != expected['valid'],r['answer'] != expected['answer'],
                r['work']['transitions'] != expected['transitions']]):
            raise AssertionError('benchmark differs from trajectory counterfactual')
        if r['arm'] in ARMS[:6] and r['arm'] != 'original_order' and r['work']['actions'] != expected['actions']:
            raise AssertionError('benchmark decisions differ from history-limited selector')
    for r in extended_rows:
        if r['arm']=='symbolic':
            case = case_map[(r['family'],r['core_seed'],r['example_id'])]
            spec = FAMILY_SPECS[r['family']][0]
            if not r['valid'] or not spec.independent_correct(torch.tensor([case['input']],dtype=torch.int64),
                    torch.tensor([r['answer']],dtype=torch.int64)):
                raise ValueError('classical follow-up changed')
            continue
        key = (r['family'],r['core_seed'],r['example_id'])
        expected = predict(case_map[key],extended[key],r['arm'])
        if any([r['valid'] != expected['valid'],r['answer'] != expected['answer'],
                r['work']['transitions'] != expected['transitions']]):
            raise AssertionError('follow-up differs from extended-trajectory counterfactual')
    live_answers, live_solves = 0, 0
    if replay:
        historical, accepted, members = sources()
        originals = {(r['family'],r['core_seed'],r['example_id'],r['arm']):r for r in rows if r['round']==0}
        followups = {(r['family'],r['core_seed'],r['example_id'],r['arm']):r for r in extended_rows if r['round']==0}
        with tempfile.TemporaryDirectory() as td:
            for family,(spec,_,_) in FAMILY_SPECS.items():
                for seed,(core,_) in load_sources(family,members,historical,accepted,Path(td)).items():
                    before = tensor_digest(core.state_dict())
                    for case in [c for c in cases if c['family']==family and c['core_seed']==seed]:
                        key = (family,seed,case['example_id'])
                        inp = torch.tensor([case['input']],dtype=torch.int64)
                        views = grid_views(spec)
                        if trajectory(core,inp,spec,views[0],32) != case['identity']:
                            raise AssertionError('identity trajectory replay changed')
                        for a in range(7):
                            if trajectory(core,inp,spec,views[a+1],4) != case['views'][a]:
                                raise AssertionError('alternative trajectory replay changed')
                        if trajectory(core,inp,spec,views[1],32) != extended[key]:
                            raise AssertionError('extended trajectory replay changed')
                        live_answers += 92
                        for arm in ARMS:
                            if arm=='symbolic':
                                actual = symbolic_solve(inp,spec,native=spec.task=='maze')
                            elif arm.startswith('identity_'):
                                actual = symmetry_solve(core,inp,spec,policy='identity',transition_budget=int(arm[9:]))
                            elif arm=='original_order':
                                actual = symmetry_solve(core,inp,spec,transition_budget=20,identity_cycles=8,view_limit=4)
                            else:
                                actual = solve_with_selector(core,inp,spec,selector_for(arm,case,policies,fit))
                            same_answer(actual,originals[(*key,arm)])
                            live_solves += 1
                        for arm in EXTENDED_ARMS:
                            same_answer(execute(core,inp,spec,arm),followups[(*key,arm)])
                            live_solves += 1
                    if tensor_digest(core.state_dict()) != before:
                        raise RuntimeError('frozen model changed during live replay')
                    print(f'REPLAY {family} {seed}',flush=True)
    if source_identity() != before_source:
        raise RuntimeError('verification source changed during execution')
    result = {'status':'PASS', 'evidence_files':files, 'cases':512,
        'independent_stored_answer_checks':independent, 'policy_counterfactuals':len(predictions),
        'timing_rows_checked_including_failed_attempt':len(rows)+len(first_rows)+len(extended_rows),
        'controller_refits':fitted, 'max_coefficient_absolute_error':max_coefficient_error,
        'coefficient_absolute_tolerance':1e-10, 'actual_trajectory_answers':live_answers,
        'actual_complete_solve_comparisons':live_solves,
        'primary_failure_information_gate_pass':False, 'new_confirmation':False,
        'seconds':time.perf_counter()-begin, 'source':before_source}
    write_json(out,result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--replay',action='store_true')
    ap.add_argument('--refit',action='store_true')
    args=ap.parse_args()
    setup()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    result=verify(args.out,args.replay,args.refit)
    print(json.dumps({k:v for k,v in result.items() if k!='source'},indent=2))


if __name__=='__main__':
    main()
