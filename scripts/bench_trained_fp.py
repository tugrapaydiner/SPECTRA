"""Prepare, measure and independently replay the declared trained FP workload."""
from __future__ import annotations
import argparse
import cProfile
import hashlib
import io
import json
import math
from pathlib import Path
import pstats
import random
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from data import sudoku
from deploy.m10_native import load_extension as checker_extension
from deploy.semantic_exit import native_semantic_exit
from scripts.m14_attempt5_dual_stream_semantic_exit import dual_stream_semantic_exit_solve
from scripts.m16_evidence import Evidence, CORE_SHA
from scripts.m17_sources import AcceptedM16
from scripts.verify_retained_results import read_bound_archive, M17_INVENTORY_SHA, M17_ZIP_SHA
from scripts.bench_integrated_runtime import setup_cpu, source_files
from spectra.fp_runtime import PreparedFPSudoku, load_extension
from spectra.fp_evidence import ARMS, ROUNDS, ORDER_SEED, analyze


def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')
def bits(t): return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
def executable():
    return {**source_files(), 'docs/TRAINED_FP_PROTOCOL.md':sha((ROOT/'docs/TRAINED_FP_PROTOCOL.md').read_bytes())}


def sources():
    historical=Evidence(ROOT/'results/m16/sources')
    m16=AcceptedM16(ROOT/'results/m16/runs/accepted-34522192590.tar.gz',historical)
    m17=read_bound_archive(ROOT/'results/m17/runs/34534009702-33e548d3e0f52d0664c00929316b2198d5fcffe4.tar.gz',
                          inventory_sha=M17_INVENTORY_SHA,zip_sha=M17_ZIP_SHA)
    cases=[]; receipts={}
    selected=[('ordinary',m16.members,'experiment/manifests/seed2026091601'),
              ('shifted',m17,'experiment/sudoku_shift/manifests/seed2026091703')]
    for family,members,prefix in selected:
        raw=members[prefix+'_arrays.npz'];manifest_bytes=members[prefix+'.json']
        manifest=json.loads(manifest_bytes)
        with np.load(io.BytesIO(raw),allow_pickle=False) as arrays: inputs=arrays['test_inputs'].copy()
        entries=manifest['splits']['test']['examples']
        if inputs.shape!=(128,16) or len(entries)!=128: raise ValueError('unexpected source geometry')
        receipts[family]={'arrays_member':prefix+'_arrays.npz','arrays_sha256':sha(raw),
                          'manifest_member':prefix+'.json','manifest_sha256':sha(manifest_bytes)}
        for seed in sorted(CORE_SHA):
            for i,(x,entry) in enumerate(zip(inputs,entries)):
                cases.append({'case_id':f'{family}:{seed}:{i}', 'family':family,'seed':seed,
                              'problem_id':f'{family}:{i}', 'source_id':entry['id'],
                              'input':[int(v) for v in x], 'model_sha256':CORE_SHA[seed],'blocks':1})
    return historical,cases,receipts


def load_models(historical,out):
    models={};prepared={};prep={}
    for seed in sorted(CORE_SHA):
        model,identity=historical.load_model('fp_recursive_dim64',seed,out)
        if identity!=CORE_SHA[seed]: raise ValueError('checkpoint mismatch')
        start=time.perf_counter_ns(); runtime=PreparedFPSudoku(model); elapsed=time.perf_counter_ns()-start
        models[seed]=model;prepared[seed]=runtime
        prep[str(seed)]={'construction_ns':elapsed,'identity':runtime.identity()}
    return models,prepared,prep


@torch.inference_mode()
def symbolic(x):
    checker=checker_extension().SudokuProblem(x.contiguous(),2)
    solved=sudoku.solve(x.numpy().reshape(4,4),2)
    answer=x.clone() if solved is None else torch.from_numpy(solved.reshape(1,16))
    valid=bool(checker.check(answer))
    return answer,{'executed_steps':0,'block_applications':0,'semantic_checks':1,
                   'checker_constructions':1,'final_semantic':valid,'target_used':False,
                   'stop_reason':'symbolic_complete' if valid else 'symbolic_unknown'}


def arm_functions(model,prepared):
    return {'python':lambda x:dual_stream_semantic_exit_solve(model,x,4),
            'native':lambda x:native_semantic_exit(model,x,4),
            'prepared':lambda x:prepared.solve(x,4),'symbolic':symbolic}


def audit_answer(x,answer):
    a=answer.numpy().reshape(4,4);p=x.numpy().reshape(4,4)
    return bool(sudoku.is_solved(a,2) and sudoku.respects_clues(p,a,2))


@torch.inference_mode()
def trajectory(model,prepared,x):
    em,states=prepared.trace(x,4)
    ref=model.token_embed(x)+model.encode_positions(x,4,4)
    if bits(em)!=bits(ref): raise ValueError('input encoding bits changed')
    y,z=torch.zeros_like(ref),torch.zeros_like(ref);hashes=[]
    for Y,Z,L in states:
        y,z=model.recursive_cycle(ref,y,z);logits=model.out_head(y).contiguous()
        if any(bits(a)!=bits(b) for a,b in [(Y,y),(Z,z),(L,logits)]):
            raise ValueError('per-step FP state/logit bits changed')
        hashes.append(sha(bits(y)+bits(z)+bits(logits)))
    return {'embedding_sha256':sha(bits(ref)),'step_sha256':hashes}


def freeze_files(out):
    inventory={str(p.relative_to(out)):sha(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'SHA256.json',inventory)


def check_files(out):
    inventory=read(out/'SHA256.json')
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}-{'SHA256.json'}
    if set(inventory)!=actual: raise ValueError('unexpected/missing evidence files')
    for name,expected in inventory.items():
        path=(out/name).resolve()
        if not path.is_relative_to(out.resolve()) or sha(path.read_bytes())!=expected:
            raise ValueError('evidence path/digest mismatch')


@torch.inference_mode()
def run(out):
    out.mkdir(parents=True,exist_ok=False); environment=setup_cpu();before=executable()
    historical,cases,receipts=sources()
    write(out/'freeze.json',{'protocol_commit':'3cb7073535d0d4a6aa06dd5f418b0b4407eeb0d6',
         'source':before,'config':{'rounds':ROUNDS,'arms':ARMS,'order_seed':ORDER_SEED},
         'sources':receipts,'cases':cases,'environment':environment})
    # Preserve the exact executable; timings cannot be reinterpreted under later edits.
    with tarfile.open(out/'execution_source.tar.gz','w:gz') as archive:
        for name in before: archive.add(ROOT/name,arcname=name)
    start=time.perf_counter_ns();load_extension();checker_extension();load_ns=time.perf_counter_ns()-start
    models,prepared,prep=load_models(historical,out/'checkpoints')
    write(out/'preparation.json',{'native_load_ns':load_ns,'library_cache_scope':'may reuse compiled extensions; not necessarily a cold compiler run',
                                  'models':prep,'energy_joules':None})
    first=[]
    for seed,model in models.items():
        x=torch.tensor(cases[0]['input'],dtype=torch.int64)[None]
        for arm,f in arm_functions(model,prepared[seed]).items():
            start=time.perf_counter_ns();a,w=f(x);elapsed=time.perf_counter_ns()-start
            first.append({'seed':seed,'arm':arm,'elapsed_ns':elapsed,'valid':audit_answer(x,a),'work':w})
    write(out/'first_calls.json',first)
    # Constant warmups, not discarded timing rounds.
    for seed,model in models.items():
        for case in cases[:4]:
            x=torch.tensor(case['input'],dtype=torch.int64)[None]
            for f in arm_functions(model,prepared[seed]).values(): f(x)
    rng=random.Random(ORDER_SEED);rows=[]
    with (out/'rows.jsonl').open('x') as stream:
        for case in cases:
            x=torch.tensor(case['input'],dtype=torch.int64)[None]
            arms=arm_functions(models[case['seed']],prepared[case['seed']])
            for round_id in range(ROUNDS):
                order=list(ARMS);rng.shuffle(order)
                for position,arm in enumerate(order):
                    start=time.perf_counter_ns();answer,work=arms[arm](x);elapsed=time.perf_counter_ns()-start
                    valid=audit_answer(x,answer)
                    if valid!=bool(work['final_semantic']): raise ValueError('external validity audit failed')
                    row={'case_id':case['case_id'],'round':round_id,'order':position,'arm':arm,
                         'elapsed_ns':elapsed,'answer':answer.flatten().tolist(),'valid':valid,
                         'steps':int(work['executed_steps']),'work':work}
                    rows.append(row);stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            stream.flush()
    summary=analyze(cases,rows)
    total_saving=(summary['arms']['native']['summed_case_median_ns']-summary['arms']['prepared']['summed_case_median_ns'])/len(cases)
    summary['cached_library_preparation_break_even_solves']=(math.ceil((load_ns+sum(p['construction_ns'] for p in prep.values()))/total_saving)
                                                           if total_saving>0 else None)
    traces={}
    for case in cases:
        x=torch.tensor(case['input'],dtype=torch.int64)[None];seed=case['seed']
        traces[case['case_id']]=trajectory(models[seed],prepared[seed],x)
    write(out/'trajectories.json',traces)
    # This diagnostic is separate from uninstrumented timing observations.
    profiles={}
    for arm in ('native','prepared'):
        profiler=cProfile.Profile();profile_start=time.perf_counter_ns();profiler.enable()
        for case in cases[:32]:
            x=torch.tensor(case['input'],dtype=torch.int64)[None]
            arm_functions(models[case['seed']],prepared[case['seed']])[arm](x)
        profiler.disable();profile_wall_ns=time.perf_counter_ns()-profile_start;profiler.dump_stats(str(out/f'{arm}.pstats'))
        output=io.StringIO();stats=pstats.Stats(profiler,stream=output).sort_stats('cumulative');stats.print_stats(25)
        profiles[arm]={'total_calls':stats.total_calls,'python_profile_attributed_seconds':stats.total_tt,'instrumented_wall_ns':profile_wall_ns,
                       'scope':'instrumented diagnostic; cProfile can omit native/released-GIL time; not an uninstrumented speedup estimate','top':output.getvalue()}
    write(out/'profile.json',profiles)
    operator_profiles={}
    for arm in ('native','prepared'):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            for case in cases[:32]:
                x=torch.tensor(case['input'],dtype=torch.int64)[None]
                arm_functions(models[case['seed']],prepared[case['seed']])[arm](x)
        operator_profiles[arm]={item.key:{'count':item.count,'self_cpu_us_instrumented':item.self_cpu_time_total}
                                for item in profiler.key_averages()}
    for key in ('aten::_native_multi_head_attention','aten::rms_norm','aten::linear','aten::gelu'):
        if operator_profiles['native'].get(key,{}).get('count')!=operator_profiles['prepared'].get(key,{}).get('count'):
            raise ValueError('instrumented recurrent operator count changed: '+key)
    write(out/'operator_profile.json',{'scope':'instrumented 32-case diagnostic, not a latency benchmark',
                                       'same_recurrent_operator_counts':True,'profiles':operator_profiles})
    if executable()!=before: raise RuntimeError('executable changed during measurement')
    write(out/'summary.json',summary);freeze_files(out)
    print(json.dumps({k:v for k,v in summary.items() if k!='case_results'},indent=2))


@torch.inference_mode()
def verify(out,replay):
    setup_cpu();check_files(out);freeze=read(out/'freeze.json')
    if freeze['source']!=executable(): raise ValueError('exact executable mismatch; use retained execution source')
    historical,cases,receipts=sources()
    if freeze['cases']!=cases or freeze['sources']!=receipts: raise ValueError('authoritative input fixture mismatch')
    if freeze['config']!={'rounds':ROUNDS,'arms':list(ARMS),'order_seed':ORDER_SEED}: raise ValueError('protocol configuration changed')
    rows=[json.loads(line) for line in (out/'rows.jsonl').read_text().splitlines()]
    summary=analyze(cases,rows);stored=read(out/'summary.json')
    prep=read(out/'preparation.json')
    saving=(summary['arms']['native']['summed_case_median_ns']-summary['arms']['prepared']['summed_case_median_ns'])/len(cases)
    summary['cached_library_preparation_break_even_solves']=(math.ceil((prep['native_load_ns']+sum(p['construction_ns'] for p in prep['models'].values()))/saving) if saving>0 else None)
    if summary!=stored: raise ValueError('summary cannot be recomputed from raw observations')
    count=steps=0
    if replay:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='spectra-fp-replay-') as directory:
            models,prepared,_=load_models(historical,Path(directory))
            lookup={(r['case_id'],r['arm']):r for r in rows if r['round']==0}
            traces=read(out/'trajectories.json')
            if set(traces)!={c['case_id'] for c in cases}: raise ValueError('incomplete trajectory coverage')
            for case in cases:
                x=torch.tensor(case['input'],dtype=torch.int64)[None];seed=case['seed']
                if trajectory(models[seed],prepared[seed],x)!=traces[case['case_id']]: raise ValueError('trajectory replay failed')
                steps+=4
                for arm,f in arm_functions(models[seed],prepared[seed]).items():
                    answer,work=f(x);record=lookup[(case['case_id'],arm)]
                    if answer.flatten().tolist()!=record['answer'] or work!=record['work'] or audit_answer(x,answer)!=record['valid']:
                        raise ValueError('answer/work replay failed')
                    count+=1
    print(json.dumps({'verification':'PASS','observations':len(rows),'execution_replays':count,
                      'full_budget_step_replays':steps,'timing_gate':summary['gate']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['run','verify'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    if args.command=='run': run(args.out)
    else: verify(args.out,args.replay)
if __name__=='__main__': main()
