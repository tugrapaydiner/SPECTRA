"""Measure real Inductor controls without weakening the existing exact runtime."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import traceback

# Freezing affects Dynamo parameter lifting at import time, not just Inductor.
if __name__ == '__main__':
    os.environ.setdefault('TORCHINDUCTOR_FREEZING','1')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts.bench_trained_fp import sources, load_models, setup_cpu, source_files, bits, audit_answer
from scripts.bench_fp_controls import PreparedEagerControl
from spectra.compiler_runtime import CompilerFPSudoku, compiler_counters
from spectra.compiler_evidence import ARMS, ROUNDS, ORDER_SEED, analyze, schedule

PROTOCOL='docs/COMPILER_BASELINE_PROTOCOL.md'
PROTOCOL_COMMIT='8b35c0879864fa8fd958d457d2442d173a32d784'


def digest(b):return hashlib.sha256(b).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def read(path):return json.loads(Path(path).read_text())
def jsonlines(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def executable():
    return {**source_files(),PROTOCOL:digest((ROOT/PROTOCOL).read_bytes())}


def seal(root):
    write(root/'SHA256.json',{str(p.relative_to(root)):digest(p.read_bytes())
                             for p in sorted(root.rglob('*')) if p.is_file()})


def integrity(root):
    root=Path(root);manifest=read(root/'SHA256.json')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}-{'SHA256.json'}
    if actual!=set(manifest):raise ValueError('missing/unexpected evidence file')
    for name,sha in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or digest(p.read_bytes())!=sha:
            raise ValueError('unsafe path or evidence digest mismatch')


@torch.inference_mode()
def reference_trace(model,x):
    emb=model.token_embed(x)+model.encode_positions(x,4,4);y,z=torch.zeros_like(emb),torch.zeros_like(emb)
    states=[]
    for _ in range(4):
        y,z=model.recursive_cycle(emb,y,z);states.append((y.clone(),z.clone(),model.out_head(y).contiguous().clone()))
    return emb,states


@torch.inference_mode()
def trace_record(cid,arm,value,ref):
    emb,states=value;ref_emb,ref_states=ref
    result={'case_id':cid,'arm':arm,'embedding_sha256':digest(bits(emb)),
            'embedding_bitwise':bits(emb)==bits(ref_emb),'steps':[]}
    for tensors,refs in zip(states,ref_states):
        finite=all(bool(torch.isfinite(t).all()) for t in tensors)
        diffs=[float((t.double()-r.double()).abs().max()) if finite else None for t,r in zip(tensors,refs)]
        result['steps'].append({'y_sha256':digest(bits(tensors[0])), 'z_sha256':digest(bits(tensors[1])),
                'logits_sha256':digest(bits(tensors[2])), 'finite':finite,
                'bitwise':all(bits(t)==bits(r) for t,r in zip(tensors,refs)),
                'max_abs_y':diffs[0], 'max_abs_z':diffs[1], 'max_abs_logits':diffs[2]})
    return result


@torch.inference_mode()
def prepare(historical,cases,out):
    models,native,prep=load_models(historical,out/'checkpoints')
    runtimes={s:{'eager':PreparedEagerControl(m),'prepared':native[s]} for s,m in models.items()}
    setup={'native':prep,'compilers':{}}
    # Whole-solve capture is a diagnostic only. Native checker/stopping are not
    # required to compile for the regional competitor to be valid.
    s=min(models);x=torch.tensor(cases[0]['input'],dtype=torch.int64)[None]
    start=time.perf_counter_ns()
    try:
        whole=torch.compile(runtimes[s]['eager'].solve,backend='inductor',fullgraph=True,dynamic=False)
        a,w=whole(x);setup['whole_solve_probe']={'status':'CAPTURED','valid':audit_answer(x,a),'work':w}
    except Exception as exc:
        setup['whole_solve_probe']={'status':'UNSUPPORTED','exception':type(exc).__name__,
                                    'message':str(exc),'traceback':traceback.format_exc()}
    setup['whole_solve_probe']['elapsed_ns']=time.perf_counter_ns()-start
    write(out/'whole_solve_probe.json',setup['whole_solve_probe'])
    for seed,m in models.items():
        for arm,mode in [('inductor_default','default'),('inductor_frozen','max-autotune')]:
            cache=Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
            files_before={str(p.relative_to(cache)) for p in cache.rglob('*.py')} if cache.exists() else set()
            before=compiler_counters();start=time.perf_counter_ns()
            try:
                rt=CompilerFPSudoku(m,mode=mode);construction=time.perf_counter_ns()-start
                start=time.perf_counter_ns();rt.trace(x,4);initial=time.perf_counter_ns()-start
                for c in cases[:2]:
                    xx=torch.tensor(c['input'],dtype=torch.int64)[None];rt.trace(xx,4);rt.solve(xx,4)
                runtimes[seed][arm]=rt
                receipt={'status':'READY','construction_ns':construction,'first_full_trace_ns':initial,
                         'identity':rt.identity(),'before':before,'after':compiler_counters(),
                         'new_generated_python':{str(p.relative_to(cache)):digest(p.read_bytes()) for p in cache.rglob('*.py')
                            if str(p.relative_to(cache)) not in files_before}}
                setup['compilers'][f'{seed}:{arm}']=receipt
                write(out/f'setup_{seed}_{arm}.json',receipt)
                print('ready',seed,arm,'first full trace seconds',initial/1e9,flush=True)
            except Exception as exc:
                write(out/f'setup_{seed}_{arm}_FAILED.json',{'status':'FAILED','exception':type(exc).__name__,
                    'message':str(exc),'traceback':traceback.format_exc(),'elapsed_ns':time.perf_counter_ns()-start})
                raise
    for seed in models:
        for c in cases[:2]:
            xx=torch.tensor(c['input'],dtype=torch.int64)[None]
            for rt in runtimes[seed].values():rt.solve(xx,4)
    return models,runtimes,setup


def snapshot_codegen(cache,out):
    manifest={};dest=out/'generated_sources';dest.mkdir()
    counts={'source_files':0,'frozen_parameter_files':0,'packed_kernel_files':0}
    for p in sorted(cache.rglob('*')):
        if not p.is_file():continue
        rel=str(p.relative_to(cache));raw=p.read_bytes();manifest[rel]={'sha256':digest(raw),'bytes':len(raw)}
        if p.suffix in {'.py','.cpp','.h'}:
            q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw);counts['source_files']+=1
            text=raw.decode('utf-8',errors='replace')
            counts['frozen_parameter_files']+=('_frozen_param' in text)
            counts['packed_kernel_files']+=('cpp_fused' in text and ('micro_gemm' in text or 'packed_gemm' in text))
    write(out/'compiler_cache_manifest.json',manifest)
    write(out/'codegen_inventory.json',counts)
    return counts


@torch.inference_mode()
def run(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    cache=out.parent/(out.name+'-inductor-cache');cache.mkdir(exist_ok=False)
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(cache)
    env=setup_cpu();before=executable();historical,cases,receipts=sources()
    import torch._dynamo.config as dc
    import torch._inductor.config as ic
    freeze={'schema':'spectra.compiler_baseline.v1','protocol_commit':PROTOCOL_COMMIT,
            'source':before,'cases':cases,'sources':receipts,'environment':env,
            'config':{'arms':list(ARMS),'rounds':ROUNDS,'order_seed':ORDER_SEED},
            'compiler_env':{'TORCHINDUCTOR_FREEZING':os.environ.get('TORCHINDUCTOR_FREEZING'),
                    'inline_inbuilt_nn_modules':dc.inline_inbuilt_nn_modules,'global_freezing':ic.freezing,'cache':str(cache),
                    'cache_created_empty':True,'cpu_capability':torch._C._get_cpu_capability()}}
    write(out/'freeze.json',freeze)
    with tarfile.open(out/'execution_source.tar.gz','w:gz') as tar:
        for name in before:tar.add(ROOT/name,arcname=name)
    models,runtimes,setup=prepare(historical,cases,out)
    write(out/'setup.json',setup);warm_before=compiler_counters();write(out/'warm_before.json',warm_before)
    rows=[]
    inputs={c['case_id']:torch.tensor(c['input'],dtype=torch.int64)[None] for c in cases}
    with (out/'rows.jsonl').open('x') as stream:
        for case,r,order,arm in schedule(cases):
            x=inputs[case['case_id']];rt=runtimes[case['seed']][arm]
            start=time.perf_counter_ns();a,w=rt.solve(x,4);elapsed=time.perf_counter_ns()-start
            row={'case_id':case['case_id'],'round':r,'order':order,'arm':arm,'elapsed_ns':elapsed,
                 'answer':a.flatten().tolist(),'work':w,'valid':audit_answer(x,a)}
            rows.append(row);stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            if order==len(ARMS)-1:stream.flush()
    warm_after=compiler_counters();write(out/'warm_after.json',warm_after)
    if warm_after.get('stats',{}).get('unique_graphs',0)!=warm_before.get('stats',{}).get('unique_graphs',0):
        raise ValueError('new compiler capture during supposedly warm timing')
    traces=[]
    with (out/'traces.jsonl').open('x') as stream:
        for case in cases:
            cid=case['case_id'];x=inputs[cid];seed=case['seed'];ref=reference_trace(models[seed],x)
            for arm in ARMS:
                value=ref if arm=='eager' else runtimes[seed][arm].trace(x,4)
                t=trace_record(cid,arm,value,ref);traces.append(t);stream.write(json.dumps(t,sort_keys=True,allow_nan=False)+'\n')
            stream.flush()
    result=analyze(cases,rows,traces);write(out/'summary.json',result)
    counts=snapshot_codegen(cache,out)
    if counts['frozen_parameter_files']==0:raise ValueError('frozen weights not demonstrated in generated code')
    if before!=executable():raise ValueError('executable changed during measurement')
    seal(out)
    print(json.dumps({k:v for k,v in result.items() if k!='case_median_ns'},indent=2),flush=True)


@torch.inference_mode()
def verify(out,replay):
    out=out.resolve();integrity(out);freeze=read(out/'freeze.json')
    historical,cases,receipts=sources()
    if (freeze['source']!=executable() or freeze['cases']!=cases or freeze['sources']!=receipts
            or freeze['protocol_commit']!=PROTOCOL_COMMIT or freeze['schema']!='spectra.compiler_baseline.v1'
            or freeze['config']!={'arms':list(ARMS),'rounds':ROUNDS,'order_seed':ORDER_SEED}):
        raise ValueError('source/fixture/protocol/configuration identity changed')
    rows=jsonlines(out/'rows.jsonl');traces=jsonlines(out/'traces.jsonl')
    if analyze(cases,rows,traces)!=read(out/'summary.json'):raise ValueError('raw analysis differs from summary')
    before,after=read(out/'warm_before.json'),read(out/'warm_after.json')
    if before.get('stats',{}).get('unique_graphs',0)<=0 or before.get('stats')!=after.get('stats'):
        raise ValueError('missing compilation or new captured graph during warm measurement')
    setup=read(out/'setup.json')
    if set(setup['compilers'])!={f'{s}:{a}' for s in (1401,2402) for a in ARMS[2:]}:
        raise ValueError('incomplete compiler configuration coverage')
    for key,v in setup['compilers'].items():
        ident=v['identity'];frozen=key.endswith('inductor_frozen')
        if (v['status']!='READY' or ident['backend']!='inductor' or ident['fullgraph'] is not True
            or ident['dynamic'] is not False or ident['fallback_on_error'] is not False
            or ident['options']['freezing'] is not frozen or (frozen and ident['options'].get('max_autotune') is not True)):
            raise ValueError('unsupported or weakened compiler configuration')
    if read(out/'codegen_inventory.json')['frozen_parameter_files']==0:raise ValueError('no frozen code evidence')
    count=steps=0
    if replay:
        import tempfile
        setup_cpu()
        with tempfile.TemporaryDirectory(prefix='spectra-compiler-replay-') as tmp:
            tmp=Path(tmp);os.environ['TORCHINDUCTOR_CACHE_DIR']=str(tmp/'cache')
            models,runtimes,_=prepare(historical,cases,tmp)
            lookup={(r['case_id'],r['arm']):r for r in rows if r['round']==0}
            saved={(t['case_id'],t['arm']):t for t in traces}
            for case in cases:
                cid=case['case_id'];seed=case['seed'];x=torch.tensor(case['input'],dtype=torch.int64)[None]
                ref=reference_trace(models[seed],x)
                for arm in ARMS:
                    rt=runtimes[seed][arm];a,w=rt.solve(x,4);r=lookup[(cid,arm)]
                    if a.flatten().tolist()!=r['answer'] or w!=r['work'] or audit_answer(x,a)!=r['valid']:
                        raise ValueError('complete solve replay mismatch')
                    val=ref if arm=='eager' else rt.trace(x,4)
                    if trace_record(cid,arm,val,ref)!=saved[(cid,arm)]:raise ValueError('strict trajectory replay mismatch')
                    count+=1;steps+=4
    print(json.dumps({'status':'PASS','observations':len(rows),'execution_replays':count,'step_replays':steps}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','verify'])
    p.add_argument('--out',required=True,type=Path);p.add_argument('--replay',action='store_true')
    args=p.parse_args()
    try:run(args.out) if args.command=='run' else verify(args.out,args.replay)
    except Exception:
        traceback.print_exc();raise
