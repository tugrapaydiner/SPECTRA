"""Separately declared matched-preparation eager ablation; does not replace primary."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from deploy.m10_native import load_extension as checker_extension
from spectra.fp_runtime import PreparedFPSudoku, _validate, _environment
from spectra.fp_evidence import quantile, correct
from scripts.bench_trained_fp import (sources,load_models,write,read,executable,freeze_files,
                                      check_files,setup_cpu,audit_answer)

ARMS=('eager_prepared','native_prepared')
ORDER_SEED=9161211


class PreparedEagerControl:
    """Owning matched-setup ablation, not a new production API."""
    @torch.inference_mode()
    def __init__(self,model):
        _validate(model);self.model=copy.deepcopy(model)
        self.position=self.model.encode_positions(torch.zeros(1,16,dtype=torch.int64),4,4)
        self.checker=checker_extension().SudokuProblem

    @torch.inference_mode()
    def solve(self,x,max_steps=4):
        _environment()
        if type(max_steps) is not int or not 1<=max_steps<=self.model.N_sup:
            raise ValueError('unsupported supervision budget')
        if (type(x) is not torch.Tensor or x.device.type!='cpu' or x.dtype!=torch.int64 or
                x.layout!=torch.strided or x.shape!=(1,16)):
            raise ValueError('expected CPU int64 [1,16]')
        x=x.contiguous();checker=self.checker(x,2)
        emb=self.model.token_embed(x)+self.position;y,z=torch.zeros_like(emb),torch.zeros_like(emb)
        for step in range(1,max_steps+1):
            y,z=self.model.recursive_cycle(emb,y,z)
            answer,valid=checker.decode(self.model.out_head(y).contiguous())
            if valid:break
        return answer,{'executed_steps':step,'block_applications':step*2*len(self.model.blocks),
                       'semantic_checks':step,'checker_constructions':1,'final_semantic':bool(valid),
                       'target_used':False,'checker':'native_exact_sudoku_v1',
                       'stop_reason':'semantic_valid' if valid else 'budget_exhausted'}


def analyze(cases,rows):
    if len(rows)!=len(cases)*7*2 or len({c['case_id'] for c in cases})!=len(cases):
        raise ValueError('incomplete ablation matrix')
    rng=random.Random(ORDER_SEED);cursor=0;times={};recorded={}
    for case in cases:
        cid=case['case_id']
        for round_id in range(7):
            order=list(ARMS);rng.shuffle(order)
            for position,arm in enumerate(order):
                r=rows[cursor];cursor+=1
                if (r['case_id'],r['round'],r['order'],r['arm'])!=(cid,round_id,position,arm):
                    raise ValueError('case/round/order/arm mismatch')
                if type(r['elapsed_ns']) is not int or r['elapsed_ns']<=0:raise ValueError('invalid timer')
                w=r['work'];steps=w['executed_steps']
                if (type(r['valid']) is not bool or r['valid']!=correct(case['input'],r['answer']) or
                    w['final_semantic'] is not r['valid'] or type(steps) is not int or not 1<=steps<=4 or
                    w['semantic_checks']!=steps or w['checker_constructions']!=1 or
                    w['block_applications']!=steps*2*case['blocks'] or w['target_used'] is not False or
                    w['stop_reason']!=('semantic_valid' if r['valid'] else 'budget_exhausted')):
                    raise ValueError('invalid answer/work contract')
                if any(type(w.get(k)) is not int for k in ('executed_steps','semantic_checks','checker_constructions','block_applications')):
                    raise ValueError('work counters must be integers, not booleans')
                value=(r['answer'],r['work'],r['valid'])
                if cid in recorded and value!=recorded[cid]:raise ValueError('ablation fidelity failed')
                recorded[cid]=value;times.setdefault((cid,arm),[]).append(r['elapsed_ns'])
    medians={k:statistics.median(v) for k,v in times.items()}
    num=sum(medians[(c['case_id'],ARMS[1])] for c in cases)
    den=sum(medians[(c['case_id'],ARMS[0])] for c in cases)
    by_problem={}
    for c in cases:
        v=by_problem.setdefault((c['family'],c['problem_id']),[0,0]);cid=c['case_id']
        v[0]+=medians[(cid,ARMS[1])];v[1]+=medians[(cid,ARMS[0])]
    families={f:[v for (family,_),v in by_problem.items() if family==f] for f,_ in by_problem}
    rng=random.Random(9161212);samples=[]
    for _ in range(2000):
        a=b=0
        for values in families.values():
            for _ in values:
                n,d=values[rng.randrange(len(values))];a+=n;b+=d
        samples.append(a/b)
    return {'scope':'separate matched-preparation eager ablation; not a replacement primary',
            'cases':len(cases),'observations':len(rows),'native_over_eager_ratio':num/den,
            'problem_bootstrap_ci95':[quantile(samples,.025),quantile(samples,.975)],
            'distinct_valid':sum(v[2] for v in recorded.values()),'exact_answers_and_work':True,
            'case_median_regressions':sum(medians[(c['case_id'],ARMS[1])]>medians[(c['case_id'],ARMS[0])] for c in cases),
            'case_p95_regressions_descriptive':sum(quantile(times[(c['case_id'],ARMS[1])],.95)>
                                                  quantile(times[(c['case_id'],ARMS[0])],.95) for c in cases),
            'case_median_ns':{c['case_id']:{a:medians[(c['case_id'],a)] for a in ARMS} for c in cases}}


@torch.inference_mode()
def run(out):
    out.mkdir(parents=True,exist_ok=False);env=setup_cpu();before=executable()
    historical,cases,receipts=sources()
    write(out/'freeze.json',{'source':before,'cases':cases,'receipts':receipts,'environment':env,
         'protocol_comment':5648284600,'rounds':7,'order_seed':ORDER_SEED})
    models,native,prep=load_models(historical,out/'checkpoints');eager={s:PreparedEagerControl(m) for s,m in models.items()}
    for c in cases[:4]:
        x=torch.tensor(c['input'],dtype=torch.int64)[None]
        for seed in models:eager[seed].solve(x);native[seed].solve(x)
    rng=random.Random(ORDER_SEED);rows=[]
    with (out/'rows.jsonl').open('x') as f:
        for c in cases:
            x=torch.tensor(c['input'],dtype=torch.int64)[None];s=c['seed']
            functions={ARMS[0]:eager[s].solve,ARMS[1]:native[s].solve}
            for round_id in range(7):
                order=list(ARMS);rng.shuffle(order)
                for position,arm in enumerate(order):
                    start=time.perf_counter_ns();a,w=functions[arm](x);elapsed=time.perf_counter_ns()-start
                    row={'case_id':c['case_id'],'round':round_id,'order':position,'arm':arm,
                         'elapsed_ns':elapsed,'answer':a.flatten().tolist(),'work':w,'valid':audit_answer(x,a)}
                    rows.append(row);f.write(json.dumps(row,sort_keys=True)+'\n')
            f.flush()
    result=analyze(cases,rows)
    if before!=executable():raise ValueError('source changed during ablation')
    write(out/'summary.json',result);freeze_files(out)
    print(json.dumps({k:v for k,v in result.items() if k!='case_median_ns'},indent=2))


@torch.inference_mode()
def verify(out,replay):
    setup_cpu();check_files(out);freeze=read(out/'freeze.json');historical,cases,receipts=sources()
    if freeze['source']!=executable() or freeze['cases']!=cases or freeze['receipts']!=receipts:
        raise ValueError('source or fixture identity changed')
    if freeze['rounds']!=7 or freeze['order_seed']!=ORDER_SEED or freeze['protocol_comment']!=5648284600:
        raise ValueError('configuration changed')
    rows=[json.loads(line) for line in (out/'rows.jsonl').read_text().splitlines()]
    if analyze(cases,rows)!=read(out/'summary.json'):raise ValueError('raw analysis mismatch')
    count=0
    if replay:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='spectra-eager-replay-') as tmp:
            models,native,_=load_models(historical,Path(tmp));eager={s:PreparedEagerControl(m) for s,m in models.items()}
            lookup={(r['case_id'],r['arm']):r for r in rows if r['round']==0}
            for c in cases:
                x=torch.tensor(c['input'],dtype=torch.int64)[None];s=c['seed']
                for arm,f in [(ARMS[0],eager[s].solve),(ARMS[1],native[s].solve)]:
                    a,w=f(x);r=lookup[(c['case_id'],arm)]
                    if a.flatten().tolist()!=r['answer'] or w!=r['work'] or audit_answer(x,a)!=r['valid']:
                        raise ValueError('ablation execution mismatch')
                    count+=1
    print(json.dumps({'status':'PASS','observations':len(rows),'execution_replays':count}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','verify'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--replay',action='store_true');a=p.parse_args()
    run(a.out) if a.command=='run' else verify(a.out,a.replay)
