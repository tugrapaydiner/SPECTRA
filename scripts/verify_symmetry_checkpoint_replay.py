#!/usr/bin/env python3
"""Replay every complete refinement arm after integrating the newer upstream base.

Frozen consumed development only; no timing claim, training or confirmation.
This verifies executable integration, separately from stored-answer validation.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
import time
if __name__=='__main__':
    os.environ.update(ATEN_CPU_CAPABILITY='avx2',MKL_ENABLE_INSTRUCTIONS='AVX2',
        ONEDNN_MAX_CPU_ISA='AVX2',DNNL_MAX_CPU_ISA='AVX2',CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from eval.symmetry_search import symmetry_solve
from scripts.audit_symmetry_proposals import REFINED_ARMS,ARM_CONFIGS
from scripts.m17_cross_task import symbolic_solve
from scripts.m17_models import tensor_digest
from scripts.verify_cpu_progress import bound_run
from scripts.verify_fixed_pool_replay import (Evidence,AcceptedM16,read_bound_archive,M17_INVENTORY_SHA,
    M17_ZIP_SHA,FAMILY_SPECS,load_sources,frozen_inputs,source_identity)
from scripts.m16_evidence import write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if str(torch.__version__)!='2.10.0+cpu' or torch.cuda.is_available():
        raise RuntimeError('frozen CPU replay requires torch 2.10.0+cpu')
    before=source_identity();start=time.perf_counter()
    summary,records=bound_run(ROOT/'results/cpu_progress/attempt2',REFINED_ARMS,
                             'docs/SYMMETRY_PROPOSAL_REFINEMENT_PROTOCOL.json')
    rows={(r['family'],r['core_seed'],r['example_id'],r['arm']):r for r in records if r['round']==0}
    if len(rows)!=2560:raise ValueError('incomplete retained refinement inventory')
    h=Evidence(ROOT/'results/m16/sources')
    a=AcceptedM16(ROOT/'results/m16/runs/accepted-34522192590.tar.gz',h)
    members=read_bound_archive(ROOT/'results/m17/runs/34534009702-33e548d3e0f52d0664c00929316b2198d5fcffe4.tar.gz',
        inventory_sha=M17_INVENTORY_SHA,zip_sha=M17_ZIP_SHA)
    count=0;counts={}
    with tempfile.TemporaryDirectory() as temp, (args.out/'comparisons.jsonl').open('x') as stream:
        for family,(spec,seed,_) in FAMILY_SPECS.items():
            x,ids,_=frozen_inputs(members,f'experiment/{family}/',seed,spec)
            cores=load_sources(family,members,h,a,Path(temp))
            for core_seed,(core,_) in cores.items():
                state_before=tensor_digest(core.state_dict());valid_by_arm={arm:0 for arm in REFINED_ARMS}
                for i,eid in enumerate(ids):
                    inp=x[i:i+1]
                    for arm in REFINED_ARMS:
                        with torch.inference_mode():
                            if arm=='symbolic':answer,work=symbolic_solve(inp,spec,native=spec.task=='maze')
                            else:answer,work=symmetry_solve(core,inp,spec,**ARM_CONFIGS[arm],cycles_per_view=4)
                        r=rows[(family,core_seed,eid,arm)]
                        raw=None if answer is None else answer.flatten().tolist()
                        valid=bool(spec.independent_correct(inp,answer))
                        result={'family':family,'core_seed':core_seed,'example_id':eid,'arm':arm,
                            'answer_equal':raw==r['answer'],'validity_equal':valid==r['valid']==work['valid'],
                            'all_work_fields_equal':work==r['work']}
                        stream.write(json.dumps(result,sort_keys=True)+'\n')
                        if not all(result[k] for k in ('answer_equal','validity_equal','all_work_fields_equal')):
                            write_json(args.out/'failure.json',result)
                            raise ValueError('integrated policy differs from retained refinement')
                        count+=1;valid_by_arm[arm]+=int(valid)
                if tensor_digest(core.state_dict())!=state_before:raise RuntimeError('model state changed')
                counts[f'{family}/{core_seed}']=valid_by_arm;stream.flush()
                print(json.dumps({'family':family,'core_seed':core_seed,'valid':valid_by_arm}),flush=True)
    if count!=2560 or source_identity()!=before:raise RuntimeError('incomplete replay or source changed')
    report={'status':'PASS','comparisons':count,'by_family_seed':counts,
        'answer_validity_and_all_work_fields_exact':True,'source':before,
        'original_refinement_source_sha256':summary['source']['source_sha256'],
        'original_refinement_rows_sha256':summary['rows_sha256'],
        'device':'cpu','torch':str(torch.__version__),'training_updates':0,'new_confirmation_generated':False,
        'elapsed_seconds_not_performance_benchmark':time.perf_counter()-start,
        'scope':'Every frozen development model-example-arm in the retained refinement replayed once through integrated code; no new timing benchmark'}
    write_json(args.out/'summary.json',report)
    print(json.dumps({'status':'PASS','comparisons':count}),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
