#!/usr/bin/env python3
"""Replay frozen M17 complete-solve answers through the current implementation.

No training, new task generation, confirmation reopening or performance claim.
All source checkpoints and inputs come from previously hash-bound archives.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import io
import json
import platform
from pathlib import Path
import sys
import tempfile
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import numpy as np
import torch
from data.ancestry import digest
from eval.checkable_tasks import MAZE11, SUDOKU_SHIFT, semantic_exit
from eval.verified_search import ValueContract, ValueTarget
from model.task_value import load_task_value
from scripts.m16_evidence import Evidence, write_json
from scripts.m16_cpu_experiment import source_identity
from scripts.m17_sources import AcceptedM16
from scripts.m17_models import load_maze_core
from scripts.m17_cross_task import UNIFORM_SHA, checked_search, symbolic_solve
from scripts.verify_retained_results import read_bound_archive, M17_INVENTORY_SHA, M17_ZIP_SHA, json_rows


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--historical-dir', type=Path, default=Path('results/m16/sources'))
    ap.add_argument('--m16', type=Path, default=Path('results/m16/runs/accepted-34522192590.tar.gz'))
    ap.add_argument('--m17', type=Path, default=Path('results/m17/runs/34534009702-33e548d3e0f52d0664c00929316b2198d5fcffe4.tar.gz'))
    ap.add_argument('--out', type=Path, required=True)
    args=ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError('checkpoint replay requires the declared CPU-only environment')
    historical=Evidence(args.historical_dir)
    accepted=AcceptedM16(args.m16, historical)
    members=read_bound_archive(args.m17, inventory_sha=M17_INVENTORY_SHA, zip_sha=M17_ZIP_SHA)
    comparisons=[]
    failures=[]
    execution_source = source_identity()
    start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spectra-frozen-replay-') as directory:
        temp=Path(directory)
        with (args.out/'rows.jsonl').open('x') as stream:
            for family,spec,data_seed in [('sudoku_shift',SUDOKU_SHIFT,2026091703),('maze',MAZE11,2026091701)]:
                prefix=f'experiment/{family}/'
                src=json.loads(members[prefix+'model_sources_and_training.json'])
                manifest_bytes=members[prefix+f'manifests/seed{data_seed}.json']
                manifest=json.loads(manifest_bytes)
                with np.load(io.BytesIO(members[prefix+f'manifests/seed{data_seed}_arrays.npz']),allow_pickle=False) as a:
                    inputs=a['test_inputs'].copy()
                by_id={r['id']:i for i,r in enumerate(manifest['splits']['test']['examples'])}
                recorded=json_rows(members[prefix+'development_closed_rows.jsonl'])
                unique={(r['core_seed'],r['example_id'],r['arm']):r for r in recorded if r['round']==0}
                if len(unique)*3!=len(recorded): raise ValueError('unexpected repeated-row geometry')
                for seed_s, core_record in src['cores'].items():
                    seed=int(seed_s); core_sha=core_record['sha256']; workdir=temp/f'{family}-{seed}'
                    workdir.mkdir()
                    if family=='sudoku_shift':
                        core, observed_sha=historical.load_model('fp_recursive_dim64',seed,workdir)
                        if observed_sha != core_sha:
                            raise ValueError('retained core identity differs from the accepted source')
                        values,contracts,_=accepted.load_values(seed,workdir/'values')
                    else:
                        name=f'maze_core_{seed}.pt'; path=workdir/name
                        path.write_bytes(members[prefix+'checkpoints/'+name])
                        core=load_maze_core(path, expected_sha=core_sha,expected_seed=seed,manifest_sha=digest(manifest_bytes))
                        values={};contracts={}
                        for record in src['values'][seed_s]:
                            target=ValueTarget(record['target'])
                            name=Path(record['path']).name; path=workdir/name
                            path.write_bytes(members[prefix+'checkpoints/'+name])
                            values[target], contracts[target]=load_task_value(path,expected_sha256=record['sha256'],
                                expected_core_sha256=core_sha,expected_training_manifest_sha256=digest(manifest_bytes),
                                spec=spec, diagnostic_improvement=target is ValueTarget.IMPROVEMENT)
                    uniform=ValueContract(ValueTarget.QUALITY,core_sha,UNIFORM_SHA,spec.transition_id,state_schema=spec.state_schema)
                    counts=defaultdict(lambda:{'comparisons':0,'valid':0,'answer_mismatches':0,'work_mismatches':0})
                    for key,r in sorted(unique.items()):
                        if key[0]!=seed: continue
                        _,eid,arm=key
                        x=torch.from_numpy(inputs[by_id[eid]:by_id[eid]+1]).long()
                        with torch.inference_mode():
                            if arm=='reference_k4': answer,work=semantic_exit(core,x,spec,4,native=False)
                            elif arm=='native_k4': answer,work=semantic_exit(core,x,spec,4,native=True)
                            elif arm=='identity_24': answer,work=semantic_exit(core,x,spec,24,native=True)
                            elif arm in ('checked_quality','checked_terminal'):
                                target=ValueTarget.QUALITY if arm=='checked_quality' else ValueTarget.TERMINAL
                                answer,work=checked_search(core,values[target],contracts[target],x,spec,core_sha)
                            elif arm=='checked_uniform': answer,work=checked_search(core,None,uniform,x,spec,core_sha)
                            elif arm in ('symbolic_python','symbolic_native_bfs'):
                                answer,work=symbolic_solve(x,spec,native=arm=='symbolic_native_bfs')
                            else: raise ValueError('unknown retained arm: '+arm)
                        raw=None if answer is None else answer.flatten().tolist()
                        valid=spec.independent_correct(x,answer)
                        answer_ok=(raw==r['answer'] and valid==r['valid'] and valid==work['valid'])
                        work_diffs={k:[r['work'].get(k),work.get(k)] for k in ('transitions','decodes','value_calls')
                                    if r['work'].get(k)!=work.get(k)}
                        result={'family':family,'core_seed':seed,'example_id':eid,'arm':arm,'answer_matches':answer_ok,
                                'valid':valid,'work_matches':not work_diffs,'work_differences':work_diffs}
                        if not answer_ok or work_diffs: failures.append(result)
                        stream.write(json.dumps(result,sort_keys=True)+'\n')
                        c=counts[arm];c['comparisons']+=1;c['valid']+=int(valid)
                        c['answer_mismatches']+=int(not answer_ok);c['work_mismatches']+=int(bool(work_diffs))
                    comparisons.append({'family':family,'core_seed':seed,'arms':dict(counts)})
                    stream.flush()
                    print(json.dumps(comparisons[-1],sort_keys=True),flush=True)
    count = sum(v['comparisons'] for r in comparisons for v in r['arms'].values())
    if count != 3840:
        raise ValueError(f'incomplete frozen replay: {count}/3840 comparisons')
    if source_identity() != execution_source:
        raise RuntimeError('executable source changed during checkpoint replay')
    report={'status':'PASS' if not failures else 'FAIL','comparisons':count,
            'by_family_seed':comparisons,'failure_count':len(failures),'failures':failures,
            'device':'cpu','python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
            'executable_source':execution_source,'m17_inventory_sha256':M17_INVENTORY_SHA,'m16_identity':accepted.identity(),
            'elapsed_seconds_not_performance_benchmark':time.perf_counter()-start,
            'scope':'All frozen M17 development complete-solve arms, once per distinct model-example-arm; exact answer/validity/work replay; no retraining or new confirmation'}
    write_json(args.out/'summary.json',report)
    print(json.dumps({'status':report['status'],'comparisons':report['comparisons'],'failure_count':report['failure_count']}),flush=True)
    return int(bool(failures))
if __name__=='__main__': raise SystemExit(main())
