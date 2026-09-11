#!/usr/bin/env python3
"""Audit retained CPU proposal runs: source bytes, raw answers, work and forecasts.

This is independent semantic/record verification, not a new inference benchmark.
The original M17 archive is read only on already-consumed development surfaces.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from data.ancestry import digest
from eval.symmetry_schedule import predict_schedule
from scripts.audit_symmetry_proposals import ARMS, REFINED_ARMS, ARM_CONFIGS, summarize
from scripts.m16_evidence import write_json
from scripts.verify_fixed_pool_replay import FAMILY_SPECS, frozen_inputs, read_bound_archive, M17_INVENTORY_SHA, M17_ZIP_SHA

ROOT=Path(__file__).resolve().parents[1]


def bound_run(directory: Path, arms: tuple[str,...], protocol_name: str):
    summary=json.loads((directory/'summary.json').read_text())
    raw=gzip.decompress((directory/'rows.jsonl.gz').read_bytes())
    if digest(raw)!=summary['rows_sha256']:
        raise ValueError('retained raw-row hash mismatch')
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)!=7680 or summary['timing_rows']!=len(rows):
        raise ValueError('incomplete retained run')
    if (summary['execution_status']!='COMPLETE' or summary['training_updates']!=0
        or summary['new_confirmation_generated'] is not False
        or summary['reference_answer_used_for_inference'] is not False
        or summary['environment']['device']!='cpu'):
        raise ValueError('unsupported retained execution scope')
    if summarize(rows,arms=arms)!=summary['families']:
        raise ValueError('retained summary differs from raw paired observations')
    with tarfile.open(directory/'executable_source.tar.gz','r:gz') as archive:
        members={}
        for item in archive.getmembers():
            p=PurePosixPath(item.name)
            if not item.isfile() or p.is_absolute() or '..' in p.parts or item.name in members:
                raise ValueError('unsafe or duplicate source archive member')
            members[item.name]=archive.extractfile(item).read()
    expected=set(summary['source']['files'])|{protocol_name}
    if set(members)!=expected:
        raise ValueError('source archive inventory is not exact')
    for name,sha in summary['source']['files'].items():
        if digest(members[name])!=sha:
            raise ValueError('archived executable source differs: '+name)
    if digest(json.dumps(summary['source']['files'],sort_keys=True).encode())!=summary['source']['source_sha256']:
        raise ValueError('invalid executable source identity')
    if digest(members[protocol_name])!=summary['protocol_sha256']:
        raise ValueError('archived protocol differs')
    protocol=json.loads(members[protocol_name])
    if tuple(protocol['arms'])!=arms or protocol['families']!=['sudoku_shift','maze']:
        raise ValueError('unexpected protocol arm/family inventory')
    return summary,rows


def verify(directory: Path):
    members=read_bound_archive(ROOT/'results/m17/runs/34534009702-33e548d3e0f52d0664c00929316b2198d5fcffe4.tar.gz',
                               inventory_sha=M17_INVENTORY_SHA,zip_sha=M17_ZIP_SHA)
    inputs={}
    manifests={}
    for family,(spec,seed,_) in FAMILY_SPECS.items():
        x,ids,sha=frozen_inputs(members,f'experiment/{family}/',seed,spec)
        inputs[family]={eid:x[i:i+1] for i,eid in enumerate(ids)}
        manifests[family]=sha
    unique={};counts={};reports={};answer_rows=0
    original=json.loads((directory/'attempt1/summary.json').read_text())
    incomplete=gzip.decompress((directory/'attempt1/rows.jsonl.gz').read_bytes())
    partial_rows=[json.loads(l) for l in incomplete.splitlines()]
    failure=json.loads((directory/'attempt1/retention_failure.json').read_text())
    if (failure['status']!='RAW_RETENTION_INTEGRITY_FAILURE' or failure['retained_rows']!=len(partial_rows)
        or failure['retained_rows_sha256']!=digest(incomplete) or failure['declared_rows_sha256']!=original['rows_sha256']
        or failure['missing_rows']!=7680-len(partial_rows) or digest(incomplete)==original['rows_sha256']):
        raise ValueError('original retention failure was altered or misrepresented')
    for name,arms,protocol in [('attempt1',ARMS,'docs/SYMMETRY_PROPOSAL_PROTOCOL.json'),
                              ('attempt2',REFINED_ARMS,'docs/SYMMETRY_PROPOSAL_REFINEMENT_PROTOCOL.json')]:
        summary,rows=bound_run(directory/('attempt1_replay' if name=='attempt1' else name),arms,protocol)
        if name=='attempt1' and summary['source']['source_sha256']!=original['source']['source_sha256']:
            raise ValueError('retention replay did not use the exact original source')
        for family in FAMILY_SPECS:
            if summary['identities'][family]['manifest_sha256']!=manifests[family]:
                raise ValueError('source input manifest differs')
        observed=set()
        expected={(f,s,e,a,r) for f in FAMILY_SPECS
                  for s in ([1401,2402] if f=='sudoku_shift' else [1701,2702])
                  for e in inputs[f] for a in arms for r in range(3)}
        for row in rows:
            f,s,e,a,r=(row[k] for k in ('family','core_seed','example_id','arm','round'))
            identity=(f,s,e,a,r)
            if identity not in expected or identity in observed:
                raise ValueError('unknown or duplicate source model-example-arm-round')
            observed.add(identity)
            spec=FAMILY_SPECS[f][0]
            answer=None if row['answer'] is None else torch.tensor([row['answer']],dtype=torch.long)
            if spec.independent_correct(inputs[f][e],answer)!=row['valid']:
                raise ValueError('stored answer differs from independent semantic validity')
            w=row['work']
            if w['target_used'] is not False or w['value_calls']!=0:
                raise ValueError('proposal run used an unexpected reference/evaluator')
            if a!='symbolic':
                if not 1<=w['transitions']<=ARM_CONFIGS[a]['transition_budget']:
                    raise ValueError('transition cap violated')
                if w['transitions']!=w['decodes'] or w['checks']!=w['transitions']+w['restoration_checks']:
                    raise ValueError('inconsistent decode/check work')
                if w['input_embeddings']!=len(w['views_started']) or w['checker_constructions']!=1+w['input_transforms']:
                    raise ValueError('inconsistent embedding/checker work')
            if r==0:unique[(name,f,s,e,a)]=row
            answer_rows+=1
        if observed!=expected:raise ValueError('missing complete source inventory')
        counts[name]={'timing_rows':len(rows),'distinct_model_example_arm_comparisons':len(rows)//3,
                     'rows_sha256':summary['rows_sha256'],'executable_source_sha256':summary['source']['source_sha256']}
        reports[name]=summary
    predictions=json.loads((directory/'selection/counterfactual_predictions.json').read_text())
    if predictions['parent_rows_sha256']!=digest(incomplete):
        raise ValueError('counterfactuals have a different parent run')
    declared={(r['family'],r['core_seed'],r['example_id']):r for r in predictions['predicted_rows']}
    if len(declared)!=len(predictions['predicted_rows']) or len(declared)!=512:
        raise ValueError('incomplete or duplicate counterfactual predictions')
    partial_seen=set()
    for r in partial_rows:
        f,s,e,a,round_id=(r[k] for k in ('family','core_seed','example_id','arm','round'))
        key=(f,s,e,a,round_id)
        if key in partial_seen or type(round_id) is not int or not 0<=round_id<3:
            raise ValueError('duplicate or malformed original partial record')
        partial_seen.add(key)
        replay=unique[('attempt1',f,s,e,a)]
        # New timings cannot replace missing historical timings. Compare only
        # deterministic answers, validity and logical work at every retained key.
        if any(r[k]!=replay[k] for k in ('answer','valid','work')):
            raise ValueError('exact-source replay changed an originally retained outcome')
    expected_round0={(f,s,e,a,0) for f in FAMILY_SPECS
        for s in ([1401,2402] if f=='sudoku_shift' else [1701,2702]) for e in inputs[f] for a in ARMS}
    if {k for k in partial_seen if k[-1]==0}!=expected_round0:
        raise ValueError('counterfactual selection lacks complete original first-round outcomes')
    comparisons=0
    for (family,seed,eid),declaration in declared.items():
        base=('attempt1',family,seed,eid)
        ident=unique[(*base,'identity_32')];views=unique[(*base,'dihedral_32')]
        actual=unique[('attempt2',family,seed,eid,'support4_prefix8')]
        predicted=predict_schedule(ident['work']['transitions'] if ident['valid'] else None,
            views['work']['transitions'] if views['valid'] else None,identity_cycles=8,view_limit=4,budget=20)
        if any(declaration[k]!=v for k,v in predicted.items()):
            raise ValueError('stored counterfactual differs from recomputation')
        if (actual['valid'],actual['work']['transitions'])!=(predicted['valid'],predicted['transitions']):
            raise ValueError('predicted validity or work differs from actual refined execution')
        if actual['valid'] and actual['answer']!=unique[(*base,predicted['answer_source'])]['answer']:
            raise ValueError('predicted successful answer differs from refined execution')
        for arm in ('identity_4','identity_32','symbolic'):
            old=unique[(*base,arm)];new=unique[('attempt2',family,seed,eid,arm)]
            if (old['answer'],old['valid'],old['work']['transitions'])!=(new['answer'],new['valid'],new['work']['transitions']):
                raise ValueError('unchanged comparison arm changed between attempts')
        comparisons+=1
    return {'status':'PASS','scope':'Stored source bytes, exact inventory, independent answer semantics, summary and counterfactual replay; not fresh model inference or confirmation',
        'runs':counts,'stored_answer_rows_checked':answer_rows,'unique_counterfactuals_checked':comparisons,
        'new_confirmation_generated':False,'training_updates':0,'m17_inventory_sha256':M17_INVENTORY_SHA,
        'original_retention_failure_preserved':True,'original_partial_rows_compared':len(partial_rows),
        'missing_original_timing_rows_not_recreated':failure['missing_rows'],
        'complete_run_scope':'attempt1 is a separately retained exact-source replay, not the original incomplete timing log; attempt2 is the original complete refinement',
        'counterfactual_prediction_file_sha256':digest((directory/'selection/counterfactual_predictions.json').read_bytes())}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence-dir',type=Path,default=ROOT/'results/cpu_progress')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    torch.set_num_threads(1)
    result=verify(args.evidence_dir)
    write_json(args.out,result)
    print(json.dumps(result,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
