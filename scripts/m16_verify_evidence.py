#!/usr/bin/env python3
"""Independently replay every retained answer through a non-Torch exact checker.

This validates recorded predictions/aggregation, not a fresh neural-model run.
It does not consult reference solutions when deciding correctness.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectra_reliability.experiment import CORE_SEEDS,RECIPE,load_dataset,load_ancestry
from spectra_reliability.identity import file_sha256,strict_json,write_json
from spectra_reliability.lineage import ExposureIndex
from spectra_reliability.sudoku import valid


def verify(output:Path,*,require_confirmation:bool=True)->dict:
    index=load_ancestry(output);sources=list(index.sources);reports={};checked=0;pool_checked=0
    stages=['fit','validation','development']+(['confirmation','shift'] if require_confirmation else [])
    for name in stages:
        ds=load_dataset(output,name);index=ExposureIndex(sources)
        index.require_disjoint(ds.source,require_input_coverage=True);sources.append(ds.source)
    for surface in stages[1:]:
        ds=load_dataset(output,surface);base=output/'evaluation'/surface
        aggregate=strict_json((base/'aggregate.json').read_bytes());totals={};timings={}
        for seed in CORE_SEEDS:
            summary=strict_json((base/f'online_seed{seed}_summary.json').read_bytes())
            path=base/f'online_seed{seed}.jsonl'
            if file_sha256(path)!=summary['raw_rows_sha256']:raise ValueError('raw row hash mismatch')
            if summary['dataset_manifest_sha256']!=ds.source.sha256 or summary['core_seed']!=seed or summary['surface']!=surface:
                raise ValueError('summary identity mismatch')
            configs=set(summary['summaries'])
            expected_configs={'semantic_exit_original','semantic_exit_native','fixed4_native','single_pass_native',
                'symbolic_exact','finite_universe_filter','mcts_legacy_learned','mcts_legacy_uniform','mcts_legacy_guard',
                'mcts_matched_improvement','mcts_matched_validity','mcts_matched_quality','mcts_matched_first_hit',
                'baseline_first_best_first_survival'}
            if configs!=expected_configs or set(aggregate['online'])!=configs:
                raise ValueError('incomplete or unexpected solver inventory')
            timed_n=min(128,len(ds.ids));rounds=RECIPE['measurement']['timing_rounds']
            expected={(name,identifier,r) for name in configs for i,identifier in enumerate(ds.ids)
                      for r in range(rounds if i<timed_n else 1)}
            seen=set();ids={identifier:i for i,identifier in enumerate(ds.ids)};first_answers={}
            with path.open() as f:
                for line in f:
                    row=strict_json(line);key=(row['config_id'],row['example_id'],row['round'])
                    if key in seen:raise ValueError('duplicate repeated-measurement identity')
                    seen.add(key)
                    if key not in expected:raise ValueError('unexpected solver/example/round')
                    if row['core_seed']!=seed or row['surface']!=surface or ids.get(row['example_id'])!=row['example_index']:
                        raise ValueError('record identity mismatch')
                    i=row['example_index'];actual=valid(ds.inputs[i],row['answer'],2)
                    if type(row['primary_timing_sample']) is not bool or row['primary_timing_sample']!=(i<timed_n):
                        raise ValueError('primary timing selection differs from frozen input-index rule')
                    if type(row['wall_ns']) is not int or type(row['process_cpu_ns']) is not int or min(row['wall_ns'],row['process_cpu_ns'])<0:
                        raise ValueError('invalid timing')
                    if row['latency_ms']!=row['wall_ns']/1e6:raise ValueError('timing unit inconsistency')
                    answer_key=(row['config_id'],row['example_id'])
                    if answer_key in first_answers and first_answers[answer_key]!=row['answer']:
                        raise ValueError('prediction changed across timing rounds')
                    first_answers[answer_key]=row['answer']
                    if type(row['semantic_success']) is not int or row['semantic_success']!=int(actual):
                        raise ValueError('independent checker disagrees with recorded correctness')
                    if row['solver_reference_target_used'] is not False or row['physical_energy_joules'] is not None:
                        raise ValueError('unsupported reference/energy metadata')
                    checked+=1
                    if row['round']==0:totals[row['config_id']]=totals.get(row['config_id'],0)+int(actual)
                    if row['primary_timing_sample']:timings.setdefault(row['config_id'],[]).append(row['latency_ms'])
            if seen!=expected or len(seen)!=summary['rows']:raise ValueError('missing repeated measurement rows')
            pool_summary=strict_json((base/f'pool_seed{seed}_summary.json').read_bytes());pool_path=base/f'pool_seed{seed}.npz'
            if file_sha256(pool_path)!=pool_summary['raw_arrays_sha256']:raise ValueError('pool array hash mismatch')
            with np.load(pool_path,allow_pickle=False) as pool:
                if not np.array_equal(pool['inputs'],ds.inputs):raise ValueError('pool puzzle mismatch')
                if pool['answers'].shape!=(len(ds.ids),12,16):raise ValueError('fixed-pool shape mismatch')
                if pool['label_validity'].dtype!=np.bool_:raise ValueError('validity labels must be Boolean')
                actual=np.asarray([[valid(x,answer,2) for answer in candidates] for x,candidates in zip(pool['inputs'],pool['answers'])])
                if not np.array_equal(actual,pool['label_validity']):raise ValueError('independent pool checker mismatch')
                pool_checked+=int(actual.size)
                for name,record in pool_summary['selections'].items():
                    selection=np.asarray(record['selected_indices'],dtype=np.int64)
                    returned=actual[np.arange(len(actual)),selection]
                    if returned.tolist()!=record['returned'] or actual.any(1).tolist()!=record['coverage']:
                        raise ValueError('selection decomposition mismatch')
        for name,count in totals.items():
            if count!=aggregate['online'][name]['valid_answers']:raise ValueError('aggregate correctness mismatch')
            if not np.isclose(np.mean(timings[name]),aggregate['online'][name]['wall_latency']['mean_ms'],rtol=0,atol=1e-12):
                raise ValueError('aggregate mean latency mismatch')
        reports[surface]={'valid_answers_by_config':totals,'distinct_puzzles':len(ds.ids),
            'aggregate_sha256':file_sha256(base/'aggregate.json'),'all_stored_predictions_rechecked':True}
    return {'schema':'spectra.m16_independent_replay.v1','pass':True,'online_answers_checked':checked,
        'fixed_pool_answers_checked':pool_checked,'surfaces':reports,'ancestral_and_new_split_input_disjointness':True,
        'neural_models_rerun':False,'reference_solution_used_by_checker':False,'symmetry_disjointness_claimed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--report',required=True,type=Path);p.add_argument('--development-only',action='store_true');a=p.parse_args()
    try:result=verify(a.out,require_confirmation=not a.development_only);write_json(a.report,result,exclusive=True)
    except (ValueError,OSError,KeyError) as exc:print(f'EVIDENCE_REPLAY_FAILED: {exc}',file=sys.stderr);return 2
    print('EVIDENCE_REPLAY_PASS',result['online_answers_checked'],result['fixed_pool_answers_checked']);return 0

if __name__=='__main__':raise SystemExit(main())
