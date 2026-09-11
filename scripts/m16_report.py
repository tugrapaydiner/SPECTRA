#!/usr/bin/env python3
"""Render a result report from frozen evidence, without changing selection/gates."""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectra_reliability.identity import file_sha256,strict_json,write_json


def render(output:Path,benchmarks:Path,destination:Path):
    destination.mkdir(parents=True,exist_ok=True)
    complete=strict_json((output/'confirmation_complete.json').read_bytes())
    freeze=strict_json((output/'confirmation_freeze.json').read_bytes())
    replay=strict_json((output/'independent_replay.json').read_bytes())
    if not replay['pass']:raise ValueError('independent prediction replay did not pass')
    surfaces={s:strict_json((output/'evaluation'/s/'aggregate.json').read_bytes()) for s in ('validation','development','confirmation','shift')}
    native=strict_json((benchmarks/'full_graph_summary.json').read_bytes());kernel=strict_json((benchmarks/'kernel_summary.json').read_bytes())
    primary=surfaces['confirmation']['comparisons_to_native_semantic_exit']['baseline_first_best_first_survival']
    mechanism={s:{'improvement_failures':r['fixed_pool']['improvement']['selection_failures_given_coverage'],
                  'quality_failures':r['fixed_pool']['quality']['selection_failures_given_coverage'],
                  'validity_failures':r['fixed_pool']['validity']['selection_failures_given_coverage']}
               for s,r in surfaces.items()}
    status={'schema':'spectra.m16_result.v1','source_commit':freeze['source_commit'],
        'freeze_sha256':file_sha256(output/'confirmation_freeze.json'),
        'confirmation_complete':True,'integrity_replay_pass':True,
        'primary_controller_twenty_percent_mean_cost_gate':primary['twenty_percent_mean_cost_matched_quality_gate'],
        'primary_controller_confirmation_regressions':primary['regressions'],
        'primary_controller_confirmation_gains':primary['gains'],
        'bounded_mechanism':mechanism,'broad_task_superiority_established':False,
        'iso_energy_superiority_established':False,'hiring_or_level_guarantee':False,
        'data_scope':'generated 4x4 diagnostic; two fixed accepted cores',
        'native_scope':native['scope']}
    write_json(destination/'summary.json',status)
    for name,value in surfaces.items():write_json(destination/f'{name}_aggregate.json',value)
    for name,path in [('confirmation_freeze.json',output/'confirmation_freeze.json'),
                      ('confirmation_opened.json',output/'confirmation_opened.json'),
                      ('independent_replay.json',output/'independent_replay.json'),
                      ('lineage_inventory.json',output/'lineage'/'inventory.json'),
                      ('environment.json',output/'confirmation_environment.json')]:
        (destination/name).write_bytes(path.read_bytes())
    write_json(destination/'native_full_graph.json',native);write_json(destination/'native_kernel.json',kernel)
    lines=['# SPECTRA M16 — verifier semantics and CPU execution','',
        '**Result boundary:** integrity and mechanism evidence are separate from useful CPU superiority. '
        'All new fitting and evaluation used CPUs. This is not an L7 certification or a 100/100 claim.','',
        f"Source commit: `{freeze['source_commit']}`. Confirmation was opened after a hash-bound source/data/checkpoint freeze.",
        f"Independent checker replay: **{replay['online_answers_checked']:,} online answers** and **{replay['fixed_pool_answers_checked']:,} fixed-pool candidates**. Neural models were not rerun by that replay.",'',
        '## Confirmation: full solves','',
        '| Configuration | Valid / model-example rows | Mean ms | Median ms | p95 ms |',
        '|---|---:|---:|---:|---:|']
    for name,row in surfaces['confirmation']['online'].items():
        t=row['wall_latency'];lines.append(f"| `{name}` | {row['valid_answers']}/{row['model_example_rows']} | {t['mean_ms']:.4f} | {t['median_ms']:.4f} | {t['p95_ms']:.4f} |")
    lines += ['', 'Quality covers 512 distinct confirmation puzzles evaluated on each of two frozen cores. '
        'Primary timing uses the first 128 puzzles per core and three randomized interleaved rounds. '
        'Confidence intervals resample common puzzle indices jointly across the fixed cores; they are not training-seed-population intervals.','',
        '## Fixed-pool selection failures','',
        '| Surface | Improvement target | Absolute quality | Current validity |', '|---|---:|---:|---:|']
    for surface,row in mechanism.items():lines.append(f"| {surface} | {row['improvement_failures']} | {row['quality_failures']} | {row['validity_failures']} |")
    lines += ['', 'A failure here means a correct answer existed in the same already-generated pool but the selector returned an invalid answer. '
        'The legacy verifier is contextual: it used raw decoding, whereas the matched new targets all restore immutable givens. '
        'Future-success probability is not current correctness; the first-hit MCTS control uses separate exploration and final-selection values.','',
        '## Primary controller decision','',
        f"The predeclared 20% mean-cost / matched-quality gate is **{'PASS' if status['primary_controller_twenty_percent_mean_cost_gate'] else 'NOT MET'}**.",
        f"Confirmation gains: {primary['gains']}; regressions from a valid baseline: {primary['regressions']}. "
        f"Mean latency ratio to the same-native-checker semantic-exit baseline: {primary['mean_latency_ratio']['point']:.4f} "
        f"(95% conditional interval {primary['mean_latency_ratio']['ci95']}).",
        'Protecting a known valid answer is a correctness property, not a latency win. The full baseline and checking costs are included.','',
        '## Native full-graph comparison','',
        '| Backend | Mean ms | p95 ms | Logits bit-identical | Answers identical |', '|---|---:|---:|---|---|']
    for name,row in native['summaries'].items():
        lines.append(f"| `{name}` | {row['wall']['mean_ms']:.4f} | {row['wall']['p95_ms']:.4f} | {row['all_logits_bit_identical']} | {row['all_answers_equal']} |")
    lines += ['', 'These are 64 retained 9x9 M10 fidelity puzzles, five rounds, including decode and an exact semantic check. '
        'The source checkpoint is undertrained and this workload is not evidence of task capability. '
        'The AVX2 handle retains an extra transposed int8 layout; it is not packed-only execution. '
        'The dense-FP32 comparator uses the same effective weights with a different accumulation order; its numerical differences are retained. '
        'No cache-residency, physical-energy, or universal-speedup claim is made.','',
        '## What remains unestablished','',
        'An important broadly useful reasoning advance, hard-family transfer, an advantage over strong task-specific solvers, '
        'physical energy improvement, independent external adoption, and any hiring outcome remain unestablished by M16. '
        'Exact input separation is not symmetry-family separation. The source model/checker assumptions and finite CPU measurements remain explicit.','',
        '## Reproduction','',
        'Run `scripts/m16_research.py` stages in order; commit the freeze before opening confirmation. '
        'Run `scripts/m16_verify_evidence.py` to independently recheck stored answers, and `scripts/m16_benchmark.py` for the original graph versus new native and dense comparators. '
        'Use `requirements-cpu.lock`, retain source artifacts and record the actual compiler/CPU environment.','']
    (destination/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    table=[]
    for surface,record in surfaces.items():
        for name,row in record['online'].items():table.append({'surface':surface,'config':name,'valid':row['valid_answers'],
            'model_example_rows':row['model_example_rows'],'unique_puzzles':row['unique_puzzles'],
            'mean_ms':row['wall_latency']['mean_ms'],'median_ms':row['wall_latency']['median_ms'],'p95_ms':row['wall_latency']['p95_ms']})
    with (destination/'result_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    write_json(destination/'file_hashes.json',{p.name:file_sha256(p) for p in sorted(destination.iterdir()) if p.is_file() and p.name!='file_hashes.json'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True,type=Path)
    p.add_argument('--benchmarks',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    render(a.experiment,a.benchmarks,a.out);return 0

if __name__=='__main__':raise SystemExit(main())
