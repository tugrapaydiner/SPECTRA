#!/usr/bin/env python3
"""Replay previously consumed M14 development data and audit input isomorphisms.

No training, new confirmation, or historical-record rewriting. The pinned source
archives have byte-identical development manifests. All regenerated row records
must match them before any symmetry or training-lookup result is reported.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from data import sudoku
from data.splits import build_reproducible_splits
from data.sudoku_symmetry import POLICY, Sudoku4OrbitLookup, canonical_sudoku4
from scripts.m16_evidence import Evidence, MANIFEST, SPECS, write_json


def audit(directory: Path) -> tuple[dict, list[dict]]:
    evidence = Evidence(directory)
    raw = evidence.archives['m14_source'].read(MANIFEST)
    if raw != evidence.archives['m14_original'].read(MANIFEST):
        raise ValueError('original/source M14 development manifests differ')
    manifest = json.loads(raw)
    datasets, observed = build_reproducible_splits(
        manifest['task'], {s: r['count'] for s, r in manifest['splits'].items()}, manifest['seed'],
        generator_kwargs=manifest['generator_kwargs'], task_scope=manifest['task_scope'],
        official_benchmark=manifest['official_benchmark'])
    for split in datasets:
        if observed['splits'][split] != manifest['splits'][split]:
            raise ValueError('replayed example inventory differs from pinned source: '+split)
    witnesses = {s: [canonical_sudoku4(x) for x in ds.inputs] for s, ds in datasets.items()}
    keys = {s: [w.key for w in ws] for s, ws in witnesses.items()}
    training = {}
    for i, key in enumerate(keys['train']):
        training.setdefault(key, i)
    lookup = Sudoku4OrbitLookup(datasets['train'].inputs, datasets['train'].targets)
    if lookup.orbit_count != len(training):
        raise AssertionError('canonical index and training lookup disagree')
    rows, splits = [], {}
    for split, ds in datasets.items():
        matched = correct = 0
        for i, (inp, key, witness) in enumerate(zip(ds.inputs, keys[split], witnesses[split])):
            train_i = training.get(key)
            if train_i is not None:
                reference = witnesses['train'][train_i]
                # Store replayable, complete bijections, not merely equal hashes.
                if not np.array_equal(reference.transform(datasets['train'].inputs[train_i]), witness.transform(inp)):
                    raise AssertionError('matching orbit hash lacks a matching spatial/digit witness')
                matched += 1
            # This receives only the held-out puzzle, never its reference answer.
            answer = lookup.solve(inp)
            valid = answer is not None and sudoku.is_solved(answer.reshape(4, 4), 2) and sudoku.respects_clues(
                inp.reshape(4, 4), answer.reshape(4, 4), 2)
            if valid != (train_i is not None):
                raise AssertionError('training-only lookup does not match independently checked orbit coverage')
            correct += int(valid)
            rows.append({'split': split, 'example_id': ds.ids[i], 'example_index': i,
                'orbit_key': key, 'in_training_orbit': train_i is not None,
                'training_example_id': None if train_i is None else datasets['train'].ids[train_i],
                'positions': list(witness.positions), 'labels': list(witness.labels),
                'lookup_valid': bool(valid)})
        splits[split] = {'examples': len(ds), 'unique_orbits': len(set(keys[split])),
            'rows_in_training_orbits': matched, 'training_orbit_fraction': matched/len(ds),
            'lookup_valid': correct, 'lookup_success': correct/len(ds),
            'reference_solution_used_at_lookup': False}
    # Descriptive stratification of the existing model outputs. This is not a
    # replacement confirmation gate, a new inference run, or a causal test of
    # whether the neural networks memorized the equivalent puzzles.
    membership = {eid: keys['test'][i] in training for i, eid in enumerate(datasets['test'].ids)}
    stratified, seen = defaultdict(list), set()
    arm_ids = defaultdict(set)
    for r in evidence.rows('m14_source', 'experiment/development_rows.jsonl'):
        identity = (r['seed'], r['config_id'], r['example_id'])
        if identity in seen or r['example_id'] not in membership:
            raise ValueError('duplicate or unknown model/example development result')
        seen.add(identity)
        if type(r['semantic_success']) is not int or r['semantic_success'] not in (0, 1):
            raise ValueError('invalid retained semantic result')
        stratum = 'training_orbit_seen' if membership[r['example_id']] else 'training_orbit_unseen'
        stratified[(r['config_id'], stratum)].append(r)
        arm_ids[r['config_id']].add((r['seed'], r['example_id']))
    comparison = {}
    for (arm, stratum), rr in sorted(stratified.items()):
        comparison.setdefault(arm, {})[stratum] = {
            'model_example_rows': len(rr), 'valid': sum(r['semantic_success'] for r in rr),
            'success': sum(r['semantic_success'] for r in rr)/len(rr),
            'model_seeds': sorted({r['seed'] for r in rr}),
            'unique_examples': len({r['example_id'] for r in rr})}
    shared = arm_ids['dual_stream_exit_k4'] & arm_ids['single_pass_fp']
    expected = arm_ids['dual_stream_exit_k4'] | arm_ids['single_pass_fp']
    if shared != expected:
        raise ValueError('primary neural comparisons are not model-example paired')
    rows_raw = ''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows).encode()
    report = {'schema': 'spectra.sudoku4_symmetry_audit.v1', 'policy': POLICY,
        'scope': 'retrospective consumed M14 training/validation/development; no confirmation opened',
        'source_archive_sha256': {k: v[1] for k, v in SPECS.items()},
        'original_source_development_manifest_identical': True,
        'manifest_sha256': hashlib.sha256(raw).hexdigest(),
        'audit_source_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
            for name in ('data/sudoku_symmetry.py', 'data/splits.py', 'scripts/_common.py', 'scripts/audit_sudoku_symmetry.py')},
        'replayed_example_records_match': True,
        'replayed_examples': sum(len(ds) for ds in datasets.values()),
        'spatial_transform_count': 128, 'digit_permutation_count': 24,
        'reference_answer_used_for_orbit_key': False,
        'orbit_lookup_train_examples': len(datasets['train']), 'splits': splits,
        'existing_development_results_by_orbit_stratum': comparison,
        'existing_result_rows_reaggregated': len(seen), 'witness_rows_sha256': hashlib.sha256(rows_raw).hexdigest(),
        'interpretation': 'Exact-disjoint generated examples need not be isomorphism-disjoint. '
            'This limits structural-generalization claims; it neither proves neural memorization '
            'nor invalidates the historical in-distribution comparison. Strata are descriptive, '
            'not a new independent confirmation or latency benchmark.'}
    return report, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=Path('results/m16/sources'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-report', type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report, rows = audit(args.evidence_dir)
    if args.verify_report is not None and report != json.loads(args.verify_report.read_text()):
        raise ValueError('recomputed symmetry audit differs from the retained report')
    args.out.mkdir(parents=True)
    write_json(args.out/'summary.json', report)
    with (args.out/'witness_rows.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True)+'\n')
    print(json.dumps({'status': 'PASS', 'replayed_examples': report['replayed_examples'],
                      'splits': report['splits']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
