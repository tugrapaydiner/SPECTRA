#!/usr/bin/env python3
"""Reassess the retained restart development result against every frozen control.

No training, new test data, repeated policy selection, or inference is performed.
Inventories come from the original pinned manifests, not the observed row file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.paired_frontier import FrontierGate, paired_frontier
from scripts.audit_symmetry_proposals import REFINED_ARMS
from scripts.verify_cpu_progress import bound_run
from scripts.verify_fixed_pool_replay import (FAMILY_SPECS, M17_INVENTORY_SHA,
    M17_ZIP_SHA, frozen_inputs, read_bound_archive)
from scripts.m16_evidence import write_json


def audit() -> dict:
    directory = ROOT / 'results/cpu_progress/attempt2'
    original, rows = bound_run(directory, REFINED_ARMS,
                              'docs/SYMMETRY_PROPOSAL_REFINEMENT_PROTOCOL.json')
    members = read_bound_archive(
        ROOT / 'results/m17/runs/34534009702-33e548d3e0f52d0664c00929316b2198d5fcffe4.tar.gz',
        inventory_sha=M17_INVENTORY_SHA, zip_sha=M17_ZIP_SHA)
    protocol_raw = (ROOT / 'docs/FINAL_CPU_CONFIRMATION_PROTOCOL.json').read_bytes()
    thresholds = json.loads(protocol_raw)['primary_gate_per_family']
    gate = FrontierGate(**{k: thresholds[k] for k in (
        'min_success_gain', 'max_mean_latency_ratio', 'max_p95_latency_ratio')})
    reports, expected = {}, set()
    for family, (spec, source_seed, _) in FAMILY_SPECS.items():
        _, ids, manifest_sha = frozen_inputs(members, f'experiment/{family}/', source_seed, spec)
        seeds = (1401, 2402) if family == 'sudoku_shift' else (1701, 2702)
        if original['identities'][family]['manifest_sha256'] != manifest_sha:
            raise ValueError('restart rows bind a different manifest')
        expected.update((family, m, e, a, r) for m in seeds for e in ids
                        for a in REFINED_ARMS for r in range(3))
        comparisons = {}
        for comparator in REFINED_ARMS:
            if comparator == 'support4_prefix8':
                continue
            selected = [r for r in rows if r['family'] == family and
                        r['arm'] in ('support4_prefix8', comparator)]
            comparisons[comparator] = paired_frontier(selected, family=family,
                candidate='support4_prefix8', comparator=comparator,
                model_seeds=seeds, example_ids=tuple(ids), gate=gate)
        reports[family] = {'manifest_sha256': manifest_sha, 'comparisons': comparisons}
    observed = [(r['family'], r['core_seed'], r['example_id'], r['arm'], r['round']) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('retained row inventory is not the complete frozen cross product')
    source = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ('eval/paired_frontier.py', 'scripts/audit_research_frontier.py')}
    return {'schema': 'spectra.research_frontier_audit.v1', 'status': 'PASS',
        'scientific_status': 'ADAPTIVE_DEVELOPMENT_ONLY',
        'retained_rows_sha256': original['rows_sha256'],
        'historical_executable_source_sha256': original['source']['source_sha256'],
        'analysis_source_sha256': source,
        'threshold_reference_sha256': hashlib.sha256(protocol_raw).hexdigest(),
        'threshold_application': 'Retrospective diagnostic on consumed development; '
            'does not execute or pass the separate locked confirmation protocol.',
        'families': reports, 'timing_rows_checked': len(observed),
        'new_inference_or_training': False, 'new_confirmation_generated': False,
        'policy_selection_adjusted': False,
        'interpretation': 'Eight comparisons are descriptive and not multiplicity adjusted. '
            'Two model seeds provide limited information about training variation. '
            'Physical energy and deployment value are not established by these intervals.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-report', type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit()
    if args.verify_report and result != json.loads(args.verify_report.read_text()):
        raise ValueError('paired frontier audit differs from the retained report')
    write_json(args.out, result)
    print(json.dumps({'status': result['status'], 'timing_rows_checked': result['timing_rows_checked'],
        'scientific_status': result['scientific_status']}))


if __name__ == '__main__':
    main()
