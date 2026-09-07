#!/usr/bin/env python3
"""Recreate and save the failed M09-v3 development policy without test access.

This utility consumes the already-generated train-only M09 target table and frozen
reasoner checkpoint. It reconstructs only the action-fit root states, deterministically
refits v3, saves a strict core-bound checkpoint, reloads it, and records its digest.
No development or test comparison is run here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.action_checkpoint import load_m09_action_checkpoint, save_m09_action_checkpoint
from eval.checkpoint_eval import load_research_trm_checkpoint
from eval.grounded_targets import assert_frozen_reasoner, generate_trajectory_states, tensor_state_sha256
from scripts._common import build_data_splits
from scripts.train_action_policy import (
    ACTION_FIT_PUZZLES, ACTION_SCALE, CANDIDATE_COUNT, CANDIDATE_SEED, DATA_SEED,
    POLICY_STEPS, TEST_N, TRAIN_N, VAL_N, id_hash, make_cfg, write_json,
)
from scripts.train_action_policy_v3 import VERSION, fit_challenger


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment-dir', required=True)
    args=ap.parse_args(); root=Path(args.experiment_dir)
    reasoner_path=root/'reasoner.pt'; table_path=root/'target_table.pt'; target_path=root/'target_generation.json'
    for p in (reasoner_path, table_path, target_path):
        if not p.is_file(): raise SystemExit(f'missing prerequisite: {p}')

    core=load_research_trm_checkpoint(reasoner_path, device='cpu', weight_identity='recorded')
    for p in core.model.parameters(): p.requires_grad_(False)
    core.model.eval(); assert_frozen_reasoner(core.model)

    cfg=make_cfg()
    datasets,_=build_data_splits(cfg, TRAIN_N, VAL_N, TEST_N, seed=DATA_SEED,
                                 manifest_path=root/'checkpoint_rebuild_manifest.json')
    fit_inputs=torch.from_numpy(datasets['train'].inputs[:ACTION_FIT_PUZZLES]).long()
    fit_ids=datasets['train'].ids[:ACTION_FIT_PUZZLES]
    rows=generate_trajectory_states(core.model, fit_inputs, fit_ids, max_depth=1)
    table=torch.load(table_path, map_location='cpu', weights_only=True)
    target_dirs=table['selected_directions'].float()
    root_u=table['restricted_train_utilities'][:ACTION_FIT_PUZZLES].float()
    module,opt,training=fit_challenger(rows, root_u, target_dirs, root/'curves'/'v3_checkpoint_rebuild.jsonl')

    target=json.loads(target_path.read_text())
    provenance={
        'reference_target_used': False,
        'candidate_bank_seed': CANDIDATE_SEED,
        'candidate_bank_count': CANDIDATE_COUNT,
        'selected_candidate_indices': target['selected_candidate_indices'],
        'target_utility_table_sha256': target['train_utility_table_sha256'],
        'reasoner_tensor_state_sha256': tensor_state_sha256(core.model),
        'train_id_sha256': target['train_id_sha256'],
        'development_id_sha256': target['development_id_sha256'],
        'test_id_sha256': target['test_id_sha256'],
        'target_generation_work': target['train_work'],
        'training_method': VERSION,
        'state_representation': module.STATE_REPRESENTATION,
        'prior_semantics': module.PRIOR_SEMANTICS,
        'candidate_scale': ACTION_SCALE,
        'utility_oracle': 'model.verifier.sudoku_score',
        'policy_fit_depths': [0],
        'purpose': 'development_failure_checkpoint; not accepted search evidence',
    }
    path=root/'development_action_policy.pt'
    save_m09_action_checkpoint(module,path,core=core,optimizer=opt,trained_steps=POLICY_STEPS,
                               fitted_version=VERSION,provenance=provenance)
    loaded=load_m09_action_checkpoint(path,core)
    record={
        'checkpoint': str(path), 'sha256': loaded.sha256,
        'class': type(loaded.module).__name__,
        'fitted_version': loaded.payload['training']['fitted_version'],
        'prior_semantics': loaded.payload['architecture']['prior_semantics'],
        'training': training,
        'action_fit_id_sha256_reconstructed': id_hash(fit_ids),
        'reference_target_used': False,
        'accepted_practical_benefit': False,
        'test_inputs_evaluated': False,
    }
    write_json(root/'development_checkpoint.json',record)
    print(json.dumps(record,indent=2,sort_keys=True))
    return 0


if __name__=='__main__': raise SystemExit(main())
