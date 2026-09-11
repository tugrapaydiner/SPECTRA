#!/usr/bin/env python3
"""CPU-only M16 experiment, with explicit stage boundaries and holdout freeze.

Examples:
  python scripts/m16_research.py prepare --archives /path/to/archives --out outputs/m16
  python scripts/m16_research.py fit --out outputs/m16
  python scripts/m16_research.py develop --out outputs/m16
  python scripts/m16_research.py freeze --out outputs/m16 --source-commit <40-hex>
  # Commit confirmation_freeze.json before the next command.
  python scripts/m16_research.py confirm --out outputs/m16 --freeze-commit <40-hex>

A failed/incomplete evaluation is retained and is NOT silently overwritten.
This command never downloads data, invokes external APIs, or uses a GPU.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectra_reliability.experiment import (CORE_SEEDS,prepare,cpu_environment,load_core,
    load_auxiliary_sources,load_dataset,load_ancestry,generate_dataset)
from spectra_reliability.identity import file_sha256,write_json,strict_json
from spectra_reliability.lineage import ExposureIndex
from spectra_reliability.targets import build_pool,fit_targets,load_targets
from spectra_reliability.evaluation import evaluate_surface
from spectra_reliability.freeze import create_freeze,open_confirmation,verify_freeze


def fit(output:Path):
    if (output/'confirmation_opened.json').exists():raise ValueError('cannot fit after opening confirmation')
    fit_data=load_dataset(output,'fit')
    done=output/'fit_complete.json'
    if done.exists():
        for seed in CORE_SEEDS:load_targets(output,seed)
        print('Verified existing fitted target checkpoints; no refitting.',flush=True);return
    env=cpu_environment()
    if not (output/'fit_environment.json').exists():write_json(output/'fit_environment.json',env,exclusive=True)
    for seed in CORE_SEEDS:
        if (output/'auxiliary'/f'fit_seed{seed}.json').exists():
            load_targets(output,seed);continue
        core=load_core(output,seed);policy,_,_=load_auxiliary_sources(output,seed)
        pool=build_pool(core,policy,fit_data)
        fit_targets(core,seed,pool,output,fit_data.source.sha256)
        print('FIT_COMPLETE',seed,flush=True)
    write_json(done,{'core_seeds':list(CORE_SEEDS),'all_targets_fitted':True,
                    'fit_manifest_sha256':fit_data.source.sha256},exclusive=True)


def develop(output:Path,native_cache:Path):
    if (output/'confirmation_opened.json').exists():raise ValueError('cannot develop after opening confirmation')
    if (output/'development_complete.json').exists():raise ValueError('development is complete; no silent rerun')
    env=cpu_environment()
    if not (output/'development_environment.json').exists():write_json(output/'development_environment.json',env,exclusive=True)
    for surface in ('validation','development'):
        if (output/'evaluation'/surface).exists():raise ValueError('existing development surface; retain partial attempt rather than overwrite')
        evaluate_surface(output,surface,native_cache=native_cache)
    write_json(output/'development_complete.json',{'surfaces':['validation','development'],
                'candidate_selection':'predeclared_not_selected_from_outcomes'},exclusive=True)


def confirm(output:Path,native_cache:Path,freeze_commit:str):
    cpu_environment();opened=open_confirmation(ROOT,output,freeze_commit)
    write_json(output/'confirmation_environment.json',cpu_environment(),exclusive=True)
    index=load_ancestry(output)
    for name in ('fit','validation','development'):
        ds=load_dataset(output,name);index=ExposureIndex((*index.sources,ds.source))
    # Both sets are identity-filtered before a new holdout prediction is observed.
    for name in ('confirmation','shift'):
        ds=generate_dataset(output,name,index);index=ExposureIndex((*index.sources,ds.source))
    for surface in ('confirmation','shift'):
        verify_freeze(ROOT,output)
        evaluate_surface(output,surface,native_cache=native_cache)
        print('HOLDOUT_SURFACE_COMPLETE',surface,flush=True)
    verify_freeze(ROOT,output)
    write_json(output/'confirmation_complete.json',{'freeze_sha256':opened['freeze_sha256'],
        'data_manifests':{name:file_sha256(output/'data'/f'{name}.json') for name in ('confirmation','shift')},
        'surfaces':['confirmation','shift'],'post_confirmation_selection':False},exclusive=True)


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage',choices=['prepare','fit','develop','freeze','confirm','verify-freeze'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--archives',type=Path)
    parser.add_argument('--native-cache',type=Path,default=Path('outputs/m16_native_cache'))
    parser.add_argument('--source-commit');parser.add_argument('--freeze-commit')
    args=parser.parse_args();output=args.out.resolve()
    try:
        if args.stage=='prepare':
            if args.archives is None:parser.error('prepare requires --archives')
            prepare(output,args.archives.resolve())
        elif args.stage=='fit':cpu_environment();fit(output)
        elif args.stage=='develop':develop(output,args.native_cache)
        elif args.stage=='freeze':
            if args.source_commit is None:parser.error('freeze requires --source-commit')
            create_freeze(ROOT,output,args.source_commit)
        elif args.stage=='confirm':
            if args.freeze_commit is None:parser.error('confirm requires --freeze-commit')
            confirm(output,args.native_cache,args.freeze_commit)
        elif args.stage=='verify-freeze':verify_freeze(ROOT,output)
    except (ValueError,OSError,RuntimeError,KeyError) as exc:
        print(f'M16 stage failed without discarding evidence: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2
    print('M16_STAGE_COMPLETE',args.stage,flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
