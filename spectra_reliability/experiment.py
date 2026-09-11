"""Frozen M16 recipe and source/data provenance utilities.

Preparation is CPU-only, offline and refuses unverified ancestor reconstruction.
A dataset cache is accepted only after its arrays and every content identity are
checked. Data generation never consults model predictions.
"""
from __future__ import annotations

import copy
import hashlib
import io
import os
import platform
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.splits import build_reproducible_splits
from data import sudoku as original_sudoku
from eval.grounded_targets import assert_frozen_reasoner, tensor_state_sha256
from model.grounded_verifier import EnsembleGroundedStateVerifier
from model.latent_action import StateConditionedLatentActionCodebook, LatentActionCodebook
from scripts.m14_primary_experiment import make_model
from .identity import (canonical_json, digest_parts, file_sha256, input_fingerprint,
                       pair_fingerprint, strict_json, write_json)
from .lineage import (ArtifactNode, ExposureIndex, ManifestSource, ancestral_index,
                      make_manifest_source)
from .sudoku import generate_unique, valid

BASE_COMMIT = "1e29cb10662cb83ba9e28e4d30164fc373a547a5"
PROTOCOL_COMMIT = "464bd19a93d733eeea656971cbefd3675d39101f"
REFINEMENT_COMMIT = "a8c46dcee609f539b2701585a12e72abfe9a7582"
CORE_SEEDS = (1401, 2402)
POOL_PATHS = ((0,), (0,0), (0,0,0), (0,0,0,0), (1,), (2,), (3,),
              (1,0), (2,0), (3,0), (1,0,0,0), (2,0,0,0))
DATA_SPECS = {
    "fit": {"seed":160101,"n":1024,"min_clues":6,"max_clues":10},
    "validation": {"seed":160102,"n":256,"min_clues":6,"max_clues":10},
    "development": {"seed":160103,"n":256,"min_clues":6,"max_clues":10},
    "confirmation": {"seed":160104,"n":512,"min_clues":6,"max_clues":10},
    "shift": {"seed":160105,"n":256,"min_clues":4,"max_clues":5},
}
RECIPE = {
    "format":"spectra.m16_recipe.v1", "base_commit":BASE_COMMIT,
    "protocol_commit":PROTOCOL_COMMIT,"refinement_commit":REFINEMENT_COMMIT,
    "core_seeds":list(CORE_SEEDS),"data":DATA_SPECS,"pool_paths":[list(p) for p in POOL_PATHS],
    "state_precision":"fp32","action_scale":0.5,"matched_targets":["improvement","validity","quality","first_hit"],
    "auxiliary":{"steps":300,"batch_size":64,"lr":0.002,"weight_decay":0.01,"clip":1.0,
                 "first_hit_max_horizon":3,"backbone_init_seed_offset":160210,"batch_seed_offset":160220},
    "first_hit_policy":"frozen_core_identity_action_clamped_decode_v1",
    "decoder":"argmax_restore_givens_v1","checker":"sudoku_exact_v1",
    "new_controller":{"max_depth":4,"transitions":20,"checks":24,"decodes":24,"values":20,"policies":20},
    "legacy_search":{"rollouts":12,"max_depth":4,"c_puct":1.5,"z_storage":"int8"},
    "bootstrap":{"replicates":2000,"seed":160990,"unit":"puzzle_shared_across_two_fixed_cores"},
    "measurement":{"threads":1,"affinity":"one_available_cpu","warmup":4,"timing_rounds":3,
                   "order_seed":160991,"physical_energy_joules":None},
}
ARCHIVES = {
    "m14":{"filename":"spectra_m14_source_evidence.zip","artifact_id":10077794916,
           "sha256":"5729932600743b2cef19d9e0b9baf112ec75b627552af4d30a089266994a2f37"},
    "m15":{"filename":"spectra_m15_evidence.zip","artifact_id":10080887785,
           "sha256":"c49e5c06e0f2822088cb9ce807fcc6bc37da6236962953ccc8c594be01c55b2e"},
}
MANIFEST_MEMBERS = {
    "m14_training":("m14","experiment/manifests/train_validation_development.json"),
    "m14_confirmation":("m14","experiment/manifests/confirmation_1.json"),
    "m15_training":("m15","experiment/manifests/train_validation_development.json"),
    "m15_confirmation":("m15","experiment/manifests/confirmation.json"),
    "m15_shift":("m15","experiment/manifests/shift_low_clue.json"),
}
CORE_FILES = {
    1401:"d5d4769726e3e45822e84a947dd4e8cb007a35374a8a40ae61a786c4c2ba55a4",
    2402:"b43f111af13bc8f7b667c9b7e97558b2b9743f522945193a7a625db77ea194ff",
}
CORE_TENSORS = {
    1401:"4cf4c93ec9d3bd688850394685924cb23d0762a8716eb3e2f42e42befd249f08",
    2402:"00a84312a4fba02e31b1954bddffe10b5933e01eaaae4226eb068490a3f97c0d",
}


def cpu_environment() -> dict[str, Any]:
    """Set explicit CPU execution; thread count is not a measured power budget."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads()!=1:
            raise
    torch.use_deterministic_algorithms(True)
    prior_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None
    if prior_affinity:
        os.sched_setaffinity(0,{prior_affinity[0]})
    cpu_model="unknown"
    try:
        cpu_model=next(line.split(":",1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    except (OSError,StopIteration):
        pass
    return {"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,
            "platform":platform.platform(),"cpu_model":cpu_model,"device":"cpu",
            "torch_threads":torch.get_num_threads(),"torch_interop_threads":torch.get_num_interop_threads(),
            "prior_affinity":prior_affinity,"affinity":sorted(os.sched_getaffinity(0)) if prior_affinity else None,
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "physical_energy_joules":None,"physical_energy_status":"not_measured"}


def verified_archives(root: Path) -> dict[str, zipfile.ZipFile]:
    out={}
    try:
        for tag,spec in ARCHIVES.items():
            path=root/spec['filename']
            if file_sha256(path)!=spec['sha256']:
                raise ValueError(f"source archive identity mismatch: {path}")
            z=zipfile.ZipFile(path)
            if len(set(z.namelist()))!=len(z.namelist()):
                z.close();raise ValueError("archive contains duplicate names")
            out[tag]=z
        return out
    except Exception:
        for z in out.values():z.close()
        raise


def _write_identical(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError(f"existing source/cache differs: {path}")
    else:
        with path.open('xb') as f:f.write(raw)


def reconstruct_ancestors(archive_root: Path, output: Path) -> ExposureIndex:
    """Recover input-only identity from exactly matched historical examples."""
    sources={};reports=[];archives=verified_archives(archive_root)
    try:
        for name,(tag,member) in MANIFEST_MEMBERS.items():
            raw=archives[tag].read(member);old=strict_json(raw)
            _write_identical(output/'sources'/'manifests'/f'{name}.json',raw)
            sizes={split:obj['count'] for split,obj in old['splits'].items()}
            ds,reconstructed=build_reproducible_splits(old['task'],sizes,old['seed'],
                generator_kwargs=old['generator_kwargs'],task_scope=old['task_scope'],official_benchmark=old['official_benchmark'])
            enriched=copy.deepcopy(old);n=0
            for split,part in old['splits'].items():
                expected=part['examples'];got=reconstructed['splits'][split]['examples']
                if len(got)!=len(expected):raise ValueError(f"ancestor count mismatch: {name}:{split}")
                for i,(a,b) in enumerate(zip(expected,got)):
                    for field in ('fingerprint','group_id','id'):
                        if a[field]!=b[field]:raise ValueError(f"ancestor reconstruction mismatch: {name}:{split}:{i}:{field}")
                    x=ds[split].inputs[i];y=ds[split].targets[i];h=ds[split].height;w=ds[split].width
                    if pair_fingerprint(old['task'],x,y,h,w)!=a['fingerprint']:
                        raise ValueError("independent fingerprint implementation disagrees")
                    enriched['splits'][split]['examples'][i]['input_fingerprint']=input_fingerprint(old['task'],x,h,w)
                    n+=1
            enriched['reconstruction']={"original_manifest_sha256":hashlib.sha256(raw).hexdigest(),
                "archive_sha256":ARCHIVES[tag]['sha256'],"all_pair_fingerprints_and_ids_verified":True,
                "input_hashes_are_derived_not_original_fields":True}
            src=make_manifest_source(name,enriched);sources[name]=src
            _write_identical(output/'lineage'/f'{name}.json',src.raw)
            reports.append({"name":name,"original_sha256":hashlib.sha256(raw).hexdigest(),
                            "enriched_sha256":src.sha256,"rows":n,"pair_and_id_exact":True})
        # Capture actual checkpoint bytes; no alternate retraining is permitted.
        checkpoint_inventory = {}
        for seed in CORE_SEEDS:
            for tag,member,filename in [
                ('m14',f'experiment/checkpoints/fp_recursive_dim64_seed{seed}.pt',f'core_{seed}.pt'),
                ('m14',f'experiment/checkpoints/single_pass_seed{seed}.pt',f'baseline_{seed}.pt'),
                ('m15',f'experiment/aux/action_policy_seed{seed}.pt',f'action_{seed}.pt'),
                ('m15',f'experiment/aux/strong_verifier_seed{seed}.pt',f'legacy_verifier_{seed}.pt')]:
                raw=archives[tag].read(member)
                if filename.startswith('core_') and hashlib.sha256(raw).hexdigest()!=CORE_FILES[seed]:
                    raise ValueError("accepted source checkpoint file hash mismatch")
                _write_identical(output/'sources'/filename,raw)
                checkpoint_inventory[filename] = {'sha256':hashlib.sha256(raw).hexdigest(), 'archive':tag, 'member':member}
        _write_identical(output/'sources'/'inventory.json',canonical_json(checkpoint_inventory))
        nodes=[ArtifactNode('m14_cores',digest_parts(*[CORE_FILES[s] for s in CORE_SEEDS]),(),('m14_training','m14_confirmation')),
               ArtifactNode('m15_aux',ARCHIVES['m15']['sha256'],('m14_cores',),('m15_training','m15_confirmation','m15_shift'))]
        index=ancestral_index(nodes,['m15_aux'],sources)
        if not index.complete_input_coverage:raise ValueError("input-only ancestral coverage incomplete")
        report={"schema":"spectra.m16_lineage.v1","sources":reports,"archive_inventory":ARCHIVES,
                "ancestor_rows":len(index.exposures),"unique_pairs":len(index.fingerprint_set),
                "complete_input_only_coverage":True,"symmetry_disjointness_established":False,
                "scope":"all declared M14/M15 historical surfaces, including development-only evaluation exposure"}
        _write_identical(output/'lineage'/'inventory.json',canonical_json(report))
        return index
    finally:
        for z in archives.values():z.close()


def load_ancestry(output: Path) -> ExposureIndex:
    inventory=strict_json((output/'lineage'/'inventory.json').read_bytes())
    return ExposureIndex(ManifestSource.from_path(row['name'],output/'lineage'/f"{row['name']}.json",row['enriched_sha256'])
                         for row in inventory['sources'])


@dataclass(frozen=True)
class Dataset:
    name: str
    inputs: np.ndarray
    targets: np.ndarray
    ids: tuple[str,...]
    source: ManifestSource


def load_dataset(output: Path,name: str) -> Dataset:
    path=output/'data'/f'{name}.json';raw=path.read_bytes();manifest=strict_json(raw)
    spec=DATA_SPECS[name]
    if manifest.get('spec')!=spec:raise ValueError("cached dataset differs from frozen specification")
    data_path=output/'data'/f'{name}.npz'
    if file_sha256(data_path)!=manifest['arrays_sha256']:raise ValueError("dataset array hash mismatch")
    with np.load(data_path,allow_pickle=False) as arrays:
        if set(arrays.files)!={'inputs','targets'}:raise ValueError("unknown dataset arrays")
        x=arrays['inputs'];y=arrays['targets']
    if x.dtype!=np.int64 or y.dtype!=np.int64 or x.shape!=(spec['n'],16) or y.shape!=x.shape:
        raise ValueError("invalid dataset arrays")
    rows=manifest['splits'][name]['examples']
    if len(rows)!=len(x):raise ValueError("dataset manifest length mismatch")
    for i,row in enumerate(rows):
        if (pair_fingerprint('sudoku',x[i],y[i],4,4)!=row['fingerprint'] or
            input_fingerprint('sudoku',x[i],4,4)!=row['input_fingerprint'] or not valid(x[i],y[i],2)):
            raise ValueError("cached dataset content/checker mismatch")
    src=ManifestSource(name,hashlib.sha256(raw).hexdigest(),raw);src.exposures()
    return Dataset(name,x,y,tuple(r['id'] for r in rows),src)


def generate_dataset(output: Path,name: str,index: ExposureIndex) -> Dataset:
    """Deterministic rejection uses identities and validity only, never scores."""
    if not index.complete_input_coverage:raise ValueError("input-only ancestry required before generation")
    spec=DATA_SPECS[name];data_dir=output/'data';data_dir.mkdir(parents=True,exist_ok=True)
    if (data_dir/f'{name}.json').exists():
        ds=load_dataset(output,name);index.require_disjoint(ds.source,require_input_coverage=True);return ds
    if (data_dir/f'{name}.npz').exists():raise ValueError("incomplete dataset output; preserve this attempt rather than overwrite")
    rng=random.Random(spec['seed']);inputs=[];targets=[];rows=[];seen_pairs=set();seen_inputs=set();rejections=[];draws=0
    start=time.perf_counter()
    while len(inputs)<spec['n']:
        if draws>spec['n']*100:raise RuntimeError("dataset rejection limit exceeded")
        clues=rng.randint(spec['min_clues'],spec['max_clues']);x,y=generate_unique(2,clues,rng);draws+=1
        fp=pair_fingerprint('sudoku',x,y,4,4);inp=input_fingerprint('sudoku',x,4,4)
        ancestors=index.matches(fp,fp,inp)
        if ancestors or fp in seen_pairs or inp in seen_inputs:
            rejections.append({"draw":draws,"fingerprint":fp,"input_fingerprint":inp,
                "reason":"ancestor" if ancestors else "within_split_duplicate",
                "ancestor_ids":[e.row_id for e in ancestors]});continue
        # An implementation independent of this generator verifies uniqueness and all clues.
        xx=np.array(x,dtype=np.int64).reshape(4,4);yy=np.array(y,dtype=np.int64).reshape(4,4)
        if not original_sudoku.is_solved(yy,2) or not original_sudoku.respects_clues(xx,yy,2) or original_sudoku.count_solutions(xx,2,2)!=1:
            raise ValueError("independent generated-label validity/uniqueness failure")
        i=len(inputs);seen_pairs.add(fp);seen_inputs.add(inp);inputs.append(x);targets.append(y)
        rows.append({"id":digest_parts('spectra.m16',name,str(spec['seed']),str(i),fp)[:24],
            "fingerprint":fp,"group_id":fp,"input_fingerprint":inp,"difficulty":{"clues":clues},"draw":draws})
    data_path=data_dir/f'{name}.npz'
    with data_path.open('xb') as f:np.savez_compressed(f,inputs=np.asarray(inputs,dtype=np.int64),targets=np.asarray(targets,dtype=np.int64))
    manifest={"schema_version":1,"task":"sudoku","spec":spec,"generator":"stdlib_random_mrv_unique_deletion_v1",
        "symmetry_disjointness_established":False,"arrays_sha256":file_sha256(data_path),
        "splits":{name:{"count":len(rows),"examples":rows}},"draws":draws,"rejections":rejections,
        "generation_seconds":time.perf_counter()-start,"predictions_used_for_generation":False}
    src=make_manifest_source(name,manifest);audit=index.require_disjoint(src,require_input_coverage=True)
    _write_identical(data_dir/f'{name}.json',src.raw);write_json(data_dir/f'{name}_audit.json',audit,exclusive=True)
    return load_dataset(output,name)


def load_core(output: Path,seed: int):
    path=output/'sources'/f'core_{seed}.pt'
    if file_sha256(path)!=CORE_FILES[seed]:raise ValueError("core file identity mismatch")
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['format']!='spectra.m14_model' or payload['version']!=1 or payload['seed']!=seed or payload['kind']!='fp_recursive_dim64':
        raise ValueError("invalid source core checkpoint")
    model=make_model('fp_recursive_dim64',seed);model.load_state_dict(payload['model_state'],strict=True);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    if tensor_state_sha256(model)!=CORE_TENSORS[seed]:raise ValueError("source tensor identity mismatch")
    assert_frozen_reasoner(model)
    return model


def load_auxiliary_sources(output: Path,seed: int):
    inventory=strict_json((output/'sources'/'inventory.json').read_bytes())
    for name in (f'action_{seed}.pt',f'legacy_verifier_{seed}.pt'):
        if file_sha256(output/'sources'/name)!=inventory[name]['sha256']:
            raise ValueError('auxiliary source-file identity mismatch')
    action_payload=torch.load(output/'sources'/f'action_{seed}.pt',map_location='cpu',weights_only=True)
    if action_payload['format']!='spectra.m15_action' or action_payload['core_seed']!=seed:raise ValueError("invalid action source")
    policy=StateConditionedLatentActionCodebook(dim=64,num_tokens=5,n_actions=4,scale=0.5,hidden_dim=64)
    policy.load_state_dict(action_payload['state'],strict=True)
    if not torch.equal(policy.directions,action_payload['directions']):raise ValueError("action direction copies disagree")
    uniform=LatentActionCodebook(dim=64,n_actions=4,scale=0.5)
    with torch.no_grad():uniform.directions.copy_(policy.directions);uniform.prior_logits.zero_()
    vf_payload=torch.load(output/'sources'/f'legacy_verifier_{seed}.pt',map_location='cpu',weights_only=True)
    if vf_payload['format']!='spectra.m15_verifier' or vf_payload['core_seed']!=seed or vf_payload['strength']!='strong':
        raise ValueError("invalid legacy verifier source")
    legacy=EnsembleGroundedStateVerifier(num_tokens=5,dim=64,n_members=3,n_layers=1,heads=4,max_grid_size=8,act_bits=8,include_y=True)
    legacy.load_state_dict(vf_payload['state'],strict=True)
    for module in (policy,uniform,legacy):
        module.eval()
        for parameter in module.parameters():parameter.requires_grad_(False)
    return policy,uniform,legacy


def prepare(output: Path,archive_root: Path) -> None:
    output.mkdir(parents=True,exist_ok=True)
    if (output/'confirmation_opened.json').exists():raise ValueError("confirmation is already open; preparation cannot mutate this run")
    _write_identical(output/'recipe.json',canonical_json(RECIPE))
    env=cpu_environment()
    if not (output/'prepare_environment.json').exists():write_json(output/'prepare_environment.json',env,exclusive=True)
    index=reconstruct_ancestors(archive_root,output)
    for name in ('fit','validation','development'):
        ds=generate_dataset(output,name,index)
        index=ExposureIndex((*index.sources,ds.source))
    write_json(output/'prepare_complete.json',{"pass":True,"recipe_sha256":file_sha256(output/'recipe.json'),
        "data_manifests":{name:file_sha256(output/'data'/f'{name}.json') for name in ('fit','validation','development')},
        "confirmation_generated":False})
