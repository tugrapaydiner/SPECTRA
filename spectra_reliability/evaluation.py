"""M16 fixed-pool and online evaluations with explicit value consumers.

All final semantic decisions are independent of reference solutions. Stored
reference targets are used only by dataset integrity checks, never by solvers.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from data import sudoku as original_sudoku
from eval.grounded_targets import tensor_state_sha256
from model.verifier import sudoku_correct, sudoku_score
from scripts.m14_primary_experiment import make_model
from scripts.m14_attempt5_dual_stream_semantic_exit import dual_stream_semantic_exit_solve
from scripts.m15_mechanism_ablations import M15Search
from .budget import Budget
from .experiment import (CORE_SEEDS, CORE_TENSORS, POOL_PATHS, RECIPE, Dataset,
                         load_core, load_auxiliary_sources, load_dataset)
from .identity import file_sha256, strict_json, write_json
from .native import NativeCPU, build
from .runtime import semantic_exit_native
from .search import ReliableSearch, Task
from .semantics import (CheckedHorizonEvaluator, FirstHitFamilyContract, TargetKind)
from .statistics import probability_metrics, pool_selection, paired_bootstrap, latency_summary
from .sudoku import all_four_by_four
from .targets import build_pool, predict_pool, load_targets, clamp_decode, HORIZON, POLICY

TIMING_PUZZLES=128


class TargetMCTS(M15Search):
    """Same serial MCTS tree mechanics, explicit matched target and consumer."""
    def __init__(self,*args,target_evaluator,**kwargs):
        self.target_evaluator=target_evaluator;self._current_selection={}
        super().__init__(*args,**kwargs)

    def _reset_search_state(self,mode):
        super()._reset_search_state(mode);self._current_selection={}
        self.last_search_stats.update({'matched_target':self.target_evaluator.target,
            'mismatched_improvement_control':self.target_evaluator.target=='improvement',
            'selection_target':'current_validity' if self.target_evaluator.target=='first_hit' else self.target_evaluator.target,
            'evaluator_member_count':1,'horizon_queries':{}})

    def _value(self,x,node):
        target=self.target_evaluator.target;z=node.latent()
        if target=='first_hit':
            horizon=min(HORIZON,max(0,self.max_depth-node.depth))
            probs=self.target_evaluator.first_hit_probabilities(x,node.y,z,continuation_policy=POLICY)
            self._current_selection[tuple(node.path)]=float(probs[:,0].item())
            self.last_search_stats['horizon_queries'][str(horizon)]=self.last_search_stats['horizon_queries'].get(str(horizon),0)+1
            return float(probs[:,:horizon+1].sum(-1).clamp(0,1).item())
        return float(self.target_evaluator.current_value(x,node.y,z,target=target).item())

    def _consider_best(self,node,value):
        if self.target_evaluator.target=='first_hit':
            # Exploration/backup value is NOT silently reused to rank current answers.
            value=self._current_selection[tuple(node.path)]
        return super()._consider_best(node,value)

    def search_batched(self,*args,**kwargs):
        raise NotImplementedError('M16 target isolation is serial; no unvalidated batched value adapter')


@dataclass
class SolverResult:
    answer: torch.Tensor
    valid: bool
    work: dict[str,Any]


@dataclass(frozen=True)
class _State:
    y: torch.Tensor
    z: torch.Tensor
    depth: int


@torch.inference_mode()
def reliable_solve(core,policy,first_hit,x,native:NativeCPU) -> SolverResult:
    # The outer complete-solve timer includes this embedding and all construction.
    embedding=core.token_embed(x)+core.encode_positions(x,4,4)
    root=_State(torch.zeros_like(embedding),torch.zeros_like(embedding),0)
    puzzle=tuple(x[0].tolist());xn=x[0].contiguous().numpy()
    family=FirstHitFamilyContract('sudoku',first_hit.core_sha256,RECIPE['decoder'],RECIPE['checker'],POLICY,HORIZON)
    def predict(state,query):
        horizon=0 if query.target is TargetKind.CURRENT_VALIDITY else query.horizon
        return float(first_hit.within_horizon(x,state.y,state.z,horizon,continuation_policy=POLICY).item())
    evaluator=CheckedHorizonEvaluator(family,family,lambda state:min(HORIZON,max(0,4-state.depth)),predict)
    def transition(state,action):
        y,z=core.recursive_cycle(embedding,state.y,policy.apply_action(state.z,action))
        return _State(y,z,state.depth+1)
    search=ReliableSearch(task=Task('sudoku',RECIPE['checker'],puzzle),
        checker=lambda px,a:native.sudoku_valid(xn,np.asarray(a,dtype=np.int64),2),
        evaluator=evaluator,actions=lambda state:range(4),transition=transition,
        decode=lambda state:clamp_decode(core,x,state.y)[0].tolist(),max_depth=4,max_actions=4)
    spec=RECIPE['new_controller']
    result=search.run(root,Budget(**{key:spec[key] for key in ('transitions','checks','decodes','values','policies')}),
                      baseline_actions=(0,0,0,0),stop_on_valid=True)
    if result.status=='callback_error':raise RuntimeError(result.error)
    if result.answer is None:raise RuntimeError('configured baseline/search returned no candidate')
    work=result.stats|{'status':result.status,'reason':result.error,'horizon_query_counts':evaluator.query_counts,
        'value_family':{'policy':POLICY,'max_horizon':HORIZON},'target_used':False,
        'embedding_and_construction_in_outer_timer':True,'embedding_excluded_from_internal_ledger':True,
        'learned_action_prior_used':False,'frozen_learned_directions_used':True}
    return SolverResult(torch.tensor([result.answer],dtype=torch.long),result.valid,work)


def _baseline_model(output:Path,seed:int):
    path=output/'sources'/f'baseline_{seed}.pt';inventory=strict_json((output/'sources'/'inventory.json').read_bytes())
    if file_sha256(path)!=inventory[path.name]['sha256']:raise ValueError('baseline source hash mismatch')
    p=torch.load(path,map_location='cpu',weights_only=True)
    if p['format']!='spectra.m14_model' or p['kind']!='single_pass' or p['seed']!=seed:raise ValueError('invalid baseline')
    m=make_model('single_pass',seed);m.load_state_dict(p['model_state'],strict=True);m.eval()
    for q in m.parameters():q.requires_grad_(False)
    return m


def solvers(output:Path,seed:int,core,policy,uniform,legacy,models,native:NativeCPU) -> dict[str,Callable]:
    base=_baseline_model(output,seed)
    universe=np.asarray(all_four_by_four(),dtype=np.int64)
    def final(answer,x,work):
        ok=native.sudoku_valid(x[0].contiguous().numpy(),answer[0].contiguous().numpy(),2)
        return SolverResult(answer,ok,work|{'target_used':False,'final_native_semantic_check':True})
    def original(x):
        answer,work=dual_stream_semantic_exit_solve(core,x,4)
        return SolverResult(answer,bool(work['final_semantic']),work)
    def native_exit(x):
        answer,work=semantic_exit_native(core,x,4,native)
        return SolverResult(answer,bool(work['final_semantic']),work)
    def fixed(x):
        emb=core.token_embed(x)+core.encode_positions(x,4,4);y=torch.zeros_like(emb);z=torch.zeros_like(emb)
        for _ in range(4):y,z=core.recursive_cycle(emb,y,z)
        return final(clamp_decode(core,x,y),x,{'recursive_cycles':4,'block_applications':8,'decode_calls':1,'semantic_checks':1})
    def single(x):
        logits,_=base(x,height=4,width=4);answer=torch.where(x!=0,x,logits.argmax(-1))
        return final(answer,x,{'block_applications':2,'decode_calls':1,'semantic_checks':1})
    def exact(x):
        answer=original_sudoku.solve(x[0].numpy().reshape(4,4),2)
        if answer is None:raise RuntimeError('generated unique puzzle became unsolvable')
        return final(torch.from_numpy(answer.reshape(1,16)),x,{'method':'numpy_mrv_backtracking','semantic_checks':1})
    def finite(x):
        # Strong context: the entire finite solution universe, not a learned model.
        xx=x[0].numpy();mask=xx!=0;matches=np.flatnonzero(np.all((universe==xx)|~mask,axis=1))
        if len(matches)!=1:raise RuntimeError('expected a unique finite-universe solution')
        return final(torch.from_numpy(universe[matches[0]].copy()[None]),x,
            {'method':'complete_288_board_filter','stored_solution_bytes':int(universe.nbytes),'semantic_checks':1})
    spec=RECIPE['legacy_search']
    def searcher(verifier,codebook,guard=False,target=None):
        kw=dict(model=core,energy_verifier=verifier,action_codebook=codebook,height=4,width=4,
                n_rollouts=spec['rollouts'],max_depth=spec['max_depth'],c_puct=spec['c_puct'],
                uncertainty_beta=0.,latent_vq=None,terminal_guard=guard)
        return M15Search(**kw) if target is None else TargetMCTS(**kw,target_evaluator=target)
    def bind_mcts(search):
        def solve(x):
            node=search.search(x);answer=torch.where(x!=0,x,search.decode(node))
            work=copy.deepcopy(search.last_search_stats)
            work['evaluator_member_count']=3 if search.verifier is legacy else 1
            return final(answer,x,work)
        return solve
    result={'semantic_exit_original':original,'semantic_exit_native':native_exit,'fixed4_native':fixed,
        'single_pass_native':single,'symbolic_exact':exact,'finite_universe_filter':finite,
        'mcts_legacy_learned':bind_mcts(searcher(legacy,policy)),
        'mcts_legacy_uniform':bind_mcts(searcher(legacy,uniform)),
        'mcts_legacy_guard':bind_mcts(searcher(legacy,policy,True)),
        'baseline_first_best_first_survival':lambda x:reliable_solve(core,policy,models['first_hit'],x,native)}
    for target in ('improvement','validity','quality','first_hit'):
        result[f'mcts_matched_{target}']=bind_mcts(searcher(models[target],policy,target=models[target]))
    return result


@torch.inference_mode()
def pool_batch_fidelity(core,pool) -> dict[str,Any]:
    disagreements=0;valid_disagreements=0;max_y_error=0.;max_z_error=0.
    for i in range(pool.puzzle_count):
        x=pool.x[i*len(POOL_PATHS):i*len(POOL_PATHS)+1]
        emb=core.token_embed(x)+core.encode_positions(x,4,4);y=torch.zeros_like(emb);z=torch.zeros_like(emb)
        for depth in range(4):
            y,z=core.recursive_cycle(emb,y,z);flat=i*len(POOL_PATHS)+depth
            max_y_error=max(max_y_error,float((y[0]-pool.y[flat]).abs().max()))
            max_z_error=max(max_z_error,float((z[0]-pool.z[flat]).abs().max()))
            answer=clamp_decode(core,x,y)
            disagreements+=int(not torch.equal(answer[0],pool.answers[flat]))
            valid_disagreements+=int(bool(sudoku_correct(x,answer,2).bool().item())!=bool(pool.validity[flat]))
    return {'compared_states':pool.puzzle_count*4,'answer_disagreements':disagreements,
            'validity_disagreements':valid_disagreements,'max_abs_y_error':max_y_error,'max_abs_z_error':max_z_error,
            'bit_identity_assumed':False,'baseline_cycle_graph_unchanged':True}


def pool_array_payload(dataset:Dataset,pool:Pool,predictions:dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    """Disjoint truth/prediction namespaces prevent label overwrite on export."""
    shape=(len(dataset.inputs),len(POOL_PATHS))
    arrays={'inputs':dataset.inputs,'answers':pool.answers.numpy().reshape(*shape,16),
        'label_validity':pool.validity.numpy().astype(bool).reshape(shape),
        'label_quality':pool.quality.numpy().reshape(shape),
        'label_improvement':pool.improvement.numpy().reshape(shape),
        'label_first_hit':pool.first_hit.numpy().reshape(shape)}
    for key,value in predictions.items():
        name='prediction_'+key
        if name in arrays or not isinstance(key,str) or not key:
            raise ValueError('invalid or colliding pool prediction name')
        array=np.asarray(value)
        if array.size!=shape[0]*shape[1] or not np.isfinite(array).all():
            raise ValueError('invalid pool prediction shape/values')
        arrays[name]=array.reshape(shape)
    return arrays


@torch.inference_mode()
def evaluate_fixed_pool(output:Path,surface:str,dataset:Dataset,seed:int,core,policy,legacy,models) -> dict[str,Any]:
    destination=output/'evaluation'/surface;destination.mkdir(parents=True,exist_ok=True)
    summary_path=destination/f'pool_seed{seed}_summary.json'
    if summary_path.exists():raise ValueError('fixed-pool result already exists; no silent overwrite')
    pool=build_pool(core,policy,dataset);predictions=predict_pool(models,legacy,pool)
    valid=pool.validity.numpy().astype(bool).reshape(len(dataset.inputs),len(POOL_PATHS))
    quality=pool.quality.numpy();improvement=pool.improvement.numpy();first=pool.first_hit.numpy()
    selections={key:pool_selection(valid,scores.reshape(valid.shape)) for key,scores in predictions.items()}
    selections['exact_validity']=pool_selection(valid,valid.astype(float))
    selections['exact_quality']=pool_selection(valid,quality.reshape(valid.shape))
    selections['oracle_improvement_mismatched']=pool_selection(valid,improvement.reshape(valid.shape))
    metrics={}
    for key,scores in predictions.items():
        target=improvement if key in ('improvement','legacy_improvement') else (quality if key=='quality' else
            (first<=int(key[-1])).astype(float) if key.startswith('first_hit_h') else pool.validity.numpy())
        metrics[key]={'declared_target':probability_metrics(target,scores),
                      'absolute_validity':probability_metrics(pool.validity.numpy(),scores)}
    arrays=pool_array_payload(dataset,pool,predictions)
    raw_path=destination/f'pool_seed{seed}.npz'
    with raw_path.open('xb') as f:np.savez_compressed(f,**arrays)
    summary={'surface':surface,'seed':seed,'dataset_manifest_sha256':dataset.source.sha256,
        'raw_arrays_sha256':file_sha256(raw_path),'array_schema':'spectra.m16_pool.v2','core_tensor_sha256':tensor_state_sha256(core),
        'work':pool.work,'metrics':metrics,'selections':selections,'b1_fidelity':pool_batch_fidelity(core,pool),
        'scope':'fixed identical candidate pools; not an online compute frontier',
        'future_horizon_used_for_immediate_selection_is_a_mismatched_diagnostic':True}
    write_json(summary_path,summary,exclusive=True)
    return summary


@torch.inference_mode()
def evaluate_online(output:Path,surface:str,dataset:Dataset,seed:int,core,policy,uniform,legacy,models,native:NativeCPU) -> dict[str,Any]:
    destination=output/'evaluation'/surface;destination.mkdir(parents=True,exist_ok=True)
    path=destination/f'online_seed{seed}.jsonl'
    if path.exists():raise ValueError('online surface already opened; preserve partial runs rather than overwrite')
    solver_map=solvers(output,seed,core,policy,uniform,legacy,models,native)
    warm=load_dataset(output,'validation').inputs[-4:]
    for solve in solver_map.values():
        for row in warm:solve(torch.from_numpy(row)[None])
    rng=np.random.default_rng(RECIPE['measurement']['order_seed']+seed)
    timed_n=min(TIMING_PUZZLES,len(dataset.inputs));rows=[];reference={};native_steps_agree=True
    with path.open('x') as fh:
        for round_index in range(RECIPE['measurement']['timing_rounds']):
            n=len(dataset.inputs) if round_index==0 else timed_n
            for i in range(n):
                x=torch.from_numpy(dataset.inputs[i:i+1]);case={}
                for name in rng.permutation(list(solver_map)):
                    start=time.perf_counter_ns();cpu=time.process_time_ns()
                    result=solver_map[name](x)
                    cpu_ns=time.process_time_ns()-cpu;wall_ns=time.perf_counter_ns()-start
                    answer=result.answer[0].tolist()
                    # Independent full check OUTSIDE timing detects checker/adapter errors.
                    independently_valid=bool(sudoku_correct(x,result.answer,2).bool().item())
                    if independently_valid!=result.valid:raise ValueError(f'independent checker mismatch: {surface}:{seed}:{i}:{name}')
                    key=(i,str(name))
                    if round_index==0:reference[key]=(answer,result.valid)
                    elif reference[key]!=(answer,result.valid):raise ValueError('deterministic solver changed its answer across timing rounds')
                    row={'surface':surface,'core_seed':seed,'example_id':dataset.ids[i],'example_index':i,
                         'config_id':str(name),'round':round_index,'primary_timing_sample':i<timed_n,
                         'semantic_success':int(result.valid),'answer':answer,
                         'wall_ns':wall_ns,'process_cpu_ns':cpu_ns,'latency_ms':wall_ns/1e6,
                         'work':result.work,'solver_reference_target_used':False,
                         'physical_energy_joules':None,'physical_energy_status':'not_measured'}
                    fh.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');rows.append(row);case[str(name)]=row
                if case['semantic_exit_original']['answer']!=case['semantic_exit_native']['answer']:
                    raise ValueError('native checker changed accepted semantic-exit answer')
                if case['semantic_exit_original']['work']['executed_steps']!=case['semantic_exit_native']['work']['executed_steps']:
                    raise ValueError('native checker changed accepted stopping step')
                if case['semantic_exit_native']['semantic_success'] and not case['baseline_first_best_first_survival']['semantic_success']:
                    raise ValueError('valid baseline answer was lost by protected-incumbent controller')
                if i%64==0:fh.flush()
    summaries={}
    for name in solver_map:
        quality_rows=[r for r in rows if r['config_id']==name and r['round']==0]
        timed_rows=[r for r in rows if r['config_id']==name and r['primary_timing_sample']]
        summaries[name]={'quality_rows':len(quality_rows),'valid_answers':sum(r['semantic_success'] for r in quality_rows),
            'strict_success':sum(r['semantic_success'] for r in quality_rows)/len(quality_rows),
            'timing_puzzles':timed_n,'timing_rounds':RECIPE['measurement']['timing_rounds'],
            'wall_latency':latency_summary([r['latency_ms'] for r in timed_rows]),
            'process_cpu_latency':latency_summary([r['process_cpu_ns']/1e6 for r in timed_rows])}
    report={'surface':surface,'core_seed':seed,'dataset_manifest_sha256':dataset.source.sha256,
        'raw_rows_sha256':file_sha256(path),'rows':len(rows),'summaries':summaries,
        'all_answers_stable_across_rounds':True,'native_checker_all_answers_and_stopping_steps_match':True,
        'valid_incumbent_regressions':0,'timing_excludes_output_json_serialization_and_independent_audit':True}
    write_json(destination/f'online_seed{seed}_summary.json',report,exclusive=True)
    return report


def aggregate_surface(output:Path,surface:str,dataset:Dataset) -> dict[str,Any]:
    destination=output/'evaluation'/surface;by_seed={};pool_by_seed={}
    for seed in CORE_SEEDS:
        rows=[strict_json(line) for line in (destination/f'online_seed{seed}.jsonl').read_text().splitlines() if line]
        summary=strict_json((destination/f'online_seed{seed}_summary.json').read_bytes())
        if file_sha256(destination/f'online_seed{seed}.jsonl')!=summary['raw_rows_sha256']:
            raise ValueError('online evidence hash mismatch')
        by_seed[seed]=rows
        pool_by_seed[seed]=strict_json((destination/f'pool_seed{seed}_summary.json').read_bytes())
    configs=sorted({r['config_id'] for r in by_seed[CORE_SEEDS[0]]});arrays={};timing_arrays={};totals={}
    for name in configs:
        aa=[];tt=[];raw_latency=[];raw_cpu=[];per_seed={}
        for seed in CORE_SEEDS:
            rows=[r for r in by_seed[seed] if r['config_id']==name];quality=sorted([r for r in rows if r['round']==0],key=lambda r:r['example_index'])
            if tuple(r['example_id'] for r in quality)!=dataset.ids:raise ValueError('paired puzzle identities differ')
            aa.append([r['semantic_success'] for r in quality]);per_seed[str(seed)]=sum(aa[-1])/len(aa[-1])
            timed=[r for r in rows if r['primary_timing_sample']];n=min(TIMING_PUZZLES,len(dataset.inputs))
            if len(timed)!=n*RECIPE['measurement']['timing_rounds']:raise ValueError('missing primary timing row')
            tt.append([float(np.mean([r['latency_ms'] for r in timed if r['example_index']==i])) for i in range(n)])
            raw_latency.extend(r['latency_ms'] for r in timed);raw_cpu.extend(r['process_cpu_ns']/1e6 for r in timed)
        arrays[name]=np.asarray(aa);timing_arrays[name]=np.asarray(tt)
        totals[name]={'valid_answers':int(arrays[name].sum()),'model_example_rows':int(arrays[name].size),
            'unique_puzzles':len(dataset.inputs),'fixed_core_seeds':list(CORE_SEEDS),
            'strict_success':float(arrays[name].mean()),'seed_specific_success':per_seed,
            'wall_latency':latency_summary(raw_latency),'process_cpu_latency':latency_summary(raw_cpu)}
    baseline='semantic_exit_native';comparisons={}
    for name in configs:
        if name==baseline:continue
        quality=paired_bootstrap(arrays[name],arrays[baseline])
        latency=paired_bootstrap(timing_arrays[name],timing_arrays[baseline],statistic='ratio')
        regressions=int(((arrays[baseline]==1)&(arrays[name]==0)).sum())
        gains=int(((arrays[baseline]==0)&(arrays[name]==1)).sum())
        comparisons[name]={'baseline':baseline,'quality':quality,'mean_latency_ratio':latency,'regressions':regressions,'gains':gains,
            'twenty_percent_mean_cost_matched_quality_gate':bool(quality['point']>=-.01 and quality['ci95'][0]>=-.01 and latency['point']<=.8 and latency['ci95'][1]<1.)}
    pool_comparisons={}
    keys=pool_by_seed[CORE_SEEDS[0]]['selections']
    for key in keys:
        selected=np.asarray([pool_by_seed[seed]['selections'][key]['returned'] for seed in CORE_SEEDS],dtype=float)
        wrong=np.asarray([pool_by_seed[seed]['selections']['improvement']['returned'] for seed in CORE_SEEDS],dtype=float)
        coverage=sum(pool_by_seed[seed]['selections'][key]['covered_puzzles'] for seed in CORE_SEEDS)
        good=int(selected.sum())
        pool_comparisons[key]={'valid_answers':good,'covered_model_examples':coverage,
            'selection_failures_given_coverage':coverage-good,'vs_matched_improvement':paired_bootstrap(selected,wrong)}
    result={'surface':surface,'dataset_manifest_sha256':dataset.source.sha256,'online':totals,
        'comparisons_to_native_semantic_exit':comparisons,'fixed_pool':pool_comparisons,
        'primary_new_controller':'baseline_first_best_first_survival',
        'scope':'two fixed accepted reasoner cores on generated 4x4 Sudoku; no training-seed-population, broad-task or iso-energy claim'}
    write_json(destination/'aggregate.json',result,exclusive=True)
    return result


def evaluate_surface(output:Path,surface:str,*,native_cache:Path) -> dict[str,Any]:
    dataset=load_dataset(output,surface);native=NativeCPU(build(native_cache))
    for seed in CORE_SEEDS:
        print('SURFACE_BEGIN',surface,seed,flush=True)
        core=load_core(output,seed);policy,uniform,legacy=load_auxiliary_sources(output,seed);models=load_targets(output,seed)
        evaluate_fixed_pool(output,surface,dataset,seed,core,policy,legacy,models)
        evaluate_online(output,surface,dataset,seed,core,policy,uniform,legacy,models,native)
        if tensor_state_sha256(core)!=CORE_TENSORS[seed]:raise ValueError('frozen core changed during evaluation')
        print('SURFACE_SEED_DONE',surface,seed,flush=True)
    return aggregate_surface(output,surface,dataset)
