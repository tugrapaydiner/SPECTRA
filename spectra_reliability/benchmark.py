"""Actual M10 graph and kernel benchmarks; reference wins and losses retained.

No quality claim is made for the undertrained M10 fidelity checkpoint. The dense
FP32 comparator uses the same effective weights but a different accumulation
order. It is not labeled bit-exact; output differences are measured explicitly.
"""
from __future__ import annotations
import json
import resource
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from data.splits import build_reproducible_splits
from deploy.m10_artifact import load_cpu_artifact
from deploy.m10_runtime import CPURecursiveRuntime
from deploy.pack_ternary import unpack_ternary_rows
from .experiment import cpu_environment
from .identity import file_sha256,strict_json,write_json
from .native import NativeCPU,build,pack_ternary
from .runtime import ValidatedCPURecursiveRuntime
from .statistics import latency_summary,paired_bootstrap

M10_ZIP_SHA='74cf5c06d2d06a9239724f97ff8ad5542854eec5857cd13d8ad7286a3a183359'
M10_ARTIFACT_SHA='e40f5d84a60ce21237a9a184c8459588b7603f991195c875ef8dcbf319a270e1'


class DenseCPURecursiveRuntime(CPURecursiveRuntime):
    def __init__(self,artifact):
        super().__init__(artifact);self.dense_weights={};self.dense_bias={}
        for name,entry in self.linears.items():
            codes=unpack_ternary_rows(entry['packed'].numpy(),int(entry['out_features']),int(entry['in_features']))
            self.dense_weights[name]=torch.from_numpy(codes.astype(np.float32))*entry['scale'][:,None]
            self.dense_bias[name]=entry['bias'].clone() if entry['bias'].numel() else None
    def _linear(self,name,x):
        weight=self.dense_weights[name]
        if x.dtype!=torch.float32 or x.device.type!='cpu' or x.requires_grad:raise ValueError('dense comparator is CPU FP32 inference only')
        self._layer_calls[name]+=1;vectors=x.numel()//weight.shape[1]
        self._native_vectors+=vectors;self._native_scalar_products+=vectors*weight.numel()
        return F.linear(x,weight,self.dense_bias[name])
    def work_record(self):
        old=super().work_record()
        return {'dense_fp32_linear_calls':old['native_linear_calls'],'linear_calls_by_layer':old['native_calls_by_layer'],
            'logical_linear_MACs':old['native_scalar_products'],'custom_cpp_linear_calls':0,
            'a8_quantization_calls':old['a8_quantization_calls'],'decode_calls':old['decode_calls'],
            'scope':'same recurrent graph, optimized torch F.linear effective-FP32 weights; not bit-exact accumulation'}
    def backend_report(self):
        return {'backend':'torch_dense_fp32_same_effective_weights','weight_payload_bytes':sum(w.numel()*w.element_size() for w in self.dense_weights.values()),
            'bias_payload_bytes':sum(b.numel()*b.element_size() for b in self.dense_bias.values() if b is not None),
            'original_artifact_retained':True,'packed_only_execution':False,'bit_identity_claimed':False}


def prepare_m10(archive:Path,destination:Path):
    if file_sha256(archive)!=M10_ZIP_SHA:raise ValueError('M10 archive identity mismatch')
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in ('spectra_cpu_v1.pt','data_manifest.json'):
            raw=z.read('experiment/'+name);path=destination/name
            if path.exists() and path.read_bytes()!=raw:raise ValueError('M10 retained source changed')
            if not path.exists():path.write_bytes(raw)
    artifact=destination/'spectra_cpu_v1.pt'
    if file_sha256(artifact)!=M10_ARTIFACT_SHA:raise ValueError('M10 inference artifact identity mismatch')
    manifest=strict_json((destination/'data_manifest.json').read_bytes())
    # Split generation uses independent RNG streams. Verify that omitting other
    # split generation still gives the exact retained test identities.
    ds,actual=build_reproducible_splits(manifest['task'],{'train':0,'validation':0,'test':128},manifest['seed'],
        generator_kwargs=manifest['generator_kwargs'],task_scope=manifest['task_scope'],official_benchmark=manifest['official_benchmark'])
    for expected,got in zip(manifest['splits']['test']['examples'],actual['splits']['test']['examples']):
        for field in ('fingerprint','id','group_id'):
            if expected[field]!=got[field]:raise ValueError('M10 test reconstruction failed: '+field)
    return artifact,ds['test'].inputs


@torch.inference_mode()
def microbench(native:NativeCPU,destination:Path):
    rng=np.random.default_rng(160601);rows=[];fidelity=[];memory=[]
    for hidden,out,vectors in ((48,48,81),(48,192,81),(192,48,81),(64,256,16),(256,64,16),(512,512,1)):
        codes=rng.integers(-1,2,(out,hidden),dtype=np.int8);packed=pack_ternary(codes)
        scales=rng.uniform(.01,1,out).astype(np.float32);bias=rng.normal(size=out).astype(np.float32)
        x=rng.normal(size=(vectors,hidden)).astype(np.float32);handle=native.weight(packed,scales,bias,hidden)
        xt=torch.from_numpy(x);wt=torch.from_numpy(codes.astype(np.float32)*scales[:,None]);bt=torch.from_numpy(bias)
        functions={'checked_scalar':lambda:native.checked_linear(x,packed,scales,bias,hidden=hidden),
            'validated_scalar':lambda:handle.linear(x),
            'dense_fp32':lambda:F.linear(xt,wt,bt).numpy()}
        if native.has_avx2:functions['validated_avx2']=lambda:handle.linear(x,avx2=True)
        expected=functions['checked_scalar']()
        for name,fn in functions.items():
            got=fn();same=bool(np.array_equal(got.view(np.uint32),expected.view(np.uint32)))
            if name!='dense_fp32' and not same:raise ValueError('native kernel bit identity failed')
            fidelity.append({'hidden':hidden,'out':out,'vectors':vectors,'backend':name,'bit_identical':same,
                'max_abs_error':float(np.max(np.abs(got-expected)))})
            for _ in range(3):fn()
        for round_index in range(21):
            for name in rng.permutation(list(functions)):
                t=time.perf_counter_ns();cpu=time.process_time_ns()
                for _ in range(3):functions[name]()
                cpu_ns=time.process_time_ns()-cpu;ns=time.perf_counter_ns()-t
                rows.append({'hidden':hidden,'out':out,'vectors':vectors,'round':round_index,'backend':str(name),
                    'iterations':3,'wall_ns_total':ns,'process_cpu_ns_total':cpu_ns,'per_call_ms':ns/3e6,
                    'MACs_per_call':vectors*hidden*out,'arithmetic_ops_per_call':2*vectors*hidden*out,
                    'scope':'available input vectors in one linear; not sequential recurrence'})
        memory.append({'hidden':hidden,'out':out,'vectors':vectors,**handle.memory});handle.close()
    raw=destination/'kernel_rows.jsonl'
    with raw.open('x') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True)+'\n')
    summaries=[]
    for shape in sorted({(r['hidden'],r['out'],r['vectors']) for r in rows}):
        for backend in sorted({r['backend'] for r in rows}):
            values=[r['per_call_ms'] for r in rows if (r['hidden'],r['out'],r['vectors'])==shape and r['backend']==backend]
            if values:summaries.append({'hidden':shape[0],'out':shape[1],'vectors':shape[2],'backend':backend,**latency_summary(values)})
    result={'raw_sha256':file_sha256(raw),'fidelity':fidelity,'memory':memory,'summaries':summaries,
        'physical_energy_joules':None,'cache_residency_established':False}
    write_json(destination/'kernel_summary.json',result,exclusive=True);return result


@torch.inference_mode()
def full_graph(archive:Path,destination:Path,native:NativeCPU):
    artifact,inputs=prepare_m10(archive,destination/'source');warm=inputs[-4:];inputs=inputs[:64]
    loaded=load_cpu_artifact(artifact);setup=[];runtimes={}
    factories={'original_checked_scalar':lambda:CPURecursiveRuntime(loaded),
        'validated_scalar':lambda:ValidatedCPURecursiveRuntime(loaded,native,avx2=False),
        'dense_fp32':lambda:DenseCPURecursiveRuntime(loaded)}
    if native.has_avx2:factories['validated_avx2']=lambda:ValidatedCPURecursiveRuntime(loaded,native,avx2=True)
    for name,factory in factories.items():
        start=time.perf_counter_ns();runtimes[name]=factory();setup.append({'backend':name,'construction_ms':(time.perf_counter_ns()-start)/1e6})
    def solve(runtime,x):
        result=runtime.forward(x)
        ok=native.sudoku_valid(x[0].contiguous().numpy(),result.answer[0].contiguous().numpy(),3)
        return result,ok
    first=[]
    for name,runtime in runtimes.items():
        start=time.perf_counter_ns();solve(runtime,torch.from_numpy(warm[0:1]))
        first.append({'backend':name,'first_solve_ms':(time.perf_counter_ns()-start)/1e6,
            'scope':'first call in this already-imported process; may include native extension load/build; not fresh-process cold start'})
        for row in warm:solve(runtime,torch.from_numpy(row)[None])
    rng=np.random.default_rng(160602);rows=[];fidelity={};reference={}
    for i,row in enumerate(inputs):
        ref=runtimes['original_checked_scalar'].forward(torch.from_numpy(row)[None])
        reference[i]=ref.logits.clone()
    raw=destination/'full_solve_rows.jsonl'
    with raw.open('x') as fh:
        for round_index in range(5):
            for i,row in enumerate(inputs):
                x=torch.from_numpy(row)[None]
                for name in rng.permutation(list(runtimes)):
                    start=time.perf_counter_ns();cpu=time.process_time_ns();result,valid=solve(runtimes[name],x)
                    cpu_ns=time.process_time_ns()-cpu;wall_ns=time.perf_counter_ns()-start
                    expected=reference[i];bits=torch.equal(result.logits.view(torch.int32),expected.view(torch.int32))
                    answer_equal=torch.equal(result.answer,expected.argmax(-1))
                    if name!='dense_fp32' and not bits:raise ValueError('full native graph bit identity failed')
                    entry={'backend':str(name),'puzzle_index':i,'round':round_index,'wall_ns':wall_ns,'process_cpu_ns':cpu_ns,
                        'latency_ms':wall_ns/1e6,'valid':bool(valid),'answer':result.answer[0].tolist(),'work':result.work,
                        'logits_bit_identical':bits,'answer_equal':answer_equal,
                        'max_abs_logit_error':float((result.logits-expected).abs().max()),'physical_energy_joules':None}
                    rows.append(entry);fh.write(json.dumps(entry,sort_keys=True,allow_nan=False)+'\n')
    summaries={};arrays={}
    for name,runtime in runtimes.items():
        rr=[r for r in rows if r['backend']==name];arrays[name]=np.asarray([[np.mean([r['latency_ms'] for r in rr if r['puzzle_index']==i]) for i in range(64)]])
        summaries[name]={'wall':latency_summary([r['latency_ms'] for r in rr]),
            'cpu':latency_summary([r['process_cpu_ns']/1e6 for r in rr]),'valid_of_64':sum(r['valid'] for r in rr if r['round']==0),
            'all_logits_bit_identical':all(r['logits_bit_identical'] for r in rr),'all_answers_equal':all(r['answer_equal'] for r in rr),
            'max_abs_logit_error':max(r['max_abs_logit_error'] for r in rr),'backend_report':runtime.backend_report()}
    comparisons={}
    for name in runtimes:
        comparisons[name]={base:paired_bootstrap(arrays[name],arrays[base],statistic='ratio') for base in ('original_checked_scalar','dense_fp32')}
    for runtime in runtimes.values():
        if hasattr(runtime,'close'):runtime.close()
    result={'source_archive_sha256':M10_ZIP_SHA,'inference_artifact_sha256':M10_ARTIFACT_SHA,
        'puzzles':64,'rounds':5,'summaries':summaries,'comparisons_mean_ratio':comparisons,
        'construction':setup,'first_calls':first,'raw_sha256':file_sha256(raw),
        'reference_targets_used':False,'physical_energy_joules':None,
        'scope':'trained M10 9x9 fidelity workload; not a task-capability or broad efficiency claim',
        'mean_cost_intervals_conditional_on_this_host':True,
        'process_lifetime_peak_rss_kib_all_loaded_arms':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(destination/'full_graph_summary.json',result,exclusive=True);return result


def run_benchmarks(archive:Path,destination:Path,native_cache:Path):
    if destination.exists() and any(destination.iterdir()):raise ValueError('benchmark output already exists; preserve prior measurements')
    destination.mkdir(parents=True,exist_ok=True);write_json(destination/'environment.json',cpu_environment(),exclusive=True)
    t=time.perf_counter_ns();library=build(native_cache);native=NativeCPU(library)
    write_json(destination/'native_load.json',{'build_or_cached_load_ms':(time.perf_counter_ns()-t)/1e6,
        'library_sha256':file_sha256(library),'build_manifest':strict_json(library.with_name('build.json').read_bytes()),
        'has_avx2':native.has_avx2},exclusive=True)
    microbench(native,destination);full_graph(archive,destination,native)
    write_json(destination/'complete.json',{'complete':True,'energy_claim':False,'capability_claim':False},exclusive=True)
